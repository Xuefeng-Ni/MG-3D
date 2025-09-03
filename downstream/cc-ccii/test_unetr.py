# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from functools import partial
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer_unetr import infer
from utils.data_utils import get_loader

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
import math

os.environ['CUDA_VISIBLE_DEVICES'] = "2, 5"
os.environ['MASTER_ADDR'] = 'localhost'
##os.environ['MASTER_PORT'] = '28890'

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
print('Setting resource limit:', str(resource.getrlimit(resource.RLIMIT_NOFILE)))
import monai

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")#"/home/xuefeng/downstream/cc-ccii/runs/logs_UNETR_pre_384_1e-3/model_final.pt"
parser.add_argument("--logdir", default="logs", type=str, help="directory to save the tensorboard logs")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--csv_list", default="./csv/", type=str, help="csv directory")
parser.add_argument("--fold", default=0, type=int, help="fold")
parser.add_argument("--data_dir", default="/home/xuefeng/CC-CCII/CC-CCII_public/", type=str, help="dataset directory")
parser.add_argument("--pretrained_model_name", default=None, type=str, help="pretrained model name",
)

parser.add_argument("--save_checkpoint", default=True, help="save checkpoint during training")
parser.add_argument("--max_epochs", default=100, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=1, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=4, type=int, help="number of workers")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=16, type=int, help="feature size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
# warmup is important !!!
parser.add_argument("--warmup_epochs", default=5, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--use_checkpoint", default=False, help="use gradient checkpointing to save memory")
parser.add_argument("--use_ssl_pretrained", default=False, help="use self-supervised pretrained weights")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")
parser.add_argument("--roi_x", default=16, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=384, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=384, type=int, help="roi size in z direction")

def adapt_position_encoding(model, patch_size=32, after_xy=384, after_z=16, suffix='patch_embedding.position_embeddings'):
    keys = [k for k in model if k.endswith(suffix)]
    assert len(keys) == 1
    key = keys[0]
    origin_pos_embed = model[key]
    origin_dim2 = False
    if len(origin_pos_embed.shape) == 2:
        origin_dim2 = True
        origin_pos_embed = origin_pos_embed.unsqueeze(0)
    grid_before = math.ceil(math.pow((origin_pos_embed.shape[1]), 1/3))
    before = int(grid_before * patch_size)
    assert (before % patch_size) == 0
    grid_after_xy = after_xy // patch_size
    grid_after_z = after_z // patch_size
    assert (after_xy % patch_size) == 0
    embed_dim = origin_pos_embed.shape[-1]

    pos_embed = origin_pos_embed.reshape((grid_before, grid_before, grid_before, embed_dim))
    new_size = (grid_after_xy, grid_after_xy, grid_after_z)
    pos_embed = torch.nn.functional.interpolate(pos_embed.permute((3, 0, 1, 2)).unsqueeze(0),
                                                size=new_size, mode='trilinear')
    pos_embed = pos_embed.squeeze(0).permute((1, 2, 3, 0)).reshape((-1, embed_dim)).unsqueeze(0)
##    pos_embed = torch.cat((origin_pos_embed[0, 0:1, :], pos_embed), dim=0).unsqueeze(0)
    assert pos_embed.shape == (1, grid_after_xy * grid_after_xy * grid_after_z, embed_dim)
    if origin_dim2:
        assert pos_embed.shape[0] == 1
        pos_embed = pos_embed.squeeze(0)
    model[key] = pos_embed
    return model


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    loader = get_loader(args)
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)

    if args.rank == 0:
        os.makedirs(args.logdir, exist_ok=True)
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    pretrained_dir = args.pretrained_dir

##    from model import Swin
##    model = Swin(args)

    from networks import UNETR
    model = UNETR(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        feature_size=args.feature_size,
        hidden_size=args.hidden_size,
        mlp_dim=args.mlp_dim,
        num_heads=args.num_heads,
        pos_embed=args.pos_embed,
        norm_name=args.norm_name,
        conv_block=True,
        res_block=True,
        dropout_rate=args.dropout_rate,
    )
    '''
##    extractor_dict = torch.load('/home/xuefeng/PTUnifier-main/result/task_pretrain_ptunifier-seed0-from_/vit3d_16_no_pseudo_language_no_normalization/checkpoints/epoch=31-step=5792.ckpt', map_location=torch.device('cpu'))
##    extractor_dict = torch.load('/home/xuefeng/PTUnifier-main/result/task_pretrain_ptunifier-seed0-from_/vit3d_16_no_pseudo_language/checkpoints/epoch=79-step=14480.ckpt', map_location=torch.device('cpu'))
##    extractor_dict = torch.load('/home/xuefeng/PTUnifier-main/result/task_pretrain_ptunifier-seed0-from_/vit3d_16_normalization_no_pseudo_language/checkpoints/epoch=93-step=17014.ckpt', map_location=torch.device('cpu'))
##    extractor_dict = torch.load('/home/xuefeng/PTUnifier-main/result/task_pretrain_ptunifier-seed0-from_/vit3d_16_pseudo_language_no_normalization/checkpoints/epoch=85-step=15566.ckpt', map_location=torch.device('cpu'))
##    extractor_dict = torch.load('/home/xuefeng/PTUnifier-main/result/task_pretrain_ptunifier-seed0-from_/vit3d_16_itc_no_norm/checkpoints/epoch=112-step=20453.ckpt', map_location=torch.device('cpu'))
    extractor_dict = torch.load('/home/xuefeng/PTUnifier-main/result/task_pretrain_ptunifier-seed0-from_/vit3d_16_itc_bf_no_norm/checkpoints/epoch=113-step=20634.ckpt', map_location=torch.device('cpu'))
##    extractor_dict = torch.load('/home/xuefeng/PTUnifier-main/result/task_pretrain_ptunifier-seed0-from_/vit3d_16_t3d/checkpoints/last.ckpt', map_location=torch.device('cpu'))

    new_extractor_dict = {}
    extractor_dict = extractor_dict['state_dict']
    for key, value in extractor_dict.items():
        name = key[15:]
        if key[:14] == 'vision_encoder':
            new_extractor_dict['vit.' + name] = value

    new_extractor_dict = adapt_position_encoding(new_extractor_dict,
                            after_xy=384,
                            after_z=16,
                            patch_size=16)
        
    model.load_state_dict(new_extractor_dict, strict=False)
    '''
    '''
    model = monai.networks.nets.ViT(in_channels=args.in_channels, num_classes=args.out_channels, patch_size=16,
                                    img_size=(args.roi_x, args.roi_y, args.roi_z),  
                                    classification=True)
    # from densenet import densenet3d
    # model = densenet3d()
    '''
##    args.resume_ckpt = True
    if args.resume_ckpt:
        model_dict = torch.load(os.path.join(args.pretrained_model_name))["state_dict"]
        model.load_state_dict(model_dict)
        print("Use pretrained weights")

    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    args.checkpoint = '/home/xuefeng/downstream/cc-ccii/runs/logs_UNETR_pre_384_1e-4_pretrain_no_pseudo_no_normalization_3/model.pt'
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

    accuracy = infer(
        model=model,
        val_loader=loader[1],
        args=args,
    )
    return accuracy

logs = set()
def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

if __name__ == "__main__":
    main()
