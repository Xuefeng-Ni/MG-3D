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
from trainer_unetr import run_training
from utils.data_utils import get_loader

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
import math

os.environ['CUDA_VISIBLE_DEVICES'] = "1,3"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '38434'

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
parser.add_argument("--pretrained_model_name", default=None, type=str, help="pretrained model name",)

parser.add_argument("--save_checkpoint", default=True, help="save checkpoint during training")
parser.add_argument("--max_epochs", default=130, type=int, help="max number of training epochs")

parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=3e-4, type=float, help="optimization learning rate")

parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=1, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--local-rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=4, type=int, help="number of workers")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
# warmup is important !!!
parser.add_argument("--warmup_epochs", default=30, type=int, help="number of warmup epochs")

parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--use_checkpoint", default=False, help="use gradient checkpointing to save memory")
parser.add_argument("--use_ssl_pretrained", default=False, help="use self-supervised pretrained weights")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")
parser.add_argument("--roi_x", default=32, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=384, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=384, type=int, help="roi size in z direction")

def main():
    args = parser.parse_args()
    args.noamp = False
    args.amp = not args.noamp
    args.distributed = True
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

    backbone = 'swin_b' # swin_l
    if backbone == 'swin_b':
        from networks.swin_b import SwinTransformer
        from monai.utils import ensure_tuple_rep
        spatial_dims = 3
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        model = SwinTransformer(            
                in_chans=args.in_channels,
                embed_dim=96,
                window_size=window_size,
                patch_size=patch_size,
                depths=(2, 2, 2, 2),
                num_heads=(3, 6, 12, 24),
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                use_checkpoint=False,
                spatial_dims=spatial_dims,
                classification=True,
                num_classes=args.out_channels)
        model_dict = dict(model.state_dict())

        extractor_dict = torch.load('/home/ubuntu/MG-3D-Swin-B.ckpt', map_location=torch.device('cpu'))

        new_extractor_dict = {}
        for key, value in extractor_dict.items():
            name = key[15:]
            if key[:14] == 'vision_encoder':
                new_extractor_dict[name] = value
                
        pretrain_dict = {k: v for k, v in new_extractor_dict.items() if k in model_dict}
        model.load_state_dict(new_extractor_dict, strict=False)
    if backbone == 'swin_l':
        from networks.swin_l import Swin
        model = Swin(args)
        model_dict = dict(model.state_dict())
        
        extractor_dict = torch.load('/home/ubuntu/MG-3D-Swin-L.ckpt', map_location=torch.device('cpu'))

        new_extractor_dict = {}
        model_dict = dict(model.state_dict())
        for key, value in extractor_dict.items():
            name = key[15:]
            if key[:14] == 'vision_encoder':
                new_extractor_dict['swinViT.' + name] = value
            if key[:22] == 'vision_encoder.encoder':
                new_extractor_dict[name] = value
        pretrain_dict = {k: v for k, v in new_extractor_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        model.load_state_dict(pretrain_dict, strict=False)

    args.resume_ckpt = False
    if args.resume_ckpt:
        model_dict = torch.load(os.path.join(args.pretrained_model_name))["state_dict"]
        model.load_state_dict(model_dict)
        print("Use pretrained weights")

    if args.use_ssl_pretrained:
        try:
            model_dict = torch.load("./pretrained_models/UNETR_model_best_acc.pth")
            state_dict = model_dict["state_dict"]
            # fix potential differences in state dict keys from pre-training to
            # fine-tuning
            if "module." in list(state_dict.keys())[0]:
                print("Tag 'module.' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("module.", "")] = state_dict.pop(key)
            if "swin_vit" in list(state_dict.keys())[0]:
                print("Tag 'swin_vit' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
            # We now load model weights, setting param `strict` to False, i.e.:
            # this load the encoder weights (Swin-ViT, SSL pre-trained), but leaves
            # the decoder weights untouched (CNN UNet decoder).
            model.load_state_dict(state_dict, strict=False)
            print("Using pretrained self-supervised Swin UNETR backbone weights !")
        except ValueError:
            raise ValueError("Self-supervised pre-trained weights not available for" + str(args.model_name))

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    best_acc = 0
    start_epoch = 0
    
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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, amsgrad=True)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        args=args,
        scheduler=scheduler,
        start_epoch=start_epoch,
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
