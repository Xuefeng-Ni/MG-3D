"""Jiaxin ZHUANG
Modified on Feb 8, 2024.
"""

import os
from functools import partial

import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data.distributed
import torch.multiprocessing

from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete

from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from networks.net import get_seg_model
from utils.helper import load_config_yaml_args
from utils.downstream_data_utils import get_loader
from utils.misc import print_with_timestamp, setup, cleanup
from utils.default_arguments import get_args
from utils.loss import get_loss_function
from utils.metric import get_metric


def main(rank, world_size, args):
    """Main function.
    """
    setup(args, rank, world_size)
    print_with_timestamp(args)
    device = torch.device(args.device)

    # Dataset.
    train_loader, val_loader, task = get_loader(args)
    args.task = task
    print_with_timestamp(f'Setting task to {args.task}')

    # Model.
    model = get_seg_model(args)
    model.to(device)
    if args.distributed:
        print_with_timestamp(torch.cuda.current_device())
        model = DDP(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
        print_with_timestamp(f'{args.rank} Using DDP.')
        # model_without_ddp = model.module
    else:
        model.cuda()
        print_with_timestamp('Using single gpu.')
        # model_without_ddp = model

    # Print model parameters.
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 10**6
    print_with_timestamp(f"Rank:{args.rank} Total parameters count: {pytorch_total_params} M")

    # Loss.
    loss_func = get_loss_function(args)

    # Optimizer.
    eff_batch_size = args.batch_size * args.sw_batch_size * world_size * args.accum_iter
    args.optim_lr = args.optim_lr * eff_batch_size / 16
    if args.rank == 0:
        print_with_timestamp(f'Effective batch size: {eff_batch_size}, learning rate: {args.optim_lr}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr,
                                  weight_decay=args.reg_weight)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
    )

    # Infer settings.
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=2,
        predictor=model,
        overlap=args.overlap,
    )
    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)

    # Get metric
    acc = get_metric(args)

    # Resume training.
    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_func=loss_func,
        acc_func=acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=args.start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
    cleanup(args)


if __name__ == "__main__":
    args = get_args().parse_args()
    args.distributed = False
    load_config_yaml_args(args.config_path, args)
    args.amp = not args.noamp
    if args.amp:
        print_with_timestamp('Training with amp')
    else:
        print_with_timestamp('Training without amp')
    WORLD_SIZE = 2  #torch.cuda.device_count()
    if WORLD_SIZE == 1:
        print_with_timestamp('Using single gpu.')
        main(0, WORLD_SIZE, args)
    else:
        print_with_timestamp('Using multiple gpus.')
        args.rank = int(os.environ["LOCAL_RANK"])
        print_with_timestamp(f'rank: {args.rank}')
        main(args.rank, WORLD_SIZE, args)
