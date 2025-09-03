"""Jiaxin ZHUANG.
Modified on Jun 26, 2023.
"""

import sys
import os
import torch
import numpy as np

from utils.misc import print_with_timestamp


def interpolate_pos_embed(args, model, checkpoint_model, new_size=None, prefix=''):
    '''Interpolate the position embedding from pretrained model.'''
    print(checkpoint_model.keys())
    pos_embed_key = prefix + 'pos_embed'
    if pos_embed_key in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model[pos_embed_key]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches if args.model_name not in ['convit3d', 'convit3d_tiny', 'MiM'] else model.patch_embed4.num_patches
        patch_size = model.patch_embed.patch_size if args.model_name not in ['convit3d', 'convit3d_tiny', 'MiM'] else [16, 16, 16]

        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        orig_size = int(np.cbrt(pos_embed_checkpoint.shape[-2] - num_extra_tokens))

        new_size_ = []
        for ns, ps in zip(new_size, patch_size):
            print_with_timestamp(ns, ps)
            new_size_.append(int(ns/ps))
        new_size = new_size_
        # class_token and dist_token are kept unchanged
        if orig_size != new_size[0] or orig_size != new_size[1] or orig_size != new_size[2]:
            print_with_timestamp(f'Position interpolate from {orig_size} to {new_size}')
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            print_with_timestamp(num_extra_tokens)
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, orig_size, embedding_size).permute(0, 4, 1, 2, 3)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=new_size, mode='trilinear', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 4, 1).flatten(1, 3)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model[pos_embed_key] = new_pos_embed
    return checkpoint_model


def load_ckpt(args, model):
    """Load checkpoint from pretrained ViT for UNETR.
    """
    if args.model_name in ['swin_unetr', 'unet']:
        return model

    if args.pretrained_path:
        if os.path.isfile(args.pretrained_path):
            checkpoint = torch.load(args.pretrained_path, map_location='cpu')

            ckpt = {}

            out = model.load_state_dict(ckpt, strict=False)
            if args.rank == 0:
                print(out)
                print(f"=> loaded checkpoint '{args.pretrained_path}' (epoch {checkpoint['epoch']})")
            del ckpt, checkpoint
        else:
            print(f"=> no checkpoint found at '{args.pretrained_path}'")
            sys.exit(-1)
    else:
        print("=> no checkpoint")
    return model



def resume_ckpt(args, model, optimizer=None, scheduler=None):
    """Resume checkpoint from previous training.
    """
    if args.resume is not None and os.path.isfile(args.resume):
        model_dict = dict(model.state_dict())
        ckpt = torch.load(args.resume, map_location="cpu")['state_dict']
        pretrain_dict = {k.replace('module.', ''): v for k, v in ckpt.items() if k.replace('module.', '') in model_dict}
        print(pretrain_dict)
        out = model.load_state_dict(pretrain_dict, strict=True)
        # ckpt = torch.load(args.resume, map_location="cpu")
        # out = model.load_state_dict(ckpt["state_dict"], strict=True)
        print_with_timestamp(f'Load model ckpt {out}')

        if 'optimizer' in ckpt and optimizer:
            out = optimizer.load_state_dict(ckpt["optimizer"])
            print_with_timestamp(f'Load optimizer: {out}')
        else:
            print_with_timestamp('No optimizer in ckpt.')

        if 'scheduler' in ckpt and scheduler:
            scheduler.load_state_dict(ckpt["scheduler"])
            print_with_timestamp(f'Load schduler: {out}')
        else:
            print_with_timestamp('No scheduler in ckpt.')

        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"]
            print_with_timestamp(f'Load start epoch: {start_epoch}')
        else:
            start_epoch = 0
            print_with_timestamp('No epoch in ckpt.')
        if "best_acc" in ckpt:
            best_acc = ckpt["best_acc"]
            print_with_timestamp(f'Load best acc: {best_acc}')
        else:
            best_acc = 0
            print_with_timestamp('No best acc in ckpt.')
        print_with_timestamp(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch}) (bestacc {best_acc})")
        args.start_epoch = start_epoch + 1
        args.best_acc = best_acc
        return model, optimizer, scheduler
    else:
        args.start_epoch = 0
        args.best_acc = 0.0
        print_with_timestamp(f"=> no checkpoint found at '{args.resume}'")
        return model, optimizer, scheduler
