"""Jiaxin ZHUANG.
Modified on Feb 8, 2024.
"""

import torch
import numpy as np
import math

def get_seg_model(args=None):
    '''Get segmentation model.'''
    if args.model_name == 'swin_unetr_b':
        from networks.swin_unetr import SwinUNETR
        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=args.dropout_path_rate,
            use_checkpoint=args.use_checkpoint,
        )    

        extractor_dict = torch.load('/home/ubuntu/MG-3D-Swin-B.ckpt', map_location=torch.device('cpu'))

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
    elif args.model_name == 'swin_unetr_l':
        from networks.swin_unetr import SwinUNETR
        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=96,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=args.dropout_path_rate,
            use_checkpoint=args.use_checkpoint,
        )    

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
    else:
        print('Require valid model name')
        raise NotImplementedError
    return model
