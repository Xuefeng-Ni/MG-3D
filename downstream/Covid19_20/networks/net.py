"""Jiaxin ZHUANG.
Modified on Feb 8, 2024.
"""

import torch
import numpy as np
import math


def get_seg_model(args=None):
    '''Get segmentation model.'''
    if args.model_name == 'swin_unetr_b':
        from monai.networks.nets import SwinUNETR
        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=48,
            use_checkpoint=True if args.pretrained_path is not None else False,
        )

        extractor_dict = torch.load('/home/ubuntu/MG-3D-Swin-B.ckpt', map_location=torch.device('cpu'))

        model_dict = dict(model.state_dict())
        new_extractor_dict = {}
        for key, value in extractor_dict.items():
            name = key[15:]
            if key[:14] == 'vision_encoder':
                new_extractor_dict['swinViT.' + name] = value
 
        pretrain_dict = {k: v for k, v in new_extractor_dict.items() if k in model_dict}
        model.load_state_dict(new_extractor_dict, strict=False)
    elif args.model_name == 'swin_unetr_l':
        from monai.networks.nets import SwinUNETR
        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=96,
            use_checkpoint=True if args.pretrained_path is not None else False,
        )

        extractor_dict = torch.load('/home/ubuntu/MG-3D-Swin-L.ckpt', map_location=torch.device('cpu'))

        model_dict = dict(model.state_dict())
        new_extractor_dict = {}
        for key, value in extractor_dict.items():
            name = key[15:]
            if key[:14] == 'vision_encoder':
                new_extractor_dict['swinViT.' + name] = value
 
        pretrain_dict = {k: v for k, v in new_extractor_dict.items() if k in model_dict}
        model.load_state_dict(new_extractor_dict, strict=False)
    else:
        print('Require valid model name')
        raise NotImplementedError
    return model
