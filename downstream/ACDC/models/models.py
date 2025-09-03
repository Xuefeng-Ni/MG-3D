import os

import torch
from monai.networks.nets import SwinUNETR


import numpy as np
import math

def SwinB(args):
    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=args.dropout_path_rate,
        use_checkpoint=args.use_checkpoint,
        use_v2=False
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

    return model

def SwinL(args):
    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=96,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=args.dropout_path_rate,
        use_checkpoint=args.use_checkpoint,
        use_v2=False
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

    return model

def load(model, model_dict):
    if "state_dict" in model_dict.keys():
        state_dict = model_dict["state_dict"]
    elif "network_weights" in model_dict.keys():
        state_dict = model_dict["network_weights"]
    elif "net" in model_dict.keys():
        state_dict = model_dict["net"]
    elif "student" in model_dict.keys():
        state_dict = model_dict["student"]
    else:
        state_dict = model_dict

    if "module." in list(state_dict.keys())[0]:
        print("Tag 'module.' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("module.", "")] = state_dict.pop(key)

    if "backbone." in list(state_dict.keys())[0]:
        print("Tag 'backbone.' found in state dict - fixing!")
    for key in list(state_dict.keys()):
        state_dict[key.replace("backbone.", "")] = state_dict.pop(key)

    if "swin_vit" in list(state_dict.keys())[0]:
        print("Tag 'swin_vit' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)

    current_model_dict = model.state_dict()

    # for k in current_model_dict.keys():
    #     if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()):
    #         print(k)

    new_state_dict = {
        k: state_dict[k] if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()) else current_model_dict[k]
        for k in current_model_dict.keys()}

    model.load_state_dict(new_state_dict, strict=True)

    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Swin models")
    parser.add_argument("--pretrained_root", default='/home/linshan/pretrained/', type=str, help="pretrained_root")
    parser.add_argument("--pretrained_path", default='model_B.pt', help="checkpoint name for Swin")

    parser.add_argument("--feature_size", default=48, type=int, help="feature size")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=4, type=int, help="number of output channels")

    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")

    parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")

    args = parser.parse_args()
    model = Swin(args)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)
    input = torch.rand(1, 1, 96, 96, 96)
    output = model(input)
    print(output.shape)

    from thop import profile
    import torch
    import torchvision.models as models

    flops, params = profile(model, inputs=(input,))
    gflops = flops / 1e9
    print(f"GFLOPS: {gflops}")


