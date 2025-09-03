import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss
from fastai.data.core import DataLoader, DataLoaders
from fastai.vision.learner import Learner
from fastai.callback.all import *
from fastai.metrics import AccumMetric

from algorithm.i3d.i3dpt import I3D
from ctdataset import CTDataset
from config import get_config
from metrics import acc_probseverecovid
from metrics import acc_probcovid
from metrics import roc_probseverecovid
from metrics import roc_probcovid
import numpy as np
import math

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import torch
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

CONFIGFILE = "/home/xuefeng/stoic2021/training/config/baseline.json"

def get_datasets(config, data_dir):
    image_dir = os.path.join("/ssd1/xuefeng/data/mha/")
    #split
    reference_train_path = os.path.join("/ssd1/xuefeng/metadata/test/tran_fold0.csv")
    reference_val_path = os.path.join("/ssd1/xuefeng/metadata/test/tran_fold0.csv")
    # reference_train_path = os.path.join("/ssd1/xuefeng/metadata/test/tran_fold1.csv")
    # reference_val_path = os.path.join("/ssd1/xuefeng/metadata/test/tran_fold1.csv")
    # reference_train_path = os.path.join("/ssd1/xuefeng/metadata/test/train_fold2.csv")
    # reference_val_path = os.path.join("/ssd1/xuefeng/metadata/test/test_fold2.csv")
    
    df_train = pd.read_csv(reference_train_path)
    df_valid = pd.read_csv(reference_val_path)
    df_train["x"] = df_train.apply(lambda row: os.path.join(image_dir, str(row["PatientID"]) + ".mha"), axis=1)
    df_train["y"] = df_train.apply(lambda row: [row["probCOVID"], row["probSevere"]], axis=1)
    df_valid["x"] = df_valid.apply(lambda row: os.path.join(image_dir, str(row["PatientID"]) + ".mha"), axis=1)
    df_valid["y"] = df_valid.apply(lambda row: [row["probCOVID"], row["probSevere"]], axis=1)

    data_train = df_train[["x", "y"]].to_dict("records")
    data_valid = df_valid[["x", "y"]].to_dict("records")

    train_ds = CTDataset(data_train, config["preprocess_dir"])
    valid_ds = CTDataset(data_valid, config["preprocess_dir"])
    return train_ds, valid_ds


def get_learn(config, data_dir, artifact_dir):
    train_ds, valid_ds = get_datasets(config, data_dir)
    train_dl = DataLoader(train_ds, bs=config["batch_size"], num_workers=config["num_workers"],
                          shuffle=True, drop_last=True)
    valid_dl = DataLoader(valid_ds, bs=config["batch_size"], num_workers=config["num_workers"])

    backbone = 'swin_unetr_b' #swin_unetr_l
    if backbone == 'swin_unetr_b':
        from algorithm.i3d.swin_b import SwinTransformer
        from monai.utils import ensure_tuple_rep
        spatial_dims = 3
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        model = SwinTransformer(            
                in_chans=1,
                embed_dim=48,
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
                num_classes=2)
        model = model.to(device)
        model_dict = dict(model.state_dict())

        extractor_dict = torch.load('/home/ubuntu/MG-3D-Swin-B.ckpt', map_location=torch.device('cpu'))

        new_extractor_dict = {}
        for key, value in extractor_dict.items():
            name = key[15:]
            if key[:14] == 'vision_encoder':
                new_extractor_dict[name] = value

        pretrain_dict = {k: v for k, v in new_extractor_dict.items() if k in model_dict}
        model.load_state_dict(new_extractor_dict, strict=False)

    elif backbone == 'swin_unetr_l':
        from algorithm.i3d.swin_l import SwinTransformer
        from monai.utils import ensure_tuple_rep
        spatial_dims = 3
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        model = SwinTransformer(            
                in_chans=1,
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
                num_classes=2)
        model = model.to(device)
        model_dict = dict(model.state_dict())

        extractor_dict = torch.load('/home/ubuntu/MG-3D-Swin-L.ckpt', map_location=torch.device('cpu'))

        new_extractor_dict = {}
        for key, value in extractor_dict.items():
            name = key[15:]
            if key[:14] == 'vision_encoder':
                new_extractor_dict[name] = value

        pretrain_dict = {k: v for k, v in new_extractor_dict.items() if k in model_dict}
        model.load_state_dict(new_extractor_dict, strict=False)
         

    metrics = [
        AccumMetric(acc_probcovid, flatten=False),
        AccumMetric(roc_probcovid, flatten=False),
        AccumMetric(acc_probseverecovid, flatten=False),
        AccumMetric(roc_probseverecovid, flatten=False)
    ]

    return Learner(DataLoaders(train_dl, valid_dl),
                   model=model,
                   metrics=metrics,
                   loss_func=BCEWithLogitsLoss(),
                   model_dir=artifact_dir
                   ).to_fp16()


def train(learn, config):
    cbs = [SaveModelCallback(monitor="roc_probseverecovid",
                             fname=config["experiment_name"]
                             )
           ]
    learn.fit(config["epochs"],
              lr=config["lr"],
              cbs=cbs)


def do_learning(data_dir, artifact_dir):
    """
    You can implement your own solution to the STOIC2021 challenge by editing this function.
    :param data_dir: Input directory that the training Docker container has read access to. This directory has the same
        structure as the stoic2021-training S3 bucket (see https://registry.opendata.aws/stoic2021-training/)
    :param artifact_dir: Output directory that, after training has completed, should contain all artifacts (e.g. model
        weights) that the inference Docker container needs. It is recommended to continuously update the contents of
        this directory during training.
    :returns: A list of filenames that are needed for the inference Docker container. These are copied into artifact_dir
        in main.py. If your model already produces all necessary artifacts into artifact_dir, an empty list can be
        returned. Note: To limit the size of your inference Docker container, please make sure to only place files that 
        are necessary for inference into artifact_dir.
    """
    config = get_config(CONFIGFILE)
    learn = get_learn(config, data_dir, artifact_dir)
    train(learn, config)

    artifacts = [] # empty list because train() already writes all artifacts to artifact_dir

    # If your code does not produce all necessary artifacts for the inference Docker container into artifact_dir, return 
    # their filenames:
    # artifacts = ["/tmp/model_checkpoint.pth", "/tmp/some_other_artifact.json"]
    
    return artifacts

