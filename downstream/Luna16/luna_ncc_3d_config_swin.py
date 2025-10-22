import os
import numpy as np
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from trainers import *
import argparse

import sys
sys.path.append('/home/csexuefeng/Luna16/')

class ncc_config:
    attr = 'class'
    gpu_ids = [0]
    benchmark = False
    manualseed = 111 
    model = 'Simple'
    network = 'swin_3d'
    init_weight_type = 'kaiming'
    note = "luna_swin_3d_swin3d_fold0_(order_xyz)"

    # data
    train_fold = [0, 1, 2, 3, 4, 5] 
    valid_fold = [6]
    test_fold = [7, 8, 9] 
    hu_min = -1000
    hu_max = 600
    input_size = [48, 48, 48]
    train_dataset = 'luna_ncc'
    eval_dataset = 'luna_ncc'
    im_channel = 1
    class_num = 1
    order_class_num = 1
    num_grids_per_axis = 2
    
    normalization = 'sigmoid' # for bce loss.
    random_sample_ratio = 1 # if none, extremely imbalanced
    sample_type = 'random'

    # model
    optimizer = 'adam'
    scheduler = None
    lr = 1.5e-4  
    patience = 10000
    verbose = 1
    train_batch = 32   
    val_batch = 24   
    val_epoch = 10
    num_workers = 16
    max_queue_size = num_workers * 1
    epochs = 10001
    loss = 'bce'

    # pretrain
    resume = None
    pretrained_model = None
    pretrained_model = '/home/xuefeng/MG-3D-Swin-B.ckpt'
    transferred_part = 'encoder'#'encoder'

    transferred_dismatched_keys = None #['module.', 'module.encoder.']
   

    def display(self, logger):
        logger.info("Configurations")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                logger.info("{:30} {}".format(a, getattr(self, a)))
                # print("\n")


if __name__ == '__main__':
    config = ncc_config()
    Trainer = ClassificationTrainer(config)
    Trainer.train()
