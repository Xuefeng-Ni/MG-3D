"""Jiaxin ZHUANG.
Modified Aug 21, 2023.
"""

import argparse


def get_args():
    '''Get arguments.'''
    parser = argparse.ArgumentParser(description="Segmentation end-to-end training")
    parser.add_argument("--logdir", default=None, type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--data_dir", default=None, type=str, help="dataset directory")
    parser.add_argument("--json_list", default=None, type=str, help="dataset json file")
    parser.add_argument("--pretrained_path", default=None, type=str, help="pretrained checkpoint path")
    parser.add_argument("--resume", default=None, type=str, help="resume checkpoint path")
    parser.add_argument("--config_path", default="./configs/downstream_configs.yaml", type=str, help="dataset config path.")
    parser.add_argument("--max_epochs", default=None, type=int, help="max number of training epochs")
    parser.add_argument("--batch_size", default=None, type=int, help="number of batch size")
    parser.add_argument("--accum_iter", default=None, type=int, help="accumulate iteration for update")
    parser.add_argument("--infer_sw_batch_size", default=2, type=int, help="number of batch size when running inference slicing windows.")
    parser.add_argument("--optim_lr", default=None, type=float, help="optimization learning rate")
    parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
    parser.add_argument("--distributed", action="store_true", help="start distributed training")
    parser.add_argument("--dist-url", default=None, type=str, help="distributed url, i.e., tcp://127.0.0.1:23456")
    parser.add_argument("--dataset_name", default=None, type=str, help="Dataset")
    parser.add_argument("--out_channels", default=None, type=int, help="number of output channels")
    parser.add_argument("--use_persistent_dataset", action="store_true", help="use monai Dataset class")
    parser.add_argument("--persistent_cache_dir", default=None, type=str, help="persistent cache directory")
    parser.add_argument("--roi_x", default=None, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=None, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=None, type=int, help="roi size in z direction")
    parser.add_argument("--space_x", default=None, type=float, help="space size in x direction")
    parser.add_argument("--space_y", default=None, type=float, help="space size in y direction")
    parser.add_argument("--space_z", default=None, type=float, help="space size in z direction")
    parser.add_argument("--num_positive", default=1, type=int, help="numbers of positive samples for each image")
    parser.add_argument("--num_negative", default=1, type=int, help="numbers of negative samples for each image")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--RandFlipd_prob", default=None, type=float, help='prob for the flipping')
    parser.add_argument("--RandRotate90d_prob", default=None, type=float, help='prob for the rotation')
    parser.add_argument("--RandScaleIntensityd_prob", default=None, type=float, help='prob for the intensigty scaling')
    parser.add_argument("--RandShiftIntensityd_prob", default=None, type=float, help='prob for the intensity shifting')
    parser.add_argument("--overlap", default=None, type=float, help="Overlap for inference.")
    parser.add_argument("--warmup_epochs", default=None, type=int, help="number of warmup epochs")
    parser.add_argument("--model_name", default="swin_unetr", choices=['swin_unetr_b', 'swin_unetr_l'], type=str, help="model name")
    parser.add_argument("--fold", default=None, type=int, help="Five cross validation.")
    parser.add_argument("--print_freq", default=None, type=int, help="print frequency")
    parser.add_argument("--val_every", default=None, type=int, help="validation frequency")
    parser.add_argument("--seed", default=None, type=int, help="Setting random seed.")
    parser.add_argument("--clip_value", default=5, type=int, help="Clip value for gradient clipping.")
    parser.add_argument("--eval_only", action="store_true", help="eval_only")
    parser.add_argument("--best_acc", default=None, type=int, help="Best accuracy.")
    parser.add_argument("--local-rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument("--use_checkpoint", default=False, help="use gradient checkpointing to save memory")

    parser.add_argument('--ensemble_list', default=None, nargs='+', help='whether to ensemble models.')
    parser.add_argument('--save_visualization', action="store_true", help='save visaulization results.')
    parser.add_argument("--pred_dir", default=None, type=str, help="eval_only mode using this.")
    parser.add_argument("--sr_ratio", default=1, type=int, help="multi scale token")
    parser.add_argument("--pred_csv", default="result.csv", type=str, help="eval_only mode using this.")

    return parser