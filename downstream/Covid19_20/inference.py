'''Jiaxin ZHUANG, Sep 14, 2023.
Only for inference with single GPU.
'''

import os
import copy
from functools import partial
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
#, SurfaceDiceMetric
from monai.utils.enums import MetricReduction
from monai.transforms import AsDiscrete, Activations, Compose


from networks.net import get_seg_model
from utils.downstream_utils import resume_ckpt
from utils.helper import load_config_yaml_args
from utils.downstream_data_utils import get_loader
from utils.misc import print_with_timestamp, calculate_time, setup, cleanup
from utils.default_arguments import get_args
from utils.metric import NSD, HD
from trainer import infer


@calculate_time
def main(rank, world_size, args):
    """Main function.
    """
    setup(args, rank, world_size)
    print_with_timestamp(args)
    device = torch.device(args.device)

    # dataset
    val_loader, val_files, args.task = get_loader(args)

    # model
    model = get_seg_model(args)
    # ensemble
    if args.ensemble_list:
        print_with_timestamp(f'Ensemble list: {args.ensemble_list}')
        model_list = []
        for ckpt_path in args.ensemble_list:
            args.resume = ckpt_path
            model_copy = copy.deepcopy(model)
            model_copy, *_ = resume_ckpt(args, model_copy)
            model_list.append(model_copy.to(device))
        model = model_list
    else:
        model, *_ = resume_ckpt(args, model)
        model = model.to(device)

    # metrics
    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    post_nsd = AsDiscrete(argmax=True)

    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=False)
    nsd_metric = NSD(spacing_mm=(args.space_x, args.space_y, args.space_z), tolerance_mm=args.tolerance_mm)
    #hd_metric = HD(spacing_mm=(args.space_x, args.space_y, args.space_z), percent=95)
    hd_metric = None
    inf_size = [args.roi_x, args.roi_y, args.roi_z]

    if isinstance(model, list):
        model_inferer_list = []
        for the_model in model:
            model_inferer = partial(
                sliding_window_inference,
                roi_size=inf_size,
                sw_batch_size=args.infer_sw_batch_size,
                predictor=the_model,
                overlap=args.overlap,
            )
            model_inferer_list.append(model_inferer)
        model_inferer = model_inferer_list
    elif args.dataset_name == 'msd_brainT':
        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        post_label = None
        inf_size = [args.roi_x, args.roi_y, args.roi_z]
        model_inferer = partial(
            sliding_window_inference,
            roi_size=inf_size,
            sw_batch_size=args.infer_sw_batch_size,
            predictor=model,
            overlap=args.overlap,
        )
    else:
        model_inferer = partial(
            sliding_window_inference,
            roi_size=inf_size,
            sw_batch_size=args.infer_sw_batch_size,
            predictor=model,
            overlap=args.overlap,
        )

    args.pred_dir = os.path.join(args.logdir, args.pred_dir)
    args.pred_csv = os.path.join(args.pred_dir, args.pred_csv)
    os.makedirs(args.pred_dir, exist_ok=True)
    print_with_timestamp(f'Prediction dir: {args.pred_dir}')
    print_with_timestamp(f'Prediction csv: {args.pred_csv}')
    os.makedirs(args.pred_dir, exist_ok=True)

    infer(model, val_loader, val_files,
          dice_acc,
          nsd_metric,
          hd_metric,
          args=args, model_inferer=model_inferer,
          post_label=post_label, post_pred=post_pred, post_nsd=post_nsd)

    print_with_timestamp('Inference done.')
    cleanup(args)


if __name__ == '__main__':
    args = get_args().parse_args()
    load_config_yaml_args(args.config_path, args)
    args.amp = True
    args.eval_only = True
    #args.dataset_name = 'input_file'
    print_with_timestamp(f'Setting dataset name to: {args.dataset_name}')

    if not args.pred_dir:
        raise ValueError('Please specify the prediction dir')

    print_with_timestamp('Using single gpu.')
    main(rank=0, world_size=1, args=args)
