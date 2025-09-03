"""Jiaxin ZHUANG.
Modified on Feb 8, 2024.
"""
import sys

def get_loss_function(args):
    '''Get loss function.'''
    if args.loss_name == 'DiceCELoss':
        from monai.losses import DiceCELoss
        loss_func = DiceCELoss(
            to_onehot_y=True, softmax=True, squared_pred=True,
            smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        ).cuda()
    elif args.loss_name == 'DiceLoss':
        from monai.losses import DiceLoss
        loss_func = DiceLoss(to_onehot_y=True, softmax=True).cuda()
    else:
        print('=> unknown loss name', flush=True)
        sys.exit(-1)
    return loss_func
