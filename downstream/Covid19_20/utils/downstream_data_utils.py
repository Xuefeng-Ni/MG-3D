"""Jiaxin ZHUANG
Modified on Feb 8, 2024.
"""

import math
from copy import deepcopy

import numpy as np
import torch
from monai import data
from monai.data import load_decathlon_datalist, PersistentDataset
import monai
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    RandAffined,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ToTensord,
    SpatialPadd,
    RandZoomd,
    RandRotate90d,
    RandGaussianNoised,
    CastToTyped,
    EnsureTyped
)

try:
    from utils.misc import print_with_timestamp
except ImportError:
    print_with_timestamp = print


def get_loader(args):
    '''Get the dataloader for the downstream segmentation task.'''

    data = get_seg_loader_covid19(args)
    task = 'seg'

    return [*data, task]

def get_seg_loader_covid19(args):
    """Get the dataloader for the Covid19-20.
    """
    def _get_xforms(mode="train", keys=("image", "label")):
        """returns a composed transform for train/val/infer."""

        xforms = [
            LoadImaged(keys, ensure_channel_first=True, image_only=True),
            Orientationd(keys, axcodes="LPS"),
            Spacingd(keys, pixdim=(1.25, 1.25, 5.0),
                     mode=("bilinear", "nearest")[: len(keys)]),
            ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0,
                                 b_min=0.0, b_max=1.0, clip=True),
        ]
        if mode == "train":
            xforms.extend(
                [
                    SpatialPadd(keys, spatial_size=(192, 192, -1), mode="reflect"),  # ensure at least 96x96x96
                    RandAffined(
                        keys,
                        prob=0.15,
                        rotate_range=(0.05, 0.05, None),  # 3 parameters control the transform on 3 dimensions
                        scale_range=(0.1, 0.1, None),
                        mode=("bilinear", "nearest"),
                    ),
                    SpatialPadd(keys, spatial_size=(192, 192, -1 if args.roi_z == 16 else args.roi_z), mode="constant"),  # ensure at least 96x96x96
                    RandCropByPosNegLabeld(keys, label_key=keys[1],
                                           spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                                           num_samples=args.sw_batch_size),
                    RandGaussianNoised(keys[0], prob=0.15, std=0.01),
                    RandFlipd(keys, spatial_axis=0, prob=0.5),
                    RandFlipd(keys, spatial_axis=1, prob=0.5),
                    RandFlipd(keys, spatial_axis=2, prob=0.5),
                ]
            )
            dtype = (torch.float32, torch.uint8)
        if mode == "val":
            dtype = (torch.float32, torch.uint8)
        # if mode == "infer":
            # dtype = (np.float32, np.uint8)
            # dtype = (np.float32,)
        xforms.extend([CastToTyped(keys, dtype=dtype), EnsureTyped(keys)])
        return monai.transforms.Compose(xforms)

    train_files, val_files = load_fold(args=args, data_dir=args.data_dir,
                                       datalist_json=args.json_list)
    keys = ("image", "label")
    train_transforms = _get_xforms("train", keys)
    train_ds = PersistentDataset(data=train_files, transform=train_transforms,
                                 cache_dir=args.persistent_cache_dir)
    train_loader = monai.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )

    val_transforms = _get_xforms("val", keys)
    val_ds = PersistentDataset(data=val_files, transform=val_transforms,
                               cache_dir=args.persistent_cache_dir)
    val_loader = monai.data.DataLoader(
        val_ds, batch_size=1, pin_memory=True,
        num_workers=args.num_workers, shuffle=False
    )

    if args.eval_only:
        # batch size must be 1 for inference!
        #return train_loader, val_loader, train_files, val_files
        return val_loader, val_files
    else:
        return train_loader, val_loader


def load_fold(args, datalist_json, data_dir):
    '''Load the fold of the dataset.'''
    datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
    if args.fold is not None:
        train_files = []
        val_files = []
        for dd in datalist:
            if dd["fold"] != args.fold:
                train_files.append(dd)
            else:
                val_files.append(dd)
    else:
        train_files = datalist
        val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)

    if args.rank == 0:
        print_with_timestamp(f"train_files: {len(train_files)}")
        print_with_timestamp(f"val_files: {len(val_files)}")
        print_with_timestamp(val_files)
    return train_files, val_files


class RandZoomdSelect(RandZoomd):
    '''Selectively apply RandZoomd to the 10_Decathlon dataset.'''
    def __init__(self, args):
        self.args = args
        super().__init__(keys=["image", "label"], prob=0.3, min_zoom=1.3,
                         max_zoom=1.5, mode=['area', 'nearest'])

    def __call__(self, data, lazy=None):
        key = self.args.dataset_name
        if key in ['10_Decathlon_Task06_Lung', '10_Decathlon_Task07_Pancreas',
                   '10_Decathlon_Task10_Colon',
                   '10_Decathlon_Task03_Liver', '10_Decathlon_Task03_Liver_tumor'
                   ]:
##            data = super().__call__(data, lazy=lazy)
            data = super().__call__(data)
        return data


class FilterLabels:
    """Filter unsed label.
    """
    def __init__(self, args):
        '''Init'''
        self.args = args

    def __call__(self, data):
        data['label'] = torch.from_numpy(data['label'])
        if hasattr(self.args, 'ignore_label'):
            if self.args.ignore_label is not None:
                label = deepcopy(data['label'])
                for key in self.args.ignore_label:
                    label = torch.where(label == key, torch.zeros_like(label), label)
                data['label'] = label
        return data


class SortLabelMap:
    """Sort the label map.
    """
    def __init__(self, args):
        '''Init'''
        self.args = args

    def __call__(self, data):
        unique_labels = torch.unique(data['label'])
        # map the label to 0, 1, 2, 3, ...
        unique_labels = unique_labels[unique_labels != 0]
        result = torch.zeros_like(data['label'])
        for i, label in enumerate(unique_labels, start=1):
            result = torch.where(data['label'] == label, i, result)
        unique_labels_after = torch.unique(result)
        unique_labels_after = unique_labels_after[unique_labels_after != 0]
        data['label'] = result
        return data


class Sampler(torch.utils.data.Sampler):
    '''Sampler for distributed training.'''
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        '''Set epoch'''
        self.epoch = epoch
