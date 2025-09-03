"""Jiaxin ZHUANG
Modified on Feb 8, 2024.
"""

import math
from copy import deepcopy

import numpy as np
import torch
from monai import data
from monai.data import load_decathlon_datalist, PersistentDataset
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
)

try:
    from utils.misc import print_with_timestamp
except ImportError:
    print_with_timestamp = print


def get_loader(args):
    '''Get the dataloader for the downstream segmentation task.'''
    task = None
    if args.dataset_name in ['10_Decathlon_Task06_Lung', '10_Decathlon_Task10_Colon']:
        data = get_seg_loader_10_decathlon_usual(args)
        task = 'seg'
    else:
        raise NotImplementedError
    return [*data, task]

def get_seg_loader_10_decathlon_usual(args):
    '''Get the dataloader
    '''
    train_files, val_files = load_fold(args=args, data_dir=args.data_dir,
                                       datalist_json=args.json_list)
    # Train transforms
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        FilterLabels(args=args),
        SortLabelMap(args=args),
        Spacingd(keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandZoomdSelect(args),
        SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    mode='constant'),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            pos=args.num_positive,
            neg=args.num_negative,
            num_samples=args.sw_batch_size,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
        RandAffined(
            keys=['image', 'label'],
            mode=('bilinear', 'nearest'),
            prob=1.0, spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            rotate_range=(0, 0, np.pi/15),
            scale_range=(0.1, 0.1, 0.1)),
        ToTensord(keys=["image", "label"]),
    ])
    # Val transforms.
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            FilterLabels(args=args),
            SortLabelMap(args=args),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z),
                     mode=("bilinear", "nearest")),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )

    train_ds = PersistentDataset(data=train_files, transform=train_transforms,
                                cache_dir=args.persistent_cache_dir)
    val_ds = PersistentDataset(data=val_files, transform=val_transforms,
                               cache_dir=args.persistent_cache_dir)
    print_with_timestamp(f'Using persistent dataset: {args.persistent_cache_dir}')

    train_sampler = Sampler(train_ds, shuffle=False) if args.distributed else None
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,   #(train_sampler is None)
        num_workers=args.num_workers,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True,
    )
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
        sampler=val_sampler, pin_memory=True, persistent_workers=True)
    if args.eval_only:
        # batch size must be 1 for inference!
        return val_loader, val_files
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
        if key in ['10_Decathlon_Task06_Lung', '10_Decathlon_Task10_Colon']:
            data = super().__call__(data)
        return data


class FilterLabels:
    """Filter unsed label.
    """
    def __init__(self, args):
        '''Init'''
        self.args = args

    def __call__(self, data):
        if type(data['label']) == np.ndarray:
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
