import os
import json
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
from monai.transforms import *

from monai.data.utils import SUPPORTED_PICKLE_MOD, pickle_hashing
import tempfile
from pathlib import Path
from monai.utils import look_up_option
import pickle
import shutil

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.num_slices = args.num_slices
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[
                :self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None):
        super(IuxrayMultiImageDataset, self).__init__(args, tokenizer, split, transform)
        self.label = self._load_data(args.label_path)

    def _load_data(self, label_path):
        label_dict = {}

        data = pd.read_csv(label_path)
        for index, row in data.iterrows():
            idx = row['id']
            label = row[1:].to_list()

            label_dict[idx] = list(map(lambda x: 1 if x == 1.0 else 0, label))

        return label_dict

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        pid = image_id.split('_')[0][3:]
        try:
            labels = torch.tensor(self.label[int(pid)], dtype=torch.float32)
        except:
            # print('Except id ', pid)
            labels = torch.tensor([0 for _ in range(14)], dtype=torch.float32)

        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        return sample


class CTRG_MultiImageDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None):
        super(CTRG_MultiImageDataset, self).__init__(args, tokenizer, split, transform)
        self.label = self._load_data(args.label_path)
        self.args = args

    def _load_data(self, label_path):
        label_dict = {}

        data = pd.read_csv(label_path)
        for index, row in data.iterrows():
            idx = row['id']
            label = row[1:].to_list()

            label_dict[idx] = list(map(lambda x: 1 if x == 1.0 else 0, label))

        return label_dict

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_paths = os.path.join(self.args.image_dir, image_id)
        '''
        # select num_slices images from image_paths
        num_slices = self.num_slices
        step = len(image_paths) // num_slices

        if len(image_paths) > num_slices:
            image_paths = [image_paths[i]
                           for i in range(0, len(image_paths), step)]
            image_paths = image_paths[:num_slices]
        else:
            image_paths = image_paths + \
                [image_paths[-1]] * (num_slices - len(image_paths))

        assert len(image_paths) == num_slices

        images = []
        for image_path in image_paths:
            image = Image.open(os.path.join(
                self.image_dir, image_path)).convert('L')
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

##        image = torch.stack(images, 0)
##        labels = torch.tensor(self.label[int(image_id)], dtype=torch.float32)
        '''
        '''
        train_transforms = Compose([LoadImaged(keys=["image"]),
                                    EnsureChannelFirstd(keys=["image"]),
##                                    Orientationd(keys=["image"], axcodes="RAS"),
                                    ScaleIntensityRanged(
                                        keys=["image"], a_min=0.0, a_max=255.0,
                                        b_min=0.0, b_max=1.0, clip=True),
##                                    CropForegroundd(keys=["image"], source_key="image"),
##                                    RandFlipd(keys=["image"], prob=1, spatial_axis=2),
                                    Resized(keys=["image"], mode="trilinear", align_corners=True,
                                            spatial_size=(96, 96, 96)), 
##                                            spatial_size=(90, 90, 90)), 
                                    RandFlipd(keys=["image"], prob=1, spatial_axis=2),
                                    ToTensord(keys=["image"])                                   
                                    ])
        transformed_image = train_transforms({'image':image_paths})
        image = transformed_image['image'].permute(0,3,2,1).contiguous()
        labels = torch.tensor(self.label[int(image_id[:-7])], dtype=torch.float32)
        '''
        cache_dir = Path('/home/xuefeng/cache/m2kt_CTRG_nii_chest_128/')
        item_transformed = example['id']
        if cache_dir is not None:
            data_item_md5 = pickle_hashing(item_transformed).decode("utf-8")
            hashfile = cache_dir / f"{data_item_md5}.pt"

        if hashfile is not None and hashfile.is_file():  # cache hit
            image = torch.load(hashfile)
        else:        
            try:
                train_transforms = Compose([LoadImaged(keys=["image"]),
                                            EnsureChannelFirstd(keys=["image"]),
        ##                                    Orientationd(keys=["image"], axcodes="RAS"),
                                            ScaleIntensityRanged(
                                                keys=["image"], a_min=0.0, a_max=255.0,
                                                b_min=0.0, b_max=1.0, clip=True),
        ##                                    CropForegroundd(keys=["image"], source_key="image"),
        ##                                    RandFlipd(keys=["image"], prob=1, spatial_axis=2),
                                            Resized(keys=["image"], mode="trilinear", align_corners=True,
                                                    spatial_size=(128, 128, 128)), 
        ##                                            spatial_size=(90, 90, 90)), 
                                            RandFlipd(keys=["image"], prob=1, spatial_axis=2),
                                            ToTensord(keys=["image"])                                   
                                            ])
                transformed_image = train_transforms({'image':image_paths})
                image = transformed_image['image'].permute(0,3,2,1).contiguous()
                
                with tempfile.TemporaryDirectory() as tmpdirname:
                    temp_hash_file = Path(tmpdirname) / hashfile.name
                    torch.save(
                        obj=image,
                        f=temp_hash_file,
                        pickle_module=look_up_option("pickle", SUPPORTED_PICKLE_MOD),
                        pickle_protocol=pickle.HIGHEST_PROTOCOL,
                    )
                    if temp_hash_file.is_file() and not hashfile.is_file():
                        shutil.move(str(temp_hash_file), hashfile)
            except PermissionError:  # project-monai/monai issue #3613
                pass
            
        labels = torch.tensor(self.label[int(image_id[:-7])], dtype=torch.float32)
        '''
        cache_dir = Path('/home/xuefeng/cache/m2kt_CTRG_nii_chest_fore_128/')
        item_transformed = example['id']
        if cache_dir is not None:
            data_item_md5 = pickle_hashing(item_transformed).decode("utf-8")
            hashfile = cache_dir / f"{data_item_md5}.pt"

        if hashfile is not None and hashfile.is_file():  # cache hit
            image = torch.load(hashfile)
        else:        
            try:
                train_transforms = Compose([LoadImaged(keys=["image"]),
                                            EnsureChannelFirstd(keys=["image"]),
        ##                                    Orientationd(keys=["image"], axcodes="RAS"),
                                            ScaleIntensityRanged(
                                                keys=["image"], a_min=0.0, a_max=255.0,
                                                b_min=0.0, b_max=1.0, clip=True),
                                            ToTensord(keys=["image"])                                   
                                            ])
                transformed_image = train_transforms({'image':image_paths})
                train_transforms_2 = Compose([CenterSpatialCrop(roi_size=(448, 336, 
                                                    0.88 * transformed_image['image'].size(3))),
                                            Resize(mode="trilinear", align_corners=True,
                                                    spatial_size=(128, 96, 128)), 
                                            RandFlip(prob=1, spatial_axis=2),
                                            ])
                transformed_image = train_transforms_2(transformed_image['image'])
                
                image = transformed_image.permute(0,3,2,1).contiguous()
        
                with tempfile.TemporaryDirectory() as tmpdirname:
                    temp_hash_file = Path(tmpdirname) / hashfile.name
                    torch.save(
                        obj=image,
                        f=temp_hash_file,
                        pickle_module=look_up_option("pickle", SUPPORTED_PICKLE_MOD),
                        pickle_protocol=pickle.HIGHEST_PROTOCOL,
                    )
                    if temp_hash_file.is_file() and not hashfile.is_file():
                        shutil.move(str(temp_hash_file), hashfile)
            except PermissionError:  # project-monai/monai issue #3613
                pass
            
        labels = torch.tensor(self.label[int(image_id[:-7])], dtype=torch.float32)
        '''
        '''
        images = []
        for image_path in image_paths:
            image = Image.open(os.path.join(self.image_dir, image_path)).convert('L')
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)
        images = images[::-1]
##        image = torch.stack(images, 0)
        image = torch.stack(images, -1)
##        image = F.interpolate(image, size=(128, 64)).permute(0,3,1,2).contiguous()
##        image = F.interpolate(image, size=(128, 96)).permute(0,3,1,2).contiguous()
        image = image[:, :, :, int(len(images)*0.06):int(len(images)*0.94)]
        image = F.interpolate(image,size=(128, 128)).permute(0,3,1,2).contiguous()     
        '''
        '''
        cache_dir = Path('/home/xuefeng/cache/m2kt_CTRG_chest_fore_128/')
        item_transformed = example['id']
        if cache_dir is not None:
            data_item_md5 = pickle_hashing(item_transformed).decode("utf-8")
            hashfile = cache_dir / f"{data_item_md5}.pt"

        if hashfile is not None and hashfile.is_file():  # cache hit
            image = torch.load(hashfile)
        else:        
            try:
                images = []
                for image_path in image_paths:
                    image = Image.open(os.path.join(self.image_dir, image_path)).convert('L')
                    if self.transform is not None:
                        image = self.transform(image)
                    images.append(image)
                images = images[::-1]
                image = torch.stack(images, -1)
                image = image[:, :, :, int(len(images)*0.06):int(len(images)*0.94)]
                image = F.interpolate(image,size=(128, 128)).permute(0,3,1,2).contiguous()
                   
                with tempfile.TemporaryDirectory() as tmpdirname:
                    temp_hash_file = Path(tmpdirname) / hashfile.name
                    torch.save(
                        obj=image,
                        f=temp_hash_file,
                        pickle_module=look_up_option("pickle", SUPPORTED_PICKLE_MOD),
                        pickle_protocol=pickle.HIGHEST_PROTOCOL,
                    )
                    if temp_hash_file.is_file() and not hashfile.is_file():
                        shutil.move(str(temp_hash_file), hashfile)
            except PermissionError:  # project-monai/monai issue #3613
                pass
        '''
        
##        labels = torch.tensor(self.label[int(image_id)], dtype=torch.float32)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        return sample

class MimiccxrSingleImageDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None):
        super(MimiccxrSingleImageDataset, self).__init__(args, tokenizer, split, transform)
        self.label = pd.read_csv(args.label_path)

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        d = self.label[self.label['dicom_id'] == image_id]
        labels = torch.tensor(d.values.tolist()[0][8:], dtype=torch.float32)
        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        return sample


class CovidSingleImageDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None):
        super(CovidSingleImageDataset, self).__init__(args, tokenizer, split, transform)
        self.label = self._load_data(args.label_path)

    def _load_data(self, label_file):
        labels = {}

        # print(f"Loading data from {label_file}")

        data = pd.read_csv(label_file)
        # data = data[data['split'] == self.subset]
        for index, row in data.iterrows():
            idx = row['idx']
            label = [1, 0] if row['label'] == '轻型' else [0, 1]
            labels[idx] = label

        return labels

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        labels = torch.tensor(self.label[image_id], dtype=torch.float32)
        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        return sample


class CovidAllImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        labels = torch.tensor(example['label'], dtype=torch.float32)
        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        return sample