import SimpleITK as sitk
from torch.utils.data import Dataset
import torch
import os
import numpy as np
from torchio.transforms import Compose, RandomAffine, RandomNoise
from algorithm.preprocess import preprocess
from monai.transforms import *

class CTDataset(Dataset):
    def __init__(self, data, preprocess_dir, augment=False):
        self.data = data
        self.preprocess_dir = os.path.join(preprocess_dir, "preprocess")
        os.makedirs(self.preprocess_dir, exist_ok=True)
        if augment:
            self.transform = Compose([
                RandomAffine(scales=0.1, degrees=15, translation=20),
                RandomNoise(std=(0, 0.05)),
            ])
        else:
            self.transform = lambda x: x

    def __len__(self):
        return len(self.data)

    def get_x_filename(self, idx):
        return self.data[idx]['x']

    def get_x_filename_preprocessed(self, idx):
        return os.path.join(self.preprocess_dir, os.path.basename(self.get_x_filename(idx)))

    def is_preprocessed(self, idx):
        return os.path.isfile(self.get_x_filename_preprocessed(idx))

    def get_x_preprocessed(self, idx):
        sitk_image = sitk.ReadImage(self.get_x_filename_preprocessed(idx))
        return sitk.GetArrayFromImage(sitk_image)

    def get_x(self, idx):
        if not self.is_preprocessed(idx):
            sitk_image = sitk.ReadImage(self.get_x_filename(idx))
            image = preprocess(sitk_image)
            sitk_image = sitk.GetImageFromArray(image)
            sitk.WriteImage(sitk_image, self.get_x_filename_preprocessed(idx), False)
            print(f"Image {idx} processed.")
        return self.get_x_preprocessed(idx)

    def get_y(self, idx):
        return np.asarray(self.data[idx]['y']).astype(np.float32)

    def __getitem__(self, idx):
        x = self.get_x(idx)
        x = self.transform(np.expand_dims(x, axis=0))
        x = torch.from_numpy(x)
        y = torch.from_numpy(self.get_y(idx))
        train_transforms = Compose([Resize(mode="trilinear", align_corners=False,
                                    spatial_size=(128, 128, 128))
##                                    spatial_size=(120, 120, 120))
                                    ])
        x = train_transforms(x).float()
        
        return x, y


if __name__ == "__main__":
    import sys
    import glob
    import matplotlib.pyplot as plt

    preprocess_dir = sys.argv[1]
    image_dir = sys.argv[2]

    data = [
        {'x': filename, 'y': [0, 0]}
        for filename in glob.glob(os.path.join(image_dir, "*.mha"))
    ]
    ctdataset = CTDataset(data, preprocess_dir)

    steps = 4
    for x, y in ctdataset:
        print(y)
        print(x.shape)
        x = x.numpy()[0]
        length = x.shape[1]
        start = length // 3
        stop = (length // 5) * 4
        step = (stop - start) // steps
        fig, axes = plt.subplots(1, steps, figsize=(15, 4))
        its = range(start, stop, step)
        for it, axis in zip(its, axes):
            screenshot = x[:, it, :][::-1]
            axis.imshow(screenshot, cmap='gray')
            axis.axis('off')
        plt.suptitle(f'label: {y}')
        plt.show()

