import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import tqdm
from monai.transforms import *

from monai.data.utils import SUPPORTED_PICKLE_MOD, pickle_hashing
import tempfile
from pathlib import Path
from monai.utils import look_up_option
import pickle
import shutil

class CTReportDatasetinfer(Dataset):
    def __init__(self, data_folder, csv_file, min_slices=20, resize_dim=500, force_num_frames=True, labels = "labels.csv"):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.labels = labels
        self.accession_to_text = self.load_accession_text(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.transform = transforms.Compose([
            transforms.Resize((resize_dim,resize_dim)),
            transforms.ToTensor()
        ])
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)

    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['VolumeName']] = row["Findings_EN"],row['Impressions_EN']
        return accession_to_text


    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        # Read labels once outside the loop
        test_df = pd.read_csv(self.labels)
        test_label_cols = list(test_df.columns[1:])
        test_df['one_hot_labels'] = list(test_df[test_label_cols].values)

        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz.npz'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]

##                    accession_number = accession_number.replace(".npz", ".nii.gz")
                    accession_number = accession_number.replace(".npz", "")
                    if accession_number not in self.accession_to_text:
                        continue

                    impression_text = self.accession_to_text[accession_number]
                    text_final = ""
                    for text in list(impression_text):
                        text = str(text)
                        if text == "Not given.":
                            text = ""

                        text_final = text_final + text

                    onehotlabels = test_df[test_df["VolumeName"] == accession_number]["one_hot_labels"].values
                    if len(onehotlabels) > 0:
                        if len(samples) < 1000:
                            samples.append((nii_file, text_final, onehotlabels[0]))
                            self.paths.append(nii_file)
                        else:
                            return samples
        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path, transform):
        
        item_transformed = path.split('/')[-1]
        cache_dir = Path('/home/xuefeng/cache/CT_RATE_Vocab_128_fore_cache')
        if cache_dir is not None:
            data_item_md5 = pickle_hashing(item_transformed).decode("utf-8")
            hashfile = cache_dir / f"{data_item_md5}.pt"
        else:
            hashfile = None

        if hashfile is not None and hashfile.is_file():  # cache hit
            tensor = torch.load(hashfile)  
        else:        
            try:
                img_data = np.load(path)['arr_0']

                img_data= torch.from_numpy(img_data).unsqueeze(0)
                train_transforms = Compose([ScaleIntensityRange(
                                                a_min=-1.0, a_max=1.0,
                                                b_min=0.0, b_max=1.0, clip=True),
                                        CenterSpatialCrop(roi_size=(0.88 * img_data.size(1), 
                                                                    0.875 * img_data.size(2), 
                                                                    0.65625 * img_data.size(3))),
                                        Resize(mode="trilinear", align_corners=False,
                                        spatial_size=(128, 128, 96))
                                            ])
                tensor = train_transforms(img_data)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    temp_hash_file = Path(tmpdirname) / hashfile.name
                    torch.save(
                        obj=tensor,
                        f=temp_hash_file,
                        pickle_module=look_up_option("pickle", SUPPORTED_PICKLE_MOD),
                        pickle_protocol=pickle.HIGHEST_PROTOCOL,
                    )
                    if temp_hash_file.is_file() and not hashfile.is_file():
                        shutil.move(str(temp_hash_file), hashfile)

            except PermissionError:  # project-monai/monai issue #3613
                pass

        return tensor

    def __getitem__(self, index):
        nii_file, input_text, onehotlabels = self.samples[index]
        video_tensor = self.nii_to_tensor(nii_file)
        input_text = input_text.replace('"', '')  
        input_text = input_text.replace('\'', '')  
        input_text = input_text.replace('(', '')  
        input_text = input_text.replace(')', '')  
        name_acc = nii_file.split("/")[-2]
        name_all = nii_file.split("/")[-1]
        return video_tensor, input_text, onehotlabels, name_acc, name_all
