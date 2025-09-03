import os
import random

import pyarrow as pa
import torch
from PIL import Image

from mg3d.datasets.utils.utils import split_into_sents
from mg3d.datasets.transforms import keys_to_transforms

import torch.nn.functional as F
import numpy as np
import torchio as tio
import SimpleITK as sitk
from monai.transforms import *
import io

from monai.data.utils import SUPPORTED_PICKLE_MOD, pickle_hashing
import tempfile
from pathlib import Path
from monai.utils import look_up_option
import pickle
import shutil

def keep_words(string, num):
    words = string.split() # 将字符串按空格分割为单词列表
    
    if len(words) <= num: # 如果单词数小于等于指定数量，则直接返回原始字符串
        return string
        
    else:
        new_string = ' '.join(words[:num]) # 只保留前num个单词并添加省略号
        return new_string

class RandomCrop:
    def __init__(self, scale=(0.8, 1)):
        for n in scale:
            if not 0 < n <= 1:
                raise ValueError(f'Invalid scale value {n}')
        self.scale = scale

    def __call__(self, sample):

        spatial_shape = torch.Tensor([sample.shape[1:]]).squeeze()
        if spatial_shape[2] >= 96:
            crop_shape = torch.FloatTensor([96,96,96]).int()
        else:
            crop_shape = torch.FloatTensor([96,96,spatial_shape[2]]).int()
        h_center = random.randint(0, spatial_shape[0] - crop_shape[0])
        w_center = random.randint(0, spatial_shape[1] - crop_shape[1])
        d_center = random.randint(0, spatial_shape[2] - crop_shape[2])

        patch = sample[:, h_center:(h_center+crop_shape[0]), 
                       w_center:(w_center+crop_shape[1]), 
                       d_center:(d_center+crop_shape[2])]
        return patch

class MaskGenerator:
    def __init__(self, input_size=224, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6, slice=32):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        self.slice = slice
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        mask = np.expand_dims(mask, axis=0).repeat(self.slice, axis=0)

        return mask

class BaseDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir: str,
            transform_keys: list,
            image_size: int,
            names: list,
            text_column_name: str = "",
            max_text_len: int = 40,
            draw_false_image: int = 0,
            draw_false_text: int = 0,
            image_only: bool = False,
            label_column_name: str = "",
            sent_level: bool = False,
            data_frac: float = 1.,
    ):
        super().__init__()
        assert len(transform_keys) >= 1
        # Hyper-Parameters
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir
        self.label_column_name = label_column_name
        self.sent_level = sent_level
        self.data_frac = data_frac

        self.mask_generator = MaskGenerator(
            input_size=128,
            mask_patch_size=32,
            model_patch_size=4,
            mask_ratio=0.6,
            slice=96,  #16
        )

        # Image Transformations
        if "train" not in names[0]:
            transform_keys = [transform_key.replace("_randaug", "") for transform_key in transform_keys]
            transform_keys = [transform_key.replace("_resizedcrop", "") for transform_key in transform_keys]
        self.transforms = keys_to_transforms(transform_keys, size=image_size)
        self.transforms_brain = keys_to_transforms(['clip_randaug'], size=image_size)
        self.clip_transform = False
        for transform_key in transform_keys:
            if 'clip' in transform_key:
                self.clip_transform = True
                break

        # Read Texts
        if len(names) != 0:
            tables = [
                pa.ipc.RecordBatchFileReader(pa.memory_map(f"{data_dir}/{name}.arrow", "r")).read_all()
                for name in names
                if os.path.isfile(f"{data_dir}/{name}.arrow")
            ]
            self.table_names = list()
            for i, name in enumerate(names):
                self.table_names += [name] * len(tables[i])
            self.table = pa.concat_tables(tables, promote=True)
            if text_column_name != "":
                self.text_column_name = text_column_name
                self.all_texts = self.table[text_column_name].to_pandas().tolist()
                assert type(self.all_texts[0][0]) == str
            else:
                self.all_texts = list()
        else:
            self.all_texts = list()

        # Record Index Mappings
        self.index_mapper = dict()
        if text_column_name != "" and not self.image_only:
            j = 0
            for i, texts in enumerate(self.all_texts):
                for _j in range(len(texts)):
                    self.index_mapper[j] = (i, _j)
                    j += 1
        else:
            for i in range(len(self.table)):
                self.index_mapper[i] = (i, None)

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.index_mapper)

    def get_raw_image(self, index, image_key="image", img_size=96):
        index, caption_index = self.index_mapper[index]

        image_bytes = self.table["image_id"][index].as_py()
        if 'RATE' in image_bytes:
            path = image_bytes.split('/', 6)[6]
            image_bytes = '/jhcnas5/nixuefeng/CT-RATE/' + path
            images = torch.from_numpy(np.load(image_bytes)['arr_0']).unsqueeze(0)
            train_transforms = Compose([ScaleIntensityRange(
                                            a_min=-1.0, a_max=1.0,
                                            b_min=0.0, b_max=1.0, clip=True),
                                    CenterSpatialCrop(roi_size=(0.88 * images.size(1), 
                                                                0.875 * images.size(2), 
                                                                0.65625 * images.size(3))),
                                    Resize(mode="trilinear", align_corners=False,
                                    spatial_size=(128, 128, 96))
                                        ])
            transformed_image = train_transforms(images)
            image_tensors_tr = transformed_image.permute(1,0,3,2).contiguous()
        else:
            images, image_tensors = [], []
                
            for i in range(len(self.table[image_key][index])):
                image_bytes = io.BytesIO(self.table[image_key][index][i].as_py())
                image_bytes.seek(0)
                if self.clip_transform:
                    image = Image.open(image_bytes).convert("L")
                else:
                    image = Image.open(image_bytes).convert("L")
                images.append(image)
                
            if 'Chest' in self.table['image_id'][index].as_py():
                for i in range(len(images)):
                    image_tensor = [tr(images[i]) for tr in self.transforms]
                    image_tensors.append(image_tensor[0])
                image_tensors = image_tensors[::-1]
            
            image_tensors = torch.stack(image_tensors, -1)
            
            # crop foreground
            if 'Chest' in self.table['image_id'][index].as_py():
                image_tensors = image_tensors[:, :, :, int(len(images)*0.06):int(len(images)*0.94)]
                image_tensors_tr = F.interpolate(image_tensors,size=(128, 128)).permute(3,0,1,2).contiguous()     
            
        return image_tensors_tr

    def get_image(self, index, image_key="image", img_size=224):
        
        image_bytes = self.table["image_id"][index].as_py()
        if 'RATE' in image_bytes:
            cache_dir = Path('/home/xuefeng/cache/CT_RATE_128_fore_cache/')
        elif 'Chest' in self.table['image_id'][index].as_py():
            cache_dir = Path('/home/xuefeng/cache/CTRG_chest_fore_128_cache/')
            
        item_transformed = self.table['image_id'][index]
        if cache_dir is not None:
            data_item_md5 = pickle_hashing(item_transformed).decode("utf-8")
            hashfile = cache_dir / f"{data_item_md5}.pt"

        if hashfile is not None and hashfile.is_file():  # cache hit
            image = [torch.load(hashfile)]  
        else:        
            try:
                image = [self.get_raw_image(index, image_key=image_key, img_size=img_size)]
                
                with tempfile.TemporaryDirectory() as tmpdirname:
                    temp_hash_file = Path(tmpdirname) / hashfile.name
                    torch.save(
                        obj=image[0],
                        f=temp_hash_file,
                        pickle_module=look_up_option("pickle", SUPPORTED_PICKLE_MOD),
                        pickle_protocol=pickle.HIGHEST_PROTOCOL,
                    )
                    if temp_hash_file.is_file() and not hashfile.is_file():
                        shutil.move(str(temp_hash_file), hashfile)
            except PermissionError:  # project-monai/monai issue #3613
                pass
            
        mask = self.mask_generator()

        return {
            "image": image,
            "img_index": self.index_mapper[index][0],
            "cap_index": self.index_mapper[index][1],
            "raw_index": index,
            "mask": torch.from_numpy(mask),
            "hashfile": hashfile
        }
    

    def get_false_image(self, rep, image_key="image", selected_index=None, img_size=224):
        random_index = random.randint(0, len(self.index_mapper) - 1)
        
        image_bytes = self.table["image_id"][random_index].as_py()
        if 'RATE' in image_bytes:
            cache_dir = Path('/home/xuefeng/cache/CT_RATE_128_fore_cache/')
        elif 'Chest' in self.table['image_id'][random_index].as_py():
            cache_dir = Path('/home/xuefeng/cache/CTRG_chest_fore_128_cache/')
            
        item_transformed = self.table['image_id'][random_index]
        if cache_dir is not None:
            data_item_md5 = pickle_hashing(item_transformed).decode("utf-8")
            hashfile = cache_dir / f"{data_item_md5}.pt"

        if hashfile is not None and hashfile.is_file():  # cache hit
            image = [torch.load(hashfile)]  
        else:        
            try:
                image = [self.get_raw_image(random_index, image_key=image_key, img_size=img_size)]
                with tempfile.TemporaryDirectory() as tmpdirname:
                    temp_hash_file = Path(tmpdirname) / hashfile.name
                    torch.save(
                        obj=image[0],
                        f=temp_hash_file,
                        pickle_module=look_up_option("pickle", SUPPORTED_PICKLE_MOD),
                        pickle_protocol=pickle.HIGHEST_PROTOCOL,
                    )
                    if temp_hash_file.is_file() and not hashfile.is_file():
                        shutil.move(str(temp_hash_file), hashfile)
            except PermissionError:  # project-monai/monai issue #3613
                pass
            
        mask = self.mask_generator()
        
        return {f"false_image_{rep}": image, 
                f"false_mask_{rep}": torch.from_numpy(mask),
                f"hashfile_{rep}": hashfile
                }

    def get_text(self, raw_index):
        index, caption_index = self.index_mapper[raw_index]
        text = self.all_texts[index][caption_index]

        sents = split_into_sents(text)
        if self.sent_level:
            if len(sents) >= 1:
                sent_idx = random.randint(0, len(sents) - 1)
                text = sents[sent_idx]
        else:
            text = " ".join(sents)

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        )
        return {
            "text": (text, encoding),
            "img_index": index,
            "cap_index": caption_index,
            "raw_index": raw_index,
        }
        
    def get_text_sen(self, raw_index):
        index, caption_index = self.index_mapper[raw_index]
        text = self.all_texts[index][caption_index].lower()
        findings_sen = self.all_findings_sen[index]
        impression_sen = self.all_impression_sen[index]
        sentences_list = []

        sents = split_into_sents(text)
        if self.sent_level:
            if len(sents) >= 1:
                sent_idx = random.randint(0, len(sents) - 1)
                text = sents[sent_idx]
        else:
            text = " ".join(sents)
        for i in range(findings_sen.size):
            sents = split_into_sents(findings_sen[i].lower())
            if self.sent_level:
                if len(sents) >= 1:
                    sent_idx = random.randint(0, len(sents) - 1)
                    findings_sen[i] = sents[sent_idx]
            else:
                findings_sen[i] = " ".join(sents)
            sentences_list.append(findings_sen[i])
        for i in range(impression_sen.size):
            sents = split_into_sents(impression_sen[i].lower())
            if self.sent_level:
                if len(sents) >= 1:
                    sent_idx = random.randint(0, len(sents) - 1)
                    impression_sen[i] = sents[sent_idx]
            else:
                impression_sen[i] = " ".join(sents)
            sentences_list.append(impression_sen[i])

        sentences = ''
        sen_text = []
        for sen in sentences_list:
            sentences += sen
            
        tokens = self.tokenizer(
            sentences,
            padding="max_length",
            truncation=False,
            max_length=self.max_text_len,
            add_special_tokens=True,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        for key, token in tokens.items():
            tokens[key] = token.squeeze(dim=0)
        num_tokens = 0
        sentence_index = torch.ones_like(tokens['input_ids']) * -1
        start_index = 1

        # assign the index of sentence to each token
        for idx, sentence in enumerate(sentences_list):
            sentnece_token = self.tokenizer(sentence, return_tensors='pt')['input_ids'][0]
            len_sentence = sentnece_token.shape[0] - 2 # Remove [CLS] and [SEP]
            sentence_index[start_index: start_index + len_sentence] = idx + 1
            start_index += len_sentence

        # truncation 
        for key, value in tokens.items():
            #print(len(value))
            tokens[key] = value
            #print(len(tokens[key]))
            if len(value) > self.max_text_len:
                tokens[key] = value[ :self.max_text_len]
        if len(sentence_index) > self.max_text_len:
            sentence_index = sentence_index[ :self.max_text_len]

        for i in range(torch.max(sentence_index)):
            if (i == (torch.max(sentence_index) - 1)) and (len(sentences.split()) > self.max_text_len):
                last_num = torch.sum(sentence_index == torch.max(sentence_index)).item()
                sentences_list[i] = keep_words(sentences_list[i], last_num)
            encoding = self.tokenizer(
                sentences_list[i],
                padding=False,
                truncation=True,
                max_length=self.max_text_len,
                return_special_tokens_mask=True,
                return_offsets_mapping=True,
                return_tensors='pt'
            )        
            sen_text.append((sentences_list[i], encoding))

        return {
            "text": (sentences, tokens),
            "sen_txt": sen_text,
            "img_index": index,
            "cap_index": caption_index,
            "raw_index": raw_index,
            "num_tokens": num_tokens, 
            "sentence_index": sentence_index
        }
    
    def get_false_text_sen(self, rep, selected_index=None):
        random_index = random.randint(0, len(self.index_mapper) - 1)
        index, caption_index = self.index_mapper[random_index]
        text = self.all_texts[index][caption_index].lower()
        findings_sen = self.all_findings_sen[index]
        impression_sen = self.all_impression_sen[index]
        sentences_list = []
        
        if self.sent_level:
            sents = split_into_sents(text)
            if len(sents) >= 1:
                sent_idx = random.randint(0, len(sents) - 1)
                text = sents[sent_idx]
        for i in range(findings_sen.size):
            if self.sent_level:
                sents = split_into_sents(findings_sen[i].lower())
                if len(sents) >= 1:
                    sent_idx = random.randint(0, len(sents) - 1)
                    findings_sen[i] = sents[sent_idx]
            sentences_list.append(findings_sen[i])
        for i in range(impression_sen.size):
            if self.sent_level:
                sents = split_into_sents(impression_sen[i].lower())
                if len(sents) >= 1:
                    sent_idx = random.randint(0, len(sents) - 1)
                    impression_sen[i] = sents[sent_idx]
            sentences_list.append(impression_sen[i])
            
        sentences = ''
        sen_text = []
        for sen in sentences_list:
            encoding = self.tokenizer(
                sen,
                padding=False,
                truncation=True,
                max_length=self.max_text_len,
                return_special_tokens_mask=True,
                return_offsets_mapping=True,
                return_tensors='pt'
            )        
            sen_text.append((sen, encoding))
            sentences += sen

        tokens = self.tokenizer(
            sentences,
            padding="max_length",
            truncation=False,
            max_length=self.max_text_len,
            add_special_tokens=True,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        for key, token in tokens.items():
            tokens[key] = token.squeeze(dim=0)
        num_tokens = 0
        sentence_index = torch.ones_like(tokens['input_ids']) * -1
        start_index = 1

        # assign the index of sentence to each token
        for idx, sentence in enumerate(sentences_list):
            sentnece_token = self.tokenizer(sentence, return_tensors='pt')['input_ids'][0]
            len_sentence = sentnece_token.shape[0] - 2 # Remove [CLS] and [SEP]
            sentence_index[start_index: start_index + len_sentence] = idx + 1
            start_index += len_sentence

        # truncation 
        for key, value in tokens.items():
            #print(len(value))
            tokens[key] = value
            #print(len(tokens[key]))
            if len(value) > self.max_text_len:
                tokens[key] = value[ :self.max_text_len]
        if len(sentence_index) > self.max_text_len:
            sentence_index = sentence_index[ :self.max_text_len]
        return {f"false_text_{rep}": (sentences, tokens, sentence_index, num_tokens, sen_text)}

    def get_suite(self, index):
        result = None
        while result is None:
            img_size = 128 #96 160
            ret = dict()
            ret.update(self.get_image(index, img_size=img_size))
            if not self.image_only:
                txt = self.get_text_sen(index)
                ret.update({"replica": True if txt["cap_index"] > 0 else False})
                ret.update(txt)
                
            for i in range(self.draw_false_image):
                ret.update(self.get_false_image(i, selected_index=index, img_size=img_size))
            for i in range(self.draw_false_text):
                ret.update(self.get_false_text_sen(i, selected_index=index))
            result = True

        return ret
    
    def collate(self, batch, mlm_collator):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        mask_keys = [k for k in list(dict_batch.keys()) if "mask" in k]

        img_sizes = list()

        for img_key in img_keys:
            img = dict_batch[img_key]
            img_sizes += [ii.shape for i in img if i is not None for ii in i]

        if len(img_keys) != 0:
            max_depth = max([i[0] for i in img_sizes])
            max_height = max([i[2] for i in img_sizes])
            max_width = max([i[3] for i in img_sizes])

        for mask_key in mask_keys:
            view_size = len(img[0])
            masks = dict_batch[mask_key]
            new_masks = [torch.zeros(batch_size, masks[0].shape[0], masks[0].shape[1], masks[0].shape[2]) for _ in range(view_size)]
            for bi in range(batch_size):
                orig_batch = dict_batch[mask_key][bi]
                for vi in range(view_size):
                    if orig_batch is None:
                        new_masks[vi][bi] = None
                    else:
                        orig = masks[bi][vi]
                        new_masks[vi][bi, :, :, :] = orig
            dict_batch[mask_key] = new_masks

        for img_key in img_keys:
            img = dict_batch[img_key]

            view_size = len(img[0])
            new_images = [torch.zeros(batch_size, max_depth, 1, max_height, max_width) for _ in range(view_size)]

            for bi in range(batch_size):
                orig_batch = img[bi]
                for vi in range(view_size):
                    if orig_batch is None:
                        new_images[vi][bi] = None
                    else:
                        orig = img[bi][vi]
                        new_images[vi][bi, : orig.shape[0], :, : orig.shape[2], : orig.shape[3]] = orig

            dict_batch[img_key] = new_images

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]
        if len(txt_keys) != 0:
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = ([d[0] for d in dict_batch[txt_key]], [d[1] for d in dict_batch[txt_key]])
                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i): batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i): batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask
                
                
        sen_keys = [k for k in list(dict_batch.keys()) if "sen_txt" in k]
        dict_batch["sen_txt_"], dict_batch["sen_txt_ids"] = [], []
        dict_batch["sen_txt_labels"], dict_batch["sen_txt_masks"] = [], []
        if len(sen_keys) != 0:
            for b in range(len(dict_batch['sen_txt'])):
                b_sen_txt, b_sen_txt_ids, b_sen_txt_labels, b_sen_txt_masks = [], [], [], []
                for sen_idx in dict_batch['sen_txt'][b]:
                    encodings = [[sen_idx[1]] for sen_key in sen_keys]
                    flatten_encodings = [e for encoding in encodings for e in encoding]

                    for i, sen_key in enumerate(sen_keys):
                        texts, encodings = (sen_idx[0], [sen_idx[1]])

                        for _i, encoding in enumerate(encodings):
                            input_ids = torch.zeros_like(encoding["input_ids"])
                            attention_mask = torch.zeros_like(encoding["input_ids"])
                            _input_ids, _attention_mask = (
                                torch.tensor(encoding["input_ids"]),
                                torch.tensor(encoding["attention_mask"]),
                            )
                            input_ids = _input_ids
                            attention_mask = _attention_mask

                        sen_txt = texts
                        sen_txt_ids = input_ids
                        sen_txt_labels = torch.full_like(input_ids, -100)
                        sen_txt_masks = attention_mask
                    
                    b_sen_txt.append(sen_txt)
                    b_sen_txt_ids.append(sen_txt_ids)
                    b_sen_txt_labels.append(sen_txt_labels)
                    b_sen_txt_masks.append(sen_txt_masks)
                        
                dict_batch[f"{sen_key}_"].append(b_sen_txt)
                dict_batch[f"{sen_key}_ids"].append(b_sen_txt_ids)
                dict_batch[f"{sen_key}_labels"].append(b_sen_txt_labels)
                dict_batch[f"{sen_key}_masks"].append(b_sen_txt_masks)

        return dict_batch