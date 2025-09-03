import os
import json
import random
import ast
from monai.data import *
from monai.apps import DecathlonDataset
from copy import deepcopy
from monai.transforms import *

data_dir = "/mnt/hdd2/xuefeng/Dataset027_ACDC"
datalist_json = "/home/xuefeng/Downstream/ACDC/ACDC.json"
train_list = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
val_list = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)

# 创建annotation字典
annotation = {}
val_list_new = []
train_list_new = train_list.copy()

for i in range(int(len(train_list) / 4)):
    val_list_new.append(train_list[i*4 + 3])
    
for i in range(int(160 / 4)):
    train_list_new.remove(train_list[i*4 + 3])
    
train_list_new = train_list_new + val_list

annotation["training"] = train_list_new
annotation["validation"] = val_list_new

# 将annotation字典保存为JSON文件
with open("/home/xuefeng/Downstream/ACDC/ACDC_fold0.json", "w") as json_file:
    json.dump(annotation, json_file)