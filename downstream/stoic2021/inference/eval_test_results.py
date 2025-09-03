import os
import pandas as pd
import shutil
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import sigmoid
from numpy import round

def acc_probcovid(preds, labels):
    preds = np.where(preds >= 0.5, 1, 0)
    return accuracy_score(labels, round(preds))

def acc_probseverecovid(preds, labels, masks):
    preds = np.where(preds >= 0.5, 1, 0)
    preds = preds[masks == 0.0]
    labels = labels[masks == 0.0]
    return accuracy_score(labels, preds)

def roc_probcovid(preds, labels):
    preds = np.where(preds >= 0.5, 1, 0)
    return roc_auc_score(labels, preds)

def roc_probseverecovid(preds, labels, masks):
    preds = preds[masks >= 0.0]
    labels = labels[masks >= 0.0]
    return roc_auc_score(labels, preds)

mask_path = os.path.join("/ssd1/xuefeng/metadata/test/test.csv")
result_path = os.path.join("/home/xuefeng/stoic2021/inference/result_swin3d.csv")
df_mask = pd.read_csv(mask_path)
df_result = pd.read_csv(result_path)
df_test_list = df_mask.apply(lambda row: int(row["PatientID"]), axis=1).to_list()
mask_probCOVID_list, mask_probSevere_list, result_probCOVID_list, result_probSevere_list = [], [], [], []
for i in range(len(df_test_list)):
    mask_probCOVID_list.extend(df_mask[(df_mask['PatientID'] == df_test_list[i])]['probCOVID'].to_list())
    mask_probSevere_list.extend(df_mask[(df_mask['PatientID'] == df_test_list[i])]['probSevere'].to_list())
    result_probCOVID_list.extend(df_result[(df_result['PatientID'] == df_test_list[i])]['probCOVID'].to_list())
    result_probSevere_list.extend(df_result[(df_result['PatientID'] == df_test_list[i])]['probSevere'].to_list())
mask_probCOVID_list = np.array(mask_probCOVID_list)
mask_probSevere_list = np.array(mask_probSevere_list)
result_probCOVID_list = np.array(result_probCOVID_list)
result_probSevere_list = np.array(result_probSevere_list)
acc_COVID = acc_probcovid(result_probCOVID_list, mask_probCOVID_list)
print('acc_COVID: ' + str(acc_COVID))
roc_COVID = roc_probcovid(result_probCOVID_list, mask_probCOVID_list)
print('roc_COVID: ' + str(roc_COVID))
acc_SEVERE = acc_probseverecovid(result_probSevere_list, mask_probSevere_list, mask_probCOVID_list)
print('acc_SEVERE: ' + str(acc_SEVERE))
roc_SEVERE = roc_probseverecovid(result_probSevere_list, mask_probSevere_list, mask_probCOVID_list)
print('roc_SEVERE: ' + str(roc_SEVERE))


