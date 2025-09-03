from pprint import pprint

import pandas as pd
import json
import csv

from modules.metrics_clinical import CheXbertMetrics

from metrics import compute_scores


def main():
    
    gt_dict = {}
    pred_dict = {}
    eval_dict = {}

    # 读取 JSON 文件，假设文件名为data.json
    with open('/home/xuefeng/M2KT/report_vis/swinmm_Enc2Dec-14_0_test_generated.json', 'r') as file:
        data = json.load(file)
    
    for i in range(len(data)):  #len(data)
        # 将ID作为键，GT作为值存储在gt_dict中
        gt_dict[data[i]['filename']] = [data[i]['ground_truth']]

        # 将ID作为键，Pred作为值存储在pred_dict中
        pred_dict[data[i]['filename']] = [data[i]['prediction']]
        
        metrics = compute_scores(gt_dict, pred_dict)
        eval_dict[data[i]['filename']] = metrics
        gt_dict = {}
        pred_dict = {}

        # 提取 "prediction" 和 "gt" 的值到列表
    id_list = [item['filename'] for item in data]
    bleu1_list = [value['BLEU_1'] for key, value in eval_dict.items()]
    bleu2_list = [value['BLEU_2'] for key, value in eval_dict.items()]
    bleu3_list = [value['BLEU_3'] for key, value in eval_dict.items()]
    bleu4_list = [value['BLEU_4'] for key, value in eval_dict.items()]
    meteor_list = [value['METEOR'] for key, value in eval_dict.items()]
    rougel_list = [value['ROUGE_L'] for key, value in eval_dict.items()]

    data_dict = {
        'ID': id_list,
        'bleu1': bleu1_list,
        'bleu2': bleu2_list,
        'bleu3': bleu3_list,
        'bleu4': bleu4_list,
        'meteor': meteor_list,
        'rougel': rougel_list,
    }


    # 创建 DataFrame
    df = pd.DataFrame(data_dict)

    # 保存 DataFrame 到 CSV 文件
    df.to_csv('/home/xuefeng/M2KT/report_vis/swinmm_Enc2Dec-14_0_test_generated.csv', index=False)


if __name__ == '__main__':
    main()

