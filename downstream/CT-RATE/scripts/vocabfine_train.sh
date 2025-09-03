CUDA_VISIBLE_DEVICES=7 python ct_vocabfine_train.py \
    --lr 1e-5 \
    --wd 0.1 \
    --epochs 10 \
    --warmup_length 10000 \
    --save /home/xuefeng/CT-CLIP-main/exps/ \
    --pretrained /jhcnas5/nixuefeng/CT-RATE/models/CT_CLIP_zeroshot.pt \
    --data-folder /jhcnas5/nixuefeng/CT-RATE/train_preprocessed/ \
    --reports-file /jhcnas5/nixuefeng/CT-RATE/dataset/radiology_text_reports/train_reports.csv \
    --labels /jhcnas5/nixuefeng/CT-RATE/dataset/multi_abnormality_labels/train_predicted_labels.csv