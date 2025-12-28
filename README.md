# MG-3D
This is the implementation of [MG-3D: Multi-Grained Knowledge-Enhanced Vision-Language Pre-training for 3D Medical Image Analysis](https://arxiv.org/abs/2412.05876).

## Table of Contents

- [Requirements](#requirements)
- [Pre-training](#pre-training)
- [Downstream Evaluation](#downstream-evaluation)

## Requirements

Run the following command to install the required packages:

```bash
pip install -r requirements.txt
```

## Preparation
You can download the CT-RATE and CTRG-Chest datasets used in this work via the Hugging Face repository (https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) and Github (https://github.com/tangyuhao2016/CTRG).

The project structure should be:
```
root:[.]
+--mg3d
| +--datasets
| +--datamodules
| +--metrics
| +--models
| +--config.py
| +--__init__.py
+--prepro
| +--glossary.py
| +--make_arrow.py
| +--prepro_finetuning_language_data.py
| +--prepro_finetuning_data.py
| +--prepro_finetuning_vision_data.py
| +--prepro_pretraining_data.py
+--data
| +--pretrain_arrows
| +--finetune_arrows
| +--finetune_vision_arrows
| +--finetune_language_arrows
+--run_scripts
| +--pretrain.sh
| +--finetune.sh
+--tools
| +--visualize_datasets.py
| +--convert_meter_weights.py
+--downstream
| +--ACDC
| +--cc-ccii
| +--Covid19_20
| +--CT-RATE
| +--CTRG
| +--Luna16
| +--MSD
| +--stoic2021
+--requirements.txt
+--README.md
+--main.py
```

## Pre-training

### 1. Pre-processing

Run the following command to pre-process the data:

```angular2
python prepro/prepro_pretraining_data.py
```

to get the following arrow files:

```angular2
root:[data]
+--pretrain_arrows
| +--clm_chest_ctrg_train.arrow
| +--clm_chest_ctrg_val.arrow
| +--clm_chest_ctrg_test.arrow
| +--clm_ct_rate_train.arrow
| +--clm_ct_rate_val.arrow
| +--clm_ct_rate_test.arrow
```

### 2. Pre-training

Now we can start to pre-train the ptunifer model:

```angular2
Single GPU:
bash run_scripts/pretrain.sh

Multiple GPUs:
bash run_scripts/pretrain_multi_gpus.sh
```

### 3. Pre-trained Models

We provide various models for downstream tasks. You can find the [3D Swin-B-47K](https://drive.google.com/file/d/1Aew0la4wPbxOKaF3BCApS15Pv-TYsRQr/view?usp=drive_link), [3D Swin-L-47K](https://drive.google.com/file/d/1--ELM9N13MvJhB82xpiB3pqf29Gg8ICj/view?usp=drive_link), [3D UNet-1.4K](https://drive.google.com/file/d/1uPFzy66FshFebZu9UcT1qse7m0_VI-FS/view?usp=drive_link), and [3D nn-UNet-1.4K](https://drive.google.com/file/d/1IhC9G6T50LhleC31K7fG649UJSVzrENV/view?usp=drive_link).

## Acknowledgement

The code is based on [PTunifier](https://github.com/zhjohnchan/PTUnifier), [MONAI](https://github.com/Project-MONAI/MONAI), [CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP), [M2KT](https://github.com/LX-doctorAI1/M2KT).

We thank the authors for their open-sourced code and encourage users to cite their works when applicable.

## Citation

If you find this repo useful for your research, please consider citing the paper as follows:
```latex
@article{ni2024mg,
  title={MG-3D: Multi-Grained Knowledge-Enhanced 3D Medical Vision-Language Pre-training},
  author={Ni, Xuefeng and Wu, Linshan and Zhuang, Jiaxin and Wang, Qiong and Wu, Mingxiang and Vardhanabhuti, Varut and Zhang, Lihai and Gao, Hanyu and Chen, Hao},
  journal={arXiv preprint arXiv:2412.05876},
  year={2024}
}
```
