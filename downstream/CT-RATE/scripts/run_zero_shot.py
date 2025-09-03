import torch
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel

from ct_clip import CTCLIP
##from ct_clip.ct_clip_latent_order import CTCLIP
##from ct_clip.ct_clip_latent_mm import CTCLIP

from zero_shot import CTClipInference
##from zero_shot_latent import CTClipInference
##from zero_shot_latent_one_to_all import CTClipInference

import accelerate
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

text_encoder.resize_token_embeddings(len(tokenizer))

backbone = 'swin_unetr'


if backbone == 'swin_unetr_l':
    from monai.utils import ensure_tuple_rep
    from baseline.swin_l import Swin
    spatial_dims = 3
    patch_size = ensure_tuple_rep(2, spatial_dims)
    window_size = ensure_tuple_rep(7, spatial_dims)
    image_encoder = Swin(in_channels=1, 
                             feature_size=96, 
                             spatial_dims=spatial_dims)   
    clip = CTCLIP(
            image_encoder=image_encoder, text_encoder=text_encoder,
            dim_image=73728, dim_text=768, dim_latent=768,  #196608 20736 36864
            extra_latent_projection=False, use_mlm=False,
            downsample_image_embeds=False, use_all_token_embeds=False
    )
    clip.load("/jhcnas5/nixuefeng/CT-CLIP-main/exps/exps_fore_rate_47k_swinl_proj_inter_no_itm_coatt_mm_all_1000_lr-1e-6_1000/checkpoint_0_epoch_10.pt")

elif backbone == 'swin_unetr_b':
    from monai.utils import ensure_tuple_rep
    from baseline.swin_b import SwinTransformer
    spatial_dims = 3
    patch_size = ensure_tuple_rep(2, spatial_dims)
    window_size = ensure_tuple_rep(7, spatial_dims)
    swin = SwinTransformer(            
            in_chans=1,
            embed_dim=48, 
            window_size=window_size,
            patch_size=patch_size,
            depths=(2, 2, 2, 2), 
            num_heads=(3, 6, 12, 24),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=False,
            spatial_dims=spatial_dims,
            classification=True,
            num_classes=18)
    clip = CTCLIP(
        image_encoder=swin, text_encoder=text_encoder,
        dim_image=36864, dim_text=768, dim_latent=768, #36864 20736
        extra_latent_projection=False, use_mlm=False,
        downsample_image_embeds=False, use_all_token_embeds=False
    )
    clip.load("/jhcnas5/nixuefeng/jhcpu5_data_files/CT-CLIP-main/exps/exps_vocabfine96_new_correct_swin3d_comma_96_rate1.4k_sen_comma_milm_ml_sfa_mh_all_fc_cml_ml_dcl_reuse_ml_fc_gl_att_mlp_all_itc_ori_split/checkpoint_0_epoch_10.pt")

inference = CTClipInference(
    clip,
    data_folder = '/jhcnas4/nixuefeng/CT-RATE/valid_preprocessed/',
    reports_file= "/home/xuefeng/datasets/CT-RATE/dataset/radiology_text_reports/validation_reports.csv",
    labels = "/home/xuefeng/datasets/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv",
    batch_size = 1,
    results_folder="/jhcnas5/nixuefeng/jhcpu5_data_files/CT-CLIP-main/exps/exps_vocabfine96_new_correct_swin3d_comma_96_rate1.4k_sen_comma_milm_ml_sfa_mh_all_fc_cml_ml_dcl_reuse_ml_fc_gl_att_mlp_all_itc_ori_split/",    #inference_zeroshot
    num_train_steps = 1,
)

inference.infer()
