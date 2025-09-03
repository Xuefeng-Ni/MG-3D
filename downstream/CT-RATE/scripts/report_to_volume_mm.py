import numpy as np
import torch
import tqdm
import torch.nn.functional as F
from ct_clip.local_attention import MIMCrossMultiHeadAttention, MLMCrossMultiHeadAttention
from einops import rearrange, repeat, reduce
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def find_top_k_indices(values,k, ks):
    # Check if the list has at least 50 values
##    if len(values) < 50:
    if len(values) < ks:
        raise ValueError("The list must contain at least 50 values")

    # Use a combination of 'sorted' and 'enumerate' to sort the values while keeping track of indices
    sorted_values_with_indices = sorted(enumerate(values), key=lambda x: x[1], reverse=True)

    # Extract the indices of the top 50 values
    top_50_indices = [index for index, value in sorted_values_with_indices[:k]]

    return top_50_indices

def l2norm(t):
    return F.normalize(t, dim = -1)


data_folder = "/jhcnas5/nixuefeng/CT-CLIP-main/exps/exps_fore_rate_47k_swinl_proj_inter_no_itm_coatt_mm_new_1000_lr-1e-6/"

image_data= np.load(data_folder+"image_latents_order.npz")["data"]
# image_data_1= np.load(data_folder+"image_latents_order_1.npz")["data"]
# image_data_2= np.load(data_folder+"image_latents_order_2.npz")["data"]
text_data = np.load(data_folder+"text_latents_order.npz")["data"]

print(image_data.shape)

from transformers import BertTokenizer, BertModel
from monai.utils import ensure_tuple_rep
from baseline.swin_b import SwinTransformer
from baseline.swin_l import Swin
from ct_clip.ct_clip_r2v_swinb_mm import CTCLIP_B
from ct_clip.ct_clip_r2v_swinl_mm import CTCLIP_L

tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
text_encoder.resize_token_embeddings(len(tokenizer))


backbone = 'swin_unetr_b' # 'swin_unetr_l'
if backbone == 'swin_unetr_b':
    spatial_dims = 3
    patch_size = ensure_tuple_rep(2, spatial_dims)
    window_size = ensure_tuple_rep(7, spatial_dims)
    swin = SwinTransformer(            
                in_chans=1,
                embed_dim=96,   #48 96
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
    clip = CTCLIP_B(
            image_encoder=swin, text_encoder=text_encoder,
            dim_image=36864, dim_text=768, dim_latent=768, #196608 36864 20736
            extra_latent_projection=False, use_mlm=False,
            downsample_image_embeds=False, use_all_token_embeds=False
    )
else:
    spatial_dims = 3
    patch_size = ensure_tuple_rep(2, spatial_dims)
    window_size = ensure_tuple_rep(7, spatial_dims)
    image_encoder = Swin(in_channels=1, 
                            feature_size=96, 
                            spatial_dims=spatial_dims)   
    clip = CTCLIP_L(
            image_encoder=image_encoder, text_encoder=text_encoder,
            dim_image=73728, dim_text=768, dim_latent=768,  #196608 20736 36864
            extra_latent_projection=False, use_mlm=False,
            downsample_image_embeds=False, use_all_token_embeds=False
    )
clip.load("/jhcnas5/nixuefeng/CT-CLIP-main/exps/exps_fore_rate_47k_swinl_proj_inter_no_itm_coatt_mm_new_1000_lr-1e-6/checkpoint_999_epoch_12.pt")
clip.cuda()
clip.eval()

list_texts = []
list_ks=[50] #5 10 50 100
with torch.no_grad():
    for value in tqdm.tqdm(list_ks):
        num_is_in=0
        num_random=0

        for i in tqdm.tqdm(range(text_data.shape[0])):
            crosses = []
            crosses_rands=[]
            for k in range(image_data.shape[0]):
                text = torch.tensor(text_data[i]).cuda()
                image = torch.tensor(image_data[k]).cuda()
                
                text_mm, image = clip(text, image, k, device='cuda', return_latents=True)
                
                cross = text_mm.squeeze(0) @ image.squeeze(0)
                # text = torch.mean(text, dim=1)
                # cross = text.squeeze(0) @ image.squeeze(0)
                # cross = text.squeeze(0) @ text_mm.squeeze(0)
                crosses.append(cross)

            top_k_indices = find_top_k_indices(crosses,value, list_ks[0])

            if i in top_k_indices:
                num_is_in=num_is_in+1

            for k in range(image_data.shape[0]):
                size = (512)
                text =  torch.rand(size).cuda()
                image = torch.rand(size).cuda()

                crosses_rand= text @ image
                crosses_rands.append(crosses_rand)
            top_k_indices = find_top_k_indices(crosses_rands,value, list_ks[0])

            if i in top_k_indices:
                num_random=num_random+1

        clip = num_is_in/text_data.shape[0]
        rand = num_random/text_data.shape[0]
        write_str = f"K={value}, clip = {clip}, rand= {rand}"
        list_texts.append(write_str)


file_path = data_folder + f"internal_accessions_t2i_{list_ks[0]}.txt"

# Open the file for writing (you can also use "a" to append if the file already exists)
with open(file_path, "w") as file:
    # Write each string from the list to the file
    for string in list_texts:
        file.write(string + "\n")

# File has been written, close it
file.close()
