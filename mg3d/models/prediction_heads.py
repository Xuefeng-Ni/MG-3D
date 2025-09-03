import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform

from mg3d.models.vision_encoders.position_embeddings import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed, get_3d_sincos_pos_embed_2
from mg3d.models.vision_encoders.clip_model import Transformer, LayerNorm
import math
import torch.nn.functional as F
import numpy as np

from mg3d.models.gather import SentenceGather
from mg3d.models.sentence_pool import SentenceAttentionPool


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class Swin_Pooler(nn.Module):
    def __init__(self, hidden_size, img_size=96):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Linear(hidden_size, hidden_size)
        
        if img_size == 96:
            self.fc = nn.Linear(27, 1)
        elif img_size == 128:
            self.fc = nn.Linear(48, 1)
        elif img_size == 160:
            self.fc = nn.Linear(50, 1)
        elif img_size == 192:
            self.fc = nn.Linear(36, 1)
            
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        hidden_states = self.fc(hidden_states.transpose(1, 2))
        hidden_states = torch.flatten(hidden_states, 1)
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class Sentence_Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Sequential(nn.Linear(hidden_size, 384),
                                    nn.BatchNorm1d(384, affine=False, track_running_stats=False),
                                    nn.GELU(),
                                    nn.Linear(384, 128)) # output layer # used for simsiam loss
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class Swin_Att_Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.global_attention = SentenceAttentionPool(32, 768, pos_embed=False) # Max sentence num: 32
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        hidden_states = self.avgpool(hidden_states.transpose(1, 2))
        hidden_states = torch.flatten(hidden_states, 1)
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class SEN_MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.item_gather = SentenceGather("avg", config.hidden_size)
        self.fc = nn.Sequential(nn.Linear(config.hidden_size, 128),
                                    nn.BatchNorm1d(128, affine=False, track_running_stats=False),
                                    nn.GELU(),
                                    nn.Linear(128, 8)) # output layer # used for simsiam loss
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x, batch):
        x = self.transform(x)
        
        batch_stacks = []
        local_text_embed_stacks = self.item_gather(x, batch)
        uni_modal_text_feats = local_text_embed_stacks

        for local_text_embed in uni_modal_text_feats:
            batch_stacks.append(self.fc(local_text_embed))

        x = self.decoder(x) + self.bias
        return x, batch_stacks
    
class MIM3DHead(nn.Module):
    def __init__(self, config, cls_token=True):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.patch_size = 16        
        self.num_patches = 8*6*8
        self.decoder_hidden_size = config["mim_decoder_hidden_size"]
        self.decoder_num_layers = config["mim_decoder_num_layers"]
        self.decoder_num_heads = config["mim_decoder_num_heads"]
        self.decoder_num_channels = 1 * config["patch_size"] ** 3

        self.decoder_embed = nn.Linear(self.hidden_size, self.decoder_hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_hidden_size))
        torch.nn.init.normal_(self.mask_token, std=.02)

        if cls_token:
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1,
                                                          self.decoder_hidden_size), requires_grad=False)
            decoder_pos_embed = get_3d_sincos_pos_embed_2(self.decoder_hidden_size, 
                                                          math.ceil(math.pow(self.num_patches, 1/3)), True)
        else:
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches,
                                                          self.decoder_hidden_size), requires_grad=False)
            decoder_pos_embed = get_3d_sincos_pos_embed_2(self.decoder_hidden_size, math.ceil(math.pow(self.num_patches, 1/3)), False)

        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        self.decoder = Transformer(self.decoder_hidden_size, self.decoder_num_layers + 1, self.decoder_num_heads)
        self.decoder_norm = LayerNorm(self.decoder_hidden_size)
        self.decoder_pred = nn.Linear(self.decoder_hidden_size, self.patch_size ** 3 * 1, bias=True)

    def forward(self, x, ids_restore, cls_token=True):
        # embed tokens
        x = self.decoder_embed(x)

        x_shape = x.shape
        depth = int(ids_restore.shape[0]/x_shape[0])
        x = x.view(x_shape[0], depth, int(x_shape[1]/depth), x_shape[2])
        x = x.view(ids_restore.shape[0], x.shape[2], x.shape[3])

        # append mask tokens to sequence
        if cls_token:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        else:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = x_

        # add pos embed
        x = x + self.decoder_pos_embed.to(x.dtype)

        # apply Transformer blocks
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.decoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        if cls_token:
            x = x[:, 1:, :]

        return x

class MIM_SWINUNETR_Head(nn.Module):
    def __init__(self, upsample="vae", dim=768):
        super().__init__()
        in_channels = 1
        if upsample == "large_kernel_deconv":
            self.conv = nn.ConvTranspose3d(dim, in_channels, kernel_size=(32, 32, 32), stride=(32, 32, 32))
        elif upsample == "deconv":
            self.conv = nn.Sequential(
                nn.ConvTranspose3d(dim, dim // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 4, dim // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 8, dim // 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 16, in_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            )
        elif upsample == "vae":
            self.conv = nn.Sequential(
                nn.Conv3d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 2),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 4),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 8),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 16, in_channels, kernel_size=1, stride=1),
            )
            self.fc = nn.Sequential(nn.Linear(dim, 128),
                                    nn.BatchNorm1d(128, affine=False, track_running_stats=False),
                                    nn.GELU(),
                                    nn.Linear(128, 8)) # output layer # used for simsiam loss

    def forward(self, x_out):
        _, c, h, w, d = x_out.shape
        x_rec = x_out.flatten(start_dim=2, end_dim=4)
        x_rec_feats = self.fc(x_rec.permute(0, 2, 1))
        x_rec = x_rec.view(-1, c, h, w, d)
        x_rec = self.conv(x_rec)
        return x_rec, x_rec_feats

class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )


allgather = AllGather.apply


class ITCHead(nn.Module):
    def __init__(self, hidden_size, temp):
        super().__init__()

        self.vision_ln = LayerNorm(hidden_size * 2)
        self.language_ln = LayerNorm(hidden_size * 2)
        self.vision_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.language_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.temp = temp

    def forward(self, image_feats, text_feats, idx=None):
        image_feats = self.vision_proj(self.vision_ln(image_feats))
        text_feats = self.language_proj(self.language_ln(text_feats))
        
        # normalized features
        image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
        text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)

        # gather features
        image_feats_all = allgather(image_feats, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feats_all = allgather(text_feats, torch.distributed.get_rank(), torch.distributed.get_world_size())

        # cosine similarity as logits
        logits_per_image = image_feats_all @ text_feats_all.t() / self.temp
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

    def proj_images(self, image_feats):
        image_feats = self.vision_proj(self.vision_ln(image_feats))
        return image_feats

    def proj_texts(self, text_feats):
        text_feats = self.language_proj(self.language_ln(text_feats))
        return text_feats

class ITCHead_Proj(nn.Module):
    def __init__(self, hidden_size, temp):
        super().__init__()
        self.vision_ln = LayerNorm(hidden_size)
        self.language_ln = LayerNorm(hidden_size)
        self.vision_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.language_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.temp = temp

    def forward(self, image_feats, text_feats, idx=None):
        image_feats = self.vision_proj(self.vision_ln(image_feats))
        text_feats = self.language_proj(self.language_ln(text_feats))
        
        # normalized features
        image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
        text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)

        # gather features
        image_feats_all = allgather(image_feats, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feats_all = allgather(text_feats, torch.distributed.get_rank(), torch.distributed.get_world_size())

        # cosine similarity as logits
        logits_per_image = image_feats_all @ text_feats_all.t() / self.temp
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

    def proj_images(self, image_feats):
        image_feats = self.vision_proj(self.vision_ln(image_feats))
        return image_feats

    def proj_texts(self, text_feats):
        text_feats = self.language_proj(self.language_ln(text_feats))
        return text_feats

class ITCHead_Single(nn.Module):
    def __init__(self, hidden_size, temp):
        super().__init__()

        self.vision_ln = LayerNorm(hidden_size)
        self.language_ln = LayerNorm(hidden_size)
        self.vision_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.language_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.temp = temp

    def forward(self, image_feats, text_feats, idx=None):
        image_feats = self.vision_proj(self.vision_ln(image_feats))
        text_feats = self.language_proj(self.language_ln(text_feats))

        # normalized features
        image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
        text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)

        # gather features
        image_feats_all = allgather(image_feats, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feats_all = allgather(text_feats, torch.distributed.get_rank(), torch.distributed.get_world_size())

        # cosine similarity as logits
        logits_per_image = image_feats_all @ text_feats_all.t() / self.temp
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

    def proj_images(self, image_feats):
        image_feats = self.vision_proj(self.vision_ln(image_feats))
        return image_feats

    def proj_texts(self, text_feats):
        text_feats = self.language_proj(self.language_ln(text_feats))
        return text_feats

class ITCHead_AGG(nn.Module):
    def __init__(self, temp):
        super().__init__()

        self.local_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temp))

    def forward(self, embed_A, embed_B, norm=True):
        logit_scale = self.local_logit_scale.exp()
        if norm:
            embed_A = F.normalize(embed_A, dim=-1, p=2)
            embed_B = F.normalize(embed_B, dim=-1, p=2)
        logits_per_text_agg = logit_scale * embed_B @ embed_A.t()
        logits_per_text = logit_scale * embed_A @ embed_B.t()

        return logits_per_text_agg, logits_per_text

class ITCHead_T3D(nn.Module):
    def __init__(self, hidden_size, temp):
        super().__init__()
        self.vision_ln = LayerNorm(hidden_size)
        self.language_ln = LayerNorm(hidden_size)
        self.vision_proj = nn.Linear(hidden_size, 128, bias=False)
        self.language_proj = nn.Linear(hidden_size, 128, bias=False)

        self.patch_pool_3d = nn.AvgPool3d(kernel_size=3)

        self.temp = 0.7 #temp

    def forward(self, image_feats, text_feats, idx=None):

        image_feats = self.vision_proj(self.vision_ln(image_feats))
        text_feats = self.language_proj(self.language_ln(text_feats))

        # normalized features
        image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
        text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)

        # gather features
        image_feats_all = allgather(image_feats, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feats_all = allgather(text_feats, torch.distributed.get_rank(), torch.distributed.get_world_size())

        # cosine similarity as logits
        logits_per_image = image_feats_all @ text_feats_all.t() / self.temp
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

    def proj_images(self, image_feats):
        image_feats = self.vision_proj(self.vision_ln(image_feats))
        return image_feats

    def proj_texts(self, text_feats):
        text_feats = self.language_proj(self.language_ln(text_feats))
        return text_feats

class IRHead_T3D(nn.Module):
    def __init__(self, upsample="vae", dim=768):
        super().__init__()
        in_channels = 1
        if upsample == "large_kernel_deconv":
            self.conv = nn.ConvTranspose3d(dim, in_channels, kernel_size=(32, 32, 32), stride=(32, 32, 32))
        elif upsample == "deconv":
            self.conv = nn.Sequential(
                nn.ConvTranspose3d(dim, dim // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 4, dim // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 8, in_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            )
        elif upsample == "vae":
            self.conv = nn.Sequential(
                nn.Conv3d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 2),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 4),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 8),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                
                nn.Conv3d(dim // 8, dim // 8, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 8),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 8, in_channels, kernel_size=1, stride=1),
            )

    def forward(self, x_out):
        _, c, h, w, d = x_out.shape
        x_rec = x_out.flatten(start_dim=2, end_dim=4)
        x_rec = x_rec.view(-1, c, h, w, d)
        x_rec = self.conv(x_rec)
        return x_rec