import functools

import torch
import torch.nn.functional as F
import tqdm
from einops import rearrange
from torch.utils.data.distributed import DistributedSampler

from mg3d.models.utils.dist_utils import all_gather
import torch.nn as nn

from monai.data.utils import SUPPORTED_PICKLE_MOD, pickle_hashing
import tempfile
from pathlib import Path
from monai.utils import look_up_option
import pickle
import shutil

def ce_loss(labels, logits):
    pos_dis = torch.abs(labels - logits)
    pos_loss = - labels * torch.log(1 - pos_dis + 1e-6)
    pos_loss = pos_loss.sum() / (labels.sum() + 1e-6)

    neg_lab = (labels == 0).long()
    neg_loss = neg_lab * (logits ** 2)
    neg_loss = neg_loss.sum() / (neg_lab.sum() + 1e-6)
    return pos_loss, neg_loss

def regularization_loss(bases):
    k, c = bases.size()
    loss_all = 0
    num = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            num += 1
            simi = F.cosine_similarity(bases[i].unsqueeze(0), bases[j].unsqueeze(0).detach(), dim=1)
            simi = F.relu(simi)
            loss_all += simi ** 2
    loss_all = loss_all / num

    return loss_all

def cosine_similarity(tensor_1, tensor_2):
    normalized_tensor_1 = F.normalize(tensor_1, p=2, dim=(1))
    normalized_tensor_2 = F.normalize(tensor_2, p=2, dim=(1))
    cosine_sim= torch.mm(normalized_tensor_1 ,normalized_tensor_2.permute(1,0))

    return cosine_sim

def text_local_loss_fn(embed_A, embed_B, norm=True):
    if norm:
        embed_A = F.normalize(embed_A, dim=-1, p=2)
        embed_B = F.normalize(embed_B, dim=-1, p=2)
    lc_labels = torch.arange(embed_B.size(0), device=embed_B.device).long()
    logits_per_image = embed_B @ embed_A.t()
    logits_per_text = embed_A @ embed_B.t()
    image_loss = F.cross_entropy(logits_per_image, lc_labels)
    text_loss = F.cross_entropy(logits_per_text, lc_labels)
    loss = (image_loss + text_loss) / 2   
    return loss

class MutualLoss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.alpha = 1.0
        self.mask_ratio = args.mask_ratio
        self.recon_loss_2 = torch.nn.MSELoss().cuda()

    def __call__(self, rec1, rec2, ask):
        mask = mask.to(dtype=rec1.dtype)
        rec1 = rec1 * mask 

        recon_loss = self.recon_loss_2(rec1, rec2) / self.mask_ratio
        return self.alpha * recon_loss

class SWIN_MIM_Loss(torch.nn.Module):
    def __init__(self, mask_ratio):
        super().__init__()

        self.recon_loss_2 = torch.nn.MSELoss().cuda()
        self.alpha3 = 1.0
        self.norm_pix_loss = False
        self.mask_ratio = mask_ratio

    def __call__(
        self,
        output_recons,
        target_recons,
        mask
    ):
        B, C, H, W, D = output_recons.shape
        target_recons = target_recons.reshape(B, C, -1)

        if self.norm_pix_loss:
            mean = target_recons.mean(dim=-1, keepdim=True)
            var = target_recons.var(dim=-1, keepdim=True)
            target_recons = (target_recons - mean) / (var + 1.0e-6) ** 0.5
        target_recons = target_recons.reshape(B, C, H, W, D)
        # masked voxels.
        mask = mask.to(dtype=target_recons.dtype)[None, ...]
        target_recons, output_recons = [val * mask for val in [target_recons, output_recons]]
        recon_loss = self.recon_loss_2(output_recons, target_recons) / self.mask_ratio
        recon_loss = self.alpha3 * recon_loss

        return recon_loss

def compute_mlm(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False, task="mlm")
    mlm_logits = pl_module.mlm_head(infer["multi_modal_text_feats"])
    mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(ret["mlm_logits"], ret["mlm_labels"])
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", acc)

    return ret


def compute_sen_mlm(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False, task="sen_mlm")
    mlm_logits, recon_sen_feats = pl_module.mlm_head(infer["multi_modal_text_feats"], batch)
    mlm_labels = infer["text_labels"]
    full_text_feats = infer["uni_modal_text_cls_feats"]
    full_img_feats = infer["uni_modal_image_feats"]
    full_sen_feats = infer["uni_modal_sentence_feats"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )


    # SAL Loss
    measure = 'cosin' #'cosin' 'distance' 'cosin_avg'
    recon_sen_losses = 0
    cosine_similarity = nn.CosineSimilarity(dim=2, eps=1e-6)
    for i in range(len(recon_sen_feats)):
        if measure == 'cosin':
            recon_sen_loss = 1 - cosine_similarity(recon_sen_feats[i], full_sen_feats[i])
        
        recon_sen_losses += torch.sum(recon_sen_loss)
    recon_sen_losses = 0.1 * recon_sen_losses / len(recon_sen_feats)
    mlm_loss = mlm_loss + recon_sen_losses

    
    ret = {
        "sen_mlm_loss": mlm_loss,
        "sen_mlm_logits": mlm_logits,
        "sen_mlm_labels": mlm_labels,
        "sen_mlm_ids": infer["text_ids"],
        "full_img_feats": full_img_feats, 
        "full_text_feats": full_text_feats, 
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_sen_mlm_loss")(ret["sen_mlm_loss"])
    acc = getattr(pl_module, f"{phase}_sen_mlm_accuracy")(ret["sen_mlm_logits"], ret["sen_mlm_labels"])
    pl_module.log(f"sen_mlm/{phase}/loss", loss)
    pl_module.log(f"sen_mlm/{phase}/accuracy", acc)

    return ret


def compute_umlm(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False, pseudo_vision=True)
    umlm_logits = pl_module.mlm_head(infer["multi_modal_text_feats"])
    umlm_labels = infer["text_labels"]

    umlm_loss = F.cross_entropy(
        umlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        umlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "umlm_loss": umlm_loss,
        "umlm_logits": umlm_logits,
        "umlm_labels": umlm_labels,
        "umlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_umlm_loss")(ret["umlm_loss"])
    acc = getattr(pl_module, f"{phase}_umlm_accuracy")(ret["umlm_logits"], ret["umlm_labels"])
    pl_module.log(f"umlm/{phase}/loss", loss)
    pl_module.log(f"umlm/{phase}/accuracy", acc)

    return ret

def compute_swin_unetr_mim(pl_module, batch, full_img_feats, full_text_feats):

    alpha = 1

    target_recons = batch['image'][0]

    feats = pl_module.infer(batch, img=batch['image'][0], pseudo_language=False, 
                            mask_image=True, full_img_feats=full_img_feats, 
                            full_text_feats=full_text_feats, task="swin_unter_mim")  
                            
    image_feats = feats["multi_modal_image_feats"] 


    b, d, c, h, w = batch['image'][0].shape
    image_feats = image_feats.view(b, int(d/32), int(h/32), int(w/32), image_feats.shape[-1])
    image_feats = rearrange(image_feats, 'b d h w c -> b c d h w')

    image_rec, rec_feats = pl_module.mim_swin_unter_head(image_feats)
    output_recons = image_rec
    
    recon_loss = SWIN_MIM_Loss(feats["mask_ratio"]).cuda()
    
    recon_loss = alpha * recon_loss(output_recons, target_recons, feats["mim_masks"])
 
    ret = {
        "swin_unetr_mim_loss": recon_loss,
        "swin_unetr_mim_output_recons": output_recons,
        "swin_unetr_mim_target_recons": target_recons,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_swin_unetr_mim_loss")(ret["swin_unetr_mim_loss"])
    acc = -loss
    pl_module.log(f"swin_unetr_mim/{phase}/loss", loss)
    pl_module.log(f"swin_unetr_mim/{phase}/accusracy", acc)
    
    return ret

def compute_itm(pl_module, batch, full_img_feats, full_text_feats):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(pl_module.device)
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack(
            [
                ti if itm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(bti, bfi))
            ]
        )
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer(batch, full_img_feats=full_img_feats,
                            full_text_feats=full_text_feats, task="itm")

    itm_logits = pl_module.itm_head(infer["multi_modal_cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(ret["itm_logits"], ret["itm_labels"])
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret

def compute_local_itc(pl_module, batch, full_img_feats, full_text_feats):
    device = pl_module.device

    feats = pl_module.infer(batch, pseudo_language=False, full_img_feats=full_img_feats,
                            full_text_feats=full_text_feats, 
                            task="local_itc")  

    measure = 'cosin' 

    # CML Loss
    cross_image_feats = feats["multi_modal_cls_feats"] 
    cross_text_feats = feats["uni_modal_cls_feats"]

    # cml normal
    itc_logits_cross_image, itc_logits_cross_text = pl_module.itc_cross_head(cross_image_feats, cross_text_feats)
    itc_labels = torch.arange(itc_logits_cross_image.size(0), device=device)
    itc_loss_cross_image = F.cross_entropy(itc_logits_cross_image, itc_labels)
    itc_loss_cross_text = F.cross_entropy(itc_logits_cross_text, itc_labels)
    itc_cross_loss = (itc_loss_cross_image + itc_loss_cross_text) / 2

    # DCL Loss
    gl_image_feats = feats["multi_modal_image_cls_feats"] 
    gl_image_agg_feats = feats["multi_modal_agg_image_cls_feats"]
    # dcl normal
    itc_logits_gl_image, itc_logits_gl_agg_image = pl_module.itc_agg_head(gl_image_feats, gl_image_agg_feats)
    agg_itc_labels = torch.arange(itc_logits_gl_image.size(0), device=device)
    itc_loss_gl_image = F.cross_entropy(itc_logits_gl_image, agg_itc_labels)
    itc_loss_gl_agg_image = F.cross_entropy(itc_logits_gl_agg_image, agg_itc_labels)
    agg_itc_loss = (itc_loss_gl_image + itc_loss_gl_agg_image) / 2

    # SSM Loss
    if gl_image_feats.size(0) > 1:
        sen_wise_image_feats = feats["multi_modal_agg_image_feats"]
        uni_modal_uni_sen_feats = feats["uni_modal_uni_sen_feats"]

        sen_sim_loss_func = torch.nn.L1Loss()

        # ssm normal
        sen_sim_matrix_gt, sen_sim_matrix_logits = {}, {}
        sen_sim_losses, pair_num = 0, 0
        for i in range(len(uni_modal_uni_sen_feats)-1):            
            sen_sim_pair_gt, sen_sim_pair_gt_id, sen_sim_pair_logits = {}, {}, {}
            for j in range(len(uni_modal_uni_sen_feats)-i-1):
                j_idx = i + 1 + j
                with torch.no_grad():
                    if measure == 'cosin':
                        gt_sen_distance = cosine_similarity(uni_modal_uni_sen_feats[i][0,:,:], uni_modal_uni_sen_feats[j_idx][0,:,:])

                    sim_matrix = torch.ones_like(gt_sen_distance).to(device)
                    dissim_matrix = torch.zeros_like(gt_sen_distance).to(device)
                    sen_sim_pair_gt[j_idx] = gt_sen_distance
                    sen_sim_pair_gt_idx = torch.where(gt_sen_distance >= 0.9, sim_matrix, dissim_matrix)
                    sen_sim_pair_gt_idx_2 = torch.where(gt_sen_distance <= 0.6, sim_matrix, dissim_matrix)

                if measure == 'cosin':
                    sen_distance = cosine_similarity(sen_wise_image_feats[i][0,:,:], sen_wise_image_feats[j_idx][0,:,:])
                sen_sim_pair_logits[j_idx] = sen_distance

                if measure == 'cosin':   # or 'cosin_avg'
                    sen_sim_pair_logits[j_idx] = sen_sim_pair_logits[j_idx] * (sen_sim_pair_gt_idx + sen_sim_pair_gt_idx_2)
                    sen_sim_pair_gt[j_idx] = sen_sim_pair_gt[j_idx] * sen_sim_pair_gt_idx

                sen_sim_loss = sen_sim_loss_func(sen_sim_pair_logits[j_idx], sen_sim_pair_gt[j_idx])
                sen_sim_losses += sen_sim_loss
                pair_num += 1

            sen_sim_matrix_gt[i] = sen_sim_pair_gt
            sen_sim_matrix_logits[i] = sen_sim_pair_logits

        sen_sim_losses = 1.0 * sen_sim_losses / (pair_num + 1e-6)

        local_itc_loss = agg_itc_loss + itc_cross_loss + sen_sim_losses
    else:
        local_itc_loss = agg_itc_loss + itc_cross_loss

    ret = {
        "local_itc_loss": local_itc_loss,
        "itc_labels": itc_labels,
        "itc_logits_cross_image": itc_logits_cross_image,
        "itc_logits_cross_text": itc_logits_cross_text,
        "agg_itc_logits_gl_image_agg": itc_logits_gl_agg_image,
        "agg_itc_logits_gl_image": itc_logits_gl_image,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_local_itc_loss")(ret["local_itc_loss"])
    acc = -loss
    pl_module.log(f"local_itc/{phase}/loss", loss)
    pl_module.log(f"local_itc/{phase}/accuracy", acc)

    return ret

def compute_vqa(pl_module, batch, test=False):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    vqa_logits = pl_module.vqa_head(infer["multi_modal_cls_feats"])
    vqa_targets = torch.zeros(len(vqa_logits), pl_module.hparams.config["vqa_label_size"], device=pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]
    vqa_answer_types = torch.tensor(batch["answer_types"], device=pl_module.device)

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets) * vqa_targets.shape[1])

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
        "vqa_answer_types": vqa_answer_types,
    }

    if test:
        phase = "test"
    else:
        phase = "train" if pl_module.training else "val"

    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(ret["vqa_logits"], ret["vqa_targets"], ret["vqa_answer_types"])
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret


def compute_cls(pl_module, batch, test=False):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False,
                            pseudo_vision=pl_module.hparams.config["language_only"],
                            pseudo_language=pl_module.hparams.config["vision_only"])

    cls_logits = pl_module.cls_head(infer["multi_modal_cls_feats"])
    cls_labels = batch["cls_labels"]
    cls_loss = F.cross_entropy(cls_logits, cls_labels)

    ret = {
        "cls_loss": cls_loss,
        "cls_logits": cls_logits,
        "cls_labels": cls_labels,
    }

    if test:
        phase = "test"
    else:
        phase = "train" if pl_module.training else "val"

    loss = getattr(pl_module, f"{phase}_cls_loss")(ret["cls_loss"])
    acc = getattr(pl_module, f"{phase}_cls_accuracy")(ret["cls_logits"], ret["cls_labels"])
    pl_module.log(f"cls/{phase}/loss", loss)
    pl_module.log(f"cls/{phase}/accuracy", acc)

    return ret


def compute_mlc(pl_module, batch, test=False):
    infer = pl_module.infer(batch,
                            pseudo_vision=pl_module.hparams.config["language_only"],
                            pseudo_language=pl_module.hparams.config["vision_only"])
    if pl_module.hparams.config["language_only"]:
        mlc_logits = pl_module.mlc_head(infer["multi_modal_cls_feats"])
    elif pl_module.hparams.config["vision_only"]:
        mlc_logits = pl_module.mlc_head(infer["multi_modal_cls_feats"])
    else:
        mlc_logits = pl_module.mlc_head(infer["multi_modal_cls_feats"])

    mlc_labels = batch["mlc_labels"]
    mlc_loss = F.binary_cross_entropy_with_logits(mlc_logits, mlc_labels.to(mlc_logits))

    ret = {
        "mlc_loss": mlc_loss,
        "mlc_logits": mlc_logits,
        "mlc_labels": mlc_labels,
    }

    if test:
        phase = "test"
    else:
        phase = "train" if pl_module.training else "val"

    loss = getattr(pl_module, f"{phase}_mlc_loss")(ret["mlc_loss"])
    getattr(pl_module, f"{phase}_mlc_aucroc").update(F.sigmoid(ret["mlc_logits"]), ret["mlc_labels"])
    getattr(pl_module, f"{phase}_mlc_f1").update(F.sigmoid(ret["mlc_logits"]), ret["mlc_labels"])

    pl_module.log(f"mlc/{phase}/loss", loss)

    return ret

def compute_clm(pl_module, batch, test=False):
    if test:
        phase = "test"
    else:
        phase = "train" if pl_module.training else "val"

    infer = pl_module.infer(batch,
                            pseudo_vision=False,
                            pseudo_language=False)

    encoder_hidden_states = torch.cat([infer["multi_modal_image_feats"], infer["multi_modal_text_feats"]], dim=1)
    encoder_hidden_states = pl_module.clm_proj(encoder_hidden_states)

    texts = batch["findings"] if pl_module.hparams.config["vision_only"] else batch["impression"]
    texts = [text.lower() for text in texts]  # Lower case the ground truth

    texts_inputs = pl_module.clm_tokenizer(texts, max_length=pl_module.hparams.config["clm_max_text_len"],
                                           truncation=True, padding=True, return_token_type_ids=False,
                                           return_tensors="pt").to(pl_module.device)

    outputs = pl_module.clm_head(input_ids=texts_inputs.input_ids,
                                 attention_mask=texts_inputs.attention_mask,
                                 encoder_hidden_states=encoder_hidden_states,
                                 use_cache=False)

    clm_logits = outputs.logits
    clm_logits = clm_logits[:, :-1, :].contiguous()
    clm_labels = texts_inputs.input_ids[:, 1:].contiguous()
    clm_loss = F.cross_entropy(clm_logits.view(-1, clm_logits.size(-1)),
                               clm_labels.view(-1),
                               ignore_index=pl_module.clm_tokenizer.pad_token_id)

    ret = {
        "clm_loss": clm_loss,
        "clm_logits": clm_logits,
        "clm_labels": clm_labels
    }

    loss = getattr(pl_module, f"{phase}_clm_loss")(ret["clm_loss"])
    pl_module.log(f"clm/{phase}/loss", loss)

    if not phase == "train":
        bs = encoder_hidden_states.size(0)
        input_ids = torch.tensor([[pl_module.clm_tokenizer.cls_token_id]] * bs, dtype=torch.long,
                                 device=pl_module.device)

        # expand for beam search
        expanded_idx = (
            torch.arange(input_ids.shape[0], device=pl_module.device).view(-1, 1).repeat(1, pl_module.hparams.config[
                "clm_num_beams"]).view(-1)
        )
        encoder_hidden_states = encoder_hidden_states.index_select(0, expanded_idx)

        gen_texts = pl_module.clm_head.generate(input_ids=input_ids,
                                                encoder_hidden_states=encoder_hidden_states,
                                                max_length=pl_module.hparams.config["clm_max_text_len"],
                                                do_sample=pl_module.hparams.config["clm_do_sample"],
                                                num_beams=pl_module.hparams.config["clm_num_beams"])

        gen_texts = pl_module.clm_tokenizer.batch_decode(gen_texts, skip_special_tokens=True)
        # Gen Texts can not be empty.
        gen_texts = [text if len(text) > 0 else "there is no evidence of pulmonary." for text in gen_texts]
        gt_texts = texts

        ret["gen_texts"] = gen_texts
        ret["gt_texts"] = gt_texts

        for i in [1, 2, 3, 4]:
            getattr(pl_module, f"{phase}_clm_bleu_{i}").update(ret["gen_texts"], [[text] for text in ret["gt_texts"]])
        getattr(pl_module, f"{phase}_clm_rouge").update(ret["gen_texts"], ret["gt_texts"])
        getattr(pl_module, f"{phase}_clm_coco_caption").update(ret["gen_texts"], ret["gt_texts"])
        getattr(pl_module, f"{phase}_clm_jb").update(ret["gen_texts"], ret["gt_texts"])

    return ret


def compute_irtr(pl_module, batch, test=False):
    is_training_phase = pl_module.training
    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack([batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1)
    text_masks = torch.stack([batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1)
    text_labels = torch.stack([batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1)

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)

    batch_infer = {
        "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
        "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
        "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
        "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
    }

    infer = pl_module.infer(batch_infer)

    score = pl_module.irtr_head(infer["multi_modal_cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {"irtr_loss": irtr_loss}

    if test:
        phase = "test"
    else:
        phase = "train" if pl_module.training else "val"

    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])
    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)

    return ret


@torch.no_grad()
def compute_irtr_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=256,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(text_dset.collate,
                                     mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator, ), )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(image_only=True)
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(image_dset.collate,
                                     mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator, ), )

    # TODO: speed up the process by caching text/image features
    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        # == Begin: Add New Keys ==
        batch_text_preload = {
            "text_ids": _b["text_ids"].to(pl_module.device),
            "text_masks": _b["text_masks"].to(pl_module.device),
            "text_labels": _b["text_labels"].to(pl_module.device),
            "img_index": _b["img_index"],
        }
        text_preload.append(batch_text_preload)
        # == End  : Add New Keys ==

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        image_preload.append((_b['image'][0], _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _im, _iid = img_batch

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            im = _im.repeat(fblen, 1, 1, 1).to(device=txt_batch['text_ids'].device)

            with torch.cuda.amp.autocast():
                # == Begin: Add New Keys ==
                batch_infer = {
                    "text_ids": txt_batch["text_ids"],
                    "text_masks": txt_batch["text_masks"],
                    "text_labels": txt_batch["text_labels"],
                }
                score = pl_module.irtr_head(pl_module.infer(batch_infer, img=im, )["multi_modal_cls_feats"])[:, 0]
                # == End  : Add New Keys ==

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)
