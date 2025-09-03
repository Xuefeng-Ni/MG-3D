import copy

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from mg3d.models import mg3d_utils, objectives
from mg3d.models import prediction_heads
from mg3d.models.language_encoders.bert_model import BertCrossLayer
from mg3d.models.language_encoders.bert_model_generation import BertGenerationDecoder
from mg3d.models.mg3d_utils import init_weights
from mg3d.models.vision_encoders.clip_model import build_model, adapt_position_encoding
from einops import rearrange
import monai
from mg3d.models.vision_encoders.ssl_head import SSLHead
from mg3d.models.ops import mask_rand_patch
from mg3d.models.gather import SentenceGather
from mg3d.models.sentence_pool import SentenceAttentionPool
from mg3d.models.local_attention import LocalCrossAttention, Unidir_LocalCrossAttention, MIMCrossMultiHeadAttention, MLMCrossMultiHeadAttention

from monai.data.utils import SUPPORTED_PICKLE_MOD, pickle_hashing
import tempfile
from pathlib import Path
from monai.utils import look_up_option
import pickle
import shutil

class MG3DTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        # == Begin: 1. Build Models ==
        self.resolution_after = config['image_size']

        backbone = 'swinunetr'
        if backbone == 'swinunetr':
            self.vision_encoder = SSLHead(upsample="vae", dim=768).swinViT
            
        bert_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=config['tokenizer'],
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
        self.language_encoder = AutoModel.from_pretrained(config['tokenizer'])
        self.item_gather = SentenceGather("avg", config['input_text_embed_size'])
        
        self.uni_global_text_attention = SentenceAttentionPool(32, config['input_text_embed_size'], pos_embed=False) # Max sentence num: 32
        self.multi_global_text_attention = SentenceAttentionPool(32, config['input_text_embed_size'], pos_embed=False) # Max sentence num: 32

        self.uni_emd_sen = nn.Sequential(nn.Linear(config['hidden_size'], config['hidden_size'] // 2),
                                    nn.ReLU(inplace=True), # hidden layer 
                                    nn.Linear(config['hidden_size'] // 2, 8)) # output layer # used for simsiam loss
        self.uni_emd_img = nn.Sequential(nn.Linear(config['hidden_size'], config['hidden_size'] // 2),
                                    nn.ReLU(inplace=True), # hidden layer 
                                    nn.Linear(config['hidden_size'] // 2, 8)) # output layer # used for simsiam loss

        self.multi_modal_language_proj = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.multi_modal_language_proj.apply(init_weights)
        self.multi_modal_vision_proj = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.multi_modal_vision_proj.apply(init_weights)

        self.modality_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.modality_type_embeddings.apply(init_weights)
        
        
        self.mim_mlm_layer = 'multi'   #'multi' single
        if self.mim_mlm_layer == 'single': 
            self.local_img_cross_multihead_attention = MIMCrossMultiHeadAttention(config['input_text_embed_size'], self.resolution_after)
            self.local_text_cross_multihead_attention = MLMCrossMultiHeadAttention(config['input_text_embed_size'], self.resolution_after)
        else:
            self.local_img_cross_multihead_attention = nn.ModuleList(
                [MIMCrossMultiHeadAttention(config['input_text_embed_size'], self.resolution_after) for _ in range(config['num_top_layer'])])
            self.local_text_cross_multihead_attention = nn.ModuleList(
                [MLMCrossMultiHeadAttention(config['input_text_embed_size'], self.resolution_after) for _ in range(config['num_top_layer'])])
        
        self.itc_layer = 'multi'     #'multi' single
        if self.itc_layer == 'single':
            self.local_cross_attention =  LocalCrossAttention(config['input_text_embed_size'])
        else:
            self.local_cross_attention = nn.ModuleList(
                [LocalCrossAttention(config['input_text_embed_size']) for _ in range(config['num_top_layer'])])

        self.dcl_layer = 'multi'   #'multi' single
        if self.dcl_layer == 'single': 
            self.agg_img_cross_attention =  Unidir_LocalCrossAttention(config['input_text_embed_size'])
            self.agg_text_cross_attention =  Unidir_LocalCrossAttention(config['input_text_embed_size'])
        else:
            self.agg_img_cross_attention = nn.ModuleList(
                [Unidir_LocalCrossAttention(config['input_text_embed_size']) for _ in range(config['num_top_layer'])])
            self.agg_text_cross_attention = nn.ModuleList(
                [Unidir_LocalCrossAttention(config['input_text_embed_size']) for _ in range(config['num_top_layer'])])


        self.multi_modal_vision_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.multi_modal_vision_layers.apply(init_weights)
        self.multi_modal_language_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.multi_modal_language_layers.apply(init_weights)

        if backbone == 'swinunetr':
            self.uni_modal_vision_pooler = prediction_heads.Swin_Pooler(config["hidden_size"], self.resolution_after)
            self.uni_modal_vision_pooler.apply(init_weights)
            
            self.multi_modal_vision_pooler = prediction_heads.Swin_Pooler(config["hidden_size"], self.resolution_after)
            self.multi_modal_vision_pooler.apply(init_weights)   
            
            self.multi_modal_agg_vision_pooler = prediction_heads.Swin_Att_Pooler(config['hidden_size']) # Max sentence num: 32
            self.multi_modal_agg_vision_pooler.apply(init_weights)
     

        self.uni_modal_language_pooler = prediction_heads.Pooler(config["hidden_size"])
        self.uni_modal_language_pooler.apply(init_weights)

        self.multi_modal_language_pooler = prediction_heads.Pooler(config["hidden_size"])
        self.multi_modal_language_pooler.apply(init_weights)
        
        self.multi_modal_sen_vision_pooler = prediction_heads.Sentence_Pooler(config["hidden_size"])
        self.multi_modal_sen_vision_pooler.apply(init_weights)      
        
        # == End  : 1. Build Models ==

        # == Begin: 2. Build Pre-Training Heads ==
        if config["loss_names"]["sen_mlm"] > 0:
            self.mlm_head = prediction_heads.SEN_MLMHead(bert_config)
            self.mlm_head.apply(init_weights)
        if config["loss_names"]["swin_unetr_mim"] > 0:
            self.mim_swin_unter_head = prediction_heads.MIM_SWINUNETR_Head(upsample="vae", dim=768)
            self.mim_swin_unter_head.apply(init_weights)
        if config["loss_names"]["itm"] > 0 or self.hparams.config["loss_names"]["irtr"] > 0:
            self.itm_head = prediction_heads.ITMHead(config["hidden_size"] * 2)
            self.itm_head.apply(init_weights)
        if config["loss_names"]["local_itc"] > 0:
            self.itc_head = prediction_heads.ITCHead_Single(config["hidden_size"], config["cl_temp"])
            self.itc_head.apply(init_weights)
            self.itc_cross_head = prediction_heads.ITCHead(config["hidden_size"], config["cl_temp"])
            self.itc_cross_head.apply(init_weights)
            self.itc_agg_head = prediction_heads.ITCHead_Single(config["hidden_size"], config["cl_temp"])
            self.itc_agg_head.apply(init_weights)

        self.pseudo_vision_token_pool_size = config["pseudo_vision_token_pool_size"]
        self.pseudo_langauge_token_pool_size = config["pseudo_langauge_token_pool_size"]
        self.num_pseudo_vision_tokens = config["num_pseudo_vision_tokens"]
        self.num_pseudo_langauge_tokens = config["num_pseudo_langauge_tokens"]

        if self.pseudo_vision_token_pool_size > 0:
            self.pseudo_vision_token_pool = torch.nn.Parameter(
                torch.empty((self.pseudo_vision_token_pool_size,
                             self.vision_encoder.visual.width)).normal_(mean=0.0, std=0.02),
                requires_grad=True if "pretrain" in config["exp_name"] else False)
        if self.pseudo_langauge_token_pool_size > 0:
            self.pseudo_language_token_pool = torch.nn.Parameter(
                torch.empty((self.pseudo_langauge_token_pool_size,
                             self.language_encoder.config.hidden_size)).normal_(mean=0.0, std=0.02),
                requires_grad=True if "pretrain" in config["exp_name"] else False)
        
        self.patch_pool_3d = nn.AvgPool3d(kernel_size=2)

        # == End  : 2. Build Pre-Training Heads ==


        # == Begin: 3. Load Models ==
        if self.hparams.config["load_path"] != "" and not self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            state_dict = adapt_position_encoding(state_dict,
                                                 after=self.resolution_after,
                                                 patch_size=self.hparams.config['patch_size'])
            self.load_state_dict(state_dict, strict=False)
        # == End  : 3. Load Models ==


        # == 4. Build Heads For Downstream Tasks ==
        hs = self.hparams.config["hidden_size"]
        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqa_label_size"]
            self.vqa_head = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_head.apply(init_weights)

        if self.hparams.config["loss_names"]["cls"] > 0:
            ms = self.hparams.config["label_size"][self.hparams.config["label_column_name"]]
            self.cls_head = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, ms),
            )
            self.cls_head.apply(init_weights)

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.irtr_head = nn.Linear(hs * 2, 1)
            self.irtr_head.weight.data = self.itm_head.fc.weight.data[1:, :]
            self.irtr_head.bias.data = self.itm_head.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_head.parameters():
                p.requires_grad = False

        if self.hparams.config["loss_names"]["mlc"] > 0:
            ms = self.hparams.config["label_size"][self.hparams.config["label_column_name"]]
            self.mlc_head = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, ms),
            )
            self.mlc_head.apply(init_weights)

        if self.hparams.config["loss_names"]["clm"] > 0:
            self.clm_tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
            self.clm_proj = nn.Sequential(
                nn.Linear(hs, self.language_encoder.config.hidden_size),
                nn.LayerNorm(self.language_encoder.config.hidden_size)
            )
            self.clm_proj.apply(init_weights)
            self.clm_head = BertGenerationDecoder(config=self.language_encoder.config,
                                                  max_length=self.hparams.config["clm_max_text_len"])
            self.clm_head.apply(init_weights)

            self.clm_head.bert.embeddings.load_state_dict(
                copy.deepcopy(self.language_encoder.embeddings.state_dict()), strict=True
            )
            self.clm_head.bert.encoder.load_state_dict(
                copy.deepcopy(self.language_encoder.encoder.state_dict()), strict=False
            )
            self.clm_head.lm_head.load_state_dict(
                copy.deepcopy(self.mlm_head.state_dict()), strict=True
            )

        mg3d_utils.set_metrics(self)
        self.current_tasks = list()
        # == End:  4. Build Heads For Downstream Tasks ==


        # == Begin: 5. Load Models For Testing ==
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            state_dict = adapt_position_encoding(state_dict, after=self.resolution_after,
                                                 patch_size=self.hparams.config['patch_size'])
            self.load_state_dict(state_dict, strict=False)
        # == End  : 5. Load Models For Testing ==


    def infer(
            self,
            batch,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            img=None,
            output_attentions=False,
            pseudo_vision=False,
            pseudo_language=False,
            full_img_feats=None,
            full_text_feats=None, 
            task="itc"
    ):
        ret = dict()

        #Text Encoder Parameter Frozen
        for p in self.language_encoder.parameters():
            p.requires_grad = False

        # == Begin: Fetch the inputs ==
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                img_key = f"image_{image_token_type_idx - 1}"
            else:
                img_key = "image"
            if pseudo_vision:
                img = None
            else:
                img = batch[img_key][0]
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        
        sen_txt_ids = batch[f"sen_txt_ids"]
        
        device = self.device
        # == End  : Fetch the inputs ==
        
        
        # == Begin: Not Important ==
        assert (pseudo_vision & pseudo_language) is False
        if pseudo_vision:
            # text token embeddings
            with torch.no_grad():
                uni_modal_text_feats = self.language_encoder.embeddings(input_ids=text_ids)
            if self.num_pseudo_vision_tokens > 0:
                # average pooling for token embeddings
                query_tensors = (uni_modal_text_feats[:, 1:] * text_masks.unsqueeze(-1)[:, 1:]).sum(1) / (
                        text_masks[:, 1:].sum(1, keepdims=True) + 1e-8)
                # find pseudo tokens
                pseudo_tokens = self.find_pseudo_tokens(query_tensors,
                                                        self.pseudo_vision_token_pool,
                                                        self.num_pseudo_vision_tokens)
                # concatenate cls embeds and pseudo tokens
                vision_cls_token = self.vision_encoder.visual.get_pos_encoded_cls_embed()
                vision_cls_token = vision_cls_token.unsqueeze(0).repeat(len(pseudo_tokens), 1, 1)
                uni_modal_image_feats = torch.cat([vision_cls_token, pseudo_tokens], dim=1)
            else:
                # use cls tokens
                vision_cls_token = self.vision_encoder.visual.get_pos_encoded_cls_embed()
                vision_cls_token = vision_cls_token.unsqueeze(0).repeat(len(uni_modal_text_feats), 1, 1)
                uni_modal_image_feats = vision_cls_token
        elif pseudo_language:
            # image token embeddings

            img = rearrange(img, 'b d c h w -> b c d h w')
            uni_modal_image_feats = self.vision_encoder.forward_patch_embed(img)

            if self.num_pseudo_langauge_tokens > 0:
                uni_modal_image_feats_reshape = uni_modal_image_feats[:, 1:]
                query_tensors = uni_modal_image_feats_reshape.mean(1)

                # find pseudo tokens
                pseudo_tokens = self.find_pseudo_tokens(query_tensors,
                                                        self.pseudo_language_token_pool,
                                                        self.num_pseudo_langauge_tokens)
                # concatenate cls embeds, pseudo tokens, and sep embeds
                cls_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
                cls_id = torch.full((len(pseudo_tokens), 1), cls_id, dtype=torch.long, device=device)
                with torch.no_grad():
                    language_cls_token = self.language_encoder.embeddings(input_ids=cls_id)
                sep_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
                sep_id = torch.full((len(pseudo_tokens), 1), sep_id, dtype=torch.long, device=device)
                with torch.no_grad():
                    language_sep_token = self.language_encoder.embeddings(input_ids=sep_id)
                uni_modal_text_feats = torch.cat([language_cls_token, pseudo_tokens, language_sep_token], dim=1)
            else:
                # use cls tokens
                cls_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
                cls_id = torch.full((len(uni_modal_image_feats), 1), cls_id, dtype=torch.long, device=device)
                with torch.no_grad():
                    language_cls_token = self.language_encoder.embeddings(input_ids=cls_id)
                    uni_modal_text_feats = language_cls_token
        # == End: Not Important ==


        # == Begin: Image and Text Embeddings ==
        else:
            with torch.no_grad():
                if full_text_feats == None:
                    
                    uni_modal_text_feats = []
                    for b in range(len(batch['sen_txt_'])):
                        if 'RATE' in str(batch["hashfile"][b]):
                            cache_dir = Path('/home/xuefeng/cache/CT_RATE_uni_text_cache/')
                        elif 'CTRG_chest' in str(batch["hashfile"][b]):
                            cache_dir = Path('/home/xuefeng/cache/CTRG_chest_uni_text_cache/')

                        if cache_dir is not None:
                            data_item_md5 = str(batch["hashfile"][b]).split('/')[-1][:-3] + '_uni_text'
                            hashfile = cache_dir / f"{data_item_md5}.pt"
                                
                        if hashfile is not None and hashfile.is_file():  # cache hit
                            uni_modal_text_feats.append(torch.load(hashfile).to(device))
                        else:        
                            try:
                                uni_modal_text_feats_batch = self.language_encoder.embeddings(input_ids=text_ids[b].unsqueeze(0))
                                uni_modal_text_feats.append(uni_modal_text_feats_batch.to(device))
                                with tempfile.TemporaryDirectory() as tmpdirname:
                                    temp_hash_file = Path(tmpdirname) / hashfile.name
                                    torch.save(
                                        obj=uni_modal_text_feats_batch.detach().cpu(),
                                        f=temp_hash_file,
                                        pickle_module=look_up_option("pickle", SUPPORTED_PICKLE_MOD),
                                        pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    )
                                    if temp_hash_file.is_file() and not hashfile.is_file():
                                        shutil.move(str(temp_hash_file), hashfile)
                            except PermissionError:  # project-monai/monai issue #3613
                                pass
                    uni_modal_text_feats = torch.cat(uni_modal_text_feats, dim=0)
                else:
                    uni_modal_text_feats = full_text_feats

                # == Loss ==
                if task == 'sen_mlm':
                    uni_modal_full_text_feats = self.language_encoder.embeddings(input_ids=batch["text_ids"])
                    uni_modal_text_cls_feats = uni_modal_full_text_feats
                    
                if task == 'local_itc':        
                    uni_modal_uni_sen_feats = []
                    for b in range(len(sen_txt_ids)):
                        s_feats = []
                        for s_id in sen_txt_ids[b]:
                            s_feats.append(torch.mean(self.language_encoder.embeddings(input_ids=s_id), dim=1, keepdim=True))
                        uni_modal_uni_sen_feats.append(torch.cat(s_feats, dim=1))
                else:
                    uni_modal_uni_sen_feats = []

            if not mask_image:
                if full_img_feats == None:
                    img = rearrange(img, 'b d c h w -> b c d h w')
        # == End: Image and Text Embeddings ==


        # == Begin: Text Encoding ==
        extended_text_masks = self.language_encoder.get_extended_attention_mask(text_masks, text_masks.size(), device)

        for layer in self.language_encoder.encoder.layer:
            uni_modal_text_feats = layer(uni_modal_text_feats, extended_text_masks)[0]
        uni_modal_text_feats = self.multi_modal_language_proj(uni_modal_text_feats)
        # == End  : Text Encoding ==
        
        # == Begin: Image Encoding ==
        # == Begin: Image Masking ==
        if mask_image:
            img = rearrange(img, 'b d c h w -> b c d h w')
            
            window_size = 16
            mask_ratio = 0.5
            window_sizes = tuple(window_size for _ in range(3))

            if self.resolution_after == 96:
                input_sizes = (96, self.resolution_after, self.resolution_after)
            elif self.resolution_after == 128:
                input_sizes = (self.resolution_after, 96, self.resolution_after)
            

            uni_modal_image_feats = self.vision_encoder.forward_patch_embed(img)

            uni_modal_image_feats, mim_masks, mim_ids_restore = self.random_masking_3d(uni_modal_image_feats,
                                                                                    mask_ratio)
            uni_modal_image_feats, _ = self.vision_encoder.forward_trans(uni_modal_image_feats)

            ret["mim_masks"] = mim_masks
            ret["mask_ratio"] = mask_ratio
            ret["mim_ids_restore"] = mim_ids_restore
        # == End  : Image Masking ==
        
        # == Begin: Original Image ==
        else:
            if full_img_feats == None:
                uni_modal_image_feats, _ = self.vision_encoder.forward(img)
            else:
                uni_modal_image_feats = full_img_feats

        uni_modal_image_feats = self.multi_modal_vision_proj(uni_modal_image_feats)
        image_masks = torch.ones((uni_modal_image_feats.size(0), uni_modal_image_feats.size(1)),
                                 dtype=torch.long, device=device)
                  
        extended_image_masks = self.language_encoder.get_extended_attention_mask(image_masks, 
                                                                                 image_masks.size(), device)
        # == End: Original Image ==
        # == End  : Image Encoding ==


        # == Begin: Assign Type Embeddings ==
        uni_modal_text_feats, uni_modal_image_feats = (
            uni_modal_text_feats + self.modality_type_embeddings(torch.zeros_like(text_masks)),
            uni_modal_image_feats + self.modality_type_embeddings(torch.full_like(image_masks, image_token_type_idx)),
        )
        # == End  : Assign Type Embeddings ==
        ret["attentions"] = {"text2image_attns": [], "image2text_attns": []} if output_attentions else None


        # == Begin: ITC Loss ==
        if task == 'local_itc':
            with torch.no_grad():
                local_text_embed_stacks = self.item_gather(uni_modal_text_feats, batch)
                uni_modal_text_feats = local_text_embed_stacks
                sentence_feats_stacks = uni_modal_text_feats
            
            batch_stacks = []
            for local_text_embed in uni_modal_text_feats:
                batch_stacks.append(self.uni_global_text_attention(local_text_embed))
            uni_modal_text_cls_feats = torch.cat(batch_stacks, dim=0)

            # == Begin: Multi-Modal Fusion ==
            x, y = uni_modal_text_feats, uni_modal_image_feats

            if self.mim_mlm_layer == 'single':
                # Single Layer
                local_text_embed_stack, local_image_embed_stack = [], []
                for idx in range(len(x)):
                    local_text_embed = x[idx] # get sentence-level representation 
                    local_image_embed = y[idx] # get patch-level representation
                    text_to_local_image_embed, _, image_to_local_text_embed, _  = \
                        self.local_img_cross_multihead_attention(local_image_embed, local_text_embed.squeeze(0)) 
                    local_text_embed_stack.append(image_to_local_text_embed.unsqueeze(0))
                    local_image_embed_stack.append(text_to_local_image_embed.unsqueeze(0))
                x = local_text_embed_stack
                y = torch.cat(local_image_embed_stack, dim=0)
            else:
                # Multiple Layers
                for layer_idx, layer in enumerate(self.local_img_cross_multihead_attention):
                    local_text_embed_stack, local_image_embed_stack = [], []
                    for idx in range(len(x)):
                        local_text_embed = x[idx] # get sentence-level representation 
                        local_image_embed = y[idx] # get patch-level representation
                        text_to_local_image_embed, _, image_to_local_text_embed, _  = \
                            layer(local_image_embed, local_text_embed.squeeze(0)) 
                        local_text_embed_stack.append(image_to_local_text_embed.unsqueeze(0))
                        local_image_embed_stack.append(text_to_local_image_embed.unsqueeze(0))
                    x = local_text_embed_stack
                    y = torch.cat(local_image_embed_stack, dim=0)
        # == End: ITC Loss ==
        
        # == Begin: MIM Loss ==
        elif task == 'mim':
            with torch.no_grad():
                local_text_embed_stacks = self.item_gather(uni_modal_text_feats, batch)
                uni_modal_text_feats = local_text_embed_stacks
                sentence_feats_stacks = uni_modal_text_feats
            
            batch_stacks = []
            for local_text_embed in uni_modal_text_feats:
                batch_stacks.append(self.uni_global_text_attention(local_text_embed))
            uni_modal_text_cls_feats = torch.cat(batch_stacks, dim=0)

            # == Begin: Multi-Modal Fusion ==
            x, y = uni_modal_text_feats, uni_modal_image_feats

            if self.mim_mlm_layer == 'single':
                # Single Layer
                local_text_embed_stack, local_image_embed_stack = [], []
                for idx in range(len(x)):
                    local_text_embed = x[idx] # get sentence-level representation 
                    local_image_embed = y[idx] # get patch-level representation
                    text_to_local_image_embed, _, image_to_local_text_embed, _  = \
                        self.local_img_cross_multihead_attention(local_image_embed, local_text_embed.squeeze(0)) 
                    local_text_embed_stack.append(image_to_local_text_embed.unsqueeze(0))
                    local_image_embed_stack.append(text_to_local_image_embed.unsqueeze(0))
                x = local_text_embed_stack
                y = torch.cat(local_image_embed_stack, dim=0)
            else:
                # Multiple Layers
                for layer_idx, layer in enumerate(self.local_img_cross_multihead_attention):
                    local_text_embed_stack, local_image_embed_stack = [], []
                    for idx in range(len(x)):
                        local_text_embed = x[idx] # get sentence-level representation 
                        local_image_embed = y[idx] # get patch-level representation
                        text_to_local_image_embed, _, image_to_local_text_embed, _  = \
                            layer(local_image_embed, local_text_embed.squeeze(0)) 
                        local_text_embed_stack.append(image_to_local_text_embed.unsqueeze(0))
                        local_image_embed_stack.append(text_to_local_image_embed.unsqueeze(0))
                    x = local_text_embed_stack
                    y = torch.cat(local_image_embed_stack, dim=0)
                    
            # == End  : Multi-Modal Fusion ==
        # == End: MIM Loss ==            


        # == Begin: MLM Loss ==
        else:
            x, y = uni_modal_text_feats, uni_modal_image_feats
            
            if task == 'sen_mlm':
                with torch.no_grad():
                    local_text_embed_stacks = self.item_gather(uni_modal_full_text_feats, batch)
                    uni_modal_full_text_feats = local_text_embed_stacks
                    sentence_feats_stacks = uni_modal_full_text_feats
                    
                batch_stacks = []
                for local_text_embed in uni_modal_full_text_feats:
                    batch_stacks.append(self.uni_emd_sen(local_text_embed))
                sentence_feats_stacks = batch_stacks
                
            else:
                with torch.no_grad():
                    sentence_feats_stacks = uni_modal_text_feats
                
            # == Begin: Multi-Modal Fusion ==
            if self.mim_mlm_layer == 'single':
                # Single Layer
                local_text_embed_stack, local_image_embed_stack = [], []
                for idx in range(len(y)):
                    local_text_embed = x[idx] # get sentence-level representation 
                    local_image_embed = y[idx] # get patch-level representation
                    image_to_local_text_embed, _, text_to_local_image_embed, _  = \
                        self.local_text_cross_multihead_attention(local_text_embed, local_image_embed) 
                    local_text_embed_stack.append(image_to_local_text_embed.unsqueeze(0))
                    local_image_embed_stack.append(text_to_local_image_embed.unsqueeze(0))
                x = torch.cat(local_text_embed_stack, dim=0) 
                y = torch.cat(local_image_embed_stack, dim=0) 
            else:
                # Multiple Layers
                for layer_idx, layer in enumerate(self.local_text_cross_multihead_attention):
                    local_text_embed_stack, local_image_embed_stack = [], []
                    for idx in range(len(y)):
                        local_text_embed = x[idx] # get sentence-level representation 
                        local_image_embed = y[idx] # get patch-level representation
                        image_to_local_text_embed, _, text_to_local_image_embed, _  = \
                            layer(local_text_embed, local_image_embed) 
                        local_text_embed_stack.append(image_to_local_text_embed.unsqueeze(0))
                        local_image_embed_stack.append(text_to_local_image_embed.unsqueeze(0))
                    x = torch.cat(local_text_embed_stack, dim=0) 
                    y = torch.cat(local_image_embed_stack, dim=0)    
            # == End: Multi-Modal Fusion ==  
        # == End: MLM Loss ==


        # == Begin: == Output Multi-Modal Features ==
        multi_modal_text_feats, multi_modal_image_feats = x, y

        multi_modal_image_cls_feats = self.multi_modal_vision_pooler(y)
        uni_modal_image_cls_feats = self.uni_modal_vision_pooler(uni_modal_image_feats)
        

        # == Begin: ITC & MIM Losses ==
        if (task == 'local_itc') or (task == 'mim'):

            multi_modal_text_embed_stack, agg_text_embed_stack, \
            agg_image_embed_stack, agg_image_cls_embed_stack = [], [], [], []
            
            for idx in range(len(x)):
                multi_modal_text_embed = self.multi_global_text_attention(multi_modal_text_feats[idx])
                multi_modal_text_embed_stack.append(multi_modal_text_embed)
                
                local_text_embed = uni_modal_text_feats[idx]

                # == Unidir_LocalCrossAttention ==

                if self.dcl_layer == 'single':
                    # Single Layer
                    text_to_local_image_embed_stack, local_image_to_text_embed_stack = [], []
                    for sen_idx in range(local_text_embed.size(1)):
                        text_to_local_image_embed, _  = \
                            self.agg_img_cross_attention(local_text_embed[:,sen_idx], uni_modal_image_cls_feats[idx].unsqueeze(0)) 
                        text_to_local_image_embed_stack.append(text_to_local_image_embed)
                                                            
                        local_image_to_text_embed, _  = \
                            self.agg_text_cross_attention(uni_modal_image_cls_feats[idx].unsqueeze(0), local_text_embed[:,sen_idx]) 
                        local_image_to_text_embed_stack.append(local_image_to_text_embed)
                
                    text_to_local_image_embed_stack = torch.cat(text_to_local_image_embed_stack, dim=0)
                    local_image_to_text_embed_stack = torch.cat(local_image_to_text_embed_stack, dim=0)

                    # == Unidir_LocalPatchCrossAttention ==
                    multi_modal_agg_image_cls_embed = self.multi_modal_agg_vision_pooler(text_to_local_image_embed_stack.unsqueeze(0))
                    text_to_local_image_embed_stack = self.multi_modal_sen_vision_pooler(text_to_local_image_embed_stack.unsqueeze(0))
                    agg_image_embed_stack.append(text_to_local_image_embed_stack)
                else:
                    # Multiple Layers
                    x, y = local_text_embed, uni_modal_image_cls_feats[idx].unsqueeze(0)
                    for layer_idx, (image_layer, text_layer) in enumerate(zip(self.agg_img_cross_attention,
                                                                            self.agg_text_cross_attention)):
                        text_to_local_image_embed_stack, local_image_to_text_embed_stack = [], []

                        for sen_idx in range(x.size(1)):
                            text_to_local_image_embed, _  = \
                                image_layer(x[:,sen_idx], y) 
                            text_to_local_image_embed_stack.append(text_to_local_image_embed)
                                                                
                            local_image_to_text_embed, _  = \
                                text_layer(y, x[:,sen_idx]) 
                            local_image_to_text_embed_stack.append(local_image_to_text_embed)
                        x = torch.cat(text_to_local_image_embed_stack, dim=0).unsqueeze(0)
                        local_image_to_text_embed_stack = torch.cat(local_image_to_text_embed_stack, dim=0)

                        y = self.multi_modal_agg_vision_pooler(x)
                    text_to_local_image_embed_stack, multi_modal_agg_image_cls_embed = x, y
                    
                    text_to_local_image_embed_stack = self.multi_modal_sen_vision_pooler(text_to_local_image_embed_stack)
                    agg_image_embed_stack.append(text_to_local_image_embed_stack)

                agg_image_cls_embed_stack.append(multi_modal_agg_image_cls_embed)
                agg_text_embed_stack.append(local_image_to_text_embed_stack.unsqueeze(0))
            multi_modal_text_cls_feats = torch.cat(multi_modal_text_embed_stack, dim=0)
            multi_modal_agg_image_feats = agg_image_embed_stack
            multi_modal_agg_text_feats = agg_text_embed_stack
            multi_modal_agg_image_cls_feats = torch.cat(agg_image_cls_embed_stack, dim=0)
            
        # == End: ITC & MIM Losses ==


        # == Begin: MLM Loss ==
        else: 
            multi_modal_text_cls_feats = self.multi_modal_language_pooler(x) 
            if task != 'sen_mlm':
                uni_modal_text_cls_feats = self.uni_modal_language_pooler(uni_modal_text_feats)
            multi_modal_agg_text_feats = multi_modal_text_feats
            multi_modal_agg_image_feats = multi_modal_image_feats
            multi_modal_agg_image_cls_feats = multi_modal_image_cls_feats

        multi_modal_cls_feats = torch.cat([multi_modal_text_cls_feats, multi_modal_image_cls_feats], dim=-1)
        
        if task == 'sen_mlm':
            uni_modal_cls_feats = uni_modal_text_cls_feats
        else:
            uni_modal_cls_feats = torch.cat([uni_modal_text_cls_feats, uni_modal_image_cls_feats], dim=-1)
        # == End: MLM Loss ==

        # == End  : == Output Multi-Modal Features ==

        ret.update({
            "images": img,
            "patched_images": self.patchify(img), # if img is not None else None,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "extended_image_masks": extended_image_masks,
            "extended_text_masks": extended_text_masks,
            "multi_modal_text_feats": multi_modal_text_feats,
            "multi_modal_image_feats": multi_modal_image_feats,
            "uni_modal_text_feats": uni_modal_text_feats,
            "uni_modal_sentence_feats": sentence_feats_stacks,
            "uni_modal_uni_sen_feats": uni_modal_uni_sen_feats,
            "uni_modal_image_feats": uni_modal_image_feats,
            "multi_modal_cls_feats": multi_modal_cls_feats,
            "multi_modal_text_cls_feats": multi_modal_text_cls_feats,
            "multi_modal_image_cls_feats": multi_modal_image_cls_feats,
            "uni_modal_text_cls_feats": uni_modal_text_cls_feats,
            "uni_modal_image_cls_feats": uni_modal_image_cls_feats,
            "uni_modal_cls_feats": uni_modal_cls_feats,
            "multi_modal_agg_text_feats": multi_modal_agg_text_feats,
            "multi_modal_agg_image_feats": multi_modal_agg_image_feats, 
            "multi_modal_agg_image_cls_feats": multi_modal_agg_image_cls_feats,
        })

        return ret

    def forward(self, batch, test=False):
        ret = dict()

        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Pre-Training: Masked Language Modeling
        if "sen_mlm" in self.current_tasks:
            ret.update(objectives.compute_sen_mlm(self, batch))

        # Pre-Training: Masked Image Modeling
        if "swin_unetr_mim" in self.current_tasks:
            ret.update(objectives.compute_swin_unetr_mim(self, batch, ret['full_img_feats'], 
                                                         ret['full_text_feats']))

        # Pre-Training: Contrastive Learning
        if "local_itc" in self.current_tasks:
            ret.update(objectives.compute_local_itc(self, batch, ret['full_img_feats'], 
                                                         ret['full_text_feats']))

        # Fine-Tuning: Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch, test=test))

        # Fine-Tuning: Image-Text Classification
        if "cls" in self.current_tasks:
            ret.update(objectives.compute_cls(self, batch, test=test))

        # Fine-Tuning: Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch, test))

        # Fine-Tuning: Image-Text Classification
        if "mlc" in self.current_tasks:
            ret.update(objectives.compute_mlc(self, batch, test=test))

        # Fine-Tuning: Causal Language Modeling
        if "clm" in self.current_tasks:
            ret.update(objectives.compute_clm(self, batch, test=test))

        return ret

    def training_step(self, batch, batch_idx):
        mg3d_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v * self.hparams.config["loss_names"][k.replace("_loss", "")]
                          for k, v in output.items() if "loss" in k])
        return total_loss

    def on_train_epoch_end(self):
        mg3d_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        mg3d_utils.set_task(self)
        output = self(batch)


    def on_validation_epoch_end(self):
        mg3d_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        mg3d_utils.set_task(self)
        output = self(batch, test=True)

    def on_test_epoch_end(self):
        mg3d_utils.epoch_wrapup(self, test=True)

    def configure_optimizers(self):
        return mg3d_utils.set_schedule(self)

    def random_masking_3d(self, x, mask_ratio, cls_token=True):
        if cls_token:
            x_ = x[:, :1]
            x = x[:, 1:]
        else:
            x_ = torch.mean(x, dim=1).unsqueeze(1)
        pos_embed = self.vision_encoder.patch_embedding.position_embeddings.to(x)

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        

        x += pos_embed
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is removed
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # append cls token
        if cls_token:
            x_ = x_ + pos_embed[:, :1]
            x_masked = torch.cat((x_, x_masked), dim=1)

        return x_masked, mask, ids_restore
    
    def patchify(self, imgs):
        p = 16
        d, h, w = 8, 8, 6
        x = imgs.reshape(shape=(imgs.shape[0], 1, d, p, h, p, w, p))
        x = torch.einsum('ncdohpwq->ndhwopqc', x)
        x = x.reshape(shape=(imgs.shape[0], d * h * w, p ** 3 * 1))
        return x

    def unpatchify(self, x):
        p = self.hparams.config["patch_size"]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs

    @torch.no_grad()
    def find_pseudo_tokens(self, query_tensors, pseudo_token_pool, num_pseudo_tokens):
        bs, dim = query_tensors.shape
        queried_idx = (query_tensors @ pseudo_token_pool.T).topk(num_pseudo_tokens, -1)[1]
        pseudo_tokens = pseudo_token_pool.unsqueeze(0).repeat(bs, 1, 1).gather(
            1, queried_idx.unsqueeze(-1).repeat(1, 1, dim))
        return pseudo_tokens
