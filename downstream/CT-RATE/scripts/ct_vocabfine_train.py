import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from data_inference import CTReportDatasetinfer

from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
import torch.nn.functional as F
from src.args import parse_arguments
from src.models.utils import cosine_lr, torch_load, LabelSmoothing
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_lr(optimizer):
    # Function to get the current learning rate of the optimizer
    for param_group in optimizer.param_groups:
        return param_group['lr']

def finetune(args):
    backbone = 'swin_unetr_b' # 'swin_unetr_l'
    # Initialize BERT tokenizer and text encoder
    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialize image encoder and clip model
    if backbone == 'swin_unetr_b':
        from monai.utils import ensure_tuple_rep
        from baseline.swin import SwinTransformer
        spatial_dims = 3
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        image_encoder = SwinTransformer(            
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
            image_encoder=image_encoder, text_encoder=text_encoder,
            dim_image=36864, dim_text=768, dim_latent=768,  #20736 36864
            extra_latent_projection=False, use_mlm=False,
            downsample_image_embeds=False, use_all_token_embeds=False
        )
        model_dict = dict(image_encoder.state_dict())

        extractor_dict = torch.load('/home/ubuntu/MG-3D-Swin-B.ckpt', map_location=torch.device('cpu'))

        new_extractor_dict, new_extractor_dict_2, new_extractor_dict_3, new_extractor_dict_4 = \
            {}, {}, {}, {}
        for key, value in extractor_dict.items():
            name = key[15:]
            if key[:14] == 'vision_encoder':
                new_extractor_dict[name] = value
            name_2 = key[36:]
            if key[:35] == 'local_img_cross_multihead_attention':
                new_extractor_dict_2[name_2] = value
            name_3 = key[24:]
            if key[:23] == 'multi_modal_vision_proj':
                new_extractor_dict_3[name_3] = value
            name_4 = key[25:]
            if key[:24] == 'modality_type_embeddings':
                new_extractor_dict_4[name_4] = value
        image_encoder.load_state_dict(new_extractor_dict, strict=False)

        pretrain_dict = {k: v for k, v in new_extractor_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        clip.visual_transformer.load_state_dict(pretrain_dict, strict=False)
        
        clip_dict = torch.load(args.pretrained, map_location=torch.device('cpu'))
        clip_dict = {k: v for k, v in clip_dict.items() if not (k.startswith('visual_transformer')
                                                                or k.startswith('to_visual_latent') 
                                                                or k.startswith('to_text_latent'))}
        clip.load_state_dict(clip_dict, strict=False)
        clip.local_img_cross_multihead_attention.load_state_dict(new_extractor_dict_2, strict=False)
        clip.multi_modal_vision_proj.load_state_dict(new_extractor_dict_3, strict=False)
        clip.modality_type_embeddings.load_state_dict(new_extractor_dict_4, strict=False)
        num_classes = 18  # Specify the number of classes
        print('Fine-tuning end-to-end')
        model = clip

    elif backbone == 'swin_unetr_l':
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

        extractor_dict = torch.load('/home/ubuntu/MG-3D-Swin-L.ckpt', map_location=torch.device('cpu'))

        model_dict = dict(image_encoder.state_dict())
        new_extractor_dict, new_extractor_dict_2, new_extractor_dict_3, new_extractor_dict_4 = \
            {}, {}, {}, {}
        for key, value in extractor_dict.items():
            name = key[15:]
            if key[:14] == 'vision_encoder':
                new_extractor_dict['swinViT.' + name] = value
            name_2 = key[36:]
            if key[:35] == 'local_img_cross_multihead_attention':
                new_extractor_dict_2[name_2] = value
            name_3 = key[24:]
            if key[:23] == 'multi_modal_vision_proj':
                new_extractor_dict_3[name_3] = value
            name_4 = key[25:]
            if key[:24] == 'modality_type_embeddings':
                new_extractor_dict_4[name_4] = value
        image_encoder.load_state_dict(new_extractor_dict, strict=False)

        pretrain_dict = {k: v for k, v in new_extractor_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        clip.visual_transformer.load_state_dict(pretrain_dict, strict=False)
        
        clip_dict = torch.load(args.pretrained, map_location=torch.device('cpu'))
        clip_dict = {k: v for k, v in clip_dict.items() if not (k.startswith('visual_transformer')
                                                                or k.startswith('to_visual_latent') 
                                                                or k.startswith('to_text_latent'))}
        clip.load_state_dict(clip_dict, strict=False)
        clip.local_img_cross_multihead_attention.load_state_dict(new_extractor_dict_2, strict=False)
        clip.multi_modal_vision_proj.load_state_dict(new_extractor_dict_3, strict=False)
        clip.modality_type_embeddings.load_state_dict(new_extractor_dict_4, strict=False)
        num_classes = 18  # Specify the number of classes
        print('Fine-tuning end-to-end')
        model = clip

    for name, param in model.named_parameters():
        if ("latent" in name) or ("visual_transformer.proj_head" in name) or \
            ("cross_multihead_attention" in name) or ("multi_modal_vision_proj" in name) or \
                ("modality_type_embeddings" in name):

            print(name, param.shape)
        else:
            param.requires_grad = False


    ds = CTReportDatasetinfer(data_folder=args.data_folder, csv_file=args.reports_file,labels=args.labels)
    dl = DataLoader(ds, num_workers=8, batch_size=1, shuffle=True)
    num_batches = len(dl)

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    model.cuda()
    if args.resume_checkpoint != None:
        model.load(args.resume_checkpoint)
    
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    loss_fn = torch.nn.MSELoss()

    params = [p for p in model.parameters() if p.requires_grad]
    
    if args.resume_checkpoint == None:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)
        start_epoch = 0
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)
        optim_checkpoint = torch.load(args.resume_optim)
        optimizer = torch.optim.AdamW(params, lr=optim_checkpoint['param_groups'][0]['lr'], 
                                      weight_decay=args.wd)
        start_epoch = 0
    
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    for epoch in range(start_epoch, args.epochs):
        for i, batch in tqdm.tqdm(enumerate(dl)):
            start_time = time.time()
            step = i + epoch * num_batches
            scheduler(step)

            inputs, _, labels, _, _ = batch

            logits = []
            labels_tensor_all = labels.float().to(torch.device('cuda'))

            for k in range(3):
                logits_list = []
                labels_list = []

                pathologies_all = ['Medical material', 'Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion',
                                   'Coronary artery wall calcification', 'Hiatal hernia', 'Lymphadenopathy', 'Emphysema',
                                   'Atelectasis', 'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion',
                                   'Mosaic attenuation pattern', 'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',
                                   'Interlobular septal thickening']
                pathologies = pathologies_all[k * 6:(k + 1) * 6]
                labels_tensor = labels_tensor_all[0][k * 6:(k + 1) * 6]

                for l in range(len(labels_tensor)):
                    text_yes = ""
                    text_no = ""
                    if labels_tensor[l] == 1:
                        text_yes = text_yes + f"{pathologies[l]}. "
                        text_no = text_no + f"not {pathologies[l]}. "
                    if labels_tensor[l] == 0:
                        text_yes = text_yes + f"not {pathologies[l]}. "
                        text_no = text_no + f"{pathologies[l]}. "
                    text = [text_yes, text_no]
                    text_tokens = tokenizer(
                        text, return_tensors="pt", padding="max_length", truncation=True, max_length=250).to(
                        torch.device('cuda'))
                    output = model(text_tokens, inputs, batch[4], device=torch.device('cuda'))

                    logits = F.softmax(output, dim=0)
                    labels = torch.tensor([1.0, 0.0]).cuda()
                    logits_list.append(logits)
                    labels_list.append(labels)

                concat_logits = torch.cat(logits_list, dim=0)
                concat_labels = torch.cat(labels_list, dim=0)

                loss = loss_fn(concat_logits, concat_labels)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            print(get_lr(optimizer))

            batch_time = time.time() - start_time

            if i % args.print_every == 0:
                percent_complete = 100 * i / len(dl)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dl)}]\t"
                    f"Loss: {loss.item():.6f}\tBatch (t) {batch_time:.3f}", flush=True
                )
            if i % args.save_every == 0:
                os.makedirs(args.save, exist_ok=True)

                # Access the underlying model to avoid the 'module.' prefix in state_dict keys
                model_to_save = model.module if hasattr(model, 'module') else model

                model_path = os.path.join(args.save, f'checkpoint_{i}_epoch_{epoch+1}.pt')
                print('Saving model to', model_path)

                # Save the state_dict of the unwrapped model
                torch.save(model_to_save.state_dict(), model_path)

                optim_path = os.path.join(args.save, f'optim_{i}_epoch_{epoch+1}.pt')

                # Save the optimizer state
                torch.save(optimizer.state_dict(), optim_path)

        # Saving model
        if args.save is not None:
            os.makedirs(args.save, exist_ok=True)

            # Access the underlying model to avoid the 'module.' prefix in state_dict keys
            model_to_save = model.module if hasattr(model, 'module') else model

            model_path = os.path.join(args.save, f'epoch_{epoch+1}.pt')
            print('Saving model to', model_path)

            # Save the state_dict of the unwrapped model
            torch.save(model_to_save.state_dict(), model_path)

            optim_path = os.path.join(args.save, f'epoch_{epoch+1}.pt')

            # Save the optimizer state
            torch.save(optimizer.state_dict(), optim_path)

    if args.save is not None:
        return model_path


if __name__ == '__main__':
    args = parse_arguments()
    finetune(args)
