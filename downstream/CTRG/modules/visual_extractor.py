import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from .efficientnet_pytorch.model import EfficientNet
from .swin import SwinTransformer
from monai.utils import ensure_tuple_rep
import numpy as np
import math



class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
        )

    def forward(self, x):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.densenet121.classifier(out)
        return out


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.args = args
        print(f"=> creating model '{args.visual_extractor}'")
        if 'swin3d_b' in args.visual_extractor:
            spatial_dims = 3
            self.visual_extractor = 48
            from .swin import Swin
            self.model = Swin(in_channels=1, feature_size=args.feature_size, spatial_dims=spatial_dims)
            self.classifier = nn.Linear(args.d_vf, args.num_labels)

            extractor_dict = torch.load('/home/ubuntu/MG-3D-Swin-B.ckpt', map_location=torch.device('cpu'))
            new_extractor_dict = {}
            model_dict = dict(self.model.state_dict())
            for key, value in extractor_dict.items():
                name = key[15:]
                if key[:14] == 'vision_encoder':
                    new_extractor_dict['swinViT.' + name] = value
                if key[:22] == 'vision_encoder.encoder':
                    new_extractor_dict[name] = value
            pretrain_dict = {k: v for k, v in new_extractor_dict.items() if k in model_dict}
            model_dict.update(pretrain_dict)
            self.model.load_state_dict(pretrain_dict, strict=False)

        elif 'swin3d_l' in args.visual_extractor:
            spatial_dims = 3
            self.visual_extractor = args.visual_extractor
            from .swin import Swin
            self.model = Swin(in_channels=1, feature_size=96, spatial_dims=spatial_dims)
            self.classifier = nn.Linear(args.d_vf, args.num_labels)

            extractor_dict = torch.load('/home/ubuntu/MG-3D-Swin-L.ckpt', map_location=torch.device('cpu'))
            new_extractor_dict = {}
            model_dict = dict(self.model.state_dict())
            for key, value in extractor_dict.items():
                name = key[15:]
                if key[:14] == 'vision_encoder':
                    new_extractor_dict['swinViT.' + name] = value
                if key[:22] == 'vision_encoder.encoder':
                    new_extractor_dict[name] = value
            pretrain_dict = {k: v for k, v in new_extractor_dict.items() if k in model_dict}
            model_dict.update(pretrain_dict)
            self.model.load_state_dict(pretrain_dict, strict=False)
        else:
            raise NotImplementedError

        # load pretrained visual extractor
        if args.pretrain_cnn_file and args.pretrain_cnn_file != "":
            checkpoint = torch.load(args.pretrain_cnn_file, map_location=torch.device('cpu'))
            if "state_dict" in checkpoint.keys():
                checkpoint = checkpoint["state_dict"]
            self.model.load_state_dict(checkpoint, strict=False)
            print(f'Load pretrained model from: VoCo pre-trained model')

        else:
            print(f'Load pretrained CNN model from: official pretrained in ImageNet')

    def forward(self, images):
        if 'swin3d' in self.visual_extractor:
            patch_feats, avg_feats = self.model(images)
            labels = self.classifier(avg_feats)
        else:
            raise NotImplementedError
        return patch_feats, avg_feats, labels
