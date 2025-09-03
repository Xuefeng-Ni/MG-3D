from typing import Dict
from pathlib import Path
from glob import glob
import SimpleITK
import torch

from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

from utils import MultiClassAlgorithm, to_input_format, unpack_single_output, device
from algorithm.preprocess import preprocess
from algorithm.i3d.i3dpt import I3D


COVID_OUTPUT_NAME = Path("probability-covid-19")
SEVERE_OUTPUT_NAME = Path("probability-severe-covid-19")


class StoicAlgorithm(MultiClassAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path=Path("/ssd1/xuefeng/data/test_mha/"),
            output_path=Path("/home/xuefeng/stoic2021/inference/output/")
        )

        # load model
        backbone = 'swin_unetr_b' 
        if backbone == 'swin_unetr':
            from algorithm.i3d.swin_b import SwinTransformer
            from monai.utils import ensure_tuple_rep
            spatial_dims = 3
            patch_size = ensure_tuple_rep(2, spatial_dims)
            window_size = ensure_tuple_rep(7, spatial_dims)
            self.model = SwinTransformer(            
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
                    num_classes=2)
        elif backbone == 'swin_unetr_l':
            from algorithm.i3d.swin_l import SwinTransformer
            from monai.utils import ensure_tuple_rep
            spatial_dims = 3
            patch_size = ensure_tuple_rep(2, spatial_dims)
            window_size = ensure_tuple_rep(7, spatial_dims)
            self.model = SwinTransformer(            
                    in_chans=1,
                    embed_dim=96,
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
                    num_classes=2)
            
        self.model = self.model.to(device)
        self.model.load_state_dict(
            torch.load("/home/xuefeng/stoic2021/training/output/swin3d/baseline.pth", map_location=torch.device(device)))
        self.model = self.model.eval()

    def predict(self, *, input_image: SimpleITK.Image) -> Dict:
        # pre-processing
        input_image = preprocess(input_image)
        input_image = to_input_format(input_image)

        # run model
        with torch.no_grad():
            output = torch.sigmoid(self.model(input_image))
        prob_covid, prob_severe = unpack_single_output(output)

        return {
            COVID_OUTPUT_NAME: prob_covid,
            SEVERE_OUTPUT_NAME: prob_severe
        }


if __name__ == "__main__":
    StoicAlgorithm().process()
