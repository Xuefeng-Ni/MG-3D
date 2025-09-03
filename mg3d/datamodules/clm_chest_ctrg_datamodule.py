from .base_datamodule import BaseDataModule
from ..datasets import CLMCHESTCTRGDataset


class CLMCHESTCTRGDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return CLMCHESTCTRGDataset

    @property
    def dataset_cls_no_false(self):
        return CLMCHESTCTRGDataset

    @property
    def dataset_name(self):
        return "clm_chest_ctrg"
