from .base_datamodule import BaseDataModule
from ..datasets import CHESTCTRGDataset


class CHESTCTRGDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return CHESTCTRGDataset

    @property
    def dataset_cls_no_false(self):
        return CHESTCTRGDataset

    @property
    def dataset_name(self):
        return "chest_ctrg"
