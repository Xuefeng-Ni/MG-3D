from transformers import BertTokenizerFast

from .base_dataset import BaseDataset


def get_chest_ctrg_tokenizer(path):
    return BertTokenizerFast(vocab_file="", tokenizer_file=path)


class CLMCHESTCTRGDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["clm_chest_ctrg_train"]
        elif split == "val":
            names = ["clm_chest_ctrg_val"]
        elif split == "test":
            names = ["clm_chest_ctrg_test"]
        else:
            raise ValueError

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")
        self.all_findings = self.table["findings"].to_pandas().tolist()
        self.all_impression = self.table["impression"].to_pandas().tolist()
        self.all_findings_sen = self.table["findings_sen"].to_pandas().tolist()
        self.all_impression_sen = self.table["impression_sen"].to_pandas().tolist()
        
    def __getitem__(self, index):
        return self.get_suite(index)

    def get_suite(self, index):
        ret = super(CLMCHESTCTRGDataset, self).get_suite(index)
        img_index, cap_index = self.index_mapper[index]
        ret["findings"] = self.all_findings[img_index][cap_index]
        ret["impression"] = self.all_impression[img_index][cap_index]
        ret["findings_sen"] = self.all_findings_sen[img_index]
        ret["impression_sen"] = self.all_impression_sen[img_index]
        
        return ret

    def collate(self, batch, mlm_collator):
        dict_batch = super(CLMCHESTCTRGDataset, self).collate(batch, mlm_collator)

        dict_batch["findings"] = [sample["findings"].lower() for sample in batch]
        dict_batch["impression"] = [sample["impression"].lower() for sample in batch]
        dict_batch["findings_sen"] = [sample["findings_sen"][i].lower() for sample in batch for i in range(sample["findings_sen"].size)]
        dict_batch["impression_sen"] = [sample["impression_sen"][i].lower() for sample in batch for i in range(sample["impression_sen"].size)]
        
        return dict_batch
