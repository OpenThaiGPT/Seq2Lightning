from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pytorch_lightning as pl
from typing import List, Any, Dict
import datasets
import re

def clean_text(text: str, max_uppercase_len: int = 8):
    """Remove artifacts left after parsing"""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[|~`]', ' ', text)
    text = re.sub(r'/ха', '', text)
    text = re.sub(r'[A-Z]{' + str(max_uppercase_len) + r',}', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def clean_text_array(texts: List[str], max_uppercase_len=8):
    """Remove artifacts left after parsing from an array of texts"""
    return list(map(lambda x: clean_text(x, max_uppercase_len), texts))


class Seq2SeqDataModule(pl.LightningDataModule):
    """Datamodule with preprocessing for sequence-to-sequence tasks"""
    def __init__(self, source_column: str, target_column: str,
                 tokenizer: str, **params):
        super().__init__()
        self.source_column = source_column
        self.target_column = target_column
        self.max_source_length = params.get("max_source_length", 512)
        self.max_target_length = params.get("max_target_length", 512)
        self.num_workers = params.get("num_workers", 2)
        self.train_batchsize = params.get("train_batchsize", 5)
        self.test_batchsize = params.get("test_batchsize", 5)
        self.params = params
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def setup(self, stage: str) -> None:
        existing_split = None
        name_or_ext = self.params.get("dataset_name", None)
        data_files = {}
        if self.params.get("train_data_file", None) is not None:
            train_data_file = self.params["train_data_file"]
            name_or_ext = train_data_file.split('.')[-1]
            data_files["train"] = train_data_file
            existing_split = "train"
        if self.params.get("val_data_file", None) is not None:
            val_data_file = self.params["val_data_file"]
            name_or_ext = val_data_file.split('.')[-1]
            data_files["validation"] = val_data_file
            existing_split = "validation"
        if self.params.get("test_data_file", None) is not None:
            test_data_file = self.params["test_data_file"]
            name_or_ext = test_data_file.split('.')[-1]
            data_files["test"] = test_data_file
            existing_split = "test"
        if name_or_ext is None:
            raise ValueError("You have to specify at least one data file or dataset name from library")
        self.dataset = datasets.load_dataset(name_or_ext,
                                             data_files=data_files)
        for split in self.dataset.keys():
            if self.params.get("max_samples", None) is not None:
                self.dataset[split] = self.dataset[split].select(range(self.params["max_samples"]))
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                num_proc=self.num_workers,
            )
            self.dataset[split].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    def convert_to_features(self, examples) -> Dict[str, Any]:
        padding_type = self.params.get("padding", "longest")
        inputs = examples[self.source_column]
        targets = examples[self.target_column]
        if self.params.get("do_cleaning"):
            inputs = clean_text_array(inputs)
            targets = clean_text_array(targets)
        features = self.tokenizer.batch_encode_plus(inputs,
                                                    max_length=self.max_source_length,
                                                    padding=padding_type, truncation=True)
        encoded_targets = self.tokenizer.batch_encode_plus(targets,
                                                           max_length=self.max_target_length,
                                                           padding=padding_type, truncation=True)
        """if padding_type == "max_length" and self.params.ignore_pad_token_for_loss:
            encoded_targets["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in target]
                for target in encoded_targets["input_ids"]]
        """
        features["labels"] = encoded_targets["input_ids"]
        return features

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset["train"], shuffle=True,
                          batch_size=self.train_batchsize,
                          num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset["validation"], shuffle=False,
                          batch_size=self.test_batchsize,
                          num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset["test"], shuffle=False,
                          batch_size=self.test_batchsize,
                          num_workers=self.num_workers)