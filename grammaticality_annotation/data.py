import os
import random

import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.utils import class_weight

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from grammaticality_annotation.tokenizer import (TEXT_FIELD, LABEL_FIELD, TRANSCRIPT_FIELD,
                                                 ERROR_LABELS_FIELD, AGE_FIELD, UTT_ID_FIELD)
from utils import PROJECT_ROOT_DIR

DATA_SPLIT_RANDOM_STATE = 8
FINE_TUNE_RANDOM_STATE = 1

DATA_FILE_ANNOTATED_CHILDES_DB = os.path.join(PROJECT_ROOT_DIR, "data", "manual_annotation", "annotated_childes_db.csv")

LABEL_GRAMMATICAL = 2
LABEL_UNGRAMMATICAL = 0

DEFAULT_BATCH_SIZE = 5


if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


def create_cv_folds(data, num_folds):
    transcript_files_sizes = data.transcript_file.value_counts()

    test_sets = [pd.DataFrame() for _ in range(num_folds)]

    while len(transcript_files_sizes) > 0:
        smallest_fold = np.argmin([len(s) for s in test_sets])
        largest_transcript = transcript_files_sizes.index[0]

        # Add the largest transcript to the smallest fold
        test_sets[smallest_fold] = pd.concat([test_sets[smallest_fold], data[data.transcript_file == largest_transcript]])
        del transcript_files_sizes[largest_transcript]

    train_sets = [data[~data.index.isin(data_test.index)].copy() for data_test in test_sets]

    print(f"Test set sizes: {[len(s) for s in test_sets]}")
    return train_sets, test_sets


def train_val_split(data, val_split_size, random_seed=DATA_SPLIT_RANDOM_STATE):
    # Make sure that test and train split do not contain data from the same transcripts
    if isinstance(val_split_size, float):
        val_split_size = int(len(data) * val_split_size)
    transcript_files = data.transcript_file.unique()
    random.seed(random_seed)
    random.shuffle(transcript_files)
    transcript_files = iter(transcript_files)
    data_val = pd.DataFrame()
    # Append transcripts until we have the approximate train data size.
    while len(data_val) < val_split_size:
        data_val = pd.concat([data_val, data[data.transcript_file == next(transcript_files)]])

    data_train = data[~data.index.isin(data_val.index)].copy()

    assert (len(set(data_train.index) & set(data_val.index)) == 0)
    print(f"Train data size: {len(data_train)} | val data size: {len(data_val)}")
    return data_train, data_val


def add_context(transcripts, context_length=0, sep_token=None):
    data = []
    rows_to_annotate = transcripts[~transcripts[LABEL_FIELD].isna()]
    for i, row in tqdm(rows_to_annotate.iterrows(), total=len(rows_to_annotate)):
        sentence = row.speaker_code + row.transcript_clean
        if sep_token and context_length >= 1:
            sentence = sep_token + sentence
        for j in range(1, context_length+1):
            if i-j in transcripts.index:
                context_sentence = transcripts.loc[i-j].speaker_code + transcripts.loc[i-j].transcript_clean
                sentence = context_sentence + sentence
        datapoint = {
            TEXT_FIELD: sentence,
            TRANSCRIPT_FIELD: row[TRANSCRIPT_FIELD],
            UTT_ID_FIELD: row["id"],
            AGE_FIELD: row[AGE_FIELD],
        }
        if LABEL_FIELD in row.index:
            datapoint[LABEL_FIELD] = row[LABEL_FIELD]
        if ERROR_LABELS_FIELD in row.index:
            datapoint[ERROR_LABELS_FIELD] = row[ERROR_LABELS_FIELD]
        data.append(datapoint)

    data = pd.DataFrame.from_records(data)

    if LABEL_FIELD in data.columns and is_numeric_dtype(data[LABEL_FIELD]):
        # Transform -1, 0, 1 to 0, 1, 2 so that they can be of dtype long
        data[LABEL_FIELD] = (data[LABEL_FIELD] + 1).astype("int64")

    print("Dataset size: ", len(data))
    return data


def create_dataset_dicts(num_cv_folds, val_split_proportion, context_length, random_seed=DATA_SPLIT_RANDOM_STATE, train_data_size=1.0, create_val_split=False, sep_token=None):
    dataset_dicts = [DatasetDict() for _ in range(num_cv_folds)]

    data_manual_annotations = pd.read_csv(DATA_FILE_ANNOTATED_CHILDES_DB)
    data_manual_annotations = add_context(data_manual_annotations, context_length=context_length, sep_token=sep_token)
    data_manual_annotations_train_splits, data_manual_annotations_test_splits = create_cv_folds(data_manual_annotations, num_cv_folds)
    if train_data_size < 1.0:
        data_manual_annotations_train_splits = [d.sample(round(len(d) * train_data_size), random_state=DATA_SPLIT_RANDOM_STATE) for d in data_manual_annotations_train_splits]

    for fold in range(num_cv_folds):
        if create_val_split:
            data_manual_annotations_train_splits[fold], data_manual_annotations_val_split = train_val_split(data_manual_annotations_train_splits[fold], val_split_proportion, random_seed)
            ds_val = Dataset.from_pandas(data_manual_annotations_val_split, preserve_index=False)
            dataset_dicts[fold]["validation"] = ds_val

        ds_train = Dataset.from_pandas(data_manual_annotations_train_splits[fold], preserve_index=False)
        dataset_dicts[fold]["train"] = ds_train

        ds_test = Dataset.from_pandas(data_manual_annotations_test_splits[fold], preserve_index=False)
        dataset_dicts[fold]["test"] = ds_test

    return dataset_dicts


class CHILDESGrammarDataModule(LightningDataModule):
    def __init__(
            self,
            train_batch_size: int = DEFAULT_BATCH_SIZE,
            eval_batch_size: int = DEFAULT_BATCH_SIZE,
            max_seq_length: int = 128,
            num_cv_folds = 5,
            val_split_proportion: float = 0.2,
            context_length: int = 8,
            random_seed = FINE_TUNE_RANDOM_STATE,
            num_workers = 8,
            add_eos_tokens = False,
            train_data_size = 1.0,
            fold = 0,
            lowercase = False,
            **kwargs,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_cv_folds = num_cv_folds
        self.val_split_proportion = val_split_proportion
        self.context_length = context_length
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.train_data_size = train_data_size
        self.fold = fold

        self.num_labels = 3
        self.add_eos_tokens = add_eos_tokens
        self.lowercase = lowercase

    def setup(self, stage: str):
        model_name = self.trainer.model.hparams.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        if "gpt2" in model_name:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.trainer.model.model.pad_token_id = self.trainer.model.model.eos_token_id

        datasets = create_dataset_dicts(self.num_cv_folds, self.val_split_proportion, self.context_length,
                                        self.train_data_size, create_val_split=True,
                                        sep_token=self.tokenizer.sep_token, train_data_size=self.train_data_size)
        self.dataset = datasets[self.fold]
        print(f"Selected dataset CV fold: {self.fold}")

        for split in self.dataset.keys():
            columns = [c for c in self.dataset[split].column_names]
            self.dataset[split].set_format(type="torch", columns=columns)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True, collate_fn=self.tokenize_batch, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size, collate_fn=self.tokenize_batch, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, collate_fn=self.tokenize_batch, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dataset["pred"], batch_size=self.eval_batch_size, collate_fn=self.tokenize_inference_batch, num_workers=self.num_workers, shuffle=False)

    def tokenize_batch(self, batch):
        return tokenize(batch, self.tokenizer, self.max_seq_length, add_eos_token=self.add_eos_tokens, lowercase=self.lowercase)

    def tokenize_inference_batch(self, batch):
        return tokenize(batch, self.tokenizer, self.max_seq_length, add_eos_token=self.add_eos_tokens,
                        add_labels=False, add_file_ids=True, add_utt_ids=True, lowercase=self.lowercase)


def tokenize(batch, tokenizer, max_seq_length, add_eos_token=False, add_labels=True, add_file_ids=False, add_utt_ids=False, lowercase=False):
    texts = [b[TEXT_FIELD] for b in batch]
    if lowercase:
        texts = [t.lower() for t in texts]
    if add_eos_token:
        texts = [t+tokenizer.eos_token for t in texts]

    features = tokenizer.batch_encode_plus(
        texts, max_length=max_seq_length, padding=True, truncation=True, return_tensors="pt"
    )
    if add_labels:
        features.data[LABEL_FIELD] = torch.tensor([b[LABEL_FIELD] for b in batch])
    if add_file_ids:
        features.data[TRANSCRIPT_FIELD] = torch.tensor([b[TRANSCRIPT_FIELD] for b in batch])
    if add_utt_ids:
        features.data[UTT_ID_FIELD] = torch.tensor([b[UTT_ID_FIELD] for b in batch])

    return features


def calc_class_weights(labels):
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)

    return class_weights
