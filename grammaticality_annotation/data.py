import os
import random
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.utils import class_weight

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from grammaticality_annotation.tokenizer import (TEXT_FIELD, LABEL_FIELD, TOKEN_SPEAKER_CHILD, TRANSCRIPT_FIELD,
                                                 TOKEN_SPEAKER_CAREGIVER, ERROR_LABELS_FIELD)
from utils import PROJECT_ROOT_DIR, SPEAKER_CODE_CHILD, SPEAKER_CODES_CAREGIVER

DATA_SPLIT_RANDOM_STATE = 8

DATA_PATH_CHILDES_ANNOTATED = os.path.join(PROJECT_ROOT_DIR, "data", "manual_annotation", "annotated")

LABEL_GRAMMATICAL = 2
LABEL_UNGRAMMATICAL = 0


if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


def speaker_code_to_speaker_token(code):
    if code == SPEAKER_CODE_CHILD:
        return TOKEN_SPEAKER_CHILD
    if code in SPEAKER_CODES_CAREGIVER:
        return TOKEN_SPEAKER_CAREGIVER
    raise RuntimeError("Unknown speaker code: ", code)


def create_cv_folds(data, num_folds):
    transcript_files_sizes = data.transcript_file.value_counts()

    test_split_size = round(len(data) / 5)
    test_sets = [pd.DataFrame() for _ in range(num_folds)]

    while len(transcript_files_sizes) > 0:
        fold = np.argmin([len(s) for s in test_sets])
        distances = transcript_files_sizes - test_split_size - len(test_sets[fold])
        distances = distances.__abs__().sort_values()

        transcript_closest_size = distances.index[0]
        test_sets[fold] = pd.concat([test_sets[fold], data[data.transcript_file == transcript_closest_size]])
        del transcript_files_sizes[transcript_files_sizes.index[0]]

    train_sets = [data[~data.index.isin(data_test.index)].copy() for data_test in test_sets]

    print(f"Test set sizes: {[len(s) for s in test_sets]}")
    return train_sets, test_sets


def train_val_split(data, val_split_size, random_seed=DATA_SPLIT_RANDOM_STATE):
    # Make sure that test and train split do not contain data from the same transcripts
    if isinstance(val_split_size, float):
        train_data_size = int(len(data) * (1 - val_split_size))
    else:
        train_data_size = len(data) - val_split_size
    transcript_files = data.transcript_file.unique()
    random.seed(random_seed)
    random.shuffle(transcript_files)
    transcript_files = iter(transcript_files)
    data_train = pd.DataFrame()
    # Append transcripts until we have the approximate train data size.
    while len(data_train) < train_data_size:
        data_train = pd.concat([data_train, data[data.transcript_file == next(transcript_files)]])

    data_val = data[~data.index.isin(data_train.index)].copy()

    assert (len(set(data_train.index) & set(data_val.index)) == 0)
    return data_train, data_val


def load_annotated_childes_data(path, exclude_test_data=False):
    transcripts = []
    file_ids_annotated = [f.name[0] for f in Path(DATA_PATH_CHILDES_ANNOTATED).glob("*.csv")]
    for f in Path(path).glob("*.csv"):
        if not exclude_test_data or (f.name.replace(".csv", "") not in file_ids_annotated):
            transcripts.append(pd.read_csv(f, index_col=0))

    transcripts = pd.concat(transcripts, ignore_index=True)
    transcripts["speaker_code"] = transcripts.speaker_code.apply(speaker_code_to_speaker_token)
    transcripts["sentence"] = transcripts.apply(lambda row: row.speaker_code + row.transcript_clean,
                                                    axis=1).values
    return transcripts


def load_annotated_childes_datasplits(context_length=0, num_cv_folds=5, sep_token=None, keep_error_labels_column=False):
    transcripts = load_annotated_childes_data(DATA_PATH_CHILDES_ANNOTATED)
    data = []
    for i, row in transcripts[~transcripts[LABEL_FIELD].isna()].iterrows():
        sentence = row.sentence
        if sep_token and context_length >= 1:
            sentence = sep_token + sentence
        for j in range(1, context_length+1):
            if i-j in transcripts.index:
                context_sentence = transcripts.loc[i-j].sentence
                sentence = context_sentence + sentence
        datapoint = {
            TEXT_FIELD: sentence,
            LABEL_FIELD: row[LABEL_FIELD],
            TRANSCRIPT_FIELD: row[TRANSCRIPT_FIELD],
        }
        if keep_error_labels_column:
            datapoint[ERROR_LABELS_FIELD] = row[ERROR_LABELS_FIELD]
        data.append(datapoint)

    data = pd.DataFrame.from_records(data)

    # Transform -1, 0, 1 to 0, 1, 2 so that they can be of dtype long
    data[LABEL_FIELD] = (data[LABEL_FIELD] + 1).astype("int64")

    print("Dataset size: ", len(data))
    return create_cv_folds(data, num_cv_folds)


LOADER_COLUMNS = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        LABEL_FIELD,
    ]


def create_dataset_dicts(num_cv_folds, val_split_proportion, context_length, random_seed=DATA_SPLIT_RANDOM_STATE, train_data_size=1.0, create_val_split=False, sep_token=None):
    dataset_dicts = [DatasetDict() for _ in range(num_cv_folds)]

    data_manual_annotations_train_splits, data_manual_annotations_test_splits = load_annotated_childes_datasplits(context_length, num_cv_folds, sep_token)
    if train_data_size < 1.0:
        data_manual_annotations_train_splits = [d.sample(round(len(d) * train_data_size), random_state=DATA_SPLIT_RANDOM_STATE) for d in data_manual_annotations_train_splits]

    for fold in range(num_cv_folds):
        if create_val_split:
            data_manual_annotations_train_splits[fold], data_manual_annotations_val_split = train_val_split(data_manual_annotations_train_splits[fold], val_split_proportion, random_seed)
            ds_val = Dataset.from_pandas(data_manual_annotations_val_split)
            dataset_dicts[fold]["validation"] = ds_val

        ds_train = Dataset.from_pandas(data_manual_annotations_train_splits[fold])
        dataset_dicts[fold]["train"] = ds_train

        ds_test = Dataset.from_pandas(data_manual_annotations_test_splits[fold])
        dataset_dicts[fold]["test"] = ds_test

    return dataset_dicts


class CHILDESGrammarDataModule(LightningDataModule):
    def __init__(
            self,
            model_name_or_path: str,
            train_batch_size: int,
            eval_batch_size: int,
            dataset: Dataset,
            tokenizer,
            max_seq_length: int = 128,
            num_cv_folds = 5,
            val_split_proportion: float = 0.2,
            context_length: int = 1,
            random_seed = 1,
            num_workers = 8,
            add_eos_tokens = False,
            train_data_size = 1.0,
            **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_cv_folds = num_cv_folds
        self.val_split_proportion = val_split_proportion
        self.context_length = context_length
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.train_data_size = train_data_size
        self.dataset = dataset

        self.num_labels = 3
        self.tokenizer = tokenizer
        self.add_eos_tokens = add_eos_tokens

    def setup(self, stage: str):
        for split in self.dataset.keys():
            columns = [c for c in self.dataset[split].column_names if c in LOADER_COLUMNS]
            self.dataset[split].set_format(type="torch", columns=columns + [TEXT_FIELD])

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True, collate_fn=self.tokenize_batch, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size, collate_fn=self.tokenize_batch, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, collate_fn=self.tokenize_batch, num_workers=self.num_workers)

    def tokenize_batch(self, batch):
        return tokenize(batch, self.tokenizer, self.max_seq_length, add_labels=True, add_eos_token=self.add_eos_tokens)


def tokenize(batch, tokenizer, max_seq_length, add_labels=False, add_eos_token=False):
    texts = [b[TEXT_FIELD] for b in batch]
    if add_eos_token:
        texts = [t+tokenizer.eos_token for t in texts]

    features = tokenizer.batch_encode_plus(
        texts, max_length=max_seq_length, padding=True, truncation=True, return_tensors="pt"
    )
    if add_labels:
        features.data[LABEL_FIELD] = torch.tensor([b[LABEL_FIELD] for b in batch])

    return features


def calc_class_weights(labels):
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)

    return class_weights
