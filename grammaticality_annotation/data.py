import os
import random
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.utils import class_weight

from grammaticality_annotation.prepare_hiller_fernandez_data import HILLER_FERNANDEZ_DATA_OUT_PATH
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from grammaticality_annotation.tokenizer import TOKEN_EOS, TEXT_FIELD, \
    LABEL_FIELD, TOKEN_SPEAKER_CHILD, TRANSCRIPT_FIELD, TOKEN_SPEAKER_CAREGIVER, TOKEN_CLS
from utils import PROJECT_ROOT_DIR, SPEAKER_CODE_CHILD, SPEAKER_CODES_CAREGIVER

DATA_PATH_ZORRO = os.path.join(PROJECT_ROOT_DIR, "Zorro", "sentences", "babyberta")

DATA_SPLIT_RANDOM_STATE = 8

DATA_PATH_CHILDES_ANNOTATED = os.path.join(PROJECT_ROOT_DIR, "data", "manual_annotation", "annotated")

LABEL_GRAMMATICAL = 2
LABEL_UNGRAMMATICAL = 0


def speaker_code_to_speaker_token(code):
    if code == SPEAKER_CODE_CHILD:
        return TOKEN_SPEAKER_CHILD
    if code in SPEAKER_CODES_CAREGIVER:
        return TOKEN_SPEAKER_CAREGIVER
    raise RuntimeError("Unknown speaker code: ", code)


def train_test_split(data, test_split_proportion, random_seed):
    # Make sure that test and train split do not contain data from the same transcripts
    train_data_size = int(len(data) * (1 - test_split_proportion))
    transcript_files = data.transcript_file.unique()
    random.seed(random_seed)
    random.shuffle(transcript_files)
    transcript_files = iter(transcript_files)
    data_train = pd.DataFrame()
    # Append transcripts until we have the approximate train data size.
    while len(data_train) < train_data_size:
        data_train = pd.concat([data_train, data[data.transcript_file == next(transcript_files)]])

    data_test = data[~data.index.isin(data_train.index)].copy()

    assert (len(set(data_train.index) & set(data_test.index)) == 0)

    return data_train, data_test


def load_annotated_childes_data(path):
    transcripts = []
    for f in Path(path).glob("*.csv"):
        if os.path.isfile(f):
            transcripts.append(pd.read_csv(f, index_col=0))

    transcripts = pd.concat(transcripts, ignore_index=True)
    transcripts["speaker_code"] = transcripts.speaker_code.apply(speaker_code_to_speaker_token)
    transcripts["sentence"] = transcripts.apply(lambda row: row.speaker_code + row.transcript_clean,
                                                    axis=1).values
    return transcripts


def load_annotated_childes_datasplits(context_length=0, test_split_proportion=0.2, random_seed=1):
    transcripts = load_annotated_childes_data(DATA_PATH_CHILDES_ANNOTATED)
    data = []
    for i, row in transcripts[~transcripts[LABEL_FIELD].isna()].iterrows():
        sentence = [row.sentence]
        for j in range(1, context_length+1):
            if i-j in transcripts.index:
                context_sentence = transcripts.loc[i-j].sentence
                sentence = [context_sentence] + sentence
        data.append({
            TEXT_FIELD: sentence,
            LABEL_FIELD: row[LABEL_FIELD],
            TRANSCRIPT_FIELD: row[TRANSCRIPT_FIELD],
        })
    data = pd.DataFrame.from_records(data)

    # Transform -1, 0, 1 to 0, 1, 2 so that they can be of dtype long
    data[LABEL_FIELD] = (data[LABEL_FIELD] + 1).astype("int64")

    print("Dataset size: ", len(data))
    if test_split_proportion:
        return train_test_split(data, test_split_proportion, random_seed)
    else:
        return data


def load_hiller_fernandez_data():
    data_h_f = pd.read_csv(HILLER_FERNANDEZ_DATA_OUT_PATH, index_col=0)
    data_h_f.dropna(subset=[LABEL_FIELD, "transcript_clean"], inplace=True)

    data_h_f[LABEL_FIELD] = data_h_f[LABEL_FIELD].astype(int)

    data_h_f.rename(columns={"transcript_clean": TEXT_FIELD, "labels": "categories"}, inplace=True)

    data_h_f[TEXT_FIELD] = data_h_f[TEXT_FIELD].apply(lambda text: TOKEN_SPEAKER_CHILD + text)
    data_h_f[LABEL_FIELD].replace({0: LABEL_UNGRAMMATICAL, 1: LABEL_GRAMMATICAL}, inplace=True)

    return data_h_f


def prepare_zorro_data():
    path = DATA_PATH_ZORRO
    data_zorro = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    for i, line in enumerate(f.readlines()):
                        data_zorro.append({
                            TEXT_FIELD: line.replace("\n", ""),
                            LABEL_FIELD: LABEL_UNGRAMMATICAL if i % 2 == 0 else LABEL_GRAMMATICAL
                        })

    data_zorro = pd.DataFrame(data_zorro)

    data_zorro[TEXT_FIELD] = data_zorro[TEXT_FIELD].apply(lambda text: TOKEN_SPEAKER_CHILD + text)

    return data_zorro


def prepare_blimp_data():
    mor = load_dataset("metaeval/blimp_classification", "morphology")["train"].to_pandas()
    syntax = load_dataset("metaeval/blimp_classification", "syntax")["train"].to_pandas()
    data_blimp = pd.concat([mor, syntax], ignore_index=True)
    data_blimp.rename(columns={"sentence": TEXT_FIELD, "label": LABEL_FIELD}, inplace=True)
    data_blimp[LABEL_FIELD].replace({0: LABEL_UNGRAMMATICAL, 1: LABEL_GRAMMATICAL}, inplace=True)

    data_blimp[TEXT_FIELD] = data_blimp[TEXT_FIELD].apply(lambda text: TOKEN_SPEAKER_CHILD + text)

    data_blimp.set_index("idx", inplace=True)
    return data_blimp


def prepare_cola_data():
    dataset = load_dataset("glue", "cola")
    data_cola = dataset["train"]
    data_cola = data_cola.rename_column("sentence", TEXT_FIELD)
    data_cola = data_cola.rename_column("label", LABEL_FIELD)
    data_cola = data_cola.to_pandas()
    data_cola[LABEL_FIELD].replace({0: LABEL_UNGRAMMATICAL, 1: LABEL_GRAMMATICAL}, inplace=True)

    data_cola[TEXT_FIELD] = data_cola[TEXT_FIELD].apply(lambda text: TOKEN_SPEAKER_CHILD + text)

    data_cola.set_index("idx", inplace=True)
    return data_cola


LOADER_COLUMNS = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        LABEL_FIELD,
    ]


def create_dataset_dict(train_datasets, test_split_proportion, context_length, random_seed, create_val_split=False):
    dataset_dict = DatasetDict()

    data_manual_annotations_train, data_manual_annotations_test = load_annotated_childes_datasplits(context_length, test_split_proportion, random_seed)
    if create_val_split:
        data_manual_annotations_train, data_manual_annotations_val = train_test_split(data_manual_annotations_train, test_split_proportion, random_seed)
        ds_val = Dataset.from_pandas(data_manual_annotations_val)
        dataset_dict["validation"] = ds_val

    def get_dataset_with_name(ds_name, test=False):
        if ds_name == "manual_annotations":
            if test:
                return data_manual_annotations_test
            else:
                return data_manual_annotations_train
        elif ds_name == "hiller_fernandez":
            return load_hiller_fernandez_data()
        elif ds_name == "cola":
            return prepare_cola_data()
        elif ds_name == "blimp":
            return prepare_blimp_data()
        elif ds_name == "zorro":
            return prepare_zorro_data()
        else:
            raise RuntimeError("Unknown dataset: ", ds_name)

    data_train = []
    for ds_name in train_datasets:
        data_train.append(get_dataset_with_name(ds_name, test=False))

    data_train = pd.concat(data_train, ignore_index=True)
    ds_train = Dataset.from_pandas(data_train)
    dataset_dict['train'] = ds_train

    ds_test = Dataset.from_pandas(data_manual_annotations_test)
    dataset_dict['test'] = ds_test

    return dataset_dict


class CHILDESGrammarDataModule(LightningDataModule):
    def __init__(
            self,
            model_name_or_path: str,
            train_batch_size: int,
            eval_batch_size: int,
            train_datasets: list,
            tokenizer,
            max_seq_length: int = 128,
            test_split_proportion: float = 0.2,
            context_length: int = 1,
            random_seed = 1,
            **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.test_split_proportion = test_split_proportion
        self.train_datasets = train_datasets
        self.context_length = context_length
        self.random_seed = random_seed

        self.num_labels = 3
        self.tokenizer = tokenizer

    def setup(self, stage: str):
        self.dataset = create_dataset_dict(self.train_datasets, self.test_split_proportion, self.context_length, self.random_seed, create_val_split=True)
        for split in self.dataset.keys():
            columns = [c for c in self.dataset[split].column_names if c in LOADER_COLUMNS]
            self.dataset[split].set_format(type="torch", columns=columns + [TEXT_FIELD])

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True, collate_fn=self.tokenize_batch)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size, collate_fn=self.tokenize_batch)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, collate_fn=self.tokenize_batch)

    def tokenize_batch(self, batch):
        return tokenize(batch, self.tokenizer, self.max_seq_length, add_labels=True)


def tokenize(batch, tokenizer, max_seq_length, add_labels=False):
    texts = [b[TEXT_FIELD] for b in batch]

    features = tokenizer.batch_encode_plus(
        texts, max_length=max_seq_length, padding=True, truncation=True, return_tensors="pt"
    )
    if add_labels:
        features.data[LABEL_FIELD] = torch.tensor([b[LABEL_FIELD] for b in batch])

    return features


def calc_class_weights(labels):
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)

    return class_weights
