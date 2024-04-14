import argparse
import glob
import os
import numpy as np
import torch
import yaml
from datasets import Dataset, DatasetDict
from pytorch_lightning import Trainer
from transformers import AutoTokenizer
import pandas as pd

from grammaticality_annotation.data import CHILDESGrammarDataModule, add_context
from grammaticality_annotation.fine_tune_grammaticality_nn import CHILDESGrammarModel
from grammaticality_annotation.tokenizer import LABEL_FIELD
from load_childes_db_data import DATA_FILE_PREPROCESSED_CHILDES_DB
from utils import PROJECT_ROOT_DIR

ANNOTATION_ANNOTATED_FILES_PATH = PROJECT_ROOT_DIR+"/data/manual_annotation/automatically_annotated"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Needs to match the number of utterances within a file to be annotated!
BATCH_SIZE = 200 #TODO


def annotate(args):
    hparams = yaml.safe_load(open(os.path.join(args.model, "hparams.yaml")))
    tokenizer = AutoTokenizer.from_pretrained(hparams["model_name_or_path"], use_fast=True)

    context_length = hparams["context_length"]
    sep_token = tokenizer.sep_token
    data = pd.read_csv(args.data_path)

    data = add_context(data, context_length=context_length, sep_token=sep_token)

    dataset = Dataset.from_pandas(data, preserve_index=False)
    dataset_dict = DatasetDict()
    dataset_dict["pred"] = dataset
    dm = CHILDESGrammarDataModule(val_split_proportion=0,
                                  num_cv_folds=0,
                                  model_name_or_path=args.model,
                                  eval_batch_size=BATCH_SIZE,
                                  train_batch_size=BATCH_SIZE,
                                  tokenizer=tokenizer,
                                  context_length=context_length,
                                  num_workers=args.num_workers,
                                  add_eos_tokens=False,
                                  train_data_size=1,
                                  ds_dict=dataset_dict)

    checkpoints = glob.glob(args.model+"/checkpoints/epoch*.ckpt")
    print(f"Model checkpoints: {checkpoints}")

    for i, checkpoint in enumerate(checkpoints):
        print(f"\n\nAnnotating with model checkpoint #{i}")
        model = CHILDESGrammarModel.load_from_checkpoint(checkpoint, predict_data_dir=args.data_dir, model_id=i)
        model.eval()

        trainer = Trainer(devices=1 if torch.cuda.is_available() else None, accelerator="auto")
        predictions = trainer.predict(model, datamodule=dm)
        torch.cat(predictions)

    # Majority voting
    data_annotated = pd.read_csv(args.data_path) #TODO ??

    def majority_vote(row):
        if row[LABEL_FIELD] == "TODO":
            votes = [row[f"is_grammatical_{i}"] for i in range(len(checkpoints))]
            return np.median(votes)
        else:
            return ""

    data_annotated[LABEL_FIELD] = data_annotated.apply(majority_vote, axis=1)

    # Append training data
    data_train = load_childes_data(DATA_PATH_CHILDES_ANNOTATED)
    data_all = pd.concat([data_train, data_annotated])

    data_all.to_csv(os.path.join(args.data_dir, "majority_vote.csv"))


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data-path",
        type=str,
        default=DATA_FILE_PREPROCESSED_CHILDES_DB,
    )
    argparser.add_argument(
        "--model",
        type=str,
        help="path to model checkpoint"
    )
    argparser.add_argument(
        "--num-workers",
        type=int,
        default=8,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    annotate(args)
