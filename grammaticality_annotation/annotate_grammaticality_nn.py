import argparse
import glob
import os
import torch
import yaml
from datasets import Dataset, DatasetDict
from pytorch_lightning import Trainer
from tqdm import tqdm
from transformers import AutoTokenizer
import pandas as pd

from grammaticality_annotation.data import CHILDESGrammarDataModule, add_context
from grammaticality_annotation.fine_tune_grammaticality_nn import CHILDESGrammarModel
from load_childes_db_data import DATA_DIR_PREPROCESSED_CHILDES_DB
from utils import PROJECT_ROOT_DIR

ANNOTATION_ANNOTATED_FILES_PATH = PROJECT_ROOT_DIR+"/data/manual_annotation/automatically_annotated"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Needs to match the number of utterances within a file to be annotated!
BATCH_SIZE = 200 #TODO

DATA_DIR_ANNOTATED = os.path.join(PROJECT_ROOT_DIR, "data", "automatically_annotated", "childes_db")


def annotate(args):
    hparams = yaml.safe_load(open(os.path.join(args.model, "hparams.yaml")))
    tokenizer = AutoTokenizer.from_pretrained(hparams["model_name_or_path"], use_fast=True)

    context_length = hparams["context_length"]
    sep_token = tokenizer.sep_token

    print('loading data...')
    transcript_files = {p: pd.read_csv(p) for p in tqdm(sorted(glob.glob(os.path.join(args.data_path, "*.csv"))))}
    data = pd.concat(transcript_files.values(), ignore_index=True)

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

    checkpoints = list(glob.glob(args.model+"/checkpoints/epoch*.ckpt"))
    assert len(checkpoints) == 1, "No or multiple checkpoints found."
    checkpoint = checkpoints[0]
    print(f"Model checkpoint: {checkpoint}")

    # copy the raw data into the output dir, the prediction loop will update the labels directly in these files
    os.makedirs(args.out_data_dir, exist_ok=True)
    for path, file in transcript_files.items():
        file_name = os.path.basename(path)
        out_path = os.path.join(args.out_data_dir, file_name)
        file.to_csv(out_path, index=False)

    model_id = int(args.model.split("_")[-1])
    model = CHILDESGrammarModel.load_from_checkpoint(checkpoint, predict_data_dir=args.out_data_dir, model_id=model_id)
    model.eval()

    trainer = Trainer(devices=1 if torch.cuda.is_available() else None, accelerator="auto")
    trainer.predict(model, datamodule=dm)


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data-path",
        type=str,
        default=DATA_DIR_PREPROCESSED_CHILDES_DB,
    )
    argparser.add_argument(
        "--out-data-dir",
        type=str,
        default=DATA_DIR_ANNOTATED,
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
