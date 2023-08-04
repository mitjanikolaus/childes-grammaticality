import argparse
import os

import numpy as np
import pandas as pd

import evaluate
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup, PreTrainedTokenizerFast,
)

from grammaticality_annotation.data import CHILDESGrammarDataModule, calc_class_weights, load_annotated_childes_data
from grammaticality_annotation.tokenizer import TOKEN_PAD, TOKEN_EOS, TOKEN_UNK, TOKEN_SEP, LABEL_FIELD
from grammaticality_annotation.pretrain_lstm import LSTMSequenceClassification, LSTM_TOKENIZER_PATH
from utils import RESULTS_FILE, RESULTS_DIR

FINE_TUNE_RANDOM_STATE = 1

DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-5

MODELS = [
    "yevheniimaslov/deberta-v3-large-cola",
    "phueb/BabyBERTa-3",
    "cointegrated/roberta-large-cola-krishna2020",  # Inverted labels!!
    "bert-base-uncased",
    "textattack/bert-base-uncased-CoLA",
]


class CHILDESGrammarModel(LightningModule):
    def __init__(
            self,
            class_weights,
            model_name_or_path: str,
            num_labels: int,
            train_datasets: list,
            train_batch_size: int,
            eval_batch_size: int,
            learning_rate: float,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            val_split_proportion: float = 0.5,
            random_seed = 1,
            **kwargs,
    ):
        super().__init__()
        self.learning_rate = learning_rate

        print(f"Model loss class weights: {class_weights}")
        self.save_hyperparameters(ignore=["tokenizer"])

        if os.path.isfile(model_name_or_path):
            self.model = LSTMSequenceClassification.load_from_checkpoint(model_name_or_path, num_labels=num_labels)
        else:
            self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)

        self.metric_mcc = evaluate.load("matthews_correlation")
        self.metric_acc = evaluate.load("accuracy")
        self.metrics = [self.metric_mcc, self.metric_acc]

        weight = torch.tensor(class_weights)
        self.loss_fct = CrossEntropyLoss(weight=weight)

        self.random_seed = random_seed

        self.test_error_analysis = False

    def forward(self, **inputs):
        output = self.model(**inputs)
        if output["logits"].dtype == torch.float32:
            output["logits"] = output["logits"].to(torch.float64)
        return output

    def training_step(self, batch, batch_idx):
        output = self(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"] if "token_type_ids" in batch.keys() else None,
            attention_mask=batch["attention_mask"],
        )
        logits = output["logits"]
        labels = batch[LABEL_FIELD]
        loss = self.loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))

        preds = torch.argmax(logits, dim=1)

        return {"loss": loss, "preds": preds, LABEL_FIELD: labels}

    def training_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x[LABEL_FIELD] for x in outputs]).detach().cpu().numpy()

        acc = self.metric_acc.compute(predictions=preds, references=labels)
        acc = {"train_" + key: value for key, value in acc.items()}
        self.log_dict(acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        output = self(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"] if "token_type_ids" in batch.keys() else None,
            attention_mask=batch["attention_mask"],
        )
        logits = output["logits"]
        labels = batch[LABEL_FIELD]
        val_loss = self.loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))

        preds = torch.argmax(logits, axis=1)

        return {"loss": val_loss, "preds": preds, LABEL_FIELD: labels}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x[LABEL_FIELD] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        self.log(f"val_loss", loss, prog_bar=True)
        for metric in self.metrics:
            metric_results = metric.compute(predictions=preds, references=labels)
            metric_results = {"val_" + key: value for key, value in metric_results.items()}

            self.log_dict(metric_results, prog_bar=True)

    def test_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x[LABEL_FIELD] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        self.log(f"test_loss", loss, prog_bar=True)
        for metric in self.metrics:
            metric_results = metric.compute(predictions=preds, references=labels)
            metric_results = {"test_" + key: value for key, value in metric_results.items()}

            self.log_dict(metric_results, prog_bar=True)

        if self.test_error_analysis:
            _, data_manual_annotations_val = load_annotated_childes_data(self.hparams.context_length, self.hparams.val_split_proportion, random_seed=self.random_seed)
            data_manual_annotations_val["pred"] = preds
            errors = data_manual_annotations_val[
                data_manual_annotations_val.pred != data_manual_annotations_val.label]
            correct = data_manual_annotations_val[
                data_manual_annotations_val.pred == data_manual_annotations_val.label]

            errors.to_csv(os.path.join(self.logger.log_dir, "manual_annotations_errors.csv"))
            correct.to_csv(os.path.join(self.logger.log_dir, "manual_annotations_correct.csv"))

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        if isinstance(self.model, LSTMSequenceClassification):
            optimizer = Adam(self.parameters(), lr=self.learning_rate, eps=self.hparams.adam_epsilon)
            return [optimizer]
        else:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.hparams.adam_epsilon)

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

            return [optimizer], [scheduler]


def main(args):
    seed_everything(FINE_TUNE_RANDOM_STATE)

    test_results = []

    random_seeds = range(args.num_cv_folds)
    for random_seed in random_seeds:
        print(f"\n\n\nStart training with random seed {random_seed}")

        if os.path.isfile(args.model):
            if not os.path.isfile(LSTM_TOKENIZER_PATH):
                raise RuntimeError(f"Tokenizer not found at {LSTM_TOKENIZER_PATH}")

            tokenizer = PreTrainedTokenizerFast(tokenizer_file=LSTM_TOKENIZER_PATH)
            tokenizer.add_special_tokens(
                {'pad_token': TOKEN_PAD, 'eos_token': TOKEN_EOS, 'unk_token': TOKEN_UNK, 'sep_token': TOKEN_SEP})
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

        dm = CHILDESGrammarDataModule(val_split_proportion=args.val_split_proportion,
                                      model_name_or_path=args.model,
                                      eval_batch_size=args.batch_size,
                                      train_batch_size=args.batch_size,
                                      train_datasets=args.train_datasets,
                                      tokenizer=tokenizer,
                                      context_length=args.context_length,
                                      random_seed=random_seed)
        dm.setup("fit")
        class_weights = calc_class_weights(dm.dataset["train"][LABEL_FIELD].numpy())

        model = CHILDESGrammarModel(
            class_weights=class_weights,
            train_datasets=args.train_datasets,
            eval_batch_size=args.batch_size,
            train_batch_size=args.batch_size,
            model_name_or_path=args.model,
            num_labels=dm.num_labels,
            val_split_proportion=args.val_split_proportion,
            learning_rate=args.learning_rate,
            random_seed=random_seed,
        )

        if args.model == "gpt2":
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        checkpoint_callback = ModelCheckpoint(monitor="val_matthews_correlation", mode="max", save_last=True,
                                                filename="{epoch:02d}-{val_matthews_correlation:.2f}")
        early_stop_callback = EarlyStopping(monitor="val_matthews_correlation", patience=10, verbose=True, mode="max",
                                            min_delta=0.01, stopping_threshold=0.99)

        trainer = Trainer(
            max_epochs=args.max_epochs,
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            callbacks=[checkpoint_callback, early_stop_callback],
        )

        print("\n\n\nInitial validation:")
        trainer.validate(model, datamodule=dm)

        print("\n\n\nTraining:")
        trainer.fit(model, datamodule=dm)

        print(f"\n\n\nFinal validation (using {checkpoint_callback.best_model_path}):")
        best_model = CHILDESGrammarModel.load_from_checkpoint(checkpoint_callback.best_model_path)

        trainer.validate(best_model, datamodule=dm)

        model.test_error_analysis = True
        test_result = trainer.test(best_model, datamodule=dm)
        test_results.append(test_result[0])

    accuracies = [results["test_accuracy"] for results in test_results]
    print(f"\n\n\nAccuracy: {np.mean(accuracies):.2f} Stddev: {np.std(accuracies):.2f}")

    mccs = [results["test_matthews_correlation"] for results in test_results]
    print(f"MCC: {np.mean(mccs):.2f} Stddev: {np.std(mccs):.2f}")

    results_df = pd.DataFrame([{"model": args.model, "mcc: mean": np.mean(mccs), "mcc: std": np.std(mccs), "accuracy: mean": np.mean(accuracies), "accuracy: std": np.std(accuracies)}])
    results_df.set_index(["model"], inplace=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.isfile(RESULTS_FILE):
        results_df.to_csv(RESULTS_FILE)
    else:
        old_res_file = pd.read_csv(RESULTS_FILE, index_col=0)
        results_df = results_df.combine_first(old_res_file)
        results_df.to_csv(RESULTS_FILE)


def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    argparser.add_argument(
        "--train-datasets",
        type=str,
        nargs="+",
        default=["manual_annotations"],
    )
    argparser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    argparser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    argparser.add_argument(
        "--val-split-proportion",
        type=float,
        default=0.2,
        help="Val split proportion (only for manually annotated data)"
    )
    argparser.add_argument(
        "--context-length",
        type=int,
        default=0,
        help="Number of preceding utterances to include as conversational context"
    )
    argparser.add_argument(
        "--num-cv-folds",
        type=int,
        default=3,
        help="Number of cross-validation folds"
    )
    argparser = Trainer.add_argparse_args(argparser)

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    main(args)
