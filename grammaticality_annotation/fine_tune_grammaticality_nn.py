import os

import numpy as np

import evaluate
import pandas as pd
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.cli import LightningCLI
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from grammaticality_annotation.data import CHILDESGrammarDataModule, calc_class_weights, \
    FINE_TUNE_RANDOM_STATE
from grammaticality_annotation.tokenizer import LABEL_FIELD, TRANSCRIPT_FIELD, UTT_ID_FIELD
from grammaticality_annotation.pretrain_lstm import LSTMSequenceClassification
from utils import RESULTS_FILE, RESULTS_DIR

DEFAULT_LEARNING_RATE = 1e-5


class CHILDESGrammarModel(LightningModule):
    def __init__(
            self,
            model_name_or_path: str,
            num_cv_folds: int = 5,
            train_data_size: float = 1.0,
            learning_rate: float = DEFAULT_LEARNING_RATE,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            val_split_proportion: float = 0.5,
            random_seed=1,
            predict_data_dir=None,
            model_id=None,
            **kwargs,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.model_name_or_path = model_name_or_path

        self.save_hyperparameters()

        self.metric_mcc = evaluate.load("matthews_correlation", experiment_id=str(torch.rand(10)))
        self.metric_pearson_r = evaluate.load("pearsonr", experiment_id=str(torch.rand(10)))
        self.metric_acc = evaluate.load("accuracy", experiment_id=str(torch.rand(10)))
        self.metrics = [self.metric_mcc, self.metric_acc, self.metric_pearson_r]

        self.random_seed = random_seed

        self.predict_data_dir = predict_data_dir
        self.model_id = model_id

    def configure_model(self):
        self.context_length = self.trainer.datamodule.context_length
        self.num_labels = self.trainer.datamodule.num_labels

        self.class_weights = calc_class_weights(self.trainer.datamodule.dataset["train"][LABEL_FIELD].cpu().numpy())
        self.loss_fct = CrossEntropyLoss(weight=torch.tensor(self.class_weights, dtype=torch.float))

        print(f"Model loss class weights: {self.class_weights}")

        if os.path.isfile(self.model_name_or_path):
            self.model = LSTMSequenceClassification.load_from_checkpoint(self.model_name_or_path,
                                                                         num_labels=self.num_labels,
                                                                         strict=False)
        else:
            self.config = AutoConfig.from_pretrained(self.model_name_or_path, num_labels=self.num_labels)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path, config=self.config)

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

        results = {"loss": loss, "preds": preds, LABEL_FIELD: labels}
        self.train_outputs["preds"].extend(preds.detach().cpu().numpy())
        self.train_outputs[LABEL_FIELD].extend(labels.detach().cpu().numpy())

        return results

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.train_outputs = {"preds": [], LABEL_FIELD: []}

    def on_train_epoch_end(self):
        preds = self.val_outputs["preds"]
        labels = self.val_outputs[LABEL_FIELD]

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

        preds = torch.argmax(logits, dim=1)

        results = {"loss": val_loss, "preds": preds, LABEL_FIELD: labels}
        self.val_outputs["loss"].append(val_loss.detach().cpu().numpy())
        self.val_outputs["preds"].extend(preds.detach().cpu().numpy())
        self.val_outputs[LABEL_FIELD].extend(labels.detach().cpu().numpy())

        return results

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.val_outputs = {"loss": [], "preds": [], LABEL_FIELD: []}

    def on_validation_epoch_end(self):
        preds = self.val_outputs["preds"]
        labels = self.val_outputs[LABEL_FIELD]
        loss = np.mean(self.val_outputs["loss"])

        self.log(f"val_loss", loss, prog_bar=True)
        for metric in self.metrics:
            metric_results = metric.compute(predictions=preds, references=labels)
            metric_results = {"val_" + key: value if not np.isnan(value) else 0 for key, value in
                              metric_results.items()}

            self.log_dict(metric_results, prog_bar=True)

    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    def on_test_epoch_end(self):
        preds = self.val_outputs["preds"]
        labels = self.val_outputs[LABEL_FIELD]
        loss = np.mean(self.val_outputs["loss"])

        self.log(f"test_loss", loss, prog_bar=True)
        for metric in self.metrics:
            metric_results = metric.compute(predictions=preds, references=labels)
            metric_results = {"test_" + key: value if not np.isnan(value) else 0 for key, value in
                              metric_results.items()}

            self.log_dict(metric_results, prog_bar=True)

        data_test = self.dataset["test"].to_pandas()
        data_test["pred"] = preds
        output_path = os.path.join(self.logger.log_dir, "test_set_predictions.csv")
        data_test.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"] if "token_type_ids" in batch.keys() else None,
            attention_mask=batch["attention_mask"],
        )
        logits = output["logits"]

        preds = torch.argmax(logits, dim=1)

        # Transform to annotation scheme (2, 1, 0) to (1, 0, -1)
        preds = preds - 1

        # Store predictions
        for transcript_file in batch[TRANSCRIPT_FIELD].unique():
            path_name = os.path.join(self.predict_data_dir, f"{transcript_file}.csv")
            preds_transcript = preds[batch[TRANSCRIPT_FIELD] == transcript_file].cpu().numpy()
            utt_ids_transcript = batch[UTT_ID_FIELD][batch[TRANSCRIPT_FIELD] == transcript_file].cpu().numpy()
            data = pd.read_csv(path_name, index_col=0)
            data.loc[utt_ids_transcript, LABEL_FIELD] = preds_transcript
            data.to_csv(path_name, index_label=data.index.name)

        return preds


if __name__ == "__main__":
    logging_dir = os.path.expanduser("~/data/childes_grammaticality")
    os.makedirs(logging_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(monitor="val_pearsonr",
                                          mode="max",
                                          filename="{epoch:02d}-{val_pearsonr:.2f}")
    early_stop_callback = EarlyStopping(monitor="val_pearsonr", patience=20, verbose=True, mode="max",
                                        min_delta=0.01, stopping_threshold=0.99)

    LightningCLI(
        CHILDESGrammarModel,
        CHILDESGrammarDataModule,
        seed_everything_default=FINE_TUNE_RANDOM_STATE,
        save_config_kwargs={"overwrite": True},
        trainer_defaults={
            "callbacks": [checkpoint_callback, early_stop_callback],
            "default_root_dir": logging_dir,
            "max_epochs": 1000,
            "accelerator": "auto",
            "val_check_interval": 0.25,
            "devices": 1,
            "accumulate_grad_batches": 20,
            "precision": "16-mixed",
        },
    )
