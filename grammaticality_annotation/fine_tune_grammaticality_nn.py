import os

import numpy as np

import evaluate
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

from grammaticality_annotation.data import CHILDESGrammarDataModule, calc_class_weights, FINE_TUNE_RANDOM_STATE
from grammaticality_annotation.tokenizer import LABEL_FIELD, TRANSCRIPT_FIELD
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

        self.test_error_analysis = False

    def configure_model(self):
        self.context_length = self.trainer.datamodule.context_length
        self.num_labels = self.trainer.datamodule.num_labels

        self.class_weights = calc_class_weights(self.trainer.datamodule.dataset["train"][LABEL_FIELD].numpy())
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

    def test_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x[LABEL_FIELD] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        self.log(f"test_loss", loss, prog_bar=True)
        for metric in self.metrics:
            metric_results = metric.compute(predictions=preds, references=labels)
            metric_results = {"test_" + key: value if not np.isnan(value) else 0 for key, value in
                              metric_results.items()}

            self.log_dict(metric_results, prog_bar=True)

        if self.test_error_analysis:
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

        preds = preds - 1

        # Store predictions
        # TODO file names? batch[FILE_ID_FIELD] transcript id?
        batch[TRANSCRIPT_FIELD]
        path_name = os.path.join(self.predict_data_dir, f"annotated_{batch_idx}.csv")
        data_raw = load_childes_data_file(path_name)
        data_raw.loc[data_raw[LABEL_FIELD] == "TODO", f"is_grammatical_{self.model_id}"] = preds.tolist()

        data_raw.to_csv(path_name)

        return preds


def gather_results(args):
    test_results = []
    val_results = []

    # for res in results:
    #     run_id = trainer.logger.version
    #
    #     print(f"\n\nFinal validation (using {checkpoint_callback.best_model_path}):")
    #     best_model = CHILDESGrammarModel.load_from_checkpoint(checkpoint_callback.best_model_path,
    #                                                           context_length=args.context_length,
    #                                                           val_split_proportion=args.val_split_proportion,
    #                                                           dataset=datasets[fold],
    #                                                           class_weights=class_weights)
    #
    #     if args.model == "gpt2":
    #         tokenizer.pad_token = tokenizer.eos_token
    #         best_model.config.pad_token_id = model.config.eos_token_id
    #
    #     val_result = trainer.validate(best_model, datamodule=dm)
    #     val_results.append(val_result[0])
    #
    #     best_model.test_error_analysis = True
    #     test_result = trainer.test(best_model, datamodule=dm)
    #     test_results.append(test_result[0])
    #
    # accuracies = [results["test_accuracy"] for results in test_results]
    # print(f"\n\n\nAccuracy: {np.mean(accuracies):.2f} Stddev: {np.std(accuracies):.2f}")
    #
    # mccs = [results["test_matthews_correlation"] for results in test_results]
    # print(f"MCC: {np.mean(mccs):.2f} Stddev: {np.std(mccs):.2f}")
    #
    # pearson_r_scores = [results["test_pearsonr"] for results in test_results]
    # print(f"Pearson r: {np.mean(pearson_r_scores):.2f} Stddev: {np.std(pearson_r_scores):.2f}")
    #
    # val_mccs = [results["val_matthews_correlation"] for results in val_results]
    #
    # val_pearsonr_scores = [results["val_pearsonr"] for results in val_results]
    #
    # results_df = pd.DataFrame([{"model": args.model, "mcc: mean": np.mean(mccs), "mcc: std": np.std(mccs),
    #                             "pearson_r: mean": np.mean(pearson_r_scores),
    #                             "pearson_r: std": np.std(pearson_r_scores), "accuracy: mean": np.mean(accuracies),
    #                             "accuracy: std": np.std(accuracies), "val_mcc: mean": np.mean(val_mccs),
    #                             "val_mcc: std": np.std(val_mccs), "val_pearsonr: mean": np.mean(val_pearsonr_scores),
    #                             "val_pearsonr: std": np.std(val_pearsonr_scores), "context_length": args.context_length,
    #                             "train_data_size": args.train_data_size,
    #                             "run_id": run_id}])
    # results_df.set_index(["model", "context_length", "train_data_size"], inplace=True)
    #
    # os.makedirs(RESULTS_DIR, exist_ok=True)
    # if not os.path.isfile(RESULTS_FILE):
    #     results_df.to_csv(RESULTS_FILE)
    # else:
    #     old_res_file = pd.read_csv(RESULTS_FILE, index_col=["model", "context_length", "train_data_size"])
    #     results_df = results_df.combine_first(old_res_file)
    #     results_df.to_csv(RESULTS_FILE)


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
