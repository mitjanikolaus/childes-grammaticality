# CHILDES Grammaticality Annotations

Automatic annotation of grammaticality for child-caregiver conversations.

## Python environment

A python environment can be created using the [environment.yml](environments/environment.yml) file (for
GPU: [environment_gpu.yml](environments/environment_gpu.yml)):

```
conda env create --file environments/environment.yml
```

To install the current repo:
```
pip install .
```

## Load data from CHILDES-DB

```
python load_childes_db_data.py
```

## Train models for annotation

### Baselines

Example for bigram-based model:
```
python grammaticality_annotation/train_grammaticality_baseline.py --model svc --max-n-gram-level 2
```

### Transformer-based models

These models are only fine-tuned on the task. Example for DeBERTa:
```
python grammaticality_annotation/fine_tune_grammaticality_nn.py fit --model.model_name_or_path microsoft/deberta-v3-large --data.fold 0
```

If you are using a small GPU you will most likely need to decrease the batch size for finetuning. The following command
can be used to train models which reach an average Pearson Correlation Coefficient (PCC) of 0.75 on the test sets, even
better than what the score reported in the paper (thanks to improved hyperparameters).
```
python grammaticality_annotation/fine_tune_grammaticality_nn.py fit --model.model_name_or_path microsoft/deberta-v3-large  --data.fold 0 --data.train_batch_size 5 --data.eval_batch_size 5 --trainer.accumulate_grad_batches 20
```

For the results in the paper, 5-fold cross-validation was performed. In order to train the models for the different
folds, set the `--fold` argument to different indices (0 to 4, default: 0).

In order to test a model you can run the following command: (make sure to specify the correct data fold!)
```
python grammaticality_annotation/fine_tune_grammaticality_nn.py test --ckpt_path ~/data/childes_grammaticality/lightning_logs/version_0/checkpoints/epoch\=11-val_pearsonr\=0.77.ckpt  --data.fold 0
```

## Annotate data

```
python grammaticality_annotation/annotate_grammaticality_nn.py --model ~/data/childes_grammaticality/lightning_logs/version_123 --data-dir data/manual_annotation/all
```

The data will be annotated with the following coding scheme:

| ungrammatical | ambiguous | grammatical |
|:-------------:|:---------:|:-----------:|
|      -1       |     0     |      1      |
