import argparse
import itertools
import os
from collections import Counter
import numpy as np

import nltk
import pandas as pd
from pytorch_lightning import Trainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, cohen_kappa_score, matthews_corrcoef
from transformers import (
    PreTrainedTokenizerFast,
)
from sklearn.svm import SVC, LinearSVC

from grammaticality_annotation.data import create_dataset_dicts
from grammaticality_annotation.tokenizer import TOKEN_PAD, TOKEN_EOS, TOKEN_UNK, TOKEN_SEP, \
    train_tokenizer, TOKENIZERS_DIR, TEXT_FIELD
from utils import RESULTS_DIR, RESULTS_FILE

RANDOM_STATE = 1


def create_features(datapoint, vocabs):
    vocab_unigrams = vocabs[0]
    unigrams = Counter([u for u in datapoint["encoded"] if u in vocab_unigrams])
    feat_unigrams = [unigrams[u] for u in vocab_unigrams]
    datapoint["features"] = feat_unigrams

    for level, vocab in zip(range(2, len(vocabs)+1), vocabs[1:]):
        ngrams = Counter(nltk.ngrams(unigrams, level))
        feats = [ngrams[b] for b in vocab]
        datapoint["features"] += feats

    return datapoint


def create_n_gram_vocabs(datasets, max_n_grams, max_n_gram_level):
    unigrams = itertools.chain(*datasets)
    unigrams = [u for u, c in Counter(unigrams).most_common(max_n_grams)]

    vocabs = [unigrams]
    for level in range(2, max_n_gram_level+1):
        ngrams = nltk.ngrams(itertools.chain(*datasets), level)
        ngrams = [b for b, c in Counter(ngrams).most_common(max_n_grams)]
        vocabs.append(ngrams)

    return vocabs


def tokenize(datapoint, tokenizer):
    encoded = tokenizer.encode(datapoint[TEXT_FIELD])
    datapoint["encoded"] = encoded
    return datapoint


def main(args):
    test_labels = np.array([])
    predictions = np.array([])
    accuracies = []
    mccs = []

    maj_class_accuracies = []
    maj_class_mccs = []

    dataset_dicts = create_dataset_dicts(args.num_cv_folds, args.val_split_proportion, args.context_length)

    for fold, dataset in enumerate(dataset_dicts):
        tokenizer_path = os.path.join(TOKENIZERS_DIR, f"tokenizer_{fold}.json")
        if not os.path.isfile(tokenizer_path):
            train_tokenizer(tokenizer_path, dataset["train"])

        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        tokenizer.add_special_tokens(
            {'pad_token': TOKEN_PAD, 'eos_token': TOKEN_EOS, 'unk_token': TOKEN_UNK, 'sep_token': TOKEN_SEP})

        dataset = dataset.map(tokenize, fn_kwargs={"tokenizer": tokenizer})
        vocabs = create_n_gram_vocabs(dataset["train"]["encoded"], args.max_n_grams, args.max_n_gram_level)
        dataset = dataset.map(create_features, fn_kwargs={"vocabs": vocabs})

        data_train = dataset["train"]
        data_test = dataset["test"]

        print("Train dataset size: ", len(data_train))
        print("Test dataset size: ", len(data_test))
        counter = Counter(data_train["is_grammatical"])
        print("Label counts: ", counter)
        most_common_label = counter.most_common()[0][0]

        labels = np.array(data_test["is_grammatical"])
        maj_class_acc = np.mean(labels == most_common_label)
        maj_class_accuracies.append(maj_class_acc)

        maj_class_mcc = matthews_corrcoef(labels, np.repeat(most_common_label, len(labels)))
        maj_class_mccs.append(maj_class_mcc)

        if args.model == "svc":
            clf = SVC(random_state=RANDOM_STATE, class_weight="balanced")
        elif args.model == "linear_svc":
            clf = LinearSVC(random_state=RANDOM_STATE, class_weight="balanced")
        elif args.model == "random_forest":
            clf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
        else:
            raise RuntimeError("Unknown model: ", args.model)

        print("Training model.. ", end="")
        clf.fit(data_train["features"], data_train["is_grammatical"])
        print("Done.\n")

        preds = clf.predict(data_test["features"])
        predictions = np.concatenate([predictions, preds])

        test_labels = np.concatenate([test_labels, labels])

        accuracy = np.mean(labels == preds)
        accuracies.append(accuracy)
        print("Accuracy: ", accuracy)

        mcc = matthews_corrcoef(labels, preds)
        mccs.append(mcc)
        print("MCC: ", mcc)

    print(f"==================================\n"
          f"Majority Classifier Accuracy: {np.mean(maj_class_accuracies):.2f} Stddev: {np.std(maj_class_accuracies):.2f}")

    print(f"Classifier Accuracy: {np.mean(accuracies):.2f} Stddev: {np.std(accuracies):.2f}")

    print(f"Classifier MCC: {np.mean(mccs):.2f} Stddev: {np.std(mccs):.2f}")

    cm = confusion_matrix(test_labels, predictions, normalize="true")
    print("Confusion matrix: \n", cm)

    kappa = cohen_kappa_score(test_labels, predictions, weights="linear")
    print(f"Cohen's kappa: {kappa:.2f}")

    model_name = f"{args.max_n_gram_level}-gram"
    results_df = pd.DataFrame([{"model": "majority_classifier", "mcc: mean": np.mean(maj_class_mccs), "mcc: std": np.std(maj_class_mccs), "accuracy: mean": np.mean(maj_class_accuracies), "accuracy: std": np.std(maj_class_accuracies), "context_length": 0, "val_mcc: mean": 0, "val_mcc: std": 0, "train_data_size": 1.0},
                               {"model": model_name, "mcc: mean": np.mean(mccs), "mcc: std": np.std(mccs), "accuracy: mean": np.mean(accuracies), "accuracy: std": np.std(accuracies), "context_length": args.context_length, "val_mcc: mean": 0, "val_mcc: std": 0, "train_data_size": 1.0}])
    results_df.set_index(["model", "context_length", "train_data_size"], inplace=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.isfile(RESULTS_FILE):
        results_df.to_csv(RESULTS_FILE)
    else:
        old_res_file = pd.read_csv(RESULTS_FILE, index_col=["model", "context_length", "train_data_size"])
        results_df = results_df.combine_first(old_res_file)
        results_df.to_csv(RESULTS_FILE)


def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--val-split-proportion",
        type=float,
        default=0.2,
        help="Val split proportion"
    )
    argparser.add_argument(
        "--model",
        type=str,
        default="svc",
        choices=["random_forest", "svc", "linear_svc"]
    )
    argparser.add_argument(
        "--max-n-gram-level",
        type=int,
        default=3,
    )
    argparser.add_argument(
        "--max-n-grams",
        type=int,
        default=1000,
        help="Maximum number of ngrams in the vocabulary"
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
        default=5,
        help="Number of cross-validation folds"
    )
    argparser = Trainer.add_argparse_args(argparser)

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    main(args)
