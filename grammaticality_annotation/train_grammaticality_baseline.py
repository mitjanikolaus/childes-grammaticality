import argparse
import itertools
import os
from collections import Counter
import numpy as np

import nltk
from pytorch_lightning import Trainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from transformers import (
    PreTrainedTokenizerFast,
)
from sklearn.svm import SVC, LinearSVC

from grammaticality_annotation.data import create_dataset_dict
from grammaticality_annotation.tokenizer import TOKEN_PAD, TOKEN_EOS, TOKEN_UNK, TOKEN_SEP, tokenize, \
    train_tokenizer, TOKENIZERS_DIR

RANDOM_STATE = 1


def create_features(datapoint, vocab_unigrams, vocab_bigrams, vocab_trigrams):
    unigrams = Counter([u for u in datapoint["encoded"] if u in vocab_unigrams])
    bigrams = Counter(nltk.ngrams(unigrams, 2))
    trigrams = Counter(nltk.ngrams(unigrams, 3))

    feat_unigrams = [unigrams[u] for u in vocab_unigrams]
    feat_bigrams = [bigrams[b] for b in vocab_bigrams]
    feat_trigrams = [trigrams[t] for t in vocab_trigrams]
    datapoint["features"] = feat_unigrams + feat_bigrams + feat_trigrams

    return datapoint


def create_n_gram_vocabs(datasets, max_n_grams):
    unigrams = itertools.chain(*datasets)
    bigrams = nltk.ngrams(itertools.chain(*datasets), 2)
    trigrams = nltk.ngrams(itertools.chain(*datasets), 3)

    unigrams = [u for u, c in Counter(unigrams).most_common(max_n_grams)]
    bigrams = [b for b, c in Counter(bigrams).most_common(max_n_grams)]
    trigrams = [t for t, c in Counter(trigrams).most_common(max_n_grams)]

    return unigrams, bigrams, trigrams


def main(args):
    test_labels = np.array([])
    predictions = np.array([])
    accuracies = []
    maj_class_accuracies = []

    random_seeds = range(args.num_cv_folds)
    for random_seed in random_seeds:

        datasets = create_dataset_dict(args.train_datasets, args.test_split_proportion, args.context_length, random_seed)

        tokenizer_path = os.path.join(TOKENIZERS_DIR, f"tokenizer_{random_seed}.json")
        if not os.path.isfile(tokenizer_path):
            train_tokenizer(tokenizer_path, datasets["train"])

        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        tokenizer.add_special_tokens(
            {'pad_token': TOKEN_PAD, 'eos_token': TOKEN_EOS, 'unk_token': TOKEN_UNK, 'sep_token': TOKEN_SEP})

        datasets = datasets.map(tokenize, fn_kwargs={"tokenizer": tokenizer})
        vocab_unigrams, vocab_bigrams, vocab_trigrams = create_n_gram_vocabs(datasets["train"]["encoded"], args.max_n_grams)
        datasets = datasets.map(create_features, fn_kwargs={"vocab_unigrams": vocab_unigrams, "vocab_bigrams": vocab_bigrams, "vocab_trigrams": vocab_trigrams})

        data_train = datasets["train"]
        data_test = datasets["test"]

        print("Train dataset size: ", len(data_train))
        print("Test dataset size: ", len(data_test))
        counter = Counter(data_train["is_grammatical"])
        print("Label counts: ", counter)
        most_common_label = counter.most_common()[0][0]

        maj_class_acc = np.mean(np.array(data_test["is_grammatical"]) == most_common_label)
        maj_class_accuracies.append(maj_class_acc)

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

        labels = np.array(datasets["test"]["is_grammatical"])
        test_labels = np.concatenate([test_labels, labels])

        accuracy = np.mean(labels == preds)
        accuracies.append(accuracy)
        print("Accuracy: ", accuracy)

    print(f"==================================\n"
          f"Majority Classifier Accuracy: {np.mean(maj_class_accuracies):.2f} Stddev: {np.std(maj_class_accuracies):.2f}")

    print(f"Classifier Accuracy: {np.mean(accuracies):.2f} Stddev: {np.std(accuracies):.2f}")

    cm = confusion_matrix(test_labels, predictions, normalize="true")
    print("Confusion matrix: \n", cm)

    kappa = cohen_kappa_score(test_labels, predictions, weights="linear")
    print(f"Cohen's kappa: {kappa:.2f}")


def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--train-datasets",
        type=str,
        nargs="+",
        default=["manual_annotations"],
    )
    argparser.add_argument(
        "--test-split-proportion",
        type=float,
        default=0.2,
        help="Test split proportion"
    )
    argparser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["random_forest", "svc", "linear_svc"]
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
        default=3,
        help="Number of cross-validation folds"
    )
    argparser = Trainer.add_argparse_args(argparser)

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    main(args)
