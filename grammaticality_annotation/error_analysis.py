import argparse
import os

import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix

from grammaticality_annotation.data import LABEL_UNGRAMMATICAL
from grammaticality_annotation.tokenizer import LABEL_FIELD, ERROR_LABELS_FIELD

import matplotlib.pyplot as plt

from utils import RESULTS_DIR

PREDICTION_FIELD = "pred"

BASE_COLOR = "#617fab"


def create_barplot(data_ungrammatical):
    counts = data_ungrammatical[ERROR_LABELS_FIELD].value_counts()

    plt.figure(figsize=(4, 5))
    axis = counts.plot(kind="barh", color="#b5c9e6")

    correctly_predicted = data_ungrammatical[data_ungrammatical[PREDICTION_FIELD] == LABEL_UNGRAMMATICAL][
        ERROR_LABELS_FIELD].value_counts()
    axis2 = correctly_predicted.plot(kind="barh", ax=axis, color="#617fab")

    handles, _ = axis2.get_legend_handles_labels()
    axis2.legend(handles, ["Annotated", "Correctly predicted"])
    plt.xlabel("Number of errors")
    plt.subplots_adjust(left=0.24)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "error_analysis.png"), dpi=300)
    plt.show()


def create_proportions_plot(data_ungrammatical):
    counts = data_ungrammatical[ERROR_LABELS_FIELD].value_counts()
    correctly_predicted_proportions = data_ungrammatical[data_ungrammatical[PREDICTION_FIELD] == LABEL_UNGRAMMATICAL][
        ERROR_LABELS_FIELD].value_counts() / counts
    plt.figure(figsize=(4, 5))
    sns.pointplot(x=PREDICTION_FIELD, y=ERROR_LABELS_FIELD, data=data_ungrammatical, color=BASE_COLOR, linestyles="",
                  errwidth=1, markers='o', estimator=lambda x: sum(x == LABEL_UNGRAMMATICAL) * 100.0 / len(x),
                  order=correctly_predicted_proportions.sort_values().index
                  )
    plt.axvline(x=(data_ungrammatical[LABEL_FIELD] == data_ungrammatical[PREDICTION_FIELD]).mean() * 100, linestyle="--")

    plt.ylabel("")
    plt.xlabel("Recall")
    plt.xlim((0, 120))
    plt.xticks([0, 25, 50, 75, 100])
    plt.subplots_adjust(left=0.24)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "error_analysis.png"), dpi=300)
    plt.show()


LABEL_NAMES = {l: name for l, name in enumerate(["Ungramm.", "Ambig.", "Gramm."])}


def main(args):
    data = pd.read_csv(args.predictions_file,  index_col=0)
    acc = (data[PREDICTION_FIELD] == data[LABEL_FIELD]).mean()
    pcc = pearsonr(data[PREDICTION_FIELD], data[LABEL_FIELD])[0]
    print(f"Acc: {acc:.2f} | PCC: {pcc:.2f}")

    cm = confusion_matrix(data[LABEL_FIELD], data[PREDICTION_FIELD], normalize='true')
    print(pd.DataFrame(cm).rename(index=LABEL_NAMES, columns=LABEL_NAMES).style.format(precision=2).to_latex(hrules=True))

    data_ungrammatical = data[data[LABEL_FIELD] == LABEL_UNGRAMMATICAL].copy()

    data_ungrammatical[ERROR_LABELS_FIELD] = data_ungrammatical[ERROR_LABELS_FIELD].apply(lambda x: x.split(", "))
    data_ungrammatical = data_ungrammatical.explode(ERROR_LABELS_FIELD)

    create_proportions_plot(data_ungrammatical)


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--predictions-file",
        type=str,
        required=True,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    main(args)
