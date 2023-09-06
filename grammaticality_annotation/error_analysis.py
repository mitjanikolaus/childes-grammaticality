import argparse
import os

import pandas as pd
from sklearn.metrics import matthews_corrcoef

from grammaticality_annotation.data import LABEL_UNGRAMMATICAL
from grammaticality_annotation.tokenizer import LABEL_FIELD, ERROR_LABELS_FIELD

import matplotlib.pyplot as plt

from utils import RESULTS_DIR

PREDICTION_FIELD = "pred"


def main(args):
    data = pd.read_csv(args.predictions_file,  index_col=0)
    acc = (data[PREDICTION_FIELD] == data[LABEL_FIELD]).mean()
    mcc = matthews_corrcoef(data[PREDICTION_FIELD], data[LABEL_FIELD])
    print(f"Acc: {acc:.2f} | MCC: {mcc:.2f}")

    data_ungrammatical = data[data[LABEL_FIELD] == False].copy()

    data_ungrammatical[ERROR_LABELS_FIELD] = data_ungrammatical[ERROR_LABELS_FIELD].apply(lambda x: x.split(", "))
    data_ungrammatical = data_ungrammatical.explode(ERROR_LABELS_FIELD)

    counts = data_ungrammatical[ERROR_LABELS_FIELD].value_counts()

    plt.figure(figsize=(4, 5))
    axis = counts.plot(kind="barh", color="#b5c9e6")

    correctly_predicted = data_ungrammatical[data_ungrammatical[PREDICTION_FIELD] == LABEL_UNGRAMMATICAL][ERROR_LABELS_FIELD].value_counts()
    axis2 = correctly_predicted.plot(kind="barh", ax=axis, color="#617fab")

    handles, _ = axis2.get_legend_handles_labels()
    axis2.legend(handles, ["# Errors annotated", "# Errors predicted"])

    plt.subplots_adjust(left=0.24)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "error_analysis.png"), dpi=300)
    plt.show()


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
