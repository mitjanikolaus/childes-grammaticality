import argparse
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from grammaticality_annotation.tokenizer import LABEL_FIELD, ERROR_LABELS_FIELD

import matplotlib.pyplot as plt

PREDICTION_FIELD = "pred"


def main(args):
    data = pd.read_csv(args.predictions_file,  index_col=0)
    acc = (data[PREDICTION_FIELD] == data[LABEL_FIELD]).mean()
    mcc = matthews_corrcoef(data[PREDICTION_FIELD], data[LABEL_FIELD])
    print(f"Acc: {acc:.2f} | MCC: {mcc:.2f}")

    data_ungrammatical = data[data[LABEL_FIELD] == False]

    # TODO:
    data_ungrammatical.dropna(subset=["labels"], inplace=True)

    data_ungrammatical[ERROR_LABELS_FIELD] = data_ungrammatical[ERROR_LABELS_FIELD].apply(lambda x: x.split(", "))
    data_ungrammatical = data_ungrammatical.explode(ERROR_LABELS_FIELD)

    data_ungrammatical[ERROR_LABELS_FIELD].hist()

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
