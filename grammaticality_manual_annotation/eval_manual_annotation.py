import argparse
import itertools
import os
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from krippendorff import krippendorff
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

from utils import PROJECT_ROOT_DIR

BASE_PATH = PROJECT_ROOT_DIR + "/data/manual_annotation/selection/"


def eval(args):
    for i in range(args.start_index, args.end_index+1):
        print(f"AGREEMENT SCORES FILE ID {i}:")
        base_file = os.path.join(BASE_PATH, f"{i}.csv")
        annotated_files = {ann: os.path.join(BASE_PATH, f"{i}_{ann}.csv") for ann in args.annotators}
        annotated_files = {a: f for a, f in annotated_files.items() if os.path.isfile(f)}

        data = pd.read_csv(base_file, index_col=0)
        for ann, file in annotated_files.items():
            data_ann = pd.read_csv(file, index_col=0)
            if len(data_ann.dropna(subset=["is_grammatical"])) != len(data.dropna(subset=["is_grammatical"])):
                missing = data_ann.is_grammatical.isna() != data.is_grammatical.isna()
                missing = [i for i, v in missing.to_dict().items() if v]
                raise RuntimeError(f"Missing annotations: Annotator {ann}: Lines {missing}")

            column_name = f"is_grammatical_{ann}"
            data[column_name] = data_ann["is_grammatical"].values

        def is_disagreement(row):
            if row.is_grammatical != "TODO":
                return 0
            for ann in args.annotators[1:]:
                if row[f"is_grammatical_{ann}"] != row[f"is_grammatical_{args.annotators[0]}"]:
                    return 1
            return 0

        data["disagreement"] = data.apply(is_disagreement, axis=1)
        data["disagreement"] = data.disagreement.replace({1: 1, 0: ""})
        names = "_".join(args.annotators)
        data.to_csv(os.path.join(BASE_PATH, f"{i}_agreement_{names}.csv"))

        data.dropna(subset=["is_grammatical"], inplace=True)

        kappa_scores = []
        mcc_scores = []
        for ann_1, ann_2 in itertools.combinations(annotated_files.keys(), 2):
            kappa = cohen_kappa_score(data[f"is_grammatical_{ann_1}"], data[f"is_grammatical_{ann_2}"], weights="linear")
            kappa_scores.append(kappa)
            print(f"Kappa {(ann_1, ann_2)}: {kappa:.2f}")

            mcc = matthews_corrcoef(data[f"is_grammatical_{ann_1}"], data[f"is_grammatical_{ann_2}"])
            mcc_scores.append(mcc)
        print(f"Kappa: {np.mean(kappa_scores):.2f}")

        print(f"MCC: {np.mean(mcc_scores):.2f}")

        rel_data = [data[f"is_grammatical_{ann}"] for ann in args.annotators]
        alpha = krippendorff.alpha(reliability_data=rel_data, level_of_measurement="ordinal")
        print(f"Alpha: {alpha:.2f}")


def eval_disagreement():
    data = pd.read_csv(os.path.join(BASE_PATH, "disagreement_annotated.csv"))
    data[data.disagreement == 1].groupby("disagreement_reason").size().sort_values(ascending=False).plot(kind="barh")
    plt.subplots_adjust(left=0.5)
    plt.show()


def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--start-index",
        type=int,
        default=0,
    )
    argparser.add_argument(
        "--end-index",
        type=int,
        required=True,
    )
    argparser.add_argument(
        "--annotators",
        type=str,
        nargs="+",
        default=[],
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    eval(args)
    # eval_disagreement()
