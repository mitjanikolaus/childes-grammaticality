import argparse
import itertools
import os
from collections import Counter

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from krippendorff import krippendorff
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

from utils import PROJECT_ROOT_DIR, RESULTS_DIR, RESULTS_FILE

import warnings
# Ignore pandas CSV writing warnings
warnings.filterwarnings("ignore")

BASE_PATH = PROJECT_ROOT_DIR + "/data/manual_annotation/selection/"


def eval(args):
    all_data = []

    for i in range(args.start_index, args.end_index+1):
        print(f"AGREEMENT SCORES FILE ID {i}:")
        base_file = os.path.join(BASE_PATH, f"{i}.csv")
        annotated_files = {ann: os.path.join(BASE_PATH, f"{i}_{ann}.csv") for ann in args.annotators}
        annotated_files = {a: f for a, f in annotated_files.items() if os.path.isfile(f)}

        data = pd.read_csv(base_file, index_col=0, dtype={"note": str})
        data["note"] = ""
        for ann, file in annotated_files.items():
            data_ann = pd.read_csv(file, index_col=0, dtype={"note": str})
            if len(data_ann.dropna(subset=["is_grammatical"])) != len(data.dropna(subset=["is_grammatical"])):
                missing = data_ann.is_grammatical.isna() != data.is_grammatical.isna()
                missing = [i for i, v in missing.to_dict().items() if v]
                raise RuntimeError(f"Missing annotations: Annotator {ann}: Lines {missing}")

            column_name = f"is_grammatical_{ann}"
            data[column_name] = data_ann["is_grammatical"].values

            # Update notes
            notes_ann = data_ann.note.apply(lambda x: x if "nat" in str(x) else "")
            data["note"] = data.note + notes_ann

        all_data.append(data)

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

        def majority_vote(row):
            if row.is_grammatical != "TODO":
                return ""
            votes = []
            for ann in args.annotators:
                votes.append(row[f"is_grammatical_{ann}"])

            counter = Counter(votes).most_common()
            if counter[0][1] > len(votes) / 2:
                return counter[0][0]
            else:
                return 0

        if len(args.annotators) == 3:
            data_maj = data.copy()
            data_maj["is_grammatical"] = data_maj.apply(majority_vote, axis=1)
            data_maj[["transcript_file", "speaker_code", "transcript_clean", "is_grammatical", "note"]].to_csv(os.path.join(BASE_PATH, f"{i}_majority_vote.csv"))

        data.dropna(subset=["is_grammatical"], inplace=True)

        kappa_scores = []
        mcc_scores = []
        for ann_1, ann_2 in itertools.combinations(annotated_files.keys(), 2):
            kappa = cohen_kappa_score(data[f"is_grammatical_{ann_1}"], data[f"is_grammatical_{ann_2}"], weights="linear")
            kappa_scores.append(kappa)
            print(f"Kappa {(ann_1, ann_2)}: {kappa:.2f}")

            mcc = matthews_corrcoef(data[f"is_grammatical_{ann_1}"], data[f"is_grammatical_{ann_2}"])
            mcc_scores.append(mcc)

    print("\n\nAll files:")
    data = pd.concat(all_data)

    kappa_scores = []
    mcc_scores = []
    acc_scores = []
    for ann_1, ann_2 in itertools.combinations(args.annotators, 2):
        kappa = cohen_kappa_score(data[f"is_grammatical_{ann_1}"], data[f"is_grammatical_{ann_2}"], weights="linear")
        kappa_scores.append(kappa)
        print(f"Kappa {(ann_1, ann_2)}: {kappa:.2f}")

        mcc = matthews_corrcoef(data[f"is_grammatical_{ann_1}"], data[f"is_grammatical_{ann_2}"])
        mcc_scores.append(mcc)

        acc = np.mean(data[f"is_grammatical_{ann_1}"] == data[f"is_grammatical_{ann_2}"])
        acc_scores.append(acc)

    print(f"Mean kappa: {np.mean(kappa_scores):.2f} Std: {np.std(kappa_scores):.2f}")
    print(f"Mean mcc: {np.mean(mcc_scores):.2f} Std: {np.std(mcc_scores):.2f}")
    print(f"Mean acc: {np.mean(acc_scores):.2f} Std: {np.std(acc_scores):.2f}")


    rel_data = [data[f"is_grammatical_{ann}"] for ann in args.annotators]
    alpha = krippendorff.alpha(reliability_data=rel_data, level_of_measurement="ordinal")
    print(f"Krippendorff's Alpha: {alpha:.2f}")

    results_df = pd.DataFrame([{"model": "human_annotators", "context_length": 0,
                                "mcc: mean": np.mean(mcc_scores), "mcc: std": np.std(mcc_scores),
                                "accuracy: mean": np.mean(acc_scores), "accuracy: std": np.std(acc_scores),
                                }])
    results_df.set_index(["model", "context_length"], inplace=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.isfile(RESULTS_FILE):
        results_df.to_csv(RESULTS_FILE)
    else:
        old_res_file = pd.read_csv(RESULTS_FILE, index_col=["model", "context_length"])
        results_df = results_df.combine_first(old_res_file)
        results_df.to_csv(RESULTS_FILE)


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
