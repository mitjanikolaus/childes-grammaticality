import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from grammaticality_annotation.error_analysis import BASE_COLOR
from utils import RESULTS_FILE, RESULTS_DIR

REFERENCE_METRIC = "val_mcc: mean"
MODELS_NO_CONTEXT = ["majority-classifier", "human-annotators", "1-gram", "2-gram", "3-gram", "4-gram", "5-gram", "6-gram"]


def create_results_table_model_comparison(results, context_length):
    print("\n\nMODEL COMPARISON:")

    results_full_train_data_size = results[(results["train_data_size"] == 1) | results.model.isin(MODELS_NO_CONTEXT)].copy()

    results_context_length = results_full_train_data_size[(results_full_train_data_size["context length"] == context_length) | results_full_train_data_size.model.isin(MODELS_NO_CONTEXT)].copy()
    results_context_length.sort_values(by="mcc: mean", inplace=True)
    results_context_length.drop(columns=[REFERENCE_METRIC, "mcc: mean", "mcc: std", "accuracy: mean", "accuracy: std", "val_mcc: std", "train_data_size", "context length"], inplace=True)
    print(results_context_length.to_markdown(index=False, floatfmt=".2f"))
    print("\n\n\n")
    print(results_context_length.style.hide(axis="index").to_latex(hrules=True))


def create_results_table_context_lengths(results, model="microsoft/deberta-v3-large"):
    print("\n\nCONTEXT LENGTHS:")

    results_full_train_data_size = results[results["train_data_size"] == 1].copy()

    results_model = results_full_train_data_size[results_full_train_data_size.model == model].copy()

    best_context_length = results_model.sort_values("val_mcc: mean", ascending=False).iloc[0]["context length"]
    print(f"\nBest context length: {best_context_length}\n")

    results_model["val_mcc: stderr"] = results_model["val_mcc: std"].apply(lambda x: x/np.sqrt(3))
    plt.figure(figsize=(4, 4))
    plt.errorbar(results_model["context length"], results_model["val_mcc: mean"], results_model["val_mcc: stderr"],
                 fmt="o", elinewidth=1, color=BASE_COLOR)
    plt.xlabel("context length")
    plt.ylabel("MCC")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "context_lengths.png"), dpi=300)

    results_model["val_mcc"] = results_model["val_mcc: mean"].apply("{:.03f}".format).apply(lambda x: x.lstrip('0')) + " $\pm$ " + results["val_mcc: std"].apply("{:.03f}".format).apply(lambda x: x.lstrip('0'))

    results_model.drop(columns=["mcc: mean", "mcc: std", "MCC", "accuracy: mean", "accuracy: std", "Accuracy", "val_mcc: mean", "val_mcc: std", "val_mcc: stderr", "train_data_size"], inplace=True)

    print(results_model.to_markdown(index=False, floatfmt=".2f"))
    print("\n\n\n")
    print(results_model.style.hide(axis="index").to_latex(hrules=True))

    return best_context_length


MAX_NUM_TRAIN_SAMPLES = 3360


def create_results_train_data_size(results, context_length, model="microsoft/deberta-v3-large"):
    results_context_length = results[results["context length"] == context_length].copy()
    results_model = results_context_length[results_context_length.model == model].copy()

    results_model["mcc: stderr"] = results_model["mcc: std"].apply(lambda x: x/np.sqrt(3))
    results_model["train_data_samples"] = results_model["train_data_size"] * MAX_NUM_TRAIN_SAMPLES
    plt.figure(figsize=(4, 4))
    plt.errorbar(results_model["train_data_samples"], results_model["mcc: mean"], results_model["mcc: stderr"],
                 fmt="o--", elinewidth=1, color=BASE_COLOR)
    plt.xlabel("train data samples")
    plt.ylabel("MCC")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "train_data_size.png"), dpi=300)


def main():
    results = pd.read_csv(RESULTS_FILE)

    results["model"] = results.model.apply(lambda x: x.replace("_", "-")) #.split("/")[-1]

    results["MCC"] = results["mcc: mean"].apply("{:.02f}".format) + " $\pm$ " + results["mcc: std"].apply("{:.02f}".format)
    results["Accuracy"] = results["accuracy: mean"].apply("{:.02f}".format) + " $\pm$ " + results["accuracy: std"].apply("{:.02f}".format)
    results["context_length"] = results["context_length"].astype(int)
    results.rename(columns={"context_length": "context length"}, inplace=True)


    best_context_length = create_results_table_context_lengths(results)

    create_results_train_data_size(results, best_context_length)

    create_results_table_model_comparison(results, best_context_length)

    # print("\n\nALL RESULTS:")
    # print(results.to_markdown(index=False, floatfmt=".2f"))



if __name__ == "__main__":
    main()
