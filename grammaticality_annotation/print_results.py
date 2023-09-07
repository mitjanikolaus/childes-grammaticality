import pandas as pd
from utils import RESULTS_FILE

REFERENCE_METRIC = "val_mcc: mean"
MODELS_NO_CONTEXT = ["majority-classifier", "human-annotators", "1-gram", "2-gram", "3-gram", "4-gram"]


def create_results_table_model_comparison(results, context_length=1):
    print("\n\nMODEL COMPARISON:")

    results_context_length = results[(results["context length"] == context_length) | results.model.isin(MODELS_NO_CONTEXT)].copy()
    results_context_length.sort_values(by="mcc: mean", inplace=True)
    results_context_length.drop(columns=[REFERENCE_METRIC, "mcc: mean", "mcc: std", "accuracy: mean", "accuracy: std", "val_mcc: std"], inplace=True)
    print(results_context_length.to_markdown(index=False, floatfmt=".2f"))
    print("\n\n\n")
    print(results_context_length.style.hide(axis="index").to_latex(hrules=True))


def create_results_table_context_lengths(results, model="microsoft/deberta-v3-large"):
    print("\n\nCONTEXT LENGTHS:")

    results_model = results[results.model == model].copy()
    best_context_length = results_model.sort_values("val_mcc: mean", ascending=False).iloc[0]["context length"]
    print(f"\nBest context length: {best_context_length}\n")

    results_model["MCC"] = results_model["val_mcc: mean"].apply("{:.03f}".format).apply(lambda x: x.lstrip('0')) + " $\pm$ " + results["val_mcc: std"].apply("{:.03f}".format).apply(lambda x: x.lstrip('0'))

    results_model.drop(columns=["mcc: mean", "mcc: std", "accuracy: mean", "accuracy: std", "Accuracy", "val_mcc: mean", "val_mcc: std"], inplace=True)

    print(results_model.to_markdown(index=False, floatfmt=".2f"))
    print("\n\n\n")
    print(results_model.style.hide(axis="index").to_latex(hrules=True))

    return best_context_length


def main():
    results = pd.read_csv(RESULTS_FILE)

    results["model"] = results.model.apply(lambda x: x.replace("_", "-")) #.split("/")[-1]

    results["MCC"] = results["mcc: mean"].apply("{:.02f}".format) + " $\pm$ " + results["mcc: std"].apply("{:.02f}".format)
    results["Accuracy"] = results["accuracy: mean"].apply("{:.02f}".format) + " $\pm$ " + results["accuracy: std"].apply("{:.02f}".format)
    results["context_length"] = results["context_length"].astype(int)
    results.rename(columns={"context_length": "context length"}, inplace=True)

    best_context_length = create_results_table_context_lengths(results)

    create_results_table_model_comparison(results, best_context_length)

    # print("\n\nALL RESULTS:")
    # print(results.to_markdown(index=False, floatfmt=".2f"))



if __name__ == "__main__":
    main()
