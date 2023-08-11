import pandas as pd
from utils import RESULTS_FILE

REFERENCE_METRIC = "val_mcc: mean"


def create_results_table_model_comparison(results, context_length=1):
    print("\n\nMODEL COMPARISON:")

    results_context_length = results[results.context_length == context_length].copy()

    print(results_context_length.to_markdown(index=False, floatfmt=".2f"))
    # print("\n\n\n")
    # print(results.to_latex(float_format="%.2f", index=False))


def create_results_table_context_lengths(results, model="roberta-large"):
    print("\n\nEXP CONTEXT LENGTHS:")

    results_model = results[results.model == model].copy()
    print(results_model.to_markdown(index=False, floatfmt=".2f"))
    # print("\n\n\n")
    # print(results.to_latex(float_format="%.2f", index=False))

    best_context_length = results_model.sort_values(REFERENCE_METRIC, ascending=False).iloc[0].context_length
    print(f"\nBest context length: {best_context_length}\n")
    return best_context_length


def main():
    results = pd.read_csv(RESULTS_FILE)

    best_context_length = create_results_table_context_lengths(results)

    create_results_table_model_comparison(results, best_context_length)

    print("\n\nALL RESULTS:")
    print(results.to_markdown(index=False, floatfmt=".2f"))



if __name__ == "__main__":
    main()
