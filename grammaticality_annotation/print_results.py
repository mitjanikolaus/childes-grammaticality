import pandas as pd
from utils import RESULTS_FILE


def main():
    results = pd.read_csv(RESULTS_FILE)
    print(results.to_markdown(index=False))
    print("\n\n\n")
    print(results.to_latex(float_format="%.2f", index=False))


if __name__ == "__main__":
    main()
