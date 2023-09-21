import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from grammaticality_annotation.tokenizer import LABEL_FIELD
from grammaticality_manual_annotation.prepare_for_hand_annotation import ANNOTATION_ALL_FILES_PATH
from utils import RESULTS_DIR

ANNOTATED_UTTS_FILE = os.path.join(ANNOTATION_ALL_FILES_PATH, "majority_vote.csv")


def age_bin(age, num_months=3):
    return int((age + num_months / 2) / num_months) * num_months


def main():
    data = pd.read_csv(ANNOTATED_UTTS_FILE, index_col=0)

    data_grammar = data.dropna(subset=[LABEL_FIELD, "age"]).copy()

    data_grammar["grammatical"] = (data_grammar[LABEL_FIELD] == 1).astype(int)
    data_grammar["ambiguous"] = (data_grammar[LABEL_FIELD] == 0).astype(int)
    data_grammar["ungrammatical"] = (data_grammar[LABEL_FIELD] == -1).astype(int)

    data_grammar["age"] = data_grammar.age.apply(age_bin).astype(int)

    sns.lineplot(data=data_grammar, x="age", y="grammatical", errorbar="se", linestyle=(0, (1, 1)))
    sns.lineplot(data=data_grammar, x="age", y="ambiguous", errorbar="se", linestyle=(0, (1, 1)))
    ax = sns.lineplot(data=data_grammar, x="age", y="ungrammatical", errorbar="se", linestyle=(0, (1, 1)))
    plt.ylabel("proportion")
    plt.xlabel("age (months)")
    plt.xticks([24, 30, 36, 42, 48, 54, 60])
    plt.legend(handles=ax.lines, labels=["grammatical", "ambiguous", "ungrammatical"])

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "annotations_grammaticality_over_age.png"), dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
