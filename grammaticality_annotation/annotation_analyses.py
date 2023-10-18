import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf

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

    transcripts_enough_data = [name for name, size in data_grammar.groupby("transcript_file").size().items() if size > 100]
    print(len(transcripts_enough_data))
    data_filtered = data_grammar[data_grammar.transcript_file.isin(transcripts_enough_data)].copy()

    data_grouped = data_filtered.groupby("transcript_file").aggregate({"age": "mean", LABEL_FIELD: "mean", "grammatical": "mean", "ambiguous": "mean", "ungrammatical": "mean"})

    data_grouped["age"] = data_grouped["age"].astype(int)

    plt.figure(figsize=(13, 5))

    for y_target in ["grammatical", "ambiguous", "ungrammatical"]:
        ax = sns.stripplot(
            data=data_grouped, x="age", y=y_target,
            jitter=.35, alpha=.3, legend=False, marker=".",
        )
        ax_x_ticks = ax.get_xticks()
        ax_x_dense = np.linspace(np.min(ax_x_ticks), np.max(ax_x_ticks), len(ax_x_ticks)*10)
        x = np.linspace(data_grouped.age.min(), data_grouped.age.max(), len(ax_x_dense))

        clf = LogisticRegression(random_state=0).fit(data_filtered.age.values.reshape(-1, 1), data_filtered[y_target])
        y = [y[1] for y in clf.predict_proba(x.reshape(-1, 1))]
        sns.lineplot(x=ax_x_dense, y=y)

    plt.ylabel("")
    plt.xlabel("Age (months)")

    plt.legend(handles=ax.lines, labels=["grammatical", "ambiguous", "ungrammatical"])

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "annotations_grammaticality_over_age.png"), dpi=300)


    md = smf.mixedlm("grammatical ~ age", data_filtered, groups=data_filtered["transcript_file"])
    mdf = md.fit()
    print(mdf.summary())

    md = smf.mixedlm("ambiguous ~ age", data_filtered, groups=data_filtered["transcript_file"])
    mdf = md.fit()
    print(mdf.summary())

    md = smf.mixedlm("ungrammatical ~ age", data_filtered, groups=data_filtered["transcript_file"])
    mdf = md.fit()
    print(mdf.summary())

    plt.show()


if __name__ == "__main__":
    main()
