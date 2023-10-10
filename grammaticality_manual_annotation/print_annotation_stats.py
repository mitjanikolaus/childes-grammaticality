import os

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from grammaticality_annotation.data import load_childes_data, DATA_PATH_CHILDES_ANNOTATED
from grammaticality_annotation.tokenizer import ERROR_LABELS_FIELD, LABEL_FIELD
from utils import RESULTS_DIR


# def age_bin(age, min_age, max_age, num_months):
#     return min(
#         max_age, max(min_age, int((age + num_months / 2) / num_months) * num_months)
#     )


def age_bin(age, num_months=12):
    return int((age + num_months / 2) / num_months) * num_months


def main():
    data = load_childes_data(DATA_PATH_CHILDES_ANNOTATED)

    labels = ["ungrammatical", "ambiguous", "grammatical"]
    label_counts = data.is_grammatical.value_counts().sort_index()
    print(label_counts)
    percentages = label_counts / label_counts.sum()
    print(f"Label counts: ")
    print([f"{label}: {count} ({round(perc*100)}\%)" for label, count, perc in zip(labels, label_counts, percentages)])

    data_ungrammatical = data[data[LABEL_FIELD] == -1].copy()
    data_ungrammatical[ERROR_LABELS_FIELD] = data_ungrammatical[ERROR_LABELS_FIELD].apply(lambda x: x.split(", "))
    data_ungrammatical = data_ungrammatical.explode(ERROR_LABELS_FIELD)

    counts = data_ungrammatical[ERROR_LABELS_FIELD].value_counts()
    print(counts)
    plt.figure(figsize=(4, 5))
    ax = counts.plot(kind="barh", color="#617fab")
    percentages = data_ungrammatical[ERROR_LABELS_FIELD].value_counts(normalize=True) * 100
    print(percentages)
    patches = ax.patches
    for i in range(len(patches)):
        x = patches[i].get_width() + 5
        y = patches[i].get_y() + 0.1
        ax.annotate(f"{percentages[i]:.1f}%", (x, y), ha='left')
    plt.xlim((0, 440))
    plt.xlabel("Number of errors")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "manual_annotation_error_labels.png"), dpi=300)

    plt.show()

    # data["age"] = data.age.apply(age_bin)
    # data.dropna(subset="is_grammatical", inplace=True)
    # print(data.age.value_counts())
    # data.age.value_counts().sort_index().plot(kind="bar")
    # plt.show()


if __name__ == "__main__":
    main()
