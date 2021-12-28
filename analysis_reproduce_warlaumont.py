import argparse
import math
import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from statsmodels.stats.weightstats import ztest
from tqdm import tqdm

from annotate import ANNOTATED_UTTERANCES_FILE, is_empty, get_response_latency
from utils import (
    filter_corpora_based_on_response_latency_length,
    get_path_of_utterances_file,
    age_bin,
)
from preprocess import (
    CANDIDATE_CORPORA,
    SPEAKER_CODE_CHILD,
    SPEAKER_CODES_CAREGIVER,
)
from utils import (
    clean_utterance,
    remove_nonspeech_events,
    CODE_UNINTELLIGIBLE,
)

DEFAULT_RESPONSE_THRESHOLD = 1000

# 1 second
DEFAULT_MAX_NEG_RESPONSE_LATENCY = -1 * 1000  # ms

DEFAULT_RESPONSE_LATENCY_MAX_STANDARD_DEVIATIONS_OFF = 1

DEFAULT_COUNT_ONLY_SPEECH_RELATED_RESPONSES = True

DEFAULT_MIN_RATIO_NONSPEECH = 0.0

DEFAULT_MIN_TRANSCRIPT_LENGTH = 0

# Ages aligned to study of Warlaumont et al.
DEFAULT_MIN_AGE = 8
DEFAULT_MAX_AGE = 48

AGE_BIN_NUM_MONTHS = 6


# 10 seconds
DEFAULT_MAX_RESPONSE_LATENCY_FOLLOW_UP = 10 * 1000  # ms


DEFAULT_EXCLUDED_CORPORA = []


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    elif v.lower() in ("none", "nan"):
        return None
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--corpora",
        nargs="+",
        type=str,
        required=False,
        choices=CANDIDATE_CORPORA,
        help="Corpora to analyze. If not given, corpora are selected based on a response time variance threshold.",
    )
    argparser.add_argument(
        "--excluded-corpora",
        nargs="+",
        type=str,
        choices=CANDIDATE_CORPORA,
        default=DEFAULT_EXCLUDED_CORPORA,
        help="Corpora to exclude from analysis",
    )
    argparser.add_argument(
        "--min-age",
        type=int,
        default=DEFAULT_MIN_AGE,
    )
    argparser.add_argument(
        "--max-age",
        type=int,
        default=DEFAULT_MAX_AGE,
    )
    argparser.add_argument(
        "--min-transcript-length",
        type=int,
        default=DEFAULT_MIN_TRANSCRIPT_LENGTH,
    )
    argparser.add_argument(
        "--min-ratio-nonspeech",
        type=int,
        default=DEFAULT_MIN_RATIO_NONSPEECH,
    )

    argparser.add_argument(
        "--response-latency-max-standard-deviations-off",
        type=int,
        default=DEFAULT_RESPONSE_LATENCY_MAX_STANDARD_DEVIATIONS_OFF,
        help="Number of standard deviations that the mean response latency of a corpus can be off the reference mean",
    )
    argparser.add_argument(
        "--response-latency",
        type=int,
        default=DEFAULT_RESPONSE_THRESHOLD,
        help="Response latency in milliseconds",
    )
    argparser.add_argument(
        "--max-neg-response-latency",
        type=int,
        default=DEFAULT_MAX_NEG_RESPONSE_LATENCY,
        help="Maximum negative response latency in milliseconds",
    )
    argparser.add_argument(
        "--max-response-latency-follow-up",
        type=int,
        default=DEFAULT_MAX_RESPONSE_LATENCY_FOLLOW_UP,
        help="Maximum response latency for the child follow-up in milliseconds",
    )
    argparser.add_argument(
        "--count-only-speech_related_responses",
        type=str2bool,
        const=True,
        nargs="?",
        default=DEFAULT_COUNT_ONLY_SPEECH_RELATED_RESPONSES,
    )

    args = argparser.parse_args()

    return args


def caregiver_response_contingent_on_speech_relatedness(row):
    return ((row["is_speech_related"] == True) & (row["has_response"] == True)) | (
        (row["is_speech_related"] == False) & (row["has_response"] == False)
    )


def perform_warlaumont_analysis(
    conversations, args, analysis_function, label_positive_valence
):
    print(f"\nFound {len(conversations)} micro-conversations")
    print("Overall analysis: ")
    (
        contingency_caregiver,
        contingency_children_pos_case,
        contingency_children_neg_case,
        _,
    ) = analysis_function(conversations)
    print(f"Caregiver contingency: {contingency_caregiver:.4f}")
    print(f"Child contingency (positive case): {contingency_children_pos_case:.4f}")
    # print(f"Child contingency (negative case): {contingency_children_neg_case:.4f}")

    print("Per-transcript analysis: ")
    results = []

    for transcript in conversations.transcript_file.unique():
        conversations_transcript = conversations[conversations.transcript_file == transcript]
        if len(conversations_transcript) > args.min_transcript_length:
            (
                contingency_caregiver,
                contingency_children_pos_case,
                contingency_children_neg_case,
                proportion_positive_valence,
            ) = analysis_function(conversations_transcript)
            results.append(
                {
                    "age": conversations_transcript.age.values[0],
                    "contingency_caregiver": contingency_caregiver,
                    "contingency_children_pos_case": contingency_children_pos_case,
                    "contingency_children_neg_case": contingency_children_neg_case,
                    label_positive_valence: proportion_positive_valence,
                }
            )
    results = pd.DataFrame(results)

    p_value = ztest(
        results.contingency_caregiver.dropna(), value=0.0, alternative="larger"
    )[1]
    print(
        f"Caregiver contingency: {results.contingency_caregiver.dropna().mean():.4f} +-{results.contingency_caregiver.dropna().std():.4f} p-value:{p_value}"
    )
    p_value = ztest(
        results.contingency_children_pos_case.dropna(), value=0.0, alternative="larger"
    )[1]
    print(
        f"Child contingency (positive case): {results.contingency_children_pos_case.dropna().mean():.4f} +-{results.contingency_children_pos_case.dropna().std():.4f} p-value:{p_value}"
    )
    # p_value = ztest(results.contingency_children_neg_case.dropna(), value=0.0, alternative="larger")[1]
    # print(
    #     f"Child contingency (negative case): {results.contingency_children_neg_case.dropna().mean():.4f} +-{results.contingency_children_neg_case.dropna().std():.4f} p-value:{p_value}"
    # )

    return results


def perform_contingency_analysis_speech_relatedness(conversations):
    # caregiver contingency
    n_responses_to_speech = len(
        conversations[conversations.is_speech_related & conversations.has_response]
    )
    n_speech = len(conversations[conversations.is_speech_related])

    n_responses_to_non_speech = len(
        conversations[
            (conversations.is_speech_related == False) & conversations.has_response
        ]
    )
    n_non_speech = len(conversations[conversations.is_speech_related == False])

    if n_non_speech > 0 and n_speech > 0:
        contingency_caregiver = (n_responses_to_speech / n_speech) - (
            n_responses_to_non_speech / n_non_speech
        )
    else:
        contingency_caregiver = np.nan

    # Contingency of child vocalization on previous adult response (positive case):
    n_follow_up_speech_related_if_response_to_speech_related = len(
        conversations[
            conversations.follow_up_speech_related
            & conversations.is_speech_related
            & conversations.has_response
        ]
    )

    n_follow_up_speech_related_if_no_response_to_speech_related = len(
        conversations[
            conversations.follow_up_speech_related
            & conversations.is_speech_related
            & (conversations.has_response == False)
        ]
    )
    n_no_responses_to_speech_related = len(
        conversations[
            conversations.is_speech_related & (conversations.has_response == False)
        ]
    )

    if n_responses_to_speech > 0 and n_no_responses_to_speech_related > 0:
        ratio_follow_up_speech_related_if_response_to_speech_related = (
            n_follow_up_speech_related_if_response_to_speech_related
            / n_responses_to_speech
        )
        ratio_follow_up_speech_related_if_no_response_to_speech_related = (
            n_follow_up_speech_related_if_no_response_to_speech_related
            / n_no_responses_to_speech_related
        )
        contingency_children_pos_case = (
            ratio_follow_up_speech_related_if_response_to_speech_related
            - ratio_follow_up_speech_related_if_no_response_to_speech_related
        )
    else:
        contingency_children_pos_case = np.nan

    # Contingency of child vocalization on previous adult response (negative case):
    n_follow_up_speech_related_if_no_response_to_non_speech_related = len(
        conversations[
            conversations.follow_up_speech_related
            & (conversations.is_speech_related == False)
            & (conversations.has_response == False)
        ]
    )
    n_no_responses_to_non_speech_related = len(
        conversations[
            (conversations.is_speech_related == False)
            & (conversations.has_response == False)
        ]
    )

    n_follow_up_speech_related_if_response_to_non_speech_related = len(
        conversations[
            conversations.follow_up_speech_related
            & (conversations.is_speech_related == False)
            & conversations.has_response
        ]
    )
    n_responses_to_non_speech_related = len(
        conversations[
            (conversations.is_speech_related == False) & conversations.has_response
        ]
    )

    if (
        n_no_responses_to_non_speech_related > 0
        and n_responses_to_non_speech_related > 0
    ):
        ratio_follow_up_speech_related_if_no_response_to_non_speech_related = (
            n_follow_up_speech_related_if_no_response_to_non_speech_related
            / n_no_responses_to_non_speech_related
        )
        ratio_follow_up_speech_related_if_response_to_non_speech_related = (
            n_follow_up_speech_related_if_response_to_non_speech_related
            / n_responses_to_non_speech_related
        )
        contingency_children_neg_case = (
            ratio_follow_up_speech_related_if_no_response_to_non_speech_related
            - ratio_follow_up_speech_related_if_response_to_non_speech_related
        )
    else:
        contingency_children_neg_case = np.nan

    proportion_speech_related = n_speech / (n_speech + n_non_speech)

    return (
        contingency_caregiver,
        contingency_children_pos_case,
        contingency_children_neg_case,
        proportion_speech_related,
    )


def has_response(
    row, response_latency, max_neg_response_latency, count_only_speech_related_responses
):
    if np.isnan(row["response_latency"]):
        return None

    # Disregard utterances with too long negative pauses
    if row["response_latency"] < max_neg_response_latency:
        return None

    if row["response_latency"] <= response_latency:
        if count_only_speech_related_responses:
            return row["response_is_speech_related"]
        else:
            return True
    return False


def get_micro_conversations(utterances, args):
    conversations = []
    print("Creating micro conversations from transcripts..")
    for transcript in tqdm(utterances.transcript_file.unique()):
        utterances_transcript = utterances[utterances.transcript_file == transcript]
        utterances_child = utterances_transcript[
            utterances_transcript.speaker_code == SPEAKER_CODE_CHILD
        ]

        # Filter for child utterances that are at the end of a turn
        utts_child_end_of_turn = utterances_child[
            ~(
                (utterances_child.speaker_code_next == SPEAKER_CODE_CHILD)
                & (
                    utterances_child.start_time_next - utterances_child.end_time
                    < args.response_latency
                )
            )
        ]

        for candidate_id in utts_child_end_of_turn.index.values:
            conversation = utterances_transcript.loc[candidate_id].to_dict()
            if np.isnan(conversation["is_speech_related"]) or is_empty(conversation["transcript_raw"]):
                continue
            if candidate_id + 1 not in utterances_transcript.index:
                # No response in transcript, ignore the last utterance
                continue

            subsequent_utt = utterances_transcript.loc[candidate_id + 1]
            if subsequent_utt.speaker_code in SPEAKER_CODES_CAREGIVER:
                utt2 = subsequent_utt
                if is_empty(utt2["transcript_raw"]) or np.isnan(utt2["is_speech_related"]):
                    continue
                conversation["response_is_speech_related"] = utt2["is_speech_related"]

                conversation["response_latency"] = get_response_latency(conversation)
                if conversation["response_latency"] is None:
                    continue

            elif subsequent_utt.speaker_code == SPEAKER_CODE_CHILD:
                conversation["response_latency"] = math.inf
                conversation["start_time_next"] = math.inf
                conversation["response_is_speech_related"] = False
                following_utts = utterances_transcript.loc[subsequent_utt.name + 1:]
                following_utts_non_child = following_utts[
                    following_utts.speaker_code != SPEAKER_CODE_CHILD
                ]
                if (
                    len(following_utts_non_child) == 0
                    or following_utts_non_child.iloc[0].speaker_code
                    not in SPEAKER_CODES_CAREGIVER
                ):
                    # Child is not speaking to its caregiver, ignore this turn
                    continue
            else:
                # Child is not speaking to its caregiver, ignore this turn
                continue

            following_utts = utterances_transcript.loc[candidate_id + 1:]
            following_utts_child = following_utts[
                following_utts.speaker_code == SPEAKER_CODE_CHILD
            ]
            if len(following_utts_child) > 0:
                utt3 = following_utts_child.iloc[0]

                if is_empty(utt3["transcript_raw"]) or np.isnan(utt3["is_speech_related"]):
                    continue

                conversation["response_latency_follow_up"] = (
                    utt3["start_time"] - conversation["end_time"]
                )
                conversation["follow_up_speech_related"] = utt3["is_speech_related"]
                conversations.append(conversation)

    conversations = pd.DataFrame(conversations)

    conversations = conversations.assign(
        has_response=conversations.apply(
            has_response,
            axis=1,
            response_latency=args.response_latency,
            max_neg_response_latency=args.max_neg_response_latency,
            count_only_speech_related_responses=args.count_only_speech_related_responses,
        )
    )

    return conversations


def perform_analysis_speech_relatedness(utterances, args):
    conversations = get_micro_conversations(utterances, args)

    # disregard conversations with follow up too far in the future
    conversations = conversations[
        (
            conversations.response_latency_follow_up
            <= args.max_response_latency_follow_up
        )
    ]

    conversations.dropna(
        subset=(
            "response_latency",
            "response_latency_follow_up",
            "has_response",
        ),
        inplace=True,
    )

    print(f"Filtering corpora based on average response latency")
    corpora = filter_corpora_based_on_response_latency_length(
        conversations,
        args.min_age,
        args.max_age,
        args.response_latency_max_standard_deviations_off,
    )

    print(f"Corpora included in analysis: {corpora}")
    # Filter by corpora
    conversations = conversations[conversations.corpus.isin(corpora)]

    counter_non_speech = Counter(
        conversations[conversations.is_speech_related == False].transcript_raw.values
    )
    print("Most common non-speech related sounds: ")
    print(counter_non_speech.most_common(20))

    # Filter for corpora that actually annotate non-speech-related sounds
    good_corpora = []
    print("Ratios nonspeech/speech for each corpus:")
    for corpus in conversations.corpus.unique():
        d_corpus = conversations[conversations.corpus == corpus]
        ratio = len(d_corpus[d_corpus.is_speech_related == False]) / len(
            d_corpus[d_corpus.is_speech_related == True]
        )
        if ratio > args.min_ratio_nonspeech:
            good_corpora.append(corpus)
        print(f"{corpus}: {ratio:.5f}")
    print("Filtered corpora: ", good_corpora)

    conversations = conversations[conversations.corpus.isin(good_corpora)]

    # Get the number of children in all corpora:
    num_children = len(conversations.child_name.unique())
    print(f"Number of children in the analysis: {num_children}")

    # Label caregiver responses as contingent on child utterance or not
    conversations = conversations.assign(
        caregiver_response_contingent=conversations[
            ["is_speech_related", "has_response"]
        ].apply(caregiver_response_contingent_on_speech_relatedness, axis=1)
    )

    results_analysis = perform_warlaumont_analysis(
        conversations,
        args,
        perform_contingency_analysis_speech_relatedness,
        "proportion_speech_related",
    )
    results_dir = "results/reproduce_warlaumont/"
    os.makedirs(results_dir, exist_ok=True)

    plt.figure()
    sns.scatterplot(data=results_analysis, x="age", y="contingency_caregiver")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "dev_contingency_caregivers.png"))

    plt.figure()
    sns.scatterplot(data=results_analysis, x="age", y="contingency_children_pos_case")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "dev_contingency_children.png"))

    # plt.figure()
    # sns.scatterplot(data=results_analysis, x="age", y="contingency_children_neg_case")

    plt.figure()
    sns.regplot(
        data=results_analysis,
        x="age",
        y="proportion_speech_related",
        marker=".",
        ci=None,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "dev_proportion_speech_related.png"))

    plt.figure()
    plt.title("Caregiver contingency")
    sns.barplot(
        data=conversations,
        x="is_speech_related",
        y="has_response",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "contingency_caregivers.png"))

    conversations["age"] = conversations.age.apply(
        age_bin, min_age=args.min_age, num_months=AGE_BIN_NUM_MONTHS
    )

    plt.figure(figsize=(6, 3))
    plt.title("Caregiver contingency - per age group")
    axis = sns.barplot(
        data=conversations,
        x="age",
        y="has_response",
        hue="is_speech_related",
        ci=None,
    )
    sns.move_legend(axis, "lower right")
    axis.set(ylabel="prob_caregiver_response")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "contingency_caregivers_per_age.png"))

    plt.figure()
    plt.title("Child contingency")
    sns.barplot(
        data=conversations[conversations.is_speech_related == True],
        x="has_response",
        y="follow_up_speech_related",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "contingency_children.png"))

    plt.figure(figsize=(6, 3))
    plt.title("Child contingency - per age group")
    axis = sns.barplot(
        data=conversations[conversations.is_speech_related == True],
        x="age",
        y="follow_up_speech_related",
        hue="has_response",
        ci=None,
    )
    sns.move_legend(axis, "lower right")
    axis.set(ylabel="prob_follow_up_speech_related")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "contingency_children_per_age.png"))

    plt.show()

    return conversations


def perform_analyses(args, analysis_function):
    utterances = pd.read_csv(ANNOTATED_UTTERANCES_FILE, index_col=None)

    print(args)

    print("Excluding corpora: ", args.excluded_corpora)
    utterances = utterances[~utterances.corpus.isin(args.excluded_corpora)]

    # Filter by age
    utterances = utterances[
        (args.min_age <= utterances.age) & (utterances.age <= args.max_age)
    ]

    if args.corpora:
        utterances = utterances[utterances.corpus.isin(args.corpora)]

    min_age = utterances.age.min()
    max_age = utterances.age.max()
    mean_age = utterances.age.mean()
    print(
        f"Mean of child age in analysis: {mean_age:.1f} (min: {min_age} max: {max_age})"
    )

    return analysis_function(utterances, args)


if __name__ == "__main__":
    args = parse_args()

    conversations = perform_analyses(args, perform_analysis_speech_relatedness)
