import argparse
import os
import pandas as pd

from utils import (
    remove_punctuation,
    str2bool,
    remove_babbling,
    ANNOTATED_UTTERANCES_FILE,
    UTTERANCES_WITH_SPEECH_ACTS_FILE, remove_events_and_non_parseable_words, replace_slang_forms, clean_disfluencies,
    split_into_words,
)
from utils import (
    remove_nonspeech_events,
    CODE_UNINTELLIGIBLE,
)

DEFAULT_LABEL_PARTIALLY_SPEECH_RELATED = True

DEFAULT_LABEL_PARTIALLY_INTELLIGIBLE = False

# Speech acts that relate to nonverbal/external events
SPEECH_ACTS_NONVERBAL_EVENTS = [
    "CR",  # Criticize or point out error in nonverbal act.
    "PM",  # Praise for motor acts i.e for nonverbal behavior.
    "WD",  # Warn of danger.
    "DS",  # Disapprove scold protest disruptive behavior.
    "AB",  # Approve of appropriate behavior.
    "TO",  # Mark transfer of object to hearer
    "ET",  # Express enthusiasm for hearer's performance.
    "ED",  # Exclaim in disapproval.
]


def is_speech_related(
        utterance,
        label_partially_speech_related=DEFAULT_LABEL_PARTIALLY_SPEECH_RELATED,
        label_unintelligible=pd.NA,
):
    """Label utterances as speech or non-speech."""
    utterance_without_punctuation = remove_punctuation(utterance)
    utt_without_nonspeech = remove_nonspeech_events(utterance_without_punctuation)

    utt_without_nonspeech = utt_without_nonspeech.strip()
    if utt_without_nonspeech == "":
        return False

    # We exclude completely unintelligible utterances (we don't know whether it's speech-related or not)
    is_completely_unintelligible = True
    for word in split_into_words(utt_without_nonspeech, remove_commas=True, remove_trailing_punctuation=False):
        if word != CODE_UNINTELLIGIBLE and word != "":
            is_completely_unintelligible = False
            break
    if is_completely_unintelligible:
        # By returning None, we can filter out these cases later
        return label_unintelligible

    is_partly_speech_related = len(utt_without_nonspeech) != len(
        utterance_without_punctuation
    )
    if is_partly_speech_related:
        return label_partially_speech_related

    return True


def is_intelligible(
        utterance,
        label_partially_intelligible=DEFAULT_LABEL_PARTIALLY_INTELLIGIBLE,
        label_empty_utterance=False,
):
    utterance_without_punctuation = remove_punctuation(utterance)
    utterance_without_nonspeech = remove_nonspeech_events(utterance_without_punctuation)
    utterance_without_nonspeech = utterance_without_nonspeech.strip()
    if utterance_without_nonspeech == "":
        return label_empty_utterance

    utt_without_babbling = remove_babbling(utterance_without_nonspeech)

    utt_without_babbling = utt_without_babbling.strip()
    if utt_without_babbling == "":
        return False

    is_partly_intelligible = len(utt_without_babbling) != len(
        utterance_without_nonspeech
    )
    if is_partly_intelligible:
        return label_partially_intelligible

    return True


def clean_preprocessed_utterance(utterance):
    final_punctuation = None
    while len(utterance) > 0 and utterance[-1] in [".", "!", "?"]:
        final_punctuation = utterance[-1]
        utterance = utterance[:-1]

    utt_clean = remove_events_and_non_parseable_words(utterance)
    utt_clean = replace_slang_forms(utt_clean)
    utt_clean = clean_disfluencies(utt_clean)

    # Remove underscores
    utt_clean = utt_clean.replace("_", " ")

    # Remove spacing before commas and double commas
    utt_clean = utt_clean.replace(" ,", ",")
    utt_clean = utt_clean.replace(",,", ",")
    utt_clean = utt_clean.replace(" .", ".")
    utt_clean = utt_clean.replace(",.", ".")

    # Strip:
    utt_clean = utt_clean.strip()
    utt_clean = utt_clean.replace("   ", " ")
    utt_clean = utt_clean.replace("  ", " ")

    # Remove remaining commas at beginning and end of utterance
    while len(utt_clean) > 0 and utt_clean[0] == ",":
        utt_clean = utt_clean[1:].strip()
    while len(utt_clean) > 0 and utt_clean[-1] == ",":
        utt_clean = utt_clean[:-1].strip()

    if final_punctuation:
        utt_clean += final_punctuation
    else:
        utt_clean += "."

    return utt_clean


def annotate(args):
    utterances = pd.read_pickle(UTTERANCES_WITH_SPEECH_ACTS_FILE)

    print("Annotating speech-relatedness..")
    utterances["is_speech_related"] = utterances.transcript_raw.apply(
        is_speech_related,
        label_partially_speech_related=args.label_partially_speech_related,
    )
    utterances.is_speech_related = utterances.is_speech_related.astype("boolean")

    print("Annotating intelligibility..")
    utterances["is_intelligible"] = utterances.transcript_raw.apply(
        is_intelligible,
        label_partially_intelligible=args.label_partially_intelligible,
    )

    print("Cleaning utterances..")
    utterances["transcript_clean"] = utterances.transcript_raw.apply(clean_preprocessed_utterance)

    return utterances


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--label-partially-speech-related",
        type=str2bool,
        const=True,
        nargs="?",
        default=DEFAULT_LABEL_PARTIALLY_SPEECH_RELATED,
        help="Label for partially speech-related utterances: Set to True to count as speech-related, False to count as "
             "not speech-related or None to exclude these utterances from the analysis",
    )
    argparser.add_argument(
        "--label-partially-intelligible",
        type=str2bool,
        const=True,
        nargs="?",
        default=DEFAULT_LABEL_PARTIALLY_INTELLIGIBLE,
        help="Label for partially intelligible utterances: Set to True to count as intelligible, False to count as unintelligible or None to exclude these utterances from the analysis",
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    annotated_utts = annotate(args)

    os.makedirs(os.path.dirname(ANNOTATED_UTTERANCES_FILE), exist_ok=True)

    annotated_utts.to_pickle(ANNOTATED_UTTERANCES_FILE)
    annotated_utts.to_csv(ANNOTATED_UTTERANCES_FILE.replace(".p", ".csv"))
