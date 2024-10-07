import argparse
import os
import re

from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

SPEAKER_CODE_CHILD = "CHI"

SPEAKER_CODES_CAREGIVER = [
    "MOT",
    "FAT",
    "DAD",
    "MOM",
    "GRA",
    "GRF",
    "GRM",
    "GMO",
    "GFA",
    "CAR",
]

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT_DIR, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "results.csv")

PREPROCESSED_UTTERANCES_FILE = os.path.expanduser(
    "~/data/communicative_feedback/utterances.csv"
)

UTTERANCES_WITH_SPEECH_ACTS_FILE = os.path.expanduser(
    "~/data/communicative_feedback/utterances_with_speech_acts.csv"
)

UTTERANCES_WITH_PREV_UTTS_FILE = os.path.expanduser(
    "~/data/communicative_feedback/utterances_with_prev_utts.csv"
)

FILE_GRAMMATICALITY_ANNOTATIONS = PROJECT_ROOT_DIR + "/data/manual_annotation/grammaticality_manually_annotated.csv"

UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_FILE = os.path.expanduser(
    "~/data/communicative_feedback/utterances_with_childes_error_annotations.csv"
)

UTTERANCES_WITH_CHILDES_ERROR_ANNOTATIONS_FOR_TRAINING_FILE = os.path.expanduser(
    "~/data/communicative_feedback/utterances_with_childes_error_annotations_for_training.csv"
)

ANNOTATED_UTTERANCES_FILE = os.path.expanduser(
    "~/data/communicative_feedback/utterances_annotated.csv"
)

MICRO_CONVERSATIONS_FILE = os.path.expanduser(
    "~/data/communicative_feedback/micro_conversations.csv"
)

MICRO_CONVERSATIONS_WITHOUT_NON_SPEECH_FILE = os.path.expanduser(
    "~/data/communicative_feedback/micro_conversations_without_non_speech.csv"
)

POS_PUNCTUATION = [
    ".",
    "?",
    "...",
    "!",
    "+/",
    "+/?",
    "" "...?",
    ",",
    "-",
    '+"/.',
    "+...",
    "++/.",
    "+/.",
]

# codes that will be excluded from analysis
IS_UNTRANSCRIBED = lambda word: word.replace(".", "") == "www"
IS_INTERRUPTION = lambda word: word.startswith("+/")
IS_SELF_INTERRUPTION = lambda word: word == "+//"
IS_TRAILING_OFF = lambda word: word == "+..."
IS_TRAILING_OFF_2 = lambda word: word == "+.."
IS_EXCLUDED_WORD = lambda word: "@x" in word
IS_OMITTED_WORD = lambda word: word.startswith("0")
IS_SATELLITE_MARKER = lambda word: word == "‡"
IS_QUOTATION_MARKER = lambda word: word in ['+"/', '+"/.', '+"', '+".']
IS_UNKNOWN_CODE = lambda word: word == "zzz"


def is_nan(value):
    return value != value


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


def age_bin(age, min_age, max_age, num_months):
    return min(
        max_age, max(min_age, int((age + num_months / 2) / num_months) * num_months)
    )


ERR_SUBJECT = "subject"
ERR_VERB = "verb"
ERR_OBJECT = "object"
ERR_POSSESSIVE = "possessive"
ERR_PLURAL = "plural"
ERR_SV_AGREEMENT = "sv_agreement"
ERR_TENSE_ASPECT = "tense_aspect"
ERR_PROGRESSIVE = "progressive"
ERR_DETERMINER = "determiner"
ERR_PREPOSITION = "preposition"
ERR_AUXILIARY = "auxiliary"
ERR_OTHER = "other"
ERR_UNKNOWN = "unk"


def split_into_words(utterance, split_on_apostrophe=True, remove_commas=False, remove_trailing_punctuation=False):
    # Copy in order not to modify the original utterance
    utt = utterance
    if remove_trailing_punctuation:
        utt = utterance[:-1]
    regex = '\s'
    if split_on_apostrophe:
        regex += '|\''
    if remove_commas:
        regex += '|,'

    words = re.split(regex, utt)

    # Filter out empty words:
    words = [word for word in words if len(word) > 0]
    return words


def get_num_words(clean_utts, remove_punctuation=True):
    return clean_utts.apply(lambda x: len(split_into_words(x, split_on_apostrophe=True, remove_commas=remove_punctuation, remove_trailing_punctuation=remove_punctuation)))


def get_num_unique_words(clean_utts, remove_punctuation=True):
    return clean_utts.apply(lambda x: len(set(split_into_words(x, split_on_apostrophe=False, remove_commas=remove_punctuation, remove_trailing_punctuation=remove_punctuation))))


# Use the symbol yyy when you plan to code all material phonologically on a %pho line.
# (usually used when utterance cannot be matched to particular words)
CODE_PHONETIC = "yyy"

CODE_BABBLING = "@b"
CODE_UNIBET_PHONOLOGICAL_TRANSCRIPTION = "@u"
CODE_INTERJECTION = "@i"
CODE_PHONOLOGICAL_CONSISTENT_FORM = "@p"
CODE_PHONOLOGICAL_FRAGMENT = "&"


# We're not fixing words such as "wanna", as it can be both "want to" and "want a"
# Also: "she's": can be either "she has" or "she is"
SLANG_WORDS = {
    "hasta": "has to",
    "hafta": "have to",
    "hadta": "had to",
    "needta": "need to",
    "dat's": "that is",
    "dat": "that",
    "dis": "this",
    "dere": "there",
    "de": "the",
    "gonna": "going to",
    "anoder": "another",
    "dunno": "don't know",
    "'cause": "because",
}


def replace_slang_forms(utterance):
    words = split_into_words(utterance, split_on_apostrophe=False, remove_commas=False, remove_trailing_punctuation=False)
    cleaned_utterance = [
        word if word.replace(",", "") not in SLANG_WORDS.keys() else SLANG_WORDS[word.replace(",", "")]
        for word in words
    ]
    cleaned_utterance = " ".join(cleaned_utterance)
    return cleaned_utterance.strip()


def filter_for_min_num_words(utterances, min_num_words):
    num_words = utterances.transcript_clean.apply(
        lambda x: len(split_into_words(x, split_on_apostrophe=False, remove_commas=True,
                                       remove_trailing_punctuation=True)))
    return utterances[num_words >= min_num_words]


def filter_transcripts_based_on_num_child_utts(
    conversations, min_child_utts_per_transcript
):
    child_utts = conversations[conversations.speaker_code == SPEAKER_CODE_CHILD]
    child_utts_per_transcript = child_utts.groupby("transcript_file").size()
    transcripts_enough_utts = child_utts_per_transcript[
        child_utts_per_transcript > min_child_utts_per_transcript
    ]

    return conversations[
        conversations.transcript_file.isin(transcripts_enough_utts.index)
    ]


def add_prev_utts_for_transcript(utterances_transcript, num_utts=1, add_speaker_codes=True):
    utts_speech_related = utterances_transcript[utterances_transcript.is_speech_related.isin([pd.NA, True])]

    def add_prev_utt(utterance):
        if utterance.name in utts_speech_related.index:
            row_number = np.where(utts_speech_related.index.values == utterance.name)[0][0]
            if row_number > 0:
                prev_utts = utts_speech_related.loc[utts_speech_related.index[:row_number][-num_utts:]]
                return " ".join(prev_utts.transcript_clean)

        return pd.NA

    def add_prev_utt_speaker_codes(utterance):
        if utterance.name in utts_speech_related.index:
            row_number = np.where(utts_speech_related.index.values == utterance.name)[0][0]
            if row_number > 0:
                prev_utts = utts_speech_related.loc[utts_speech_related.index[:row_number][-num_utts:]]
                return " ".join(prev_utts.speaker_code)

        return pd.NA

    column_name = "prev_transcript_clean"
    if num_utts > 1:
        column_name = "prev_transcript_clean_" + str(num_utts)
    utterances_transcript[column_name] = utterances_transcript.apply(
        add_prev_utt,
        axis=1
    )

    if add_speaker_codes:
        column_name = "prev_speaker_code"
        if num_utts > 1:
            column_name = "prev_speaker_code_" + str(num_utts)
        utterances_transcript[column_name] = utterances_transcript.apply(
            add_prev_utt_speaker_codes,
            axis=1
        )

    return utterances_transcript


def add_prev_utts(utterances, num_utts=1):
    # Single-process version for debugging:
    # results = [add_prev_utts_for_transcript(utts_transcript, num_utts)
    #     for utts_transcript in tqdm([group for _, group in utterances.groupby("transcript_file")])]
    utterances_grouped = [[group, num_utts] for _, group in utterances.groupby("transcript_file")]
    with Pool(processes=8) as pool:
        results = pool.starmap(
            add_prev_utts_for_transcript,
            tqdm(utterances_grouped, total=len(utterances_grouped)),
        )

    utterances = pd.concat(results)

    return utterances
