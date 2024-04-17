"""Load and store transcripts from childes-db."""
import argparse
import os

import pandas as pd

from grammaticality_annotation.tokenizer import TOKEN_SPEAKER_CHILD, TOKEN_SPEAKER_CAREGIVER, LABEL_FIELD
from grammaticality_manual_annotation.prepare_for_hand_annotation import ALL_EXCLUDED_CORPORA
from utils import SPEAKER_CODES_CAREGIVER, SPEAKER_CODE_CHILD, split_into_words, PROJECT_ROOT_DIR

DATA_DIR_PREPROCESSED_CHILDES_DB = os.path.join(PROJECT_ROOT_DIR, "data", "preprocessed", "childes_db")

DB_VERSION = "2021.1"
DB_ARGS = None
# Change if you are using local db access:
# {
#     "hostname": "localhost",
#     "user": "childesdb",
#     "password": "tmp",
#     "db_name": "childes-db-version-0.1.2",
# }

TYPES_QUESTION = {
    "question",
    "interruption question",
    "trail off question",
    "question exclamation",
    "self interruption question",
    "trail off",
}
TYPES_EXCLAMATION = {"imperative_emphatic"}
TYPES_STATEMENT = {
    "declarative",
    "quotation next line",
    "quotation precedes",
    "self interruption",
    "interruption",
}


def speaker_code_to_speaker_token(code):
    if code in [TOKEN_SPEAKER_CHILD, TOKEN_SPEAKER_CAREGIVER]:
        return code
    if code == SPEAKER_CODE_CHILD:
        return TOKEN_SPEAKER_CHILD
    if code in SPEAKER_CODES_CAREGIVER:
        return TOKEN_SPEAKER_CAREGIVER
    raise RuntimeError("Unknown speaker code: ", code)


def transform_childes_db_transcripts(data):
    data.rename(columns={"transcript_id": "transcript_file"}, inplace=True)
    data["transcript_clean"] = data.gloss + data.type.apply(parse_punctuation)
    data["age"] = data["target_child_age"].round()
    data["speaker_code"] = data.speaker_code.apply(speaker_code_to_speaker_token)
    return data


def parse_punctuation(utterance_type):
    if utterance_type in TYPES_QUESTION:
        return "?"
    elif utterance_type in TYPES_EXCLAMATION:
        return "!"
    elif utterance_type in TYPES_STATEMENT:
        return "."
    else:
        print("Unknown utterance type: ", utterance_type)
        return "."


def load_and_save():
    from childespy.childespy import get_utterances, get_corpora

    corpora = get_corpora()
    corpora = corpora[corpora.collection_name.isin(["Eng-NA", "Eng-UK"])]
    corpora = corpora[~corpora.corpus_name.isin(ALL_EXCLUDED_CORPORA)]

    for corpus in corpora.corpus_name.unique():
        print("\ncorpus: ", corpus)
        utt_corpus = get_utterances(
            corpus=corpus, language="eng", db_args=DB_ARGS, db_version=DB_VERSION,
        )

        utt_corpus["gloss"] = utt_corpus["gloss"].apply(
            lambda x: x.replace("xxx", "").replace("www", "").replace("yyy", "").replace("  ", " ").strip())
        utt_corpus = utt_corpus[~utt_corpus.gloss.isin([""])]

        utt_corpus = utt_corpus[utt_corpus.speaker_code.isin(SPEAKER_CODES_CAREGIVER + [SPEAKER_CODE_CHILD])]
        utt_corpus = transform_childes_db_transcripts(utt_corpus)

        # Filter for transcript that contain at least one caregiver utterance
        transcripts_with_caregiver_utts = utt_corpus[utt_corpus.speaker_code != TOKEN_SPEAKER_CHILD].transcript_file
        utt_corpus = utt_corpus[utt_corpus.transcript_file.isin(transcripts_with_caregiver_utts.unique())]

        utt_corpus["num_words"] = utt_corpus.transcript_clean.apply(
            lambda x: len(split_into_words(x, split_on_apostrophe=False, remove_commas=True,
                                           remove_trailing_punctuation=True))
        )

        utt_corpus[LABEL_FIELD] = ""
        utt_corpus.loc[
            (utt_corpus.speaker_code == TOKEN_SPEAKER_CHILD) & (utt_corpus.num_words > 1), LABEL_FIELD
        ] = "TODO"

        utt_corpus.sort_values(["transcript_file", "utterance_order"], inplace=True)
        utt_corpus = utt_corpus[["id", "transcript_file", "speaker_code", "transcript_clean", LABEL_FIELD, "age"]]

        for transcript_id in utt_corpus.transcript_file.unique():
            out_file_path = os.path.join(args.out_dir, f"{transcript_id}.csv")
            utt_corpus[utt_corpus.transcript_file == transcript_id].to_csv(out_file_path, index=False)


def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--output-dir", type=str, default=DATA_DIR_PREPROCESSED_CHILDES_DB
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    load_and_save()
