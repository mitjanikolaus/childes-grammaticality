"""Load and store transcripts from childes-db."""
import argparse

import pandas as pd

from grammaticality_annotation.data import DATA_FILE_ALL_CHILDES_DB, transform_childes_db_transcripts
from grammaticality_manual_annotation.prepare_for_hand_annotation import ALL_EXCLUDED_CORPORA
from utils import SPEAKER_CODES_CAREGIVER, SPEAKER_CODE_CHILD

DB_VERSION = "2021.1"
DB_ARGS = None
# Change if you are using local db access:
# {
#     "hostname": "localhost",
#     "user": "childesdb",
#     "password": "tmp",
#     "db_name": "childes-db-version-0.1.2",
# }


def load():
    from childespy.childespy import get_utterances, get_corpora

    corpora = get_corpora()
    corpora = corpora[corpora.collection_name.isin(["Eng-NA", "Eng-UK"])]
    corpora = corpora[~corpora.corpus_name.isin(ALL_EXCLUDED_CORPORA)]

    data_childes_db = []

    for corpus in corpora.corpus_name.unique():
        print("\ncorpus: ", corpus)
        # transcripts = get_transcripts(corpus=corpus, db_args=DB_ARGS, db_version=DB_VERSION)
        utt_corpus = get_utterances(
            corpus=corpus, language="eng", db_args=DB_ARGS, db_version=DB_VERSION,
        )

        utt_corpus["gloss"] = utt_corpus["gloss"].apply(
            lambda x: x.replace("xxx", "").replace("www", "").replace("yyy", "").replace("  ", " ").strip())
        utt_corpus = utt_corpus[~utt_corpus.gloss.isin([""])]

        utt_corpus = utt_corpus[utt_corpus.speaker_code.isin(SPEAKER_CODES_CAREGIVER + [SPEAKER_CODE_CHILD])]
        utt_corpus = transform_childes_db_transcripts(utt_corpus)

        utt_corpus = utt_corpus[["id", "transcript_file", "speaker_code", "transcript_clean", "age"]]
        data_childes_db.append(utt_corpus)

    data_childes_db = pd.concat(data_childes_db, ignore_index=True)
    return data_childes_db


def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--output-path", type=str, default=DATA_FILE_ALL_CHILDES_DB
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    data = load()
    data.to_csv(args.output_path, index=False)
