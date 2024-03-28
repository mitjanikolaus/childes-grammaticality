"""Load and store transcripts from childes-db."""
import argparse

import pandas as pd

from grammaticality_annotation.data import load_childes_data, \
    DATA_FILE_ANNOTATED_CHILDES_DB, DATA_PATH_CHILDES_ANNOTATED_FIXES_FOR_CHILDES_DB

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


CHILD = "CHI"
ADULT = "ADU"


def parse_speaker_role(role):
    if role == "Target_Child":
        return CHILD
    else:
        return ADULT


def add_punctuation(tokens, utterance_type):
    if utterance_type in TYPES_QUESTION:
        tokens += "?"
    elif utterance_type in TYPES_EXCLAMATION:
        tokens += "!"
    elif utterance_type in TYPES_STATEMENT:
        tokens += "."
    else:
        print("Unknown utterance type: ", utterance_type)
        tokens += "."

    return tokens


def approx_match(utt1, utt2):
    if utt1.replace("_", "").replace(" ", "") == utt2.replace(",", "").replace(" ", ""):
        return True
    return False


def transform():
    from childespy.childespy import get_transcripts, get_utterances

    data_annotated = load_childes_data(DATA_PATH_CHILDES_ANNOTATED_FIXES_FOR_CHILDES_DB)
    data_annotated["corpus"] = data_annotated.transcript_file.apply(lambda x: x.split('/')[0])
    data_annotated["transcript_file_name"] = data_annotated.transcript_file.apply(lambda x: "/".join(x.split('/')[1:]))
    data_annotated["transcript_no_punct"] = [u[:-1] for u in data_annotated.transcript_clean.values]

    data_annotated_childes_db = []

    for corpus in data_annotated.corpus.unique():
        print("\ncorpus: ", corpus)
        transcripts = get_transcripts(corpus=corpus, db_args=DB_ARGS, db_version="2021.1")
        utt_corpus = get_utterances(
            corpus=corpus, language="eng", db_args=DB_ARGS, db_version="2021.1",
        )

        data_annotated_corpus = data_annotated[data_annotated.corpus == corpus]
        for transcript_file_name in data_annotated_corpus.transcript_file_name.unique():
            print(transcript_file_name)
            transcript_ids = transcripts[transcripts.filename.str.contains(f"{corpus}/{transcript_file_name.replace('.cha', '')}")].transcript_id.values
            if not len(transcript_ids) == 1:
                print(len(transcript_ids))
                raise RuntimeError("Transcript id error")
            transcript_id = transcript_ids[0]

            utts_transcript = utt_corpus[utt_corpus.transcript_id == transcript_id].copy()
            utts_transcript["gloss"] = utts_transcript["gloss"].apply(lambda x: x.replace("xxx", "").replace("www", "").replace("yyy", "").replace("  ", " ").strip())
            utts_transcript = utts_transcript[~utts_transcript.gloss.isin([""])]

            data_annotated_transcript = data_annotated_corpus[data_annotated_corpus.transcript_file_name == transcript_file_name]

            utterances_selection = utts_transcript.iloc[:len(data_annotated_transcript)].copy()

            assert len(utterances_selection) == len(data_annotated_transcript)

            utterances_selection["is_grammatical"] = data_annotated_transcript["is_grammatical"].values
            utterances_selection["labels"] = data_annotated_transcript["labels"].values
            data_annotated_childes_db.append(utterances_selection)

    data_annotated_childes_db = pd.concat(data_annotated_childes_db, ignore_index=True)
    return data_annotated_childes_db


def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--output-path", type=str, default=DATA_FILE_ANNOTATED_CHILDES_DB
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    data = transform()
    data.to_csv(args.output_path, index=False)
