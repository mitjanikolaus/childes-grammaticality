import argparse
import os

import numpy as np
import pandas as pd

from utils import (
    SPEAKER_CODE_CHILD, get_num_unique_words,
    SPEAKER_CODES_CAREGIVER, ANNOTATED_UTTERANCES_FILE
)

from tqdm import tqdm
tqdm.pandas()


TOKEN_CHILD = "[CHI]"
TOKEN_CAREGIVER = "[CAR]"
TOKEN_OTHER = "[OTH]"


def speaker_code_to_special_token(code):
    if code == SPEAKER_CODE_CHILD:
        return TOKEN_CHILD
    elif code in SPEAKER_CODES_CAREGIVER:
        return TOKEN_CAREGIVER
    else:
        return TOKEN_OTHER


def prepare(args):
    utterances = pd.read_csv(args.utterances_file, index_col=0, dtype={"error": object})

    utterances = utterances.iloc[args.num_utts_ignore:]

    utterances = utterances[utterances.corpus == "Providence"].copy()
    transcripts = utterances.transcript_file.unique()
    np.random.seed(4)
    transcript = np.random.choice(transcripts)
    utts_transcript = utterances[utterances.transcript_file == transcript].copy()
    utts_transcript_child = utts_transcript[utts_transcript.speaker_code == SPEAKER_CODE_CHILD]
    while (utts_transcript.age.min() < 36) or (utts_transcript.age.min() > 40) or len(utts_transcript_child) < 10:
        transcript = np.random.choice(transcripts)
        utts_transcript = utterances[utterances.transcript_file == transcript].copy()
        utts_transcript_child = utts_transcript[utts_transcript.speaker_code == SPEAKER_CODE_CHILD]

    print("Transcript: ", transcript)
    print("Num child utts: ", len(utts_transcript_child))

    utts_transcript["utterance"] = utts_transcript["speaker_code"].apply(speaker_code_to_special_token) + " " + utts_transcript["transcript_clean"]

    utts_transcript["is_grammatical"] = ""
    utts_transcript["labels"] = ""
    utts_transcript["note"] = ""

    utts_transcript["num_unique_words"] = get_num_unique_words(utts_transcript.transcript_clean)
    utts_transcript.loc[(utts_transcript.speaker_code == SPEAKER_CODE_CHILD) & (utts_transcript.num_unique_words > 1) & (utts_transcript.is_speech_related == True), "is_grammatical"] = "TODO"
    print("Num utts to annotate: ", len(utts_transcript[(utts_transcript.speaker_code == SPEAKER_CODE_CHILD) & (utts_transcript.num_unique_words > 1) & (utts_transcript.is_speech_related == True)]))
    utts_transcript = utts_transcript[["utterance", "is_grammatical", "labels", "note"]]

    transcript_name = transcript.replace("/", "_")
    return utts_transcript, transcript_name


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--utterances-file",
        type=str,
        default=ANNOTATED_UTTERANCES_FILE,
    )
    argparser.add_argument(
        "--num-utts-ignore",
        type=int,
        default=0,
        help="First x utts to ignore (already annotated)"
    )
    argparser.add_argument(
        "--num-utts",
        type=int,
        default=None,
        help="Number of utts to annotate"
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    utterances, transcript_name = prepare(args)

    base_path = os.path.expanduser("~/data/communicative_feedback/annotations_")
    file_name = base_path + transcript_name.replace(".cha", ".csv")
    utterances.to_csv(file_name)
