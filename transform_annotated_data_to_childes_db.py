"""Load and store transcripts from childes-db."""
import argparse
import math
import pickle

from tqdm import tqdm

import pandas as pd

from grammaticality_annotation.data import load_childes_data, DATA_PATH_CHILDES_ANNOTATED, \
    DATA_PATH_CHILDES_DB_ANNOTATED, DATA_PATH_CHILDES_ANNOTATED_FIXES_FOR_CHILDES_DB
from preprocess import get_pos_tag
from utils import POS_PUNCTUATION

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
    # else:
        # print(utt1)
    return False


def transform(args):
    # from childespy.childespy import get_transcripts, get_utterances
    data_annotated = load_childes_data(DATA_PATH_CHILDES_ANNOTATED_FIXES_FOR_CHILDES_DB)
    data_annotated["corpus"] = data_annotated.transcript_file.apply(lambda x: x.split('/')[0])
    data_annotated["transcript_file_name"] = data_annotated.transcript_file.apply(lambda x: "/".join(x.split('/')[1:]))
    data_annotated["transcript_no_punct"] = [u[:-1] for u in data_annotated.transcript_clean.values]

    for corpus in data_annotated.corpus.unique():
        print("\ncorpus: ", corpus)
        # transcripts = get_transcripts(corpus=corpus, db_args=DB_ARGS, db_version="2021.1")
        # utt_corpus = get_utterances(
        #     corpus=corpus, language="eng", db_args=DB_ARGS, db_version="2021.1",
        # )
        transcripts = pickle.load(open(f"corpora/transcripts_{corpus}.pickle", "rb"))
        utt_corpus = pickle.load(open(f"corpora/{corpus}.pickle", "rb")) #TODO

        data_annotated_corpus = data_annotated[data_annotated.corpus == corpus]
        for transcript_file_name in data_annotated_corpus.transcript_file_name.unique():
            print(transcript_file_name)
            transcript_ids = transcripts[transcripts.filename.str.contains(f"{corpus}/{transcript_file_name.replace('.cha', '')}")].transcript_id.values
            if not len(transcript_ids) == 1:
                print(len(transcript_ids))
            transcript_id = transcript_ids[0]
            utts_transcript = utt_corpus[utt_corpus.transcript_id == transcript_id].copy()
            utts_transcript["gloss"] = utts_transcript["gloss"].apply(lambda x: x.replace("xxx", "").replace("www", "").replace("yyy", "").replace("  ", "").strip())
            utts_transcript = utts_transcript[~utts_transcript.gloss.isin([""])]

            data_annotated_transcript = data_annotated_corpus[data_annotated_corpus.transcript_file_name == transcript_file_name]
            # first_3_utts = data_annotated_transcript.transcript_no_punct.values[:3]
            # first_3_utts = "\n".join(first_3_utts)
            #
            # start_index = 0
            # while start_index < len(utts_transcript)-2:
            #     first_3_utts_childes_db = "\n".join(utts_transcript.gloss.values[start_index:start_index+3])
            #     if first_3_utts_childes_db == first_3_utts:
            #         print("start index: ", start_index)
            #         break
            #     start_index+=1
            # end_index=start_index+len(data_annotated_transcript)

            utterances_selection = utts_transcript.iloc[:len(data_annotated_transcript)]

            #
            # if transcript_file_name == "fatsib/fatsibcharles.cha":
            #     for i in range(0, len(data_annotated_transcript)):
            #         if not approx_match(utterances_selection.gloss.values[i], data_annotated_transcript.transcript_no_punct.values[i]):
            #             print(f"mismatch: {i}")
            #             print(utterances_selection.gloss.values[i])
            #             print(data_annotated_transcript.transcript_no_punct.values[i])
            #             print("")

            assert len(utterances_selection) == len(data_annotated_transcript)

            for i in range(min(len(data_annotated_transcript), 5)):
                if not approx_match(utterances_selection.gloss.values[i], data_annotated_transcript.transcript_no_punct.values[i]):
                    print("mismatch:")
                    print(utterances_selection.gloss.values[i])
                    print(data_annotated_transcript.transcript_no_punct.values[i])
                    print("")
                if not approx_match(utterances_selection.gloss.values[-i], data_annotated_transcript.transcript_no_punct.values[-i]):
                    print("mismatch:")
                    print(utterances_selection.gloss.values[-i])
                    print(data_annotated_transcript.transcript_no_punct.values[-i])
                    print("")

            #TODO
        # for _, transcript in tqdm(transcripts.iterrows(), total=len(transcripts)):
        #
        #     # Make sure we know the age of the child
        #     if not math.isnan(transcript["target_child_age"]):
        #
        #         # Filter utterances for current transcript
        #         utts_transcript = utterances.loc[
        #             (utterances["transcript_id"] == transcript["transcript_id"])
        #         ]
        #
        #         if len(utts_transcript) > 0:
        #             utts_transcript = utts_transcript.sort_values(
        #                 by=["utterance_order"]
        #             )
        #             for _, utt in utts_transcript.iterrows():
        #                 tokenized_utterance = utt["gloss"].split(" ")
        #                 if "punctuation" in utt.keys():
        #                     tokenized_utterance += [utt["punctuation"]]
        #                 else:
        #                     tokenized_utterance = add_punctuation(tokenized_utterance, utt["type"])
        #                 pos = [get_pos_tag(p) for p in utt["part_of_speech"].split(" ") if p not in POS_PUNCTUATION]
        #                 speaker_code = parse_speaker_role(utt["speaker_role"])
        #                 data.append(
        #                     {
        #                         "utterance_id": utt["id"],
        #                         "transcript_id": transcript["transcript_id"],
        #                         "corpus_id": transcript["corpus_id"],
        #                         "child_id": utt["target_child_id"],
        #                         "age": round(transcript["target_child_age"]),
        #                         "tokens": tokenized_utterance,
        #                         "pos": pos,
        #                         "speaker_code": speaker_code,
        #                     }
        #                 )

    return pd.DataFrame(data)


def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--output-path", type=str, default=DATA_PATH_CHILDES_DB_ANNOTATED
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    data = transform(args)

    data.to_pickle(args.output_path)
