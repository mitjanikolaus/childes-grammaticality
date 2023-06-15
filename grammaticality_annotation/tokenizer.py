import os
from pathlib import Path

import pandas as pd
from tokenizers.pre_tokenizers import Whitespace

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from utils import PROJECT_ROOT_DIR, SPEAKER_CODE_CHILD, SPEAKER_CODES_CAREGIVER

TOKENIZER_PATH = PROJECT_ROOT_DIR+"/data/tokenizer-childes.json"

TOKEN_PAD = "[PAD]"
TOKEN_EOS = "[EOS]"
TOKEN_UNK = "[UNK]"
TOKEN_SEP = "[SEP]"
TOKEN_SPEAKER_CHILD = "[CHI]"
TOKEN_SPEAKER_CAREGIVER = "[CAR]"

TEXT_FIELD = "transcript"
LABEL_FIELD = "is_grammatical"

LM_DATA = os.path.expanduser("~/data/childes_grammaticality/sentences.txt")
DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "manual_annotation", "all")


def speaker_code_to_speaker_token(code):
    if code == SPEAKER_CODE_CHILD:
        return TOKEN_SPEAKER_CHILD
    if code in SPEAKER_CODES_CAREGIVER:
        return TOKEN_SPEAKER_CAREGIVER
    raise RuntimeError("Unknown speaker code: ", code)


def prepare_data():
    print("Preparing data...")
    os.makedirs(os.path.dirname(LM_DATA), exist_ok=True)
    data = []
    for f in Path(DATA_DIR).glob("*.csv"):
        if os.path.isfile(f):
            data.append(pd.read_csv(f, index_col=0))

    data = pd.concat(data, ignore_index=True)
    data["speaker_code"] = data.speaker_code.apply(speaker_code_to_speaker_token)
    sentences = data.apply(lambda row: row.speaker_code + row.transcript_clean + TOKEN_EOS, axis=1).values
    with open(LM_DATA, 'w') as f:
        f.write("\n".join(sentences))


def train_tokenizer():
    print("Training tokenizer...")
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=[TOKEN_PAD, TOKEN_UNK, TOKEN_EOS, TOKEN_SEP], show_progress=True, vocab_size=10000)
    tokenizer.train(files=[LM_DATA], trainer=trainer)

    tokenizer.save(TOKENIZER_PATH)


def tokenize(datapoint, tokenizer):
    encoded = tokenizer.encode(datapoint[TEXT_FIELD])
    datapoint["encoded"] = encoded
    return datapoint

