import os

from tokenizers.pre_tokenizers import Whitespace

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from utils import PROJECT_ROOT_DIR

TOKENIZERS_DIR = PROJECT_ROOT_DIR+"/data/tokenizers/"

TOKEN_PAD = "[PAD]"
TOKEN_EOS = "[EOS]"
TOKEN_UNK = "[UNK]"
TOKEN_SEP = "[SEP]"
TOKEN_CLS = "[CLS]"
TOKEN_SPEAKER_CHILD = "[CHI]"
TOKEN_SPEAKER_CAREGIVER = "[CAR]"

TEXT_FIELD = "transcript"
LABEL_FIELD = "is_grammatical"
TRANSCRIPT_FIELD = "transcript_file"


def train_tokenizer(path, train_data):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=[TOKEN_PAD, TOKEN_UNK, TOKEN_EOS, TOKEN_SEP], show_progress=True, vocab_size=10000)

    if isinstance(train_data, str) and os.path.isfile(train_data):
        tokenizer.train(files=[train_data], trainer=trainer)
    else:
        tokenizer.train_from_iterator(train_data[TEXT_FIELD], trainer=trainer, length=len(train_data))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tokenizer.save(path)


def tokenize(datapoint, tokenizer):
    encoded = tokenizer.encode(datapoint[TEXT_FIELD])
    datapoint["encoded"] = encoded
    return datapoint

