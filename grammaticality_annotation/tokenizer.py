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
ERROR_LABELS_FIELD = "labels"
AGE_FIELD = "age"
FILE_ID_FIELD = "file_id"

VOCAB_SIZE = 10000


def train_tokenizer(path, train_data, add_eos_token=False):
    print("Training tokenizer..")
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    special_tokens = [TOKEN_PAD,TOKEN_SPEAKER_CAREGIVER,TOKEN_SPEAKER_CHILD]
    if add_eos_token:
        special_tokens.append(TOKEN_EOS)
    trainer = BpeTrainer(special_tokens=special_tokens, show_progress=True, vocab_size=VOCAB_SIZE)

    if isinstance(train_data, str) and os.path.isfile(train_data):
        tokenizer.train(files=[train_data], trainer=trainer)
    else:
        tokenizer.train_from_iterator(train_data[TEXT_FIELD], trainer=trainer, length=len(train_data))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tokenizer.save(path)
