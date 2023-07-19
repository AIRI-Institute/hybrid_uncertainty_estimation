# adapted from https://github.com/HanXudong/fairlib/blob/main/data/src/Moji/encode_text.ipynb
# used for texts encoding with TorchMoji model
import json
import numpy as np
from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_feature_encoding

# from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
import logging

log = logging.getLogger(__name__)

TEST_SENTENCES = [
    "I love mom's cooking",
    "I love how you never reply back..",
    "I love cruising with my homies",
    "I love messing with yo mind!!",
    "I love you and now you're just gone..",
    "This is shit",
    "This is the shit",
]


def encode_moji(texts, config):
    # load tokenizer
    vocab_path = config.data.get("moji_vocab_path", None)
    pretrained_path = config.data.get("moji_pretrained_path", None)
    if vocab_path is None or pretrained_path is None:
        log.info("Path to TorchMoji model didn't specified!")
    log.info("Tokenizing using dictionary from {}".format(vocab_path))
    with open(vocab_path, "r") as f:
        vocabulary = json.load(f)
    st = SentenceTokenizer(vocabulary, 32)
    # texts = TEST_SENTENCES
    # tokenizing
    tokenized_texts, _, _ = st.tokenize_sentences(texts)
    # load model
    log.info("Loading model from {}.".format(pretrained_path))
    model = torchmoji_feature_encoding(pretrained_path)
    log.info("Encoding texts..")
    # print(np.stack(np.array(tokenized_texts).tolist(), axis=1).T)
    encoding = model(tokenized_texts)
    # encoding = model(np.stack(np.array(tokenized_texts).tolist(), axis=1).T)
    log.info("Texts encoded")
    return encoding
