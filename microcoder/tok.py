"""Small byte-level BPE tokenizer trained on the training split only."""
import json

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

SPECIALS = ["<pad>", "<q>", "<c>", "<b>", "<r>", "<e>"]


def train_tokenizer(texts, vocab_size, path):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), min_frequency=2)
    tok.train_from_iterator(texts, tr)
    tok.save(path)
    return tok


class Tok:
    def __init__(self, path):
        self.t = Tokenizer.from_file(path)
        self.sp = {s: self.t.token_to_id(s) for s in SPECIALS}
        self.vocab = self.t.get_vocab_size()

    def encode(self, s):
        return self.t.encode(s).ids

    def decode(self, ids):
        return self.t.decode(ids, skip_special_tokens=False)


def corpus_texts(gen_rows, rep_rows):
    from .dataset import gen_text, repair_text
    return [gen_text(r["instr"], r["code"]) for r in gen_rows] + \
           [repair_text(r["instr"], r["broken"], r["err"], r["code"]) for r in rep_rows]
