"""ctypes wrapper so the evaluation harness can run the deployable INT8 C engine."""
import ctypes, os
import numpy as np
import torch

LIB = os.path.join(os.path.dirname(__file__), "..", "esp32", "engine", "libmicrocoder.so")


class CEngine:
    def __init__(self, model_bin, cfg, tok_bin=None):
        self.lib = ctypes.CDLL(LIB)
        L = self.lib
        L.mch_open.restype = ctypes.c_void_p
        L.mch_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        L.mch_forward.restype = ctypes.POINTER(ctypes.c_float)
        L.mch_forward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        L.mch_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        for f in ("mch_weight_bytes", "mch_state_bytes"):
            getattr(L, f).restype = ctypes.c_long
            getattr(L, f).argtypes = [ctypes.c_void_p]
        self.h = L.mch_open(model_bin.encode(), tok_bin.encode() if tok_bin else None)
        assert self.h, "failed to load model blob"
        self.c = cfg
        self.V = cfg.vocab

    def logits(self, tok, pos):
        p = self.lib.mch_forward(self.h, tok, pos)
        return np.ctypeslib.as_array(p, shape=(self.V,)).copy()

    def encode(self, s, max_ids=4096):
        buf = (ctypes.c_int * max_ids)()
        n = self.lib.mch_encode(self.h, s.encode(), buf, max_ids)
        return list(buf[:n])

    def generate(self, ids, max_new, stop_id, temperature=0.0, top_k=0, gen=None):
        ids = ids[-self.c.ctx:]
        for pos, t in enumerate(ids):
            lg = self.logits(t, pos)
        pos, out = len(ids), []
        for _ in range(max_new):
            if temperature <= 0:
                nxt = int(lg.argmax())
            else:
                p = torch.softmax(torch.from_numpy(lg) / temperature, -1)
                nxt = int(torch.multinomial(p, 1, generator=gen))
            if nxt == stop_id:
                break
            out.append(nxt)
            if pos >= self.c.ctx:
                break
            lg = self.logits(nxt, pos)
            pos += 1
        return out
