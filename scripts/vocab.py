"""A cut-down word table (shrink.py's kept_ids.json) in the teacher's id space.

Restrict.encode(text) splits text as the teacher's tokenizer does, except that a word using a
dropped token is spelled instead with the fewest kept pieces (single characters and bytes are
always kept, so this never fails). Ids stay the ORIGINAL ids, all of them kept, so the teacher
reads the same sequence; lut maps them to the cut-down model's ids."""
import functools
import torch


class Restrict:
    def __init__(self, tok, keep):
        self.tok = tok; self.keep = set(keep)
        self.lut = torch.full((max(len(tok), max(keep) + 1),), -1, dtype=torch.long)
        self.lut[torch.tensor(keep)] = torch.arange(len(keep))
        self.ids = torch.tensor(keep)
        names = tok.convert_ids_to_tokens(keep)
        self.piece = {s: i for s, i in zip(names, keep) if s is not None and not s.startswith("<")}
        self.longest = max(len(s) for s in self.piece)
        self.byte = {int(s[3:5], 16): i for s, i in zip(names, keep) if s is not None and len(s) == 6 and s.startswith("<0x")}

    def encode(self, text):
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if all(i in self.keep for i in ids):
            return ids
        # group the teacher's tokens into words (a word starts at a token beginning with '▁');
        # a word holding a dropped token is spelled again from kept pieces
        out, word = [], []
        for i, s in zip(ids, self.tok.convert_ids_to_tokens(ids)):
            if word and s.startswith("▁"):
                out += self._word(tuple(word)); word = []
            word.append(i)
        return out + self._word(tuple(word))

    @functools.lru_cache(maxsize=200000)
    def _word(self, ids):
        if all(i in self.keep for i in ids):
            return list(ids)
        w = "".join(self.tok.convert_ids_to_tokens(list(ids)))
        # fewest kept pieces spelling w (dynamic programme over character positions); a
        # character with no piece of its own goes in as its UTF-8 bytes
        n = len(w); best = [None] * (n + 1); best[0] = ()
        for i in range(n):
            if best[i] is None:
                continue
            for j in range(i + 1, min(n, i + self.longest) + 1):
                p = self.piece.get(w[i:j])
                if p is not None and (best[j] is None or len(best[j]) > len(best[i]) + 1):
                    best[j] = best[i] + (p,)
            if w[i] not in self.piece and best[i + 1] is None:
                best[i + 1] = best[i] + tuple(self.byte[b] for b in w[i].encode("utf-8"))
        return list(best[n])

    def __call__(self, text, add_special_tokens=False):
        """Stands in for the tokenizer where training targets are made (targets.Utt)."""
        from transformers import BatchEncoding
        return BatchEncoding({"input_ids": self.encode(text)})
