"""What the model should have on screen after hearing the first T seconds of an utterance.

A word is due once the audio reaches its emit time

    emit = max(start + min(0.5 * dur, MIN_HEARD), end - lead)

so with lead = 0.12 a word of ordinary length is due 120 ms before the speaker finishes it,
and a very short word once half of it (at most 100 ms) has been heard. The target for a
prefix is every word that is due, spelled as in the transcript, followed by end-of-text.
Punctuation that trails the last due word is left off until the next word is due (or, for
the last word, until END_PAUSE of silence has followed it), so a comma never appears and
then disappears.

After end-of-text the target continues with the next `ghost` words of the transcript:
the guess for what is being said and what comes next. A recognizer that stops at
end-of-text never sees them; one that keeps decoding gets the grey ghost words."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from text import norm

MIN_HEARD = 0.10
END_PAUSE = 0.16
TRAIL = ",;:.?!\""


def pieces(row):
    """The transcript split at spaces, if that lines up one to one with the timed words."""
    ps = row["text"].split()
    if len(ps) != len(row["words"]) or any(norm(p) != w["w"] for p, w in zip(ps, row["words"])):
        return None
    return ps


def emit_times(words, lead):
    return [max(w["start"] + min(0.5 * (w["end"] - w["start"]), MIN_HEARD), w["end"] - lead) for w in words]


def n_due(emit, t):
    k = 0
    while k < len(emit) and emit[k] <= t + 1e-6:
        k += 1
    return k


class Utt:
    """Token ids of every word of one utterance, with and without trailing punctuation."""

    def __init__(self, row, tok, lead, cache=None):
        self.ps = pieces(row)
        self.ok = self.ps is not None
        if not self.ok:
            return
        cache = {} if cache is None else cache   # shared between utterances: most words repeat

        def enc(s):
            if s not in cache:
                cache[s] = tok(s, add_special_tokens=False).input_ids
            return cache[s]
        self.full = [enc(p) for p in self.ps]
        self.bare = [enc(p.rstrip(TRAIL)) for p in self.ps]
        self.emit = emit_times(row["words"], lead)
        self.end = row["words"][-1]["end"]

    def target(self, t, ghost=0):
        """(token ids on screen at time t, ghost token ids)."""
        n = len(self.ps); k = n_due(self.emit, t)
        if k == 0:
            heard = []
        elif k == n and t >= self.end + END_PAUSE - 1e-6:
            heard = [i for p in self.full for i in p]
        else:
            heard = [i for p in self.full[: k - 1] for i in p] + self.bare[k - 1]
        g = []
        if ghost:
            nxt = self.bare[k: k + ghost]
            g = [i for p in nxt for i in p]
        return heard, g, k
