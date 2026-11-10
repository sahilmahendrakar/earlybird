"""Word start and end times from a forced alignment of a known transcript.

A wav2vec 2.0 CTC model (torchaudio's WAV2VEC2_ASR_BASE_960H, MIT licence, 20 ms frames)
scores every letter at every frame; the best path through the transcript's letters gives
each letter its frames. A word starts at the first frame of its first letter and ends
after the last frame of its last letter. check_boundaries.py measures how far these times
sit from Kokoro's own and from Montreal Forced Aligner's.

Measured that way (150 utterances each, 2026-10-07), the raw times are off by a steady
amount: starts 73 ms late against Kokoro and 59 ms against MFA, ends 4 to 8 ms early
against Kokoro and 26 ms against MFA. calibrated() removes START_SHIFT and END_SHIFT."""
import torch, torchaudio

START_SHIFT, END_SHIFT = -0.065, 0.015


def calibrated(times):
    """Raw word_times() output as manifest words' (start, end): shifted, kept in order."""
    out, prev = [], 0.0
    for s, e, _ in times:
        s = max(prev, s + START_SHIFT); e = max(s + 0.02, e + END_SHIFT)
        out.append((round(s, 3), round(e, 3))); prev = s
    return out


class Aligner:
    def __init__(self, device="cpu"):
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = bundle.get_model().to(device).eval(); self.device = device
        self.idx = {c: i for i, c in enumerate(bundle.get_labels())}

    @torch.no_grad()
    def word_times(self, x, words):
        """[(start, end, score)] in seconds for normalized `words` over 16 kHz audio `x`;
        score is the mean probability of the word's letters. None if a word cannot be spelled
        with the model's letters or the audio is too short for the transcript."""
        words = [w.upper() for w in words]
        if not words or any(c not in self.idx for w in words for c in w):
            return None
        em = torch.log_softmax(self.model(torch.as_tensor(x, device=self.device)[None])[0], -1)
        toks = [self.idx[c] for c in "|".join(words)]
        try:
            ali, sc = torchaudio.functional.forced_align(em.cpu(), torch.tensor([toks]), blank=0)
        except RuntimeError:
            return None
        ali, sc = ali[0].tolist(), sc[0].exp().tolist(); sec = len(x) / 16000 / em.shape[1]
        first, last, tot, k, prev = {}, {}, {}, -1, 0   # per letter: first frame, last frame, summed probability
        for t, a in enumerate(ali):
            if a != 0 and a != prev:
                k += 1; first[k] = t; tot[k] = [0.0, 0]
            if a != 0:
                last[k] = t; tot[k][0] += sc[t]; tot[k][1] += 1
            prev = a
        out, pos = [], 0
        for w in words:
            ls = range(pos, pos + len(w))
            out.append((first[pos] * sec, (last[pos + len(w) - 1] + 1) * sec,
                        sum(tot[l][0] for l in ls) / sum(tot[l][1] for l in ls)))
            pos += len(w) + 1
        return out
