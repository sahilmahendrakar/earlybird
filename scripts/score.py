"""Score partial logs from run_stream.py.

For every reference word:
  first   the first moment a partial shows the word correctly (in its right place)
  stable  the moment it is shown correctly and stays correct in every later partial
Both are reported as an offset from the word's acoustic end: negative means the word was
on screen before the speaker finished saying it.

"In its right place" is decided by aligning each partial against the reference with
edit distance, letting the partial stop anywhere in the reference (it is a prefix of the
utterance, not all of it). A reference word counts when the alignment pairs it with an
identical word.

Also: WER of the final transcript, and revisions per word = words taken back (the part
of the previous partial that is not a prefix of the new one), summed over all updates
and divided by the number of reference words.

Words without an "end" (utterances whose alignment does not cover the transcript, kept in
the *_full sets so WER is over every utterance) count for WER and revisions only. A set
with no results of its own is scored from the results on NAME_full, restricted to its ids.

usage: score.py [--md results/baselines.md]"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from text import norm


def matched(hyp, ref):
    """Reference indices paired with an identical hypothesis word, hypothesis may end early."""
    m, n = len(hyp), len(ref)
    if not m or not n:
        return []
    vocab = {w: i for i, w in enumerate(dict.fromkeys(hyp + ref))}
    h = np.array([vocab[w] for w in hyp]); r = np.array([vocab[w] for w in ref])
    ar = np.arange(n + 1)
    D = np.empty((m + 1, n + 1), dtype=np.int32); D[0] = ar
    for i in range(1, m + 1):
        y = np.empty(n + 1, dtype=np.int32); y[0] = i
        y[1:] = np.minimum(D[i - 1, :-1] + (r != h[i - 1]), D[i - 1, 1:] + 1)
        D[i] = np.minimum.accumulate(y - ar) + ar
    i, j, out = m, int(np.argmin(D[m])), []
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + (h[i - 1] != r[j - 1]):
            if h[i - 1] == r[j - 1]:
                out.append(j - 1)
            i -= 1; j -= 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def score_utt(log, ref_words):
    ref = [w["w"] for w in ref_words]; n = len(ref)
    first = [None] * n; stable = [None] * n
    prev, erased = [], 0
    for t, text in log["partials"]:
        hyp = norm(text).split()
        lcp = 0
        while lcp < min(len(prev), len(hyp)) and prev[lcp] == hyp[lcp]:
            lcp += 1
        erased += len(prev) - lcp; prev = hyp
        ok = set(matched(hyp, ref))
        for i in range(n):
            if i in ok:
                if first[i] is None: first[i] = t
                if stable[i] is None: stable[i] = t
            else:
                stable[i] = None
    ends = [w.get("end") for w in ref_words]   # no end: an untimed word (WER and revisions only)
    return ([None if f is None or e is None else f - e for f, e in zip(first, ends)],
            [None if s is None or e is None else s - e for s, e in zip(stable, ends)], erased)


def score_file(path, manifest):
    import jiwer
    from whisper.normalizers import EnglishTextNormalizer
    wn = EnglishTextNormalizer()
    first, stable, erased, nref, ntimed, refs, hyps = [], [], 0, 0, 0, [], []
    for line in open(path):
        log = json.loads(line)
        if log["id"] not in manifest:   # results of a larger set that contains this one
            continue
        r = manifest[log["id"]]
        f, s, e = score_utt(log, r["words"])
        first += f; stable += s; erased += e; nref += len(r["words"]); ntimed += sum("end" in w for w in r["words"])
        refs.append(wn(r["text"]) or "x"); hyps.append(wn(log["final"]))
    got = np.array([x for x in first if x is not None]) * 1e3
    stab = np.array([x for x in stable if x is not None]) * 1e3
    found = len(got) / ntimed * 100 if ntimed else float("nan")
    if not len(got):
        got = stab = np.array([float("nan")])
    return dict(utts=len(refs), words=nref, wer=jiwer.wer(refs, hyps) * 100,
                found=found, first_med=np.median(got), first_p95=np.percentile(got, 95),
                ahead=(got < 0).mean() * 100, stable_med=np.median(stab), stable_p95=np.percentile(stab, 95),
                rev=erased / nref)


def timing(path):
    steps, total, audio = [], 0.0, 0.0
    for line in open(path):
        log = json.loads(line)
        steps += log["step_ms"]; total += sum(log["step_ms"]) + log["finish_ms"]; audio += len(log["step_ms"]) * 80
    s = np.array(steps)
    return dict(step_mean=s.mean(), step_p95=np.percentile(s, 95), rtf=total / audio)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--md"); a = ap.parse_args()
    from recognizers import MODELS
    sets = [d for d in sorted(glob.glob("data/*/")) if os.path.exists(d + "manifest.jsonl")]
    out = []
    for d in sets:
        name = os.path.basename(d.rstrip("/"))
        manifest = {r["id"]: r for r in map(json.loads, open(d + "manifest.jsonl"))}
        rows = []
        for sysdir in sorted(glob.glob("results/raw/*/")):
            p = f"{sysdir}{name}.jsonl"
            if not os.path.exists(p):
                p = f"{sysdir}{name}_full.jsonl"   # a subset is scored from the full set's run
                if not os.path.exists(p):
                    continue
            m = score_file(p, manifest); m["sys"] = os.path.basename(sysdir.rstrip("/"))
            tp = f"results/raw/{m['sys']}/{name}.timing.jsonl"
            tp = tp if os.path.exists(tp) else f"results/raw/{m['sys']}/{name}_full.timing.jsonl"
            m.update(timing(tp) if os.path.exists(tp) else dict(step_mean=float("nan"), step_p95=float("nan"), rtf=float("nan")))
            rows.append(m)
        if not rows:
            continue
        out.append(f"### {name}\n")
        out.append("| System | Utterances | WER % | Words found % | First correct: median ms | p95 ms | Ahead of word end % | Stable: median ms | p95 ms | Revisions / word | Compute per 80 ms step: mean ms | p95 ms | Compute / audio |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for m in sorted(rows, key=lambda m: m["first_med"]):
            out.append(f"| {m['sys']} | {m['utts']} | {m['wer']:.1f} | {m['found']:.1f} | {m['first_med']:+.0f} | {m['first_p95']:+.0f} | "
                       f"{m['ahead']:.1f} | {m['stable_med']:+.0f} | {m['stable_p95']:+.0f} | {m['rev']:.2f} | "
                       f"{m['step_mean']:.1f} | {m['step_p95']:.1f} | {m['rtf']:.2f} |")
        out.append("")
    text = "\n".join(out)
    print(text)
    if a.md:
        open(a.md, "w").write(text)


if __name__ == "__main__":
    main()
