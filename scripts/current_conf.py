"""Confidence for the words on screen, not only the guesses: is the probability Earlybird gives
each transcript word, while that word is still being spoken, calibrated?

run      stream utterances through Engine and log, at every 80 ms step, the transcript words with
         a probability each (product of the word's token probabilities from the decoder pass that
         produced them). One JSON line per utterance: {id, steps: [[t, [[word, p], ...]], ...]}.
analyze  for every step and every transcript word whose reference word has not ended yet (so the
         word is shown early), with all the words before it right: is it the reference word?
         Reports how often it is right at each confidence, the calibration error (ECE, 10 bins),
         and the same for every transcript word and for the second guessed word (from ghost logs).

usage: current_conf.py run MODEL DATA_DIR OUT.jsonl [--limit 200] [--pad] [--shard i/n]
       current_conf.py analyze MANIFEST LOG.jsonl [LOG.jsonl ...] [--ghost GHOST.jsonl] [--json OUT]"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from text import norm

SR, CHUNK = 16000, 1280


def word_probs(eng):
    """[(normalized word, p)] for the transcript, grouping tokens into words."""
    toks, probs = eng.last["tokens"], eng.last["probs"]
    out, s = [], 0
    for i in range(1, len(toks) + 1):
        if i == len(toks) or eng._is_word_start(toks[i]):
            p = float(np.prod(probs[s:i]))
            for w in norm(eng.tok.decode(toks[s:i], skip_special_tokens=True)).split():
                out.append([w, round(p, 4)])
            s = i
    return out


def run(a):
    import soundfile as sf
    from engine import Engine
    eng = Engine(a.model, pad=a.pad, ghost=2)
    rows = [json.loads(l) for l in open(f"{a.data}/manifest.jsonl")]
    rows = [r for r in rows if all("end" in w for w in r["words"])]
    if a.ids:
        want = set(open(a.ids).read().split()); rows = [r for r in rows if r["id"] in want]
    else:
        step = max(1, len(rows) // a.limit) if a.limit else 1
        rows = rows[::step][: a.limit] if a.limit else rows
    i, n = map(int, a.shard.split("/"))
    rows = rows[i::n]
    with open(a.out, "w") as f:
        for r in rows:
            x, _ = sf.read(f"{a.data}/{r['audio']}", dtype="float32")
            x = np.concatenate([x, np.zeros(SR // 2, dtype=np.float32)])
            eng.start(); steps = []
            for k in range(0, len(x), CHUNK):
                eng.step(x[k: k + CHUNK])
                st = [round(min(k + CHUNK, len(x)) / SR, 3), word_probs(eng)]
                if a.guesses:   # also the guess after the transcript, one probability per guessed word
                    st.append([[w, round(p, 4)] for w, p in zip(norm(eng.ghost).split(), eng.ghost_word_p)])
                steps.append(st)
            f.write(json.dumps(dict(id=r["id"], steps=steps)) + "\n"); f.flush()


def ece(ps, ys, bins=10):
    ps, ys = np.asarray(ps), np.asarray(ys, float)
    if not len(ps):
        return float("nan"), []
    edges = np.linspace(0, 1, bins + 1); e = 0.0; rows = []
    for lo, hi in zip(edges, edges[1:]):
        m = (ps >= lo) & ((ps < hi) if hi < 1 else (ps <= hi))
        if m.any():
            e += m.mean() * abs(ps[m].mean() - ys[m].mean())
            rows.append((lo, hi, int(m.sum()), float(ps[m].mean()), float(ys[m].mean())))
    return 100 * e, rows


def analyze(a):
    man = {r["id"]: r for r in map(json.loads, open(a.manifest))}
    early_p, early_y, all_p, all_y = [], [], [], []
    for path in a.logs:
        for u in map(json.loads, open(path)):
            r = man.get(u["id"])
            if not r:
                continue
            ref = [norm(w["w"]) for w in r["words"]]; ends = [w["end"] for w in r["words"]]
            for st in u["steps"]:
                t, ws = st[0], st[1]
                for j, (w, p) in enumerate(ws):
                    if j >= len(ref) or any(ws[k][0] != ref[k] for k in range(j)):
                        break   # context wrong: what this word "should" be is undefined
                    y = w == ref[j]
                    all_p.append(p); all_y.append(y)
                    if ends[j] > t:
                        early_p.append(p); early_y.append(y)
    res = {}
    for name, ps, ys in (("words still being spoken", early_p, early_y), ("every transcript word", all_p, all_y)):
        e, rows = ece(ps, ys); res[name] = dict(n=len(ps), right=100 * float(np.mean(ys)) if ys else None, ece=e, bins=rows)
        print(f"\n{name}: {len(ps)} word-steps, {res[name]['right']:.1f}% right, calibration error {e:.1f} points")
        print("| confidence | word-steps | mean confidence | right % |\n|---|---:|---:|---:|")
        for lo, hi, n, mp, my in rows:
            print(f"| {lo:.1f}–{hi:.1f} | {n} | {100 * mp:.1f} | {100 * my:.1f} |")
    if a.ghost:
        p2, y2 = [], []
        for path in a.ghost:
            for u in map(json.loads, open(path)):
                r = man.get(u["id"])
                if not r:
                    continue
                ref = [norm(w["w"]) for w in r["words"]]
                for ev in u["events"]:
                    hyp = norm(ev["screen"]).split(); g = norm(ev["guess"]).split(); wp = ev.get("wp") or []
                    if hyp != ref[: len(hyp)] or len(g) < 2 or len(wp) < 2 or len(hyp) + 1 >= len(ref):
                        continue
                    if g[0] != ref[len(hyp)]:
                        continue   # second word judged only when the first guess is right
                    p2.append(wp[1]); y2.append(g[1] == ref[len(hyp) + 1])
        e, rows = ece(p2, y2)
        res["second guessed word"] = dict(n=len(p2), right=100 * float(np.mean(y2)) if y2 else None, ece=e, bins=rows)
        print(f"\nsecond guessed word (first one right): {len(p2)} guesses, calibration error {e:.1f} points")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("model"); r.add_argument("data"); r.add_argument("out")
    r.add_argument("--limit", type=int, default=200); r.add_argument("--pad", action="store_true"); r.add_argument("--shard", default="0/1")
    r.add_argument("--guesses", action="store_true", help="also log the guessed words with their probabilities")
    r.add_argument("--ids", help="file with utterance ids to run (one per line) instead of --limit")
    z = sub.add_parser("analyze"); z.add_argument("manifest"); z.add_argument("logs", nargs="+"); z.add_argument("--ghost", nargs="*"); z.add_argument("--json")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else analyze(a)
