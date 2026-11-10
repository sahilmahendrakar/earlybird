"""What a voice agent gets from guesses: precision against earliness, calibration, and when
the whole command is known. Reads ghost_demo.py logs (results/ghost/NAME.SET.jsonl).

Per system and test set:
  next word   for each confidence threshold: share of moments a guess is shown, how often it
              is right, and how long before the word ends a right guess was on screen.
  calibration expected calibration error of the first guessed word's probability, raw and
              after Platt scaling fitted on the system's slurp_devel log (so test sets never
              fit their own calibration).
  completion  the model claims the utterance is complete when its guess ends in end-of-text.
              For each threshold on that claim's probability: share of utterances whose first
              claim above it is right, how long before the last word ends that right claim
              came, and share of utterances with a wrong claim above it first (an agent would
              have acted on the wrong command). Right = matches the model's own final transcript
              (recognition errors are the recognizer's, not the anticipation's); also reported
              against the reference.

usage: analyze_ghost.py DIR [--systems antic_base antic_slurp qwen-on-p29] [--json OUT.json]"""
import argparse, glob, json, math, os
import numpy as np

THR = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95)


def load(path):
    return [json.loads(l) for l in open(path)]


def judged(utts):
    return [e for u in utts for e in u["events"] if e["ctx_ok"] and e.get("next") and e["guess"]]


def ece(P, Y, bins=10):
    b = np.minimum((P * bins).astype(int), bins - 1)
    return sum(abs(Y[b == k].mean() - P[b == k].mean()) * (b == k).sum() for k in range(bins) if (b == k).any()) / len(P)


def platt_fit(P, Y, iters=500, lr=0.5):
    """Logistic regression of the outcome on logit(p): p' = sigmoid(a * logit(p) + c)."""
    x = np.log(np.clip(P, 1e-6, 1 - 1e-6) / (1 - np.clip(P, 1e-6, 1 - 1e-6)))
    a, c = 1.0, 0.0
    for _ in range(iters):
        q = 1 / (1 + np.exp(-(a * x + c)))
        a -= lr * np.mean((q - Y) * x); c -= lr * np.mean(q - Y)
    return a, c


def platt(P, ac):
    a, c = ac
    x = np.log(np.clip(P, 1e-6, 1 - 1e-6) / (1 - np.clip(P, 1e-6, 1 - 1e-6)))
    return 1 / (1 + np.exp(-(a * x + c)))


def next_word(utts, ac=None):
    ev = judged(utts)
    n_all = sum(1 for u in utts for e in u["events"] if e["ctx_ok"] and e.get("next"))
    if not ev:
        return {}
    P = np.array([e["p"] for e in ev]); Y = np.array([e["ok"] for e in ev], float)
    Pc = platt(P, ac) if ac else P
    rows = []
    for t in THR:
        s = Pc >= t
        if not s.any():
            continue
        ok = s & (Y == 1)
        lead = [ev[i]["lead_ms"] for i in np.flatnonzero(ok)]
        rows.append(dict(thr=t, shown=round(100 * s.sum() / n_all, 1), precision=round(100 * Y[s].mean(), 1),
                         lead_ms=int(np.median(lead)) if lead else None))
    out = dict(guesses=len(ev), rows=rows, ece_raw=round(100 * ece(P, Y), 1))
    if ac:
        out["ece_platt"] = round(100 * ece(Pc, Y), 1)
    return out


def claim_text(e):
    return norm(e["screen"] + " " + e["guess"])


def final_text(u):
    return norm(u["events"][-1]["screen"]) if u["events"] else ""


def norm(s):
    return " ".join("".join(c if c.isalnum() or c in " '" else " " for c in s.lower()).split())


def completion(utts, ac=None, vs_final=True):
    """First completion claim (end-of-text after the guess) above each threshold, per utterance.
    vs_final: a claim is right when it matches the model's own final transcript (what an agent
    would otherwise act on, so recognition errors are not counted against anticipation);
    otherwise it must match the reference."""
    rows = []
    for t in THR:
        right = wrong = 0; leads = []
        for u in utts:
            for e in u["events"]:
                if not e.get("eos") or not (e["screen"].strip() or e["guess"].strip()):
                    continue   # a claim that nothing at all was said is no command to act on
                p = e["done_p"]
                if ac:
                    p = float(platt(np.array([p]), ac)[0])
                if p < t:
                    continue
                if (claim_text(e) == final_text(u)) if vs_final else e["done_ok"]:
                    right += 1; leads.append(round((u["end"] - e["t"]) * 1000))
                else:
                    wrong += 1
                break
        n = len(utts)
        rows.append(dict(thr=t, right=round(100 * right / n, 1), wrong=round(100 * wrong / n, 1),
                         lead_ms=int(np.median(leads)) if leads else None,
                         early=round(100 * sum(1 for l in leads if l > 0) / n, 1)))
    return rows


def per_word(utts, ac=None):
    """Per spoken word (after the first of each utterance): was it guessed right, at or above the
    threshold, before it ended, and how early was the first such guess? A word is identified by
    its start time (t + before_start_ms), the same across all guesses aimed at it."""
    total = sum(max(0, len(u["text"].split()) - 1) for u in utts)
    rows = []
    for t in THR:
        first = {}
        for u in utts:
            for e in u["events"]:
                if not (e["ctx_ok"] and e.get("next") and e["guess"] and e["ok"]):
                    continue
                p = float(platt(np.array([e["p"]]), ac)[0]) if ac else e["p"]
                if p < t or e["lead_ms"] <= 0:
                    continue
                key = (u["id"], round(e["t"] * 1000 + e["before_start_ms"]))
                first[key] = max(first.get(key, -1), e["lead_ms"])
        leads = sorted(first.values())
        rows.append(dict(thr=t, words=round(100 * len(leads) / max(1, total), 1),
                         lead_ms=int(np.median(leads)) if leads else None))
    return rows


def completion_fit(utts):
    ev = [(e, u) for u in utts for e in u["events"] if e.get("eos") and (e["screen"].strip() or e["guess"].strip())]
    P = [e["done_p"] for e, _ in ev]; Y = [float(claim_text(e) == final_text(u)) for e, u in ev]
    return platt_fit(np.array(P), np.array(Y)) if len(P) > 50 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("--systems", nargs="*", default=["antic_base", "antic_slurp", "qwen-on-p29"])
    ap.add_argument("--json")
    a = ap.parse_args()
    out = {}
    for s in a.systems:
        dev = f"{a.dir}/{s}.slurp_devel.jsonl"
        ac = cc = None
        if os.path.exists(dev):
            d = load(dev); ev = judged(d)
            if len(ev) > 50:
                ac = platt_fit(np.array([e["p"] for e in ev]), np.array([e["ok"] for e in ev], float))
            cc = completion_fit(d)
        for f in sorted(glob.glob(f"{a.dir}/{s}.*.jsonl")):
            ds = f[len(f"{a.dir}/{s}."):-len(".jsonl")]
            if ds == "slurp_devel":
                continue
            u = load(f)
            r = dict(utterances=len(u), next=next_word(u, ac), words=per_word(u))
            if any(e.get("eos") for x in u for e in x["events"]):
                r["completion_raw"] = completion(u); r["completion_platt"] = completion(u, cc) if cc else None
                r["completion_vs_reference"] = completion(u, vs_final=False)
            out[f"{s}/{ds}"] = r
            print(f"\n## {s} on {ds} ({len(u)} utterances)")
            nw = r["next"]
            if nw:
                print(f"next word: {nw['guesses']} guesses; calibration error {nw['ece_raw']} points raw"
                      + (f", {nw['ece_platt']} after Platt (fit on slurp_devel)" if "ece_platt" in nw else ""))
                print("| confidence ≥ | shown % | right % | right guess on screen before word ends, ms |")
                print("|---:|---:|---:|---:|")
                for x in nw["rows"]:
                    print(f"| {x['thr']} | {x['shown']} | {x['precision']} | {x['lead_ms']} |")
            print("words guessed right before they ended (raw confidence ≥ threshold), and how early the first right guess was")
            print("| confidence ≥ | words % | median, ms before the word ends |")
            print("|---:|---:|---:|")
            for x in r["words"]:
                print(f"| {x['thr']} | {x['words']} | {x['lead_ms']} |")
            for key in ("completion_raw", "completion_platt", "completion_vs_reference"):
                if r.get(key):
                    print(f"whole utterance claimed ({key.split('_')[1]} confidence): first claim above threshold")
                    print("| confidence ≥ | right % of utterances | wrong % | known before the last word ends % | median lead, ms |")
                    print("|---:|---:|---:|---:|---:|")
                    for x in r[key]:
                        print(f"| {x['thr']} | {x['right']} | {x['wrong']} | {x['early']} | {x['lead_ms']} |")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
