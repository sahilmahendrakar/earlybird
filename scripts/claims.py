"""When can a voice agent act? Whole-command claims, a better confidence for them, and intents.

A claim is a moment where the model's guess ends in end-of-text: "screen + guess is the whole
command" (ghost_demo.py logs it with done_p). Three things here:

  claim confidence  done_p alone is poorly calibrated. A small logistic model rescored it from
                    features of the claim (done_p, the guess words' probabilities, how many words
                    are guessed rather than heard, how many 80 ms steps the same claim has
                    stood, how long since the screen last changed), fitted on slurp_devel and
                    applied to the test set: the agent acts on the first claim above a threshold.
  intent            a bag-of-words intent classifier (multinomial logistic regression, numpy),
                    trained on SLURP training sentences, says what the agent would do with a
                    text. A claim acts on the right intent when its intent equals the intent of
                    the model's own final transcript (and, reported too, the reference intent).
  outcome           per threshold: share of commands acted on before the speaker finished with
                    the right text, with the right intent, and with the wrong intent; median
                    time before the last word ended.

usage: claims.py GHOST_DIR SYSTEM SLURP_TRAIN_JSONL [--json OUT]
(SLURP_TRAIN_JSONL: dataset/slurp/train.jsonl from github.com/pswietojanski/slurp)"""
import argparse, collections, json, math, re
import numpy as np

THR = (0.5, 0.7, 0.8, 0.9, 0.95)


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", s.lower()).split())


# ---- intent classifier ----
def feats(text):
    w = norm(text).split()
    return w + [a + "_" + b for a, b in zip(w, w[1:])]


class Intent:
    def __init__(self, rows, epochs=30, lr=0.5, l2=1e-4):
        self.labels = sorted({r["intent"] for r in rows}); li = {l: i for i, l in enumerate(self.labels)}
        cnt = collections.Counter(f for r in rows for f in set(feats(r["sentence"])))
        self.vocab = {f: i for i, (f, c) in enumerate(cnt.items()) if c >= 2}
        self.vocab = {f: i for i, f in enumerate(self.vocab)}
        X = [self._x(r["sentence"]) for r in rows]; Y = np.array([li[r["intent"]] for r in rows])
        V, K = len(self.vocab), len(self.labels)
        self.W = np.zeros((V + 1, K))
        rng = np.random.default_rng(0)
        for ep in range(epochs):
            for i in rng.permutation(len(X)):
                idx = X[i]; z = self.W[idx].sum(0); p = np.exp(z - z.max()); p /= p.sum()
                p[Y[i]] -= 1
                self.W[idx] -= lr * (p + l2 * self.W[idx]) / max(1, len(idx)) ** 0.5
            lr *= 0.85

    def _x(self, text):
        return [self.vocab[f] for f in feats(text) if f in self.vocab] + [len(self.vocab)]

    def __call__(self, text):
        return self.labels[int(self.W[self._x(text)].sum(0).argmax())]


# ---- claims ----
def claim_text(e):
    return norm(e["screen"] + " " + e["guess"])


def claims(u):
    """Claim events of one utterance, each with its features."""
    out, prev, stood, last_change, last_screen = [], None, 0, 0.0, None
    for e in u["events"]:
        if e["screen"] != last_screen:
            last_change, last_screen = e["t"], e["screen"]
        if not e.get("eos") or not claim_text(e):
            prev, stood = None, 0
            continue
        c = claim_text(e)
        stood = stood + 1 if c == prev else 1; prev = c
        wp = e.get("wp") or []
        f = [math.log(max(e["done_p"], 1e-6)), math.log(1 - min(e["done_p"], 1 - 1e-6)),
             min(wp) if wp else 1.0, len(norm(e["guess"]).split()), min(stood, 6), min(e["t"] - last_change, 1.0),
             len(c.split())]
        out.append((e, c, f))
    return out


def fit_logistic(X, Y, iters=3000, lr=0.1):
    X = np.asarray(X, float); mu, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd
    w = np.zeros(Z.shape[1]); b = 0.0
    for _ in range(iters):
        q = 1 / (1 + np.exp(-(Z @ w + b)))
        w -= lr * (Z.T @ (q - Y) / len(Y) + 1e-3 * w); b -= lr * np.mean(q - Y)
    return lambda F: 1 / (1 + np.exp(-(((np.asarray(F, float) - mu) / sd) @ w + b)))


def outcome(utts, score, intent, gold, thr):
    n = len(utts); right_text = right_int = wrong_int = 0; leads = []
    for u in utts:
        final = norm(u["events"][-1]["screen"]) if u["events"] else ""
        for e, c, f in claims(u):
            if score(e, f) < thr:
                continue
            lead = u["end"] - e["t"]
            if lead <= 0:
                break   # first claim above threshold came after the speaker finished: not early
            if c == final:
                right_text += 1
            if intent(c) == intent(final):
                right_int += 1; leads.append(lead)
            else:
                wrong_int += 1
            break
    pct = lambda x: round(100 * x / n, 1)
    return dict(thr=thr, early_right_text=pct(right_text), early_right_intent=pct(right_int),
                early_wrong_intent=pct(wrong_int), lead_ms=int(1000 * np.median(leads)) if leads else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ghost"); ap.add_argument("system"); ap.add_argument("train"); ap.add_argument("--json")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.train)]
    intent = Intent(rows)
    load = lambda s: [json.loads(l) for l in open(f"{a.ghost}/{a.system}.{s}.jsonl")]
    dev, test = load("slurp_devel"), load("slurp_test")
    for name, us in (("devel", dev), ("test", test)):
        g = [u for u in us if u.get("intent")]
        acc = np.mean([intent(u["text"]) == u["intent"] for u in g])
        print(f"intent classifier on SLURP {name} references: {100 * acc:.1f}% right ({len(g)} commands)")
    X, Y = [], []
    for u in dev:
        final = norm(u["events"][-1]["screen"]) if u["events"] else ""
        for e, c, f in claims(u):
            X.append(f); Y.append(float(c == final))
    model = fit_logistic(X, np.array(Y))
    print(f"claim model fitted on {len(X)} devel claims ({100 * np.mean(Y):.0f}% right)")
    res = {}
    for label, score in (("raw done_p", lambda e, f: e["done_p"]), ("claim model", lambda e, f: float(model([f])[0]))):
        print(f"\n{label}: act on the first claim above the threshold, before the speaker finishes ({len(test)} test commands)")
        print("| act at ≥ | early, right text % | early, right intent % | early, wrong intent % | median ms before the end |")
        print("|---:|---:|---:|---:|---:|")
        res[label] = []
        for t in THR:
            r = outcome(test, score, intent, None, t); res[label].append(r)
            print(f"| {t} | {r['early_right_text']} | {r['early_right_intent']} | {r['early_wrong_intent']} | {r['lead_ms']} |")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
