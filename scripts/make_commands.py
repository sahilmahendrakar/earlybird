"""New spoken-command texts for synthesis: SLURP training sentences with their slots refilled.

Each SLURP training annotation ("wake me up at [time : five am] [date : this week]") is a
template; every slot is replaced by a value of the same slot type drawn from the training
annotations (of the same intent when it has at least 5 distinct values, else of any), so "wake me up at [time : half past six] [date : tomorrow]" and the like. Only
the training split is used, and any sentence that equals a test or devel sentence is
dropped, so nothing synthesized can leak into the SLURP test set. The original training
sentences are included too. Writes one sentence per line, plus a .jsonl with the intent.

usage: make_commands.py SLURP_DIR OUT.txt [--n 30000] [--seed 0]
SLURP_DIR holds train.jsonl, devel.jsonl, test.jsonl from github.com/pswietojanski/slurp
(dataset/slurp; CC BY 4.0)."""
import argparse, collections, json, random, re

SLOT = re.compile(r"\[(\w+) : ([^\]]+)\]")


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", s.lower()).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slurp"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=30000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = [json.loads(l) for l in open(f"{a.slurp}/train.jsonl")]
    held = {norm(json.loads(l)["sentence"]) for s in ("test", "devel") for l in open(f"{a.slurp}/{s}.jsonl")}
    pool = collections.defaultdict(list); ipool = collections.defaultdict(list)
    for r in rows:
        for typ, val in SLOT.findall(r["sentence_annotation"]):
            pool[typ].append(val.strip()); ipool[(r["intent"], typ)].append(val.strip())

    def value(intent, typ):   # a value seen with the same intent when there are enough, so the sentence stays sensible
        p = ipool[(intent, typ)]
        return rng.choice(p if len(set(p)) >= 5 else pool[typ])
    templates = [r for r in rows if SLOT.search(r["sentence_annotation"])]
    out = {}
    for r in rows:   # the training sentences as they are
        t = norm(r["sentence"])
        if t and t not in held:
            out[t] = r["intent"]
    tries = 0
    while len(out) < a.n and tries < 20 * a.n:
        tries += 1
        r = rng.choice(templates)
        t = norm(SLOT.sub(lambda m: value(r["intent"], m.group(1)), r["sentence_annotation"]))
        if t and t not in held and len(t.split()) <= 20:
            out.setdefault(t, r["intent"])
    with open(a.out, "w") as f, open(a.out.rsplit(".", 1)[0] + ".jsonl", "w") as g:
        for t, i in out.items():
            f.write(t + "\n"); g.write(json.dumps(dict(text=t, intent=i)) + "\n")
    print(f"{len(out)} sentences ({len(rows)} training sentences, {len(templates)} templates, "
          f"{sum(len(v) for v in pool.values())} slot values in {len(pool)} types); held-out sentences excluded: {len(held)}")


if __name__ == "__main__":
    main()
