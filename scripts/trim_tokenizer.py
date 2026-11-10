"""Cut the tokenizer down to the 19,007 tokens Earlybird's word table keeps.

The checkpoint carries Moonshine's full tokenizer (32,000 tokens, 61k merges, 3.8 MB) plus
kept_ids.json, which maps Earlybird's ids to the original ones (engine.py's _Remap). This writes a
tokenizer whose ids ARE Earlybird's ids: only the kept tokens, renumbered in kept_ids order, and only
the merges whose two parts and result are all kept. kept_ids.json is then no longer needed.

usage: trim_tokenizer.py MODEL_DIR OUT_DIR      (copies everything else in MODEL_DIR unchanged)
       trim_tokenizer.py --check MODEL_DIR OUT_DIR   decode a few thousand id sequences both ways"""
import json, os, random, shutil, sys


def trim(src, out):
    os.makedirs(out, exist_ok=True)
    kept = json.load(open(f"{src}/kept_ids.json"))
    new_id = {old: i for i, old in enumerate(kept)}
    t = json.load(open(f"{src}/tokenizer.json"))
    vocab = t["model"]["vocab"]; inv = {v: k for k, v in vocab.items()}
    for a in t.get("added_tokens") or []:
        inv.setdefault(a["id"], a["content"])
    keep_tok = {inv[o] for o in kept}
    t["model"]["vocab"] = {inv[o]: new_id[o] for o in kept if inv[o] in vocab}
    merges = []
    for m in t["model"]["merges"]:
        a, b = m.split(" ", 1) if isinstance(m, str) else m
        if a in keep_tok and b in keep_tok and a + b in keep_tok:
            merges.append(m)
    t["model"]["merges"] = merges
    t["added_tokens"] = [dict(a, id=new_id[a["id"]]) for a in (t.get("added_tokens") or []) if a["id"] in new_id]
    for f in os.listdir(src):
        if f not in ("tokenizer.json", "kept_ids.json") and os.path.isfile(f"{src}/{f}"):
            shutil.copy(f"{src}/{f}", out)
    json.dump(t, open(f"{out}/tokenizer.json", "w"), ensure_ascii=False, separators=(",", ":"))
    print(f"{len(t['model']['vocab'])} tokens, {len(merges)} merges, {len(t['added_tokens'])} added; "
          f"tokenizer.json {os.path.getsize(f'{src}/tokenizer.json') / 1e6:.2f} -> {os.path.getsize(f'{out}/tokenizer.json') / 1e6:.2f} MB")


def check(src, out):
    sys.path.insert(0, os.path.dirname(__file__))
    from transformers import AutoTokenizer
    from engine import _Remap
    old = _Remap(AutoTokenizer.from_pretrained(src), json.load(open(f"{src}/kept_ids.json")))
    new = AutoTokenizer.from_pretrained(out)
    rng = random.Random(0); n = len(json.load(open(f"{src}/kept_ids.json"))); bad = 0
    for _ in range(3000):
        ids = [rng.randrange(3, n) for _ in range(rng.randrange(1, 12))]
        a, b = old.decode(ids, skip_special_tokens=True), new.decode(ids, skip_special_tokens=True)
        bad += a != b
        if bad and bad < 4 and a != b:
            print("differs:", ids, repr(a), repr(b))
    starts = sum(old.convert_ids_to_tokens(i).startswith("▁") != new.convert_ids_to_tokens(i).startswith("▁") for i in range(n))
    print(f"decode mismatches: {bad} / 3000; word-start mismatches: {starts} / {n}")


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        check(sys.argv[2], sys.argv[3])
    else:
        trim(sys.argv[1], sys.argv[2])
