"""Share of words shown right AND kept right from before the speaker finished them, as transcript text
or (for anticipating systems) as a guess at or above a confidence threshold. Compares ghost_demo
logs (own guesses) with plain recognizers' partial logs (results/raw) on the same utterances.
usage: kept_early.py MANIFEST [--ghost NAME=LOG.jsonl ...] [--raw NAME=RAW.jsonl ...] [--thr 0.5]"""
import argparse, json, re


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", (s or "").lower()).split()).split()


def shown(screen, guess, wp, ref, thr):
    h = norm(screen); k = 0
    while k < len(h) and k < len(ref) and h[k] == ref[k]:
        k += 1
    s = set(range(k))
    if guess and k == len(h):
        g = norm(guess)
        for j in range(len(g)):
            if k + j >= len(ref) or j >= len(wp) or wp[j] < thr or g[j] != ref[k + j]:
                break
            s.add(k + j)
    return s


def kept_times(events, ref, thr):
    cur = {}
    for t, screen, guess, wp in events:
        s = shown(screen, guess, wp, ref, thr)
        for j in list(cur):
            if j not in s:
                del cur[j]
        for j in s:
            cur.setdefault(j, t)
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("--ghost", nargs="*", default=[]); ap.add_argument("--raw", nargs="*", default=[])
    ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()
    man = {json.loads(l)["id"]: json.loads(l) for l in open(a.manifest)}
    sys = {}
    for spec in a.ghost:
        n, p = spec.split("=")
        sys[n] = {u["id"]: [[e["t"], e["screen"], e["guess"], e.get("wp", [])] for e in u["events"]] for u in map(json.loads, open(p))}
    for spec in a.raw:
        n, p = spec.split("=")
        sys[n] = {u["id"]: [[t, x, "", []] for t, x in u["partials"]] for u in map(json.loads, open(p))}
    ids = [i for i in man if all(i in v for v in sys.values()) and all("end" in w for w in man[i]["words"])]
    tot = sum(len(man[i]["words"]) for i in ids)
    print(f"{len(ids)} utterances, {tot} words; guesses count at >= {a.thr}")
    for n, v in sys.items():
        early = 0
        for i in ids:
            ws = man[i]["words"]; ref = [w["w"] for w in ws]
            kt = kept_times(v[i], ref, a.thr)
            early += sum(1 for j, t in kt.items() if t < ws[j]["end"])
        print(f"  {n:22s} {100 * early / tot:.1f}% of words shown right and kept right before they ended")


if __name__ == "__main__":
    main()
