"""One comparison table on exactly the same utterances: every system's partial log scored on
the intersection of the utterance ids all of them have for a set (score.py's metrics).

usage: unified.py --sets librispeech_test_clean,kokoro_test --sys NAME[=Label] ... [--md OUT]
(raw logs from results/raw/NAME/{SET,SET_full}.jsonl; a system without the set is left out of that set)"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from score import score_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True); ap.add_argument("--sys", nargs="+", required=True); ap.add_argument("--md")
    a = ap.parse_args()
    out = []
    for s in a.sets.split(","):
        man = {r["id"]: r for r in map(json.loads, open(f"data/{s}/manifest.jsonl"))}
        have = {}
        for spec in a.sys:
            name, _, label = spec.partition("=")
            for fn in (s, s + "_full"):
                p = f"results/raw/{name}/{fn}.jsonl"
                if os.path.exists(p):
                    have[name] = (label or name, p, {json.loads(l)["id"] for l in open(p)}); break
        ids = set(man)
        for _, _, i in have.values():
            ids &= i
        sub = {i: man[i] for i in ids}
        out += [f"### {s}: {len(sub)} utterances every system has\n",
                "| System | WER % | First correct: median ms | Ahead of word end % | Stable: median ms | Stable p95 ms | Revisions / word |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        rows = []
        for name, (label, p, _) in have.items():
            m = score_file(p, sub); rows.append((m["stable_med"], label, m))
        for _, label, m in sorted(rows, key=lambda r: r[0]):
            out.append(f"| {label} | {m['wer']:.1f} | {m['first_med']:+.0f} | {m['ahead']:.1f} | {m['stable_med']:+.0f} | {m['stable_p95']:+.0f} | {m['rev']:.2f} |")
        out.append("")
    text = "\n".join(out); print(text)
    if a.md:
        open(a.md, "w").write(text)


if __name__ == "__main__":
    main()
