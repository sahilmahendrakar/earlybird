"""Score one log against its test set.  usage: score_one.py LOG.jsonl DATA_DIR [...more pairs]"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from score import score_file

for log, d in zip(sys.argv[1::2], sys.argv[2::2]):
    man = {r["id"]: r for r in map(json.loads, open(f"{d.rstrip('/')}/manifest.jsonl"))}
    m = score_file(log, man)
    print(f"{log}: utts {m['utts']}  WER {m['wer']:.1f}  first {m['first_med']:+.0f} / p95 {m['first_p95']:+.0f} ms  "
          f"ahead {m['ahead']:.1f}%  stable {m['stable_med']:+.0f} / p95 {m['stable_p95']:+.0f} ms  revisions/word {m['rev']:.2f}  found {m['found']:.1f}%")
