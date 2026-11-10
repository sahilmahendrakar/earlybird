"""Compare a manifest's word times with an independent forced aligner (align.py).

On the Kokoro sets this checks the aligner against Kokoro's own phoneme durations, on
LibriSpeech against Montreal Forced Aligner. The same aligner times the real speech that
has no alignment of its own, so these gaps say how exact those training times are.
usage: check_boundaries.py DATA_DIR [N] [field ...]   (fields default to: start end)"""
import json, os, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(__file__))
from align import Aligner

data = sys.argv[1].rstrip("/"); n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
fields = sys.argv[3:] or ["start", "end"]
al = Aligner()
rows = [json.loads(l) for l in open(f"{data}/manifest.jsonl") if l.endswith("\n")]
rows = rows[:: max(1, len(rows) // n)][:n]
diff = {f: [] for f in fields}; worst = []
for r in rows:
    x, sr = sf.read(f"{data}/{r['audio']}", dtype="float32")
    times = al.word_times(x, [w["w"] for w in r["words"]])
    if times is None:
        continue
    for f in fields:
        for w, (s, e, _) in zip(r["words"], times):
            t = s if f == "start" else e
            diff[f].append(w[f] - t)
            if f == fields[-1]: worst.append((round(abs(w[f] - t), 2), r["id"], w["w"], round(w[f], 2), round(t, 2)))
for f in fields:
    d = np.array(diff[f]) * 1e3
    print(f"{data} {f}: {len(d)} words, manifest minus aligner: median {np.median(d):+.0f} ms, "
          f"mean {d.mean():+.0f}, p5 {np.percentile(d,5):+.0f}, p95 {np.percentile(d,95):+.0f}, "
          f"|d|>50ms {np.mean(abs(d)>50)*100:.1f}%, |d|>100ms {np.mean(abs(d)>100)*100:.1f}%")
print("largest gaps:", sorted(worst, reverse=True)[:6])
