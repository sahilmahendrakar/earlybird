"""Out-of-domain test sets from the Hub, for evaluation only (never training):

  tedlium     TED-LIUM 3 test (TED talks; CC BY-NC-ND, so it may not be trained on), from a
              parquet copy (sanchit-gandhi/tedlium-data). Segments marked
              ignore_time_segment_in_scoring are dropped, as in the standard test set.
  earnings22  Earnings-22 test (earnings calls, accented business English), from the Open
              ASR Leaderboard's copy (hf-audio/open-asr-leaderboard), a seeded --n sample.

Every utterance is kept for WER. Word times come from the aligner (align.py) and are kept
only when it places every word with confidence (--min-score); otherwise the words are
written without times, so latency is measured on the confidently aligned part and WER on
all of it (score.py). Each row records align_min.

usage: prep_hf_test.py tedlium OUT_DIR | prep_hf_test.py earnings22 OUT_DIR [--n 1000]"""
import argparse, io, json, os, random, re, sys
import numpy as np, pyarrow.parquet as pq, soundfile as sf, torch
sys.path.insert(0, os.path.dirname(__file__))
from text import norm
from align import Aligner, calibrated

SOURCES = dict(
    tedlium=("sanchit-gandhi/tedlium-data", ["data/test-00000-of-00001-05350587dc944f81.parquet"]),
    earnings22=("hf-audio/open-asr-leaderboard", [f"earnings22/test-0000{k}-of-00005.parquet" for k in range(5)]),
)
ap = argparse.ArgumentParser()
ap.add_argument("set", choices=SOURCES); ap.add_argument("out")
ap.add_argument("--n", type=int, default=0, help="seeded sample of this many utterances (0: all)")
ap.add_argument("--min-score", type=float, default=0.2)
a = ap.parse_args()
from huggingface_hub import hf_hub_download

repo, files = SOURCES[a.set]
rows = []
for f in files:
    src = hf_hub_download(repo, f, repo_type="dataset", local_dir=f"{a.out}/dl")
    t = pq.read_table(src)
    cols = [c for c in ("id", "text", "audio") if c in t.column_names]
    rows += t.select(cols).to_pylist(); os.remove(src)
rows = [r for r in rows if "ignore_time_segment_in_scoring" not in r["text"]]
for i, r in enumerate(rows):
    r.setdefault("id", f"{a.set}_{i:05d}")
if a.n and a.n < len(rows):
    rows = random.Random(0).sample(rows, a.n)
dev = "cuda" if torch.cuda.is_available() else "cpu"
aligner = Aligner(dev)
os.makedirs(f"{a.out}/audio", exist_ok=True)
kept = timed = 0; secs = 0.0
with open(f"{a.out}/manifest.jsonl", "w") as mf:
    for r in rows:
        text = r["text"].replace(" '", "'")     # TED-LIUM writes "it 's"
        words = norm(text).split()
        if not words:
            continue
        au = r["audio"]
        x, sr = sf.read(io.BytesIO(au["bytes"]), dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        if sr != 16000:
            import torchaudio
            x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy(); sr = 16000
        times = aligner.word_times(x, words)
        lo = min(t[2] for t in times) if times else 0.0
        if times is not None and lo >= a.min_score:
            ws = [dict(w=w, start=s, end=e) for w, (s, e) in zip(words, calibrated(times))]; timed += 1
        else:
            ws = [dict(w=w) for w in words]
        uid = re.sub(r"[^A-Za-z0-9._-]+", "_", re.sub(r"\.(wav|flac|mp3)$", "", str(r["id"]))).strip("_")
        rel = f"audio/{uid}.wav"; sf.write(f"{a.out}/{rel}", x.astype(np.float32), 16000, subtype="PCM_16")
        mf.write(json.dumps(dict(id=uid, audio=rel, text=text, voice="", dur=round(len(x) / 16000, 3),
                                 align_min=round(lo, 3), words=ws)) + "\n")
        kept += 1; secs += len(x) / 16000
print(f"{a.set}: {kept} utts ({timed} with word times), {secs/3600:.2f} h", flush=True)
