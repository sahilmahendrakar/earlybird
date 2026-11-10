"""Held-out spontaneous test set: People's Speech `test` (conversational and public talk,
never used in training: training draws only from the `clean` subset) in our manifest format.

Unlike prep_real.py there is no teacher-agreement filter: keeping only utterances a
recognizer already gets right would make the test easy. An utterance is dropped only when
the aligner cannot place its reference words with confidence (--min-score), since word
times are needed for the latency metrics. The references are People's Speech's own
(lower-case, no punctuation, sometimes wrong), which score.py normalizes anyway.

The test split has few recordings (about 11 per file, ~90 in all), so all 8 files are read
and each recording gives at most --per-rec pieces.

With --min-score 0 nothing is dropped for alignment: that is the looser WER set
(ps_test_wer), where word times are whatever the aligner gives (spread evenly if it fails)
and only the final WER and revisions per word should be read. Each row records align_min,
the aligner's lowest word score.

usage: prep_ps_test.py OUT_DIR [--n 500] [--per-rec 8] [--min-score 0.2]"""
import argparse, io, json, os, random, sys
import pyarrow.parquet as pq, soundfile as sf, torch
sys.path.insert(0, os.path.dirname(__file__))
from text import norm
from align import Aligner, calibrated

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--n", type=int, default=500)
ap.add_argument("--per-rec", type=int, default=8, help="pieces kept per recording at most")
ap.add_argument("--min-score", type=float, default=0.2); ap.add_argument("--max-dur", type=float, default=20.0)
a = ap.parse_args()
from huggingface_hub import hf_hub_download

dev = "cuda" if torch.cuda.is_available() else "cpu"
aligner = Aligner(dev)
os.makedirs(f"{a.out}/audio", exist_ok=True)
kept = tried = 0; secs = 0.0; recs = {}
with open(f"{a.out}/manifest.jsonl", "w") as mf:
    for k in range(8):
        if kept >= a.n:
            break
        src = hf_hub_download("MLCommons/peoples_speech", f"test/test-{k:05d}-of-00008.parquet",
                              repo_type="dataset", local_dir=f"{a.out}/dl")
        rows = pq.read_table(src, columns=["id", "text", "audio"]).to_pylist(); os.remove(src)
        random.Random(k).shuffle(rows)
        for r in rows:
            if kept >= a.n:
                break
            uid = os.path.splitext(r["id"])[0].replace("/", "_"); rec = uid.rsplit("_", 1)[0]
            words = norm(r["text"]).split()
            if len(words) < 3 or recs.get(rec, 0) >= a.per_rec:   # cap per recording, so speakers vary
                continue
            x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            if sr != 16000 or x.ndim != 1 or len(x) / sr > a.max_dur:
                continue
            tried += 1
            times = aligner.word_times(x, words)
            lo = min(t[2] for t in times) if times else 0.0
            if a.min_score > 0 and (times is None or lo < a.min_score):
                continue
            if times is None:   # spread evenly; timing metrics are not read on this set
                d = len(x) / 16000 / len(words); times = [(i * d, (i + 1) * d, 0.0) for i in range(len(words))]
            rel = f"audio/{uid}.wav"; sf.write(f"{a.out}/{rel}", x, 16000, subtype="PCM_16")
            mf.write(json.dumps(dict(id=uid, audio=rel, text=" ".join(words), voice=rec, dur=round(len(x) / 16000, 3), align_min=round(lo, 3),
                                     words=[dict(w=w, start=s, end=e) for w, (s, e) in zip(words, calibrated(times))])) + "\n")
            kept += 1; secs += len(x) / 16000; recs[rec] = recs.get(rec, 0) + 1
        print(f"file {k+1}/8: kept {kept} of {tried} aligned attempts, {secs/3600:.2f} h, {len(recs)} recordings", flush=True)
