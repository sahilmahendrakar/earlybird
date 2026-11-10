"""LibriSpeech test-clean with Montreal Forced Aligner word times -> our manifest format.

Source: gilkeyio/librispeech-alignments on the Hugging Face Hub (MFA alignments from
zenodo.org/records/2619474). A seeded random subset keeps the baseline runs short.
With N = all, every utterance is kept (the full test set, for WER comparable with published
numbers): no length limit, and an utterance whose alignment does not cover its transcript
keeps its words without times (score.py counts them for WER and revisions only).
usage: prep_librispeech.py PARQUET OUT_DIR [N|all]"""
import io, json, os, random, sys
import pyarrow.parquet as pq, soundfile as sf
sys.path.insert(0, os.path.dirname(__file__))
from text import norm

src, out = sys.argv[1], sys.argv[2]
full = len(sys.argv) > 3 and sys.argv[3] == "all"
n = 0 if full else int(sys.argv[3]) if len(sys.argv) > 3 else 500
rows = pq.read_table(src, columns=["id", "sex", "transcript", "words"]).to_pylist()
keep = set(range(len(rows))) if full else set(random.Random(0).sample(range(len(rows)), n))
MAX_DUR = 1e9 if full else 25.0  # Whisper sees 30 s at most; keep every system on the same utterances
untimed = 0
os.makedirs(f"{out}/audio", exist_ok=True)
audio = pq.read_table(src, columns=["audio"]).column("audio")
kept = secs = 0
with open(f"{out}/manifest.jsonl", "w") as mf:
    for i in sorted(keep):
        r = rows[i]
        words = [dict(w=norm(w["word"]), start=w["start"], end=w["end"]) for w in r["words"] if norm(w["word"])]
        if [w["w"] for w in words] != norm(r["transcript"]).split():
            if not full:
                continue  # alignment does not cover the transcript word for word
            words = [dict(w=w) for w in norm(r["transcript"]).split()]; untimed += 1
        x, sr = sf.read(io.BytesIO(audio[i]["bytes"].as_py()), dtype="int16")
        assert sr == 16000
        if len(x) / sr > MAX_DUR:
            continue
        rel = f"audio/{r['id']}.wav"; sf.write(f"{out}/{rel}", x, sr)
        mf.write(json.dumps(dict(id=r["id"], audio=rel, text=r["transcript"], voice=r["sex"],
                                 dur=round(len(x) / sr, 3), words=words)) + "\n")
        kept += 1; secs += len(x) / sr
print(f"{kept} utts ({untimed} without word times), {secs/60:.1f} min")
