"""Real training speech with word times and cased, punctuated transcripts.

  --source ls   LibriSpeech (CC BY 4.0) from gilkeyio/librispeech-alignments, word times
                from Montreal Forced Aligner. Splits: train_clean_100, train_clean_360,
                train_other_500 (960 h together).
  --source slurp  SLURP (CC BY 4.0, from marcel-gohsen/slurp): spoken commands to a home
                assistant, several recordings (headset and far-field) of each sentence, with
                the intent and slots, which the manifest keeps. Splits: train, devel, test.
                No word times; align.py supplies them. --keep-all keeps every utterance the
                aligner can place (no teacher filter; the reference text as given), and
                --one-per-sentence the first recording of each sentence: the test set.
  --source ps   People's Speech (MLCommons/peoples_speech), conversational and public
                talk in roughly 15 s pieces. Subsets: clean (CC BY), clean_sa (CC BY-SA).
                It has no word times; align.py supplies them.

Both come with lower- or upper-case transcripts and no punctuation, which is not what the
recognizer prints, so a larger recognizer (the teacher) writes each transcript, and an
utterance is kept only when the teacher's words are exactly the reference words. For
People's Speech that also drops the pieces whose reference is wrong.

Files are downloaded one at a time and deleted after use. Resumable.
--hours counts what OUT_DIR already holds, so a restarted run stops at the same total.
--reverse takes the files from the last one back, so a second run on the same split (into
another OUT_DIR) does not meet the first until both are nearly through.
usage: prep_real.py OUT_DIR --source ls --split train_clean_360 [--shard 0/2] [--hours H] [--reverse]"""
import argparse, io, json, os, sys, time
import numpy as np, pyarrow.parquet as pq, soundfile as sf, torch, torchaudio
sys.path.insert(0, os.path.dirname(__file__))
from text import norm

REPO = dict(ls="gilkeyio/librispeech-alignments", ps="MLCommons/peoples_speech", slurp="marcel-gohsen/slurp")
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--source", choices=REPO, default="ls")
ap.add_argument("--split", default="train_clean_100", help="ls: split name; ps: subset (clean, clean_sa, microset)")
ap.add_argument("--shard", default="0/1", help="i/n: take every n-th file starting at the i-th")
ap.add_argument("--hours", type=float, default=0, help="stop once OUT_DIR holds this much audio (0: no limit)")
ap.add_argument("--reverse", action="store_true", help="take the files last first")
ap.add_argument("--teacher", default="UsefulSensors/moonshine-streaming-medium")
ap.add_argument("--batch", type=int, default=32); ap.add_argument("--max-dur", type=float, default=20.0)
ap.add_argument("--min-score", type=float, default=0.2, help="ps: drop an utterance if the aligner is less sure than this of any word")
ap.add_argument("--limit", type=int, default=0, help="rows per file, for a quick trial")
ap.add_argument("--keep-all", action="store_true", help="no teacher filter: keep the reference text (test sets)")
ap.add_argument("--one-per-sentence", action="store_true", help="slurp: only the first recording of each sentence")
ap.add_argument("--max-utts", type=int, default=0, help="stop after this many kept utterances")
a = ap.parse_args()
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = MoonshineStreamingForConditionalGeneration.from_pretrained(a.teacher).to(dev).eval()
proc = AutoProcessor.from_pretrained(a.teacher)
if a.source in ("ps", "slurp"):
    from align import Aligner, calibrated
    aligner = Aligner(dev)

sub = a.split if a.source == "ps" else "data"
prefix = "train" if a.source == "ps" else a.split + "-" if a.source == "slurp" else a.split
files = sorted(f.path for f in HfApi().list_repo_tree(REPO[a.source], sub, repo_type="dataset")
               if f.path.endswith(".parquet") and os.path.basename(f.path).startswith(prefix))
i, n = map(int, a.shard.split("/")); files = files[i::n][::-1] if a.reverse else files[i::n]
os.makedirs(f"{a.out}/audio", exist_ok=True)
mpath = f"{a.out}/manifest.jsonl"; fpath = f"{a.out}/files_done.txt"
old = [json.loads(l) for l in open(mpath) if l.endswith("\n")] if os.path.exists(mpath) else []
done = {r["id"] for r in old}
files_done = set(open(fpath).read().split()) if os.path.exists(fpath) else set()


def rows(src):
    """(id, voice, reference words with times or None, audio) for each usable row of one file."""
    cols = dict(ls=["id", "sex", "transcript", "words", "audio"], ps=["id", "text", "audio"],
                slurp=["id", "transcript", "audio", "intent", "slots:"])[a.source]
    t = pq.read_table(src, columns=cols)
    stem = os.path.basename(src).split(".")[0]
    for j, r in enumerate((t.slice(0, a.limit) if a.limit else t).to_pylist()):
        if a.source == "slurp":
            if a.one_per_sentence and r["id"] in seen_sentences:
                continue
            uid = f"slurp_{r['id']}_{stem}_{j}"
        else:
            uid = os.path.splitext(r["id"])[0].replace("/", "_")
        if uid in done:
            continue
        extra = {}
        if a.source == "slurp":
            words = [dict(w=w) for w in norm(r["transcript"]).split()]
            voice = str(r["id"])   # the sentence (several recordings of each)
            extra = dict(intent=r["intent"], slots=r["slots:"], sentence=r["id"], ref_text=r["transcript"])
        elif a.source == "ls":
            words = [dict(w=norm(w["word"]), start=w["start"], end=w["end"]) for w in r["words"] if norm(w["word"])]
            if [w["w"] for w in words] != norm(r["transcript"]).split():
                continue
            voice = r["sex"]
        else:
            words = [dict(w=w) for w in norm(r["text"]).split()]
            voice = uid.rsplit("_", 1)[0]   # the recording the piece was cut from
        if len(words) < 2:
            continue
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        if sr != 16000:
            x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy(); sr = 16000
        if len(x) / sr <= a.max_dur:
            if a.source == "slurp":
                seen_sentences.add(r["id"])
            yield uid, voice, words, x, extra


seen = kept = 0; secs = sum(r["dur"] for r in old); t0 = time.time()
seen_sentences = {r.get("sentence") for r in old}
with open(mpath, "a") as mf:
    for k, name in enumerate(files):
        if name in files_done:
            continue
        if (a.hours and secs >= a.hours * 3600) or (a.max_utts and kept + len(old) >= a.max_utts):
            break
        src = hf_hub_download(REPO[a.source], name, repo_type="dataset", local_dir=f"{a.out}/dl")
        items = sorted(rows(src), key=lambda it: len(it[3]))
        seen += len(items)
        for b in range(0, len(items), a.batch):
            chunk = items[b: b + a.batch]
            if a.keep_all:
                texts = [" ".join(w["w"] for w in it[2]) for it in chunk]
            else:
                inp = proc([it[3] for it in chunk], sampling_rate=16000, return_tensors="pt", padding=True).to(dev)
                with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    ids = model.generate(**inp, max_new_tokens=int(len(chunk[-1][3]) / 16000 * 6.5) + 8, do_sample=False)
                texts = proc.batch_decode(ids, skip_special_tokens=True)
            for (uid, voice, words, x, extra), text in zip(chunk, texts):
                text = " ".join(text.split()); ps = text.split()
                if len(ps) != len(words) or any(norm(p) != w["w"] for p, w in zip(ps, words)):
                    continue
                if a.source in ("ps", "slurp"):
                    times = aligner.word_times(x, [w["w"] for w in words])
                    if times is None or min(t[2] for t in times) < a.min_score:
                        continue
                    words = [dict(w=w["w"], start=s, end=e) for w, (s, e) in zip(words, calibrated(times))]
                rel = f"audio/{uid}.flac"; sf.write(f"{a.out}/{rel}", x, 16000, subtype="PCM_16")
                mf.write(json.dumps(dict(id=uid, audio=rel, text=text, voice=voice,
                                         dur=round(len(x) / 16000, 3), words=words, **extra)) + "\n")
                kept += 1; secs += len(x) / 16000
        mf.flush(); os.remove(src)
        if not a.limit:
            open(fpath, "a").write(name + "\n")
        print(f"file {k+1}/{len(files)}: kept {kept} of {seen} utterances, {secs/3600:.1f} h, {time.time()-t0:.0f}s", flush=True)
