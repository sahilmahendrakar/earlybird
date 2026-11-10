"""Open ASR Leaderboard English sets: one final transcript per utterance, scored the leaderboard's way.

The sets are the leaderboard's own copies (hf-audio/open-asr-leaderboard, read straight from its
parquet files), all utterances, except TED-LIUM: the leaderboard repo no longer ships its files, so
it comes from data/tedlium_test (scripts/prep_hf_test.py: the same 1,155 test segments).

Every system gets the audio followed by TAIL s of silence (as in run_stream.py) and returns its
final transcript:
  earlybird      the streaming engine (scripts/engine.py, with the padding frames) fed the whole clip
                 at once. Its text after any step is what greedy decoding of the whole prefix gives,
                 so this is exactly the final text of a streaming run (checked by --check).
  moonshine-tiny the official moonshine-voice runtime: the clip in 80 ms chunks, then stop()
  whisper-base.en  openai-whisper, greedy, no timestamps, one pass per <=30 s piece
  whistle        Cactus Whistle as published (offline, 5-beam), one pass per <=30 s piece

Scoring follows the leaderboard (github.com/huggingface/open_asr_leaderboard, run_eval.py): its
EnglishTextNormalizer on references and predictions, references that normalize to nothing (or to
"ignore time segment in scoring") dropped, corpus WER over the set (jiwer). Mean = plain average of
the eight sets.

usage: open_asr.py run SYSTEM SET [--model CKPT] [--workers N] [--limit N]
         -> results/open_asr/SYSTEM/SET.jsonl  {id, ref, hyp, dur, ms}  (appends; skips ids already there)
       open_asr.py check SET --model CKPT --raw results/raw/NAME/SET.jsonl  (offline vs streaming finals)
       open_asr.py score [--md FILE]   (needs OAL_DIR = a clone of open_asr_leaderboard)"""
import argparse, fnmatch, glob, io, json, os, sys, time, zlib
os.environ.setdefault("MOONSHINE_ORT_SINGLE_THREAD", "1")
os.environ.setdefault("NEEDLE_TELEMETRY", "0")
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(__file__))

SR, CHUNK, TAIL = 16000, 1280, 1.0
REPO = "hf-audio/open-asr-leaderboard"
# name -> parquet glob in REPO (or a local manifest dir), smallest first
SETS = {
    "tedlium": "local:data/tedlium_test",
    "voxpopuli": "voxpopuli/test-*",
    "librispeech_clean": "librispeech/test.clean-*",
    "earnings22": "earnings22/test-*",
    "librispeech_other": "librispeech/test.other-*",
    "ami": "ami/test-*",
    "gigaspeech": "gigaspeech/test-*",
    "spgispeech": "spgispeech/test-*",
}


# ---- systems: make(model) -> transcribe(float32 16 kHz audio) -> text ----
def _pieces(x, n=30 * SR):
    k = -(-len(x) // n); m = -(-len(x) // k)
    return [x[i:i + m] for i in range(0, len(x), m)]


def make_earlybird(model):
    import torch
    from engine import Engine
    e = Engine(model, threads=1, pad=True)

    def tr(x):
        e.start()
        with torch.no_grad():
            e.step(x)
        return e.finish()
    return tr


def make_moonshine(_):
    from recognizers import Moonshine
    m = Moonshine("TINY_STREAMING", force=False)

    def tr(x):
        m.start()
        for i in range(0, len(x), CHUNK):
            m.stream.add_audio(x[i:i + CHUNK].tolist(), SR)
        return m.finish()
    return tr


def make_whisper(_):
    import torch, whisper
    torch.set_num_threads(1)
    m = whisper.load_model("base.en", device="cpu")
    opts = whisper.DecodingOptions(language="en", without_timestamps=True, fp16=False, temperature=0.0)

    def tr(x):
        out = []
        for p in _pieces(x):
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(p), n_mels=m.dims.n_mels)
            with torch.no_grad():
                out.append(whisper.decode(m, mel, opts).text.strip())
        return " ".join(out).strip()
    return tr


def make_whistle(_):
    from needle.agent.whistle import Whistle
    w = Whistle()
    return lambda x: " ".join(w.transcribe(p, language="en")["text"].strip() for p in _pieces(x)).strip()


SYSTEMS = {"earlybird": make_earlybird, "moonshine-tiny": make_moonshine,
           "whisper-base.en": make_whisper, "whistle": make_whistle}


# ---- data ----
def rows(set_name, limit=0):
    """Yield (id, ref, audio bytes or path) for every utterance of a set."""
    src = SETS[set_name]
    n = 0
    if src.startswith("local:"):
        d = src[6:]
        for l in open(f"{d}/manifest.jsonl"):
            r = json.loads(l)
            yield r["id"], r["text"], f"{d}/{r['audio']}"
            n += 1
            if limit and n >= limit:
                return
        return
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                   if fnmatch.fnmatch(f, src))
    for f in files:
        p = hf_hub_download(REPO, f, repo_type="dataset", local_dir="data/oasr")
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=64, columns=["id", "text", "audio"]):
            for r in b.to_pylist():
                yield str(r["id"]), r["text"], r["audio"]["bytes"]
                n += 1
                if limit and n >= limit:
                    return
        os.remove(p)


def load_audio(a):
    x, sr = sf.read(io.BytesIO(a) if isinstance(a, bytes) else a, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        import torch, torchaudio
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
    return x


# ---- workers ----
_tr = None


def _init(system, model):
    global _tr
    _tr = SYSTEMS[system](model)


def _work(item):
    uid, ref, a = item
    x = load_audio(a)
    t0 = time.perf_counter()
    hyp = _tr(np.concatenate([x, np.zeros(int(TAIL * SR), dtype=np.float32)]))
    return dict(id=uid, ref=ref, hyp=hyp, dur=round(len(x) / SR, 3), ms=round((time.perf_counter() - t0) * 1e3, 1))


# The two big sets (35 h and 100 h of audio) are too slow for Whisper base and Whistle on our budget:
# those two get a fixed sample (the same ids for both, chosen by a hash of the id); `score` also
# reports every system on just the ids all systems share.
SLOW = ("whisper-base.en", "whistle")
SAMPLE = {"gigaspeech": 0.1, "spgispeech": 0.05}


def sampled(uid, frac):
    return zlib.crc32(uid.encode()) / 2 ** 32 < frac


def run(a):
    if a.system == "whistle" and os.path.exists("/tmp/skip_whistle"):   # box-local switch
        return
    frac = a.frac or (SAMPLE.get(a.set, 1.0) if a.system in SLOW else 1.0)
    out = f"results/open_asr/{a.name or a.system}/{a.set}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
    import threading
    gate = threading.BoundedSemaphore(a.workers * 16)   # Pool's feeder thread would read the whole set ahead

    def todo():
        for r in rows(a.set, a.limit):
            if r[0] not in done and (frac >= 1 or sampled(r[0], frac)):
                gate.acquire(); yield r
    SYSTEMS[a.system](a.model)   # fail here: a Pool whose initializer raises respawns its workers forever
    t0 = time.time(); n = 0
    from multiprocessing import Pool
    with open(out, "a") as f, Pool(a.workers, _init, (a.system, a.model)) as pool:
        for r in pool.imap_unordered(_work, todo(), chunksize=4):
            gate.release()
            f.write(json.dumps(r) + "\n"); n += 1
            if n % 500 == 0:
                f.flush(); print(f"{a.set}: {n} utts, {time.time() - t0:.0f}s", flush=True)
    print(f"{a.name or a.system} {a.set}: {n} new utts ({len(done)} already) in {time.time() - t0:.0f}s", flush=True)


def check(a):
    """Offline (whole clip at once) vs the finals of an earlier streaming run (80 ms steps)."""
    raw = {json.loads(l)["id"]: json.loads(l)["final"] for l in open(a.raw)}
    tr = make_earlybird(a.model); same = n = 0
    for uid, _, p in rows(a.set):
        if uid not in raw:
            continue
        x = load_audio(p)
        same += tr(np.concatenate([x, np.zeros(int(TAIL * SR), dtype=np.float32)])) == raw[uid]; n += 1
        if n >= (a.limit or 200):
            break
    print(f"offline == streaming final: {same}/{n}")


# ---- scoring ----
def score(a):
    sys.path.insert(0, os.environ.get("OAL_DIR", "open_asr_leaderboard"))
    import jiwer
    from normalizer.normalizer import EnglishTextNormalizer   # = data_utils.normalizer
    norm = EnglishTextNormalizer()
    table, table_ids, pairs = {}, {}, {}
    for path in sorted(glob.glob("results/open_asr/*/*.jsonl")):
        system, s = path.split("/")[-2], os.path.basename(path)[:-6]
        refs, hyps, dur, ms = [], [], 0.0, 0.0
        seen = set()
        for l in open(path):
            r = json.loads(l)
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            ref = norm(r["ref"])
            if ref.strip() in ("", "ignore time segment in scoring"):
                continue
            refs.append(ref); hyps.append(norm(r["hyp"])); dur += r["dur"]; ms += r["ms"]
            pairs.setdefault((system, s), []).append((r["id"], ref, hyps[-1]))
        if not refs:
            continue
        table_ids[(system, s)] = {i for i, _, _ in pairs[(system, s)]}
        table.setdefault(system, {})[s] = (100 * jiwer.wer(refs, hyps), len(refs), dur / (ms / 1e3))
    sets = list(SETS)
    lines = ["Full sets (Whisper base and Whistle: a fixed sample of GigaSpeech and SPGISpeech, see SAMPLE):", "",
             "| System | " + " | ".join(sets) + " | Mean |", "|---|" + "---:|" * (len(sets) + 1)]
    for system, d in table.items():
        cells = [f"{d[s][0]:.2f}" if s in d else "–" for s in sets]
        mean = f"{np.mean([d[s][0] for s in sets]):.2f}" if all(s in d for s in sets) else "–"
        lines.append(f"| {system} | " + " | ".join(cells) + f" | {mean} |")
    # the same utterances for every system, per set
    common = {}
    for s_ in sets:
        ids = [table_ids[(sy, s_)] for sy in table if (sy, s_) in table_ids]
        if len(ids) == len(table):
            common[s_] = set.intersection(*ids)
    lines += ["", "Same utterances for every system (the ids all systems have, per set):", "",
              "| System | " + " | ".join(f"{s_} ({len(common[s_])})" for s_ in sets if s_ in common) + " |",
              "|---|" + "---:|" * len(common)]
    for system in table:
        cells = []
        for s_ in sets:
            if s_ in common:
                rr = pairs[(system, s_)]
                ref = [r for i, r, h in rr if i in common[s_]]; hyp = [h for i, r, h in rr if i in common[s_]]
                cells.append(f"{100 * jiwer.wer(ref, hyp):.2f}")
        lines.append(f"| {system} | " + " | ".join(cells) + " |")
    lines += ["", "Utterances scored / RTFx per core (audio seconds per compute second, one thread):", ""]
    for system, d in table.items():
        lines.append(f"- {system}: " + ", ".join(f"{s} {d[s][1]} / {d[s][2]:.0f}" for s in sets if s in d))
    md = "\n".join(lines) + "\n"
    print(md)
    if a.md:
        open(a.md, "w").write(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "check", "score"])
    ap.add_argument("system", nargs="?"); ap.add_argument("set", nargs="?")
    ap.add_argument("--model", help="checkpoint dir for earlybird")
    ap.add_argument("--name", help="results directory name (default: SYSTEM)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--frac", type=float, default=0, help="keep this fraction of ids (hash sample; nested: a smaller frac is a subset of a larger one)")
    ap.add_argument("--raw"); ap.add_argument("--md")
    a = ap.parse_args()
    if a.cmd == "check":
        a.set = a.system
        return check(a)
    {"run": run, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
