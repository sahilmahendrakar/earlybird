"""Feed a test set to a streaming recognizer in 80 ms chunks and log every partial.

Each utterance is followed by TAIL seconds of silence, fed the same way, and then the
recognizer is told the audio has ended. A partial is stamped with the audio clock: the
time of the last sample the recognizer had been given when it produced that text. The
compute a step took is logged separately, so the two can be combined later.

usage: run_stream.py MODEL DATA_DIR [--workers N] [--limit N] [--tag timing] [--name SYSTEM]
Writes results/raw/<MODEL>[.<variant>]/<data>[.<tag>].jsonl, one line per utterance:
  {id, dur, partials: [[t, text], ...] (logged when the text changes), final, step_ms: [...]}"""
import argparse, json, os, sys, time
# One thread is faster than the runtime's default for these small graphs (26 vs 40 ms per
# step for Tiny on an M4 Pro) and lets several workers share the machine.
os.environ.setdefault("MOONSHINE_ORT_SINGLE_THREAD", "1")
from multiprocessing import Process
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(__file__))
from recognizers import MODELS, SR, CHUNK

TAIL = 1.0


def run_utt(rec, x):
    x = np.concatenate([x, np.zeros(int(TAIL * SR), dtype=np.float32)])
    logs, step_ms = {}, []

    def note(t, out):
        for k, text in (out.items() if isinstance(out, dict) else [("", out)]):
            p = logs.setdefault(k, [])
            if not p or p[-1][1] != text:
                p.append([round(t, 3), text])

    rec.start()
    for i in range(0, len(x), CHUNK):
        chunk = x[i:i + CHUNK]
        t0 = time.perf_counter(); out = rec.step(chunk); step_ms.append((time.perf_counter() - t0) * 1e3)
        note(min(i + CHUNK, len(x)) / SR, out)
    t0 = time.perf_counter(); out = rec.finish(); fin_ms = (time.perf_counter() - t0) * 1e3
    note(len(x) / SR, out)
    finals = out if isinstance(out, dict) else {"": out}
    return logs, finals, step_ms, fin_ms


def out_path(model, name, variant, part=None):
    d = f"results/raw/{model}{'.' + variant if variant else ''}"
    os.makedirs(d, exist_ok=True)
    return f"{d}/{name}.jsonl" + (f".part{part}" if part is not None else "")


def worker(model, data, rows, name, part, threads, sysname=None):
    paths = lambda k: out_path(sysname or model, name, k, part)
    rec = MODELS[model][0](threads=threads) if threads else MODELS[model][0]()
    files = {}
    for r in rows:
        x, sr = sf.read(f"{data}/{r['audio']}", dtype="float32"); assert sr == SR
        logs, finals, step_ms, fin_ms = run_utt(rec, x)
        for k in logs:
            if k not in files:
                files[k] = open(paths(k), "a")
            files[k].write(json.dumps(dict(id=r["id"], dur=r["dur"], partials=logs[k], final=finals[k],
                                           step_ms=[round(s, 2) for s in step_ms],
                                           finish_ms=round(fin_ms, 2))) + "\n")
            files[k].flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=MODELS); ap.add_argument("data")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0, help="threads per worker, where the backend lets us set it")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--name", help="system name for the results directory (default: MODEL), e.g. one per checkpoint")
    a = ap.parse_args()
    data = a.data.rstrip("/")
    rows = [json.loads(l) for l in open(f"{data}/manifest.jsonl")]
    if a.limit:
        rows = rows[:: max(1, len(rows) // a.limit)][: a.limit]
    name = os.path.basename(data) + (f".{a.tag}" if a.tag else "")

    import glob
    sysname = a.name or a.model
    for f in glob.glob(f"results/raw/{sysname}/{name}.jsonl*") + glob.glob(f"results/raw/{sysname}.*/{name}.jsonl*"):
        os.remove(f)
    t0 = time.time()
    if a.workers == 1:
        worker(a.model, data, rows, name, None, a.threads, sysname)
    else:
        ps = [Process(target=worker, args=(a.model, data, rows[w::a.workers], name, w, a.threads, sysname), daemon=True) for w in range(a.workers)]
        [p.start() for p in ps]; [p.join() for p in ps]
        if any(p.exitcode for p in ps):
            raise SystemExit("a worker failed")
        for part0 in glob.glob(f"results/raw/{sysname}/{name}.jsonl.part0") + glob.glob(f"results/raw/{sysname}.*/{name}.jsonl.part0"):
            base = part0[: -len(".part0")]
            with open(base, "w") as out:
                for w in range(a.workers):
                    out.write(open(f"{base}.part{w}").read()); os.remove(f"{base}.part{w}")
    audio = sum(r["dur"] + TAIL for r in rows)
    print(f"{sysname} {name}: {len(rows)} utts, {audio/60:.1f} min audio in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
