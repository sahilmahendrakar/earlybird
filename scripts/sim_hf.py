"""Streaming logs for a PyTorch Moonshine checkpoint, by decoding every audio prefix.

For a Hugging Face `MoonshineStreamingForConditionalGeneration` checkpoint (the released
weights or a fine-tune) this writes the same log format as run_stream.py, without an ONNX
export. At each 80 ms step the model sees the audio up to that point and nothing after,
re-encodes it and decodes greedily; the prefixes of an utterance are decoded as batches.
There is no voice-activity detector and no line splitting, so this is the model alone;
the official runtime adds both. Compute per step is not measured here.

usage: sim_hf.py NAME MODEL DATA_DIR [--limit N] [--step-ms 80] [--batch 48] [--device mps]"""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, torch
sys.path.insert(0, os.path.dirname(__file__))
from run_stream import TAIL, SR, CHUNK


@torch.no_grad()
def decode_prefixes(model, proc, x, ends, device, batch=48, tok_per_s=6.5):
    texts = []
    for b in range(0, len(ends), batch):
        es = ends[b:b + batch]
        inp = proc([x[:e] for e in es], sampling_rate=SR, return_tensors="pt", padding=True)
        inp = {k: v.to(device) for k, v in inp.items()}
        ids = model.generate(**inp, max_new_tokens=int(es[-1] / SR * tok_per_s) + 4, do_sample=False)
        for e, row in zip(es, ids.cpu()):
            row = row[: int(e / SR * tok_per_s) + 5]  # the runtime's cap on tokens per second of audio
            texts.append(proc.decode(row, skip_special_tokens=True).strip())
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("model"); ap.add_argument("data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--step-ms", type=int, default=80)
    ap.add_argument("--min-ms", type=int, default=240, help="shortest prefix decoded")
    ap.add_argument("--batch", type=int, default=48, help="prefixes decoded together; lower it for long utterances")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(a.model).to(a.device).eval()
    proc = AutoProcessor.from_pretrained(a.model)
    data = a.data.rstrip("/")
    rows = [json.loads(l) for l in open(f"{data}/manifest.jsonl")]
    if a.limit:
        rows = rows[:: max(1, len(rows) // a.limit)][: a.limit]
    os.makedirs(f"results/raw/{a.name}", exist_ok=True)
    step = a.step_ms * SR // 1000
    t0 = time.time()
    with open(f"results/raw/{a.name}/{os.path.basename(data)}.jsonl", "w") as f:
        for n, r in enumerate(rows):
            x, sr = sf.read(f"{data}/{r['audio']}", dtype="float32")
            x = np.concatenate([x, np.zeros(int(TAIL * SR), dtype=np.float32)])
            ends = [e for e in range(step, len(x), step) if e >= a.min_ms * SR // 1000] + [len(x)]
            texts = decode_prefixes(model, proc, x, ends, a.device, batch=a.batch)
            partials = [[0.0, ""]]
            for e, t in zip(ends, texts):
                if t != partials[-1][1]:
                    partials.append([round(e / SR, 3), t])
            f.write(json.dumps(dict(id=r["id"], dur=r["dur"], partials=partials, final=texts[-1],
                                    step_ms=[], finish_ms=0)) + "\n"); f.flush()
            if (n + 1) % 20 == 0:
                print(f"{n+1}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
