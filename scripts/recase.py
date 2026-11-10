"""Give a synthesized set the recognizer's own written form: casing and punctuation.

Synthesis texts are lower-case command sentences with no punctuation, which is not what a
recognizer prints, so (as prep_real.py does for real speech) a larger recognizer, the
teacher, transcribes each utterance and the utterance is kept only when the teacher's words
are exactly the sentence's words; its transcript then becomes the text. Word times stay as
synthesized. Rewrites DIR/manifest.jsonl (the original is kept as manifest.raw.jsonl).

usage: recase.py DIR [--teacher UsefulSensors/moonshine-streaming-medium] [--batch 32]"""
import argparse, json, os, sys
import soundfile as sf, torch
sys.path.insert(0, os.path.dirname(__file__))
from text import norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("--teacher", default="UsefulSensors/moonshine-streaming-medium")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(a.teacher).to(dev).eval()
    proc = AutoProcessor.from_pretrained(a.teacher)
    raw = f"{a.dir}/manifest.raw.jsonl"
    if not os.path.exists(raw):
        os.replace(f"{a.dir}/manifest.jsonl", raw)
    rows = [json.loads(l) for l in open(raw)]
    rows.sort(key=lambda r: r["dur"])
    kept = 0
    with open(f"{a.dir}/manifest.jsonl", "w") as f:
        for b in range(0, len(rows), a.batch):
            chunk = rows[b: b + a.batch]
            xs = [sf.read(f"{a.dir}/{r['audio']}", dtype="float32")[0] for r in chunk]
            inp = proc(xs, sampling_rate=16000, return_tensors="pt", padding=True).to(dev)
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                ids = model.generate(**inp, max_new_tokens=int(max(len(x) for x in xs) / 16000 * 6.5) + 8, do_sample=False)
            for r, text in zip(chunk, proc.batch_decode(ids, skip_special_tokens=True)):
                text = " ".join(text.split())
                if norm(text).split() == [w["w"] for w in r["words"]]:
                    r["text"] = text; f.write(json.dumps(r) + "\n"); kept += 1
    print(f"{kept} of {len(rows)} kept with the teacher's written form")


if __name__ == "__main__":
    main()
