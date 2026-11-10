"""Noisy copies of a test set, same transcripts and word times: white noise at a given SNR, and
babble (three other utterances of the set mixed together) at a given SNR.
usage: make_noisy.py SRC_DIR OUT_DIR --kind white|babble --snr 10 [--seed 0]"""
import argparse, json, os, random
import numpy as np, soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--kind", choices=["white", "babble"], required=True)
    ap.add_argument("--snr", type=float, required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed); nrng = np.random.default_rng(a.seed)
    rows = [json.loads(l) for l in open(f"{a.src}/manifest.jsonl")]
    os.makedirs(f"{a.out}/audio", exist_ok=True)
    with open(f"{a.out}/manifest.jsonl", "w") as f:
        for r in rows:
            x, sr = sf.read(f"{a.src}/{r['audio']}", dtype="float32")
            if a.kind == "white":
                n = nrng.standard_normal(len(x)).astype(np.float32)
            else:
                n = np.zeros(len(x), dtype=np.float32)
                for o in rng.sample([q for q in rows if q["id"] != r["id"]], 3):
                    y, _ = sf.read(f"{a.src}/{o['audio']}", dtype="float32")
                    y = np.resize(y, len(x)); n += y / (np.sqrt((y ** 2).mean()) + 1e-9)
            ps, pn = float((x ** 2).mean()) + 1e-12, float((n ** 2).mean()) + 1e-12
            z = x + n * np.sqrt(ps / (pn * 10 ** (a.snr / 10)))
            z = z / max(1.0, float(np.abs(z).max()))
            rel = f"audio/{r['id']}.wav"; sf.write(f"{a.out}/{rel}", z, sr, subtype="PCM_16")
            f.write(json.dumps(dict(r, audio=rel)) + "\n")
    print(f"{len(rows)} utterances -> {a.out}")


if __name__ == "__main__":
    main()
