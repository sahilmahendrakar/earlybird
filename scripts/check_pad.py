"""Check that train.py --pad encodes a cut exactly as Engine(pad=True) streams it.

For a few utterances and cut points, compares the encoder frames train.encode gives (PAD on)
with the engine's final + provisional frames after streaming up to the cut.
usage: check_pad.py MODEL DATA_DIR"""
import json, sys, os
import numpy as np, soundfile as sf, torch
sys.path.insert(0, os.path.dirname(__file__))
import train
from engine import Engine

model_dir, data = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(f"{data}/manifest.jsonl")][:3]
eng = Engine(model_dir, pad=True)
seen = {}
orig = eng._encode
def spy(new):
    n, prov = orig(new); seen["prov"] = prov; return n, prov
eng._encode = spy
train.PAD = True
worst = 0.0
for r in rows:
    x, sr = sf.read(f"{data}/{r['audio']}", dtype="float32")
    x = x[: len(x) // 1280 * 1280]
    for cut in sorted({min(1280 * 10, len(x)), min(1280 * 17, len(x)), len(x)}):
        eng.start(); seen.clear()
        with torch.no_grad():
            for i in range(0, cut, 1280):
                eng.step(x[i:i + 1280])
            e = eng.enc.final_norm(torch.cat([eng.fin[-1], seen["prov"]], dim=1))[0]
            wav = torch.from_numpy(np.pad(x[:cut], (0, 4000)))[None]
            h, valid = train.encode(eng.m, wav, torch.tensor([cut]))
        h = h[0, valid[0]]
        n = min(len(e), len(h)); d = (e[:n] - h[:n]).abs().max().item(); worst = max(worst, d)
        print(f"{r['id']} cut {cut / 16000:.2f}s: frames train {len(h)} engine {len(e)}, max diff {d:.2e}", flush=True)
print("worst", worst, "OK" if worst < 1e-3 else "MISMATCH")
