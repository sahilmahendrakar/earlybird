"""Why does a student checkpoint stream so badly? Three checks on a sample of test utterances.

  offline  greedy decoding of the whole utterance (+1 s of silence, as run_stream.py feeds it)
           with train.py's own forward pass: no streaming, no caches
  stream   the streaming engine fed 80 ms chunks, final text (what the eval scores)
  same     whether the two produce the same tokens

plus WER by duration, and the norm of the decoder's per-frame position embedding (pos_emb)
by frame index, which shows how far into an utterance training has reached.
usage: diag_student.py CKPT OUT.jsonl DATA_DIR[:N[:MIN_SECONDS]] ... [--workers W]"""
import argparse, json, os, sys
from multiprocessing import Pool
import numpy as np, soundfile as sf, torch
sys.path.insert(0, os.path.dirname(__file__))
from text import norm

SR, CHUNK, TAIL = 16000, 1280, 1.0
G = {}


def init(ckpt):
    torch.set_num_threads(1)
    from transformers import MoonshineStreamingForConditionalGeneration
    from engine import Engine
    G["m"] = MoonshineStreamingForConditionalGeneration.from_pretrained(ckpt).eval()
    G["eng"] = Engine(ckpt, threads=1)


@torch.no_grad()
def offline(x):
    from train import encode
    m, eng = G["m"], G["eng"]
    wav = torch.from_numpy(x)[None]; ns = torch.tensor([len(x)])
    h, valid = encode(m, wav, ns)   # train.py's forward, with the encoder run once (the decoder
    # adds its position table to encoder_hidden_states in place, hence the copy each step)
    ids = [eng.bos]; max_new = int(len(x) / SR * eng.tok_per_s) + 4
    while len(ids) - 1 < max_new:
        out = m.model.decoder(input_ids=torch.tensor([ids]), encoder_hidden_states=h.clone(), encoder_attention_mask=valid.long())
        t = int(m.proj_out(out.last_hidden_state)[0, -1].argmax())
        if t == eng.eos:
            break
        ids.append(t)
    return ids[1:]


@torch.no_grad()
def stream(x):
    eng = G["eng"]; eng.start()
    for i in range(0, len(x), CHUNK):
        eng.step(x[i:i + CHUNK])
    return list(eng.hyp)


def wer(r, h):
    r, h = r.split(), h.split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return d[len(h)], len(r)


def one(job):
    d, r = job
    x, sr = sf.read(f"{d}/{r['audio']}", dtype="float32"); assert sr == SR
    x = np.concatenate([x, np.zeros(int(TAIL * SR), dtype=np.float32)])
    x = x[: len(x) // CHUNK * CHUNK]   # whole 80 ms chunks, as train.py cuts them (the embedder needs whole frames)
    off, st = offline(x), stream(x)
    tok = G["eng"].tok
    o, s = tok.decode(off, skip_special_tokens=True).strip(), tok.decode(st, skip_special_tokens=True).strip()
    ref = norm(r["text"])
    return dict(set=os.path.basename(d), id=r["id"], dur=r["dur"], ref=ref, offline=o, stream=s, same=off == st,
                e_off=wer(ref, norm(o))[0], e_st=wer(ref, norm(s))[0], n=len(ref.split()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("out"); ap.add_argument("sets", nargs="+")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    jobs = []
    for s in a.sets:
        d, n, lo = (s.split(":") + ["", ""])[:3]   # DIR:N:MIN_SECONDS
        n, lo = n or "300", lo or "0"
        rows = [r for r in map(json.loads, open(f"{d}/manifest.jsonl")) if r["dur"] >= float(lo)]
        n = int(n)
        jobs += [(d, r) for r in rows[:: max(1, len(rows) // n)][:n] if (d, r["id"]) not in {(j[0], j[1]["id"]) for j in jobs}]
    with Pool(a.workers, init, (a.ckpt,)) as p, open(a.out, "w") as f:
        res = []
        for r in p.imap_unordered(one, jobs):
            res.append(r); f.write(json.dumps(r) + "\n"); f.flush()
    from transformers import MoonshineStreamingForConditionalGeneration
    m = MoonshineStreamingForConditionalGeneration.from_pretrained(a.ckpt)
    pe = m.model.decoder.pos_emb.weight.detach().norm(dim=-1)
    print("pos_emb row norm by frame (50 frames/s):",
          " ".join(f"{i // 50}s:{float(pe[i:i + 50].mean()):.2f}" for i in range(0, 2000, 250)))
    for s in sorted({r["set"] for r in res}):
        rs = [r for r in res if r["set"] == s]
        n = sum(r["n"] for r in rs)
        print(f"{s}: {len(rs)} utts, offline WER {100 * sum(r['e_off'] for r in rs) / n:.1f}, "
              f"stream WER {100 * sum(r['e_st'] for r in rs) / n:.1f}, identical tokens {sum(r['same'] for r in rs)}/{len(rs)}")
        for lo in (0, 5, 10, 15, 20):
            b = [r for r in rs if lo <= r["dur"] < lo + 5] if lo < 20 else [r for r in rs if r["dur"] >= 20]
            if b:
                nb = sum(r["n"] for r in b)
                print(f"   {lo:2d}{'-%ds' % (lo + 5) if lo < 20 else '+ s'}: n={len(b):3d} offline {100 * sum(r['e_off'] for r in b) / nb:5.1f} "
                      f"stream {100 * sum(r['e_st'] for r in b) / nb:5.1f}")


if __name__ == "__main__":
    main()
