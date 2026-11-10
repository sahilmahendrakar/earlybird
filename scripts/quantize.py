"""Weight-only quantization, simulated: round every weight matrix to BITS bits in groups of
GROUP along its input dimension (asymmetric, one fp16 scale and zero point per group), and
save the result back in float so Engine runs it unchanged. WER of the saved checkpoint is
the WER a real BITS-bit build would have (the arithmetic is the same up to float rounding);
the size printed (and written to OUT/size.txt) is what that build would take on disk.

Quantized: every weight with two or more dimensions (linear layers, the word table, the
per-frame position table, convolutions flattened to out x in*k). Kept in fp16: norms and biases.
--keep REGEX leaves the matching weights in fp16 (e.g. the audio front end), or rounds them to
--keep-bits instead.
usage: quantize.py MODEL OUT --bits 4 [--group 128] [--keep REGEX [--keep-bits 8]] [--list]"""
import argparse, json, os, re, shutil
import torch


def fake_quant(w, bits, group):
    shape = w.shape; w2 = w.reshape(shape[0], -1).float()
    n = w2.shape[1]; pad = (-n) % group
    if pad:
        w2 = torch.cat([w2, w2[:, -1:].expand(-1, pad)], 1)
    g = w2.reshape(w2.shape[0], -1, group)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    q = 2 ** bits - 1
    scale = ((hi - lo) / q).clamp(min=1e-8).half().float(); zero = lo.half().float()
    out = (((g - zero) / scale).round().clamp(0, q) * scale + zero).reshape(w2.shape[0], -1)[:, :n]
    return out.reshape(shape).to(w.dtype), g.shape[0] * g.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--bits", type=int, required=True); ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--keep", help="weights whose name matches stay fp16 (or get --keep-bits)")
    ap.add_argument("--keep-bits", type=int, default=0)
    ap.add_argument("--list", action="store_true", help="print every weight's name and shape")
    a = ap.parse_args()
    from transformers import AutoTokenizer, MoonshineStreamingForConditionalGeneration as M
    m = M.from_pretrained(a.model).eval()
    seen = set(); bits_total = 0; nq = nk = 0; err = []
    with torch.no_grad():
        for name, p in m.named_parameters():
            if id(p) in seen:   # tied word table: once
                continue
            seen.add(id(p))
            if a.list:
                print(name, tuple(p.shape))
            kept = bool(a.keep and re.search(a.keep, name))
            bits = a.keep_bits if kept else a.bits
            if p.dim() >= 2 and bits:
                q, groups = fake_quant(p.data, bits, a.group)
                err.append(float((q - p.data).norm() / p.data.norm()))
                p.data.copy_(q); bits_total += p.numel() * bits + groups * 32; nq += p.numel()
            else:
                bits_total += p.numel() * 16; nk += p.numel()
    os.makedirs(a.out, exist_ok=True)
    m.save_pretrained(a.out)
    AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)
    if os.path.exists(f"{a.model}/kept_ids.json"):
        shutil.copy(f"{a.model}/kept_ids.json", a.out)
    mb = bits_total / 8 / 1e6
    line = (f"{a.out}: {a.bits}-bit" + (f" ({a.keep} {a.keep_bits}-bit)" if a.keep and a.keep_bits else "") + f", group {a.group}: {nq:,} weights quantized, {nk:,} kept in fp16; "
            f"{mb:.1f} MB; relative weight error mean {sum(err) / len(err):.4f}")
    open(f"{a.out}/size.txt", "w").write(line + "\n"); print(line)


if __name__ == "__main__":
    main()
