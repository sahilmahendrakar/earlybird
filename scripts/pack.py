"""A real low-bit file: store a quantized Earlybird checkpoint as packed integer codes (two 4-bit
codes per byte, or one byte for 8-bit) with one fp16 scale and zero point per group of GROUP
input weights, the exact scheme quantize.py simulates. Norms and biases stay fp16.

pack    MODEL OUT --bits 4 [--keep REGEX --keep-bits 8] [--group 128]
        writes OUT/earlybird.safetensors (+ config, tokenizer, kept_ids) and OUT/size.txt with the
        real bytes on disk.
unpack  PACKED DIR   rebuilds an ordinary float checkpoint the engine loads, from the packed file only.
check   PACKED REF   unpacks in memory and reports the largest difference to REF's weights (REF = the
        simulated checkpoint quantize.py wrote); 0 means the file is exactly what was evaluated."""
import argparse, glob, json, os, re, shutil
import torch
from safetensors.torch import save_file, load_file

SIDE = ["config.json", "generation_config.json", "kept_ids.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "tokenizer.model", "vocab.json", "merges.txt", "added_tokens.json"]


def quant(w, bits, group):
    shape = w.shape; w2 = w.reshape(shape[0], -1).float()
    n = w2.shape[1]; pad = (-n) % group
    if pad:
        w2 = torch.cat([w2, w2[:, -1:].expand(-1, pad)], 1)
    g = w2.reshape(w2.shape[0], -1, group)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    q = 2 ** bits - 1
    scale = ((hi - lo) / q).clamp(min=1e-8).half(); zero = lo.half()
    codes = ((g - zero.float()) / scale.float()).round().clamp(0, q).to(torch.uint8)
    return codes, scale.squeeze(-1), zero.squeeze(-1), n


def nib(codes):   # uint8 codes 0..15 -> two per byte
    c = codes.reshape(-1)
    if c.numel() % 2:
        c = torch.cat([c, c.new_zeros(1)])
    return (c[0::2] | (c[1::2] << 4)).contiguous()


def unnib(b, count):
    out = torch.empty(b.numel() * 2, dtype=torch.uint8); out[0::2] = b & 15; out[1::2] = b >> 4
    return out[:count]


def pack(a):
    from transformers import MoonshineStreamingForConditionalGeneration as M
    m = M.from_pretrained(a.model).eval()
    tensors, meta, seen = {}, {}, set()
    for name, p in m.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        kept = bool(a.keep and re.search(a.keep, name)); bits = a.keep_bits if kept else a.bits
        if p.dim() >= 2 and bits:
            codes, scale, zero, n = quant(p.data, bits, a.group)
            codes = codes.reshape(codes.shape[0], -1)[:, :n]   # the padding codes are never read back
            tensors[name + ".q"] = nib(codes) if bits == 4 else codes.reshape(-1).contiguous()
            tensors[name + ".s"] = scale.contiguous(); tensors[name + ".z"] = zero.contiguous()
            meta[name] = dict(bits=bits, shape=list(p.shape), n=n, group=a.group, codes=codes.numel())
        else:
            tensors[name] = p.data.half().contiguous(); meta[name] = dict(bits=16, shape=list(p.shape))
    os.makedirs(a.out, exist_ok=True)
    save_file(tensors, f"{a.out}/earlybird.safetensors", metadata={"earlybird": json.dumps(meta)})
    for f in SIDE:
        if os.path.exists(f"{a.model}/{f}"):
            shutil.copy(f"{a.model}/{f}", a.out)
    w = os.path.getsize(f"{a.out}/earlybird.safetensors")
    tot = sum(os.path.getsize(p) for p in glob.glob(f"{a.out}/*") if not p.endswith("size.txt"))
    line = f"{a.out}: weights file {w / 1e6:.2f} MB; with config and tokenizer {tot / 1e6:.2f} MB"
    open(f"{a.out}/size.txt", "w").write(line + "\n"); print(line)


def state(packed):
    from safetensors import safe_open
    with safe_open(f"{packed}/earlybird.safetensors", "pt") as f:
        meta = json.loads(f.metadata()["earlybird"])
    t = load_file(f"{packed}/earlybird.safetensors"); sd = {}
    for name, m in meta.items():
        if m["bits"] == 16:
            sd[name] = t[name].float(); continue
        codes = unnib(t[name + ".q"], m["codes"]) if m["bits"] == 4 else t[name + ".q"]
        s, z = t[name + ".s"].float(), t[name + ".z"].float()
        c = codes.reshape(s.shape[0], m["n"]).float()
        pad = s.shape[1] * m["group"] - m["n"]
        if pad:   # the last group's padding repeated the row's last weight; its codes are not stored
            c = torch.cat([c, c[:, -1:].expand(-1, pad)], 1)
        g = c.reshape(s.shape[0], s.shape[1], m["group"]) * s[..., None] + z[..., None]
        sd[name] = g.reshape(s.shape[0], -1)[:, : m["n"]].reshape(m["shape"])
    return sd


def model_from(packed):
    from transformers import AutoConfig, MoonshineStreamingForConditionalGeneration as M
    m = M(AutoConfig.from_pretrained(packed)).eval()
    missing, unexpected = m.load_state_dict(state(packed), strict=False)
    m.tie_weights()
    bad = [k for k in missing if not k.endswith("proj_out.weight")]
    assert not bad and not unexpected, (bad, unexpected)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack"); p.add_argument("model"); p.add_argument("out")
    p.add_argument("--bits", type=int, default=4); p.add_argument("--group", type=int, default=128)
    p.add_argument("--keep"); p.add_argument("--keep-bits", type=int, default=0)
    u = sub.add_parser("unpack"); u.add_argument("packed"); u.add_argument("dir")
    c = sub.add_parser("check"); c.add_argument("packed"); c.add_argument("ref")
    a = ap.parse_args()
    if a.cmd == "pack":
        pack(a)
    elif a.cmd == "unpack":
        m = model_from(a.packed); m.save_pretrained(a.dir)
        for f in SIDE:
            if os.path.exists(f"{a.packed}/{f}"):
                shutil.copy(f"{a.packed}/{f}", a.dir)
        print("unpacked to", a.dir)
    else:
        from transformers import MoonshineStreamingForConditionalGeneration as M
        ref = M.from_pretrained(a.ref).eval(); got = model_from(a.packed)
        rs = dict(ref.named_parameters()); worst = 0.0; wn = ""
        for k, v in got.named_parameters():
            d = float((v.float() - rs[k].float()).abs().max())
            if d > worst:
                worst, wn = d, k
        print(f"largest weight difference vs {a.ref}: {worst:.3g} ({wn})")
