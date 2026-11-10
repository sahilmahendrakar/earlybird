"""Feasibility export for an in-browser Earlybird: the two halves of one 80 ms step as ONNX graphs.

encoder.onnx   audio window [1, N samples] -> encoder frames [1, F, d] (front end + all encoder
               layers with their sliding windows + final norm; no state: the browser re-encodes a
               rolling window of recent audio each step)
decoder.onnx   token ids [1, T] + encoder frames [1, F, d] -> next-token logits [1, T, V] (the
               verify pass over the hypothesis; greedy extension re-runs it one token longer)
Both are written in float32 and with 4-bit weights (MatMulNBits, block 128) if onnxruntime's
quantizer is available.
usage: export_onnx.py MODEL OUT_DIR"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
import train


class Enc(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, wav):
        n = torch.full((1,), wav.shape[1], dtype=torch.long)
        h, _ = train.encode(self.m, wav, n)
        return h


class Dec(torch.nn.Module):
    """The engine's decoder pass (engine.Engine._dec + _cross_kv) over a whole hypothesis, written
    with plain tensor ops so it traces: causal self-attention with rotary positions, cross-attention
    to the (already final-normed) encoder frames plus their position table."""
    def __init__(self, m):
        super().__init__(); self.m = m; self.dec = m.model.decoder; self.H = m.config.num_attention_heads
        self.rot_n = 2 * self.dec.rotary_emb.inv_freq.shape[0]

    def rope(self, x, pos):
        r = self.dec.rotary_emb
        fr = pos[:, None].float() * r.inv_freq[None, :]
        c, s_ = fr.cos(), fr.sin()
        cos = torch.stack((c, c), dim=-1).reshape(c.shape[0], -1); sin = torch.stack((s_, s_), dim=-1).reshape(c.shape[0], -1)
        n = self.rot_n; xr = x[..., :n]
        ev, od = xr[..., 0::2], xr[..., 1::2]
        rot = torch.stack((-od, ev), dim=-1).reshape(xr.shape)
        y = xr * cos + rot * sin
        return y if n == x.shape[-1] else torch.cat([y, x[..., n:]], dim=-1)

    def forward(self, ids, enc):
        F_ = torch.nn.functional; L = ids.shape[1]; Fn = enc.shape[1]
        e = self.dec.proj(enc + self.dec.pos_emb.weight[:Fn][None])
        x = self.dec.embed_tokens(ids); pos = torch.arange(L)
        causal = torch.ones(L, L, dtype=torch.bool).tril()
        for layer in self.dec.layers:
            a = layer.self_attn; hd = a.head_dim
            shp = lambda t: t.reshape(1, -1, self.H, hd).transpose(1, 2)
            h = layer.input_layernorm(x)
            q, k, v = self.rope(shp(a.q_proj(h)), pos), self.rope(shp(a.k_proj(h)), pos), shp(a.v_proj(h))
            o = F_.scaled_dot_product_attention(q, k, v, attn_mask=causal)
            x = x + a.o_proj(o.transpose(1, 2).reshape(1, L, -1))
            c = layer.encoder_attn
            q = shp(c.q_proj(layer.post_attention_layernorm(x)))
            o = F_.scaled_dot_product_attention(q, shp(c.k_proj(e)), shp(c.v_proj(e)))
            x = x + c.o_proj(o.transpose(1, 2).reshape(1, L, -1))
            x = x + layer.mlp(layer.final_layernorm(x))
        return self.m.proj_out(self.dec.norm(x))


def _asinh(x):   # the TorchScript ONNX exporter has no asinh; same function, exportable ops
    return torch.sign(x) * torch.log(x.abs() + torch.sqrt(x * x + 1))


def main():
    torch.asinh = _asinh; torch.Tensor.asinh = _asinh
    from transformers import MoonshineStreamingForConditionalGeneration as M
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    m = M.from_pretrained(src, attn_implementation="eager").eval()
    wav = torch.zeros(1, 16000 * 2); ids = torch.ones(1, 12, dtype=torch.long)
    with torch.no_grad():
        enc = Enc(m)(wav)
        torch.onnx.export(Enc(m), (wav,), f"{out}/encoder.onnx", input_names=["wav"], output_names=["frames"],
                          dynamic_axes={"wav": {1: "n"}, "frames": {1: "f"}}, opset_version=17, dynamo=False)
        torch.onnx.export(Dec(m), (ids, enc), f"{out}/decoder.onnx", input_names=["ids", "frames"], output_names=["logits"],
                          dynamic_axes={"ids": {1: "t"}, "frames": {1: "f"}, "logits": {1: "t"}}, opset_version=17, dynamo=False)
    for f in ("encoder", "decoder"):
        print(f, f"{os.path.getsize(f'{out}/{f}.onnx') / 1e6:.1f} MB float32")
    try:
        import onnx
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
        for f in ("encoder", "decoder"):
            q = MatMulNBitsQuantizer(onnx.load(f"{out}/{f}.onnx"), block_size=128, is_symmetric=False, bits=4)
            q.process(); q.model.save_model_to_file(f"{out}/{f}.q4.onnx", use_external_data_format=False)
            print(f, f"{os.path.getsize(f'{out}/{f}.q4.onnx') / 1e6:.1f} MB 4-bit")
    except Exception as e:
        print("4-bit export skipped:", repr(e)[:300])
    # reference timings on this CPU, 1 thread, for comparison with the browser
    import time, onnxruntime as ort, numpy as np
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    for tag in ("", ".q4"):
        if not os.path.exists(f"{out}/encoder{tag}.onnx"):
            continue
        e = ort.InferenceSession(f"{out}/encoder{tag}.onnx", so); d = ort.InferenceSession(f"{out}/decoder{tag}.onnx", so)
        w = np.random.randn(1, 38400).astype(np.float32) * 0.05
        fr = e.run(None, {"wav": w})[0]; big = np.random.randn(1, 500, fr.shape[2]).astype(np.float32)
        t0 = time.perf_counter()
        for _ in range(10): e.run(None, {"wav": w})
        te = (time.perf_counter() - t0) / 10 * 1e3
        t0 = time.perf_counter()
        for _ in range(10): d.run(None, {"ids": np.ones((1, 40), np.int64), "frames": big})
        td = (time.perf_counter() - t0) / 10 * 1e3
        t0 = time.perf_counter()
        for _ in range(10): d.run(None, {"ids": np.ones((1, 41), np.int64), "frames": big})
        print(f"onnxruntime CPU 1 thread{tag or ' fp32'}: encoder 2.4 s window {te:.1f} ms; decoder 40 tokens x 500 frames {td:.1f} ms")


if __name__ == "__main__":
    main()
