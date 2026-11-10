"""Earlybird: predictive streaming speech-to-text in 17 MB.

    pip install torch "transformers>=5" soundfile huggingface_hub

    from earlybird import Earlybird
    eb = Earlybird.from_pretrained("sahilmahendrakar/Earlybird-29M")        # or a local folder
    print(eb.transcribe("speech.wav"))

    stream = eb.stream()                             # live: feed 16 kHz mono float32 audio
    for chunk in mic_chunks:                         # any chunk size
        for u in stream.feed(chunk):                 # one update per 80 ms of audio
            print(u.text, u.words, u.guess)

Each update has
    text   the transcript so far, including the word being spoken
    words  [(word, confidence)] for every word of `text`; the last one may be unfinished
    guess  [(word, confidence)] the next words Earlybird expects (up to 2)
    time   seconds of audio heard

Confidence is the product of the word's token probabilities, straight from the model.

From the command line:  python earlybird.py speech.wav [--model REPO_OR_DIR] [--live]

The weights file stores 4-bit codes (two per byte) with an fp16 scale and zero point per group of
128 weights; this file unpacks them in memory into the Moonshine streaming architecture in
Transformers and decodes incrementally, re-using the encoder and decoder work of earlier steps.
MIT license. Built on Moonshine by Useful Sensors (MIT)."""
import json, math, os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

SR = 16000
STEP = 1280          # 80 ms
WEIGHTS = "earlybird.safetensors"
FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", WEIGHTS]


# ---------------------------------------------------------------- loading

def _unpack(path):
    """Float state dict from the packed 4-bit file."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(path, "pt") as f:
        meta = json.loads(f.metadata()["earlybird"])
    t = load_file(path); sd = {}
    for name, m in meta.items():
        if m["bits"] == 16:
            sd[name] = t[name].float(); continue
        q = t[name + ".q"]
        if m["bits"] == 4:
            codes = torch.empty(q.numel() * 2, dtype=torch.uint8)
            codes[0::2] = q & 15; codes[1::2] = q >> 4
            codes = codes[: m["codes"]]
        else:
            codes = q
        s, z = t[name + ".s"].float(), t[name + ".z"].float()
        c = codes.reshape(s.shape[0], m["n"]).float()
        pad = s.shape[1] * m["group"] - m["n"]
        if pad:   # the last group was padded with the row's last weight when packing
            c = torch.cat([c, c[:, -1:].expand(-1, pad)], 1)
        g = c.reshape(s.shape[0], s.shape[1], m["group"]) * s[..., None] + z[..., None]
        sd[name] = g.reshape(s.shape[0], -1)[:, : m["n"]].reshape(m["shape"])
    return sd


def _local(repo):
    if os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, allow_patterns=FILES)


@dataclass
class Update:
    time: float
    text: str
    words: list = field(default_factory=list)
    guess: list = field(default_factory=list)


class Earlybird:
    @classmethod
    def from_pretrained(cls, repo, threads=1):
        from transformers import AutoConfig, AutoTokenizer, MoonshineStreamingForConditionalGeneration
        d = _local(repo)
        torch.set_num_threads(threads)
        m = MoonshineStreamingForConditionalGeneration(AutoConfig.from_pretrained(d)).eval()
        missing, unexpected = m.load_state_dict(_unpack(os.path.join(d, WEIGHTS)), strict=False)
        m.tie_weights()
        bad = [k for k in missing if not k.endswith("proj_out.weight")]
        assert not bad and not unexpected, (bad, unexpected)
        return cls(m, AutoTokenizer.from_pretrained(d))

    def __init__(self, model, tokenizer):
        self.model, self.tok = model, tokenizer

    def stream(self, guess=2):
        """A new live session. guess: how many next words to guess (0 to turn guessing off)."""
        return Stream(self, guess)

    def transcribe(self, audio, sr=SR):
        """Final text of a file path or a 1-D float array. Decodes the whole audio at once, which
        gives the same text as streaming it and is much faster."""
        s = self.stream(guess=0)
        s._step(load_audio(audio, sr))
        return s.update().text

    def updates(self, audio, sr=SR, guess=2):
        """Every 80 ms update for a whole file or array, as if it were streamed live."""
        x = load_audio(audio, sr)
        s = self.stream(guess)
        return s.feed(x) + [s.finish()]


def load_audio(audio, sr=SR):
    if isinstance(audio, (str, os.PathLike)):
        import soundfile as sf
        audio, sr = sf.read(audio, dtype="float32", always_2d=True)
        audio = audio.mean(1)
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(-1)
    if sr != SR:
        try:
            from scipy.signal import resample_poly
            g = math.gcd(sr, SR); x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        except ImportError:
            x = np.interp(np.arange(0, len(x) * SR / sr) * sr / SR, np.arange(len(x)), x).astype(np.float32)
    return x


# ---------------------------------------------------------------- streaming

class Stream:
    """Incremental decoding: gives the same text as re-encoding the whole audio heard so far and
    greedily re-decoding it at every 80 ms step, at a small, roughly constant cost per step.

    Encoder: causal convolutions and attention windows of 16 frames back and at most 3 ahead, so
    only the newest frames of each layer are recomputed. After the last real frame it appends the
    three frames the convolutions produce from silence padding (no future audio), as in training.
    Decoder: the previous text is checked in one parallel pass and decoding resumes at the first
    token that changed. After end-of-text it keeps decoding the guessed next words."""

    def __init__(self, eb, guess=2, tok_per_s=6.5, min_s=0.24):
        self.eb, self.n_guess = eb, guess
        m = eb.model
        self.m, self.enc, self.dec, self.tok = m, m.model.encoder, m.model.decoder, eb.tok
        self.cfg = m.config
        ecfg = self.cfg.encoder_config
        self.frame = self.enc.embedder.frame_len
        self.back = [w[0] for w in ecfg.sliding_windows]
        self.ahead = [max(0, w[1] - 1) for w in ecfg.sliding_windows]
        self.H = self.cfg.num_attention_heads
        self.tok_per_s, self.min_s = tok_per_s, min_s
        self.bos, self.eos = self.cfg.decoder_start_token_id, self.cfg.eos_token_id
        self._starts = {}
        self.reset()

    def reset(self):
        d = self.cfg.encoder_config.hidden_size
        self.n_samples = 0
        self.buf = np.zeros(0, dtype=np.float32)      # audio short of a whole 80 ms step
        self.rem = np.zeros(0, dtype=np.float32)      # samples short of a whole 5 ms frame
        self.c0, self.c0_n = torch.zeros(1, d, 0), 0
        self.c1, self.c1_n = torch.zeros(1, 2 * d, 0), 0
        nl = len(self.enc.layers)
        self.fin = [torch.zeros(1, 0, d) for _ in range(nl + 1)]
        hd = self.cfg.head_dim
        self.cross = [(torch.zeros(1, self.H, 0, hd), torch.zeros(1, self.H, 0, hd)) for _ in self.dec.layers]
        self.hyp, self.probs = [], []
        self.guess_toks, self.guess_probs = [], []

    # ---- public ----
    def feed(self, audio):
        """Add 16 kHz mono float32 samples; returns one Update per completed 80 ms step."""
        self.buf = np.concatenate([self.buf, np.asarray(audio, dtype=np.float32).reshape(-1)])
        out = []
        while len(self.buf) >= STEP:
            chunk, self.buf = self.buf[:STEP], self.buf[STEP:]
            self._step(chunk)
            out.append(self.update())
        return out

    def finish(self):
        """Decode whatever audio is left (less than one step) and return the last update."""
        if len(self.buf):
            chunk, self.buf = self.buf, np.zeros(0, dtype=np.float32)
            self._step(chunk)
        return self.update()

    def update(self):
        return Update(round(self.n_samples / SR, 3), self._text(self.hyp),
                      self._words(self.hyp, self.probs), self._words(self.guess_toks, self.guess_probs))

    # ---- words ----
    def _text(self, toks):
        return self.tok.decode(toks, skip_special_tokens=True).strip()

    def _is_start(self, t):
        if t not in self._starts:
            self._starts[t] = self.tok.convert_ids_to_tokens(t).startswith("▁")
        return self._starts[t]

    def _words(self, toks, probs):
        out, s = [], 0
        for i in range(1, len(toks) + 1):
            if i == len(toks) or self._is_start(toks[i]):
                w = self._text(toks[s:i])
                if w:
                    out.append((w, float(np.prod(probs[s:i]))))
                s = i
        return out

    # ---- front end ----
    def _conv(self, conv, buf, n_seen, new):
        x = torch.cat([buf, new], dim=2)
        first = n_seen + new.shape[2] - x.shape[2]
        k, s = conv.kernel_size[0], conv.stride[0]
        total = n_seen + new.shape[2]
        i0 = -(-n_seen // s)
        i1 = (total - 1) // s + 1 if total else 0
        if i1 <= i0:
            return x, total, x[:, :0, :0]
        lo = s * i0 - (k - 1)
        seg = x[:, :, max(0, lo - first): s * (i1 - 1) - first + 1]
        if lo < first:
            seg = F.pad(seg, (first - lo, 0))
        y = F.conv1d(seg, conv.weight, conv.bias, stride=s)
        keep = s * i1 - (k - 1)
        return x[:, :, max(0, keep - first):], total, y

    def _embed(self, chunk):
        e = self.enc.embedder
        x = np.concatenate([self.rem, chunk])
        n = len(x) // self.frame
        self.rem = x[n * self.frame:]
        if n == 0:
            return torch.zeros(1, 0, self.c0.shape[1])
        f = torch.from_numpy(x[: n * self.frame]).reshape(1, n, self.frame)
        h = F.silu(e.linear(e.comp(e.cmvn(f)))).transpose(1, 2)
        self.c0, self.c0_n, y = self._conv(e.conv1, self.c0, self.c0_n, h)
        if y.shape[2] == 0:
            return torch.zeros(1, 0, self.c0.shape[1])
        self.c1, self.c1_n, z = self._conv(e.conv2, self.c1, self.c1_n, F.silu(y))
        return z.transpose(1, 2)

    def _pad_frames(self):
        e = self.enc.embedder
        d = self.c0.shape[1]
        k1, s1 = e.conv1.kernel_size[0], e.conv1.stride[0]
        i0 = -(-self.c0_n // s1); i1 = (self.c0_n + k1 - 2) // s1 + 1
        if self.c0_n == 0 or i1 <= i0:
            return torch.zeros(1, 0, d)
        need = s1 * (i1 - 1) + 1 - self.c0_n
        _, _, y = self._conv(e.conv1, self.c0, self.c0_n, torch.zeros(1, d, need))
        y = F.silu(y)
        k2, s2 = e.conv2.kernel_size[0], e.conv2.stride[0]
        n1 = self.c1_n + y.shape[2]
        j1 = (n1 + k2 - 2) // s2 + 1
        need = s2 * (j1 - 1) + 1 - n1
        _, _, z = self._conv(e.conv2, self.c1, self.c1_n, torch.cat([y, torch.zeros(1, 2 * d, max(0, need))], dim=2))
        return z.transpose(1, 2)

    # ---- encoder ----
    def _enc_layer(self, layer, x, q0, back, ahead):
        a = layer.self_attn
        h = layer.input_layernorm(x)
        T, nq = h.shape[1], h.shape[1] - q0
        shp = lambda t: t.view(1, -1, self.H, a.head_dim).transpose(1, 2)
        q, k, v = shp(a.q_proj(h[:, q0:])), shp(a.k_proj(h)), shp(a.v_proj(h))
        dist = torch.arange(q0, T)[:, None] - torch.arange(T)[None, :]
        mask = ((dist >= 0) & (dist < back)) | ((dist < 0) & (-dist <= ahead))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = x[:, q0:] + a.o_proj(o.transpose(1, 2).reshape(1, nq, -1))
        return y + layer.mlp(layer.post_attention_layernorm(y))

    def _encode(self, new):
        self.fin[0] = torch.cat([self.fin[0], new], dim=1)
        prov = self._pad_frames()
        n_new_fin = 0
        for l, layer in enumerate(self.enc.layers):
            n_fin_in, done = self.fin[l].shape[1], self.fin[l + 1].shape[1]
            x = torch.cat([self.fin[l], prov], dim=1)
            lo = max(0, done - (self.back[l] - 1))
            y = self._enc_layer(layer, x[:, lo:], done - lo, self.back[l], self.ahead[l])
            n_new_fin = max(0, n_fin_in - self.ahead[l] - done)
            self.fin[l + 1] = torch.cat([self.fin[l + 1], y[:, :n_new_fin]], dim=1)
            prov = y[:, n_new_fin:]
        return n_new_fin, prov

    def _cross_kv(self, x, start):
        x = self.enc.final_norm(x)
        x = self.dec.proj(x + self.dec.pos_emb.weight[start: start + x.shape[1]])
        out = []
        for layer in self.dec.layers:
            a = layer.encoder_attn
            shp = lambda t: t.view(1, -1, self.H, a.head_dim).transpose(1, 2)
            out.append((shp(a.k_proj(x)), shp(a.v_proj(x))))
        return out

    # ---- decoder ----
    def _rope(self, x, pos):
        r = self.dec.rotary_emb
        fr = pos[:, None].float() * r.inv_freq[None, :]
        cos = fr.cos().repeat_interleave(2, dim=-1); sin = fr.sin().repeat_interleave(2, dim=-1)
        n = cos.shape[-1]
        xr, xp = x[..., :n], x[..., n:]
        rot = torch.stack((-xr[..., 1::2], xr[..., 0::2]), dim=-1).flatten(-2)
        return torch.cat([xr * cos + rot * sin, xp], dim=-1)

    def _dec(self, ids, pos0, cache, cross):
        x = self.dec.embed_tokens(torch.tensor([ids]))
        L = len(ids); pos = torch.arange(pos0, pos0 + L)
        for i, layer in enumerate(self.dec.layers):
            a = layer.self_attn
            shp = lambda t: t.view(1, -1, self.H, a.head_dim).transpose(1, 2)
            h = layer.input_layernorm(x)
            q, k, v = self._rope(shp(a.q_proj(h)), pos), self._rope(shp(a.k_proj(h)), pos), shp(a.v_proj(h))
            if cache[i] is not None:
                k = torch.cat([cache[i][0], k], dim=2); v = torch.cat([cache[i][1], v], dim=2)
            cache[i] = (k, v)
            if L > 1:
                P = k.shape[2] - L
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=torch.ones(L, P + L, dtype=torch.bool).tril(P))
            else:
                o = F.scaled_dot_product_attention(q, k, v)
            x = x + a.o_proj(o.transpose(1, 2).reshape(1, L, -1))
            c = layer.encoder_attn
            q = shp(c.q_proj(layer.post_attention_layernorm(x)))
            o = F.scaled_dot_product_attention(q, cross[i][0], cross[i][1])
            x = x + c.o_proj(o.transpose(1, 2).reshape(1, L, -1))
            x = x + layer.mlp(layer.final_layernorm(x))
        return self.m.proj_out(self.dec.norm(x))[0]

    def _decode(self, cross, max_new):
        prev = self.hyp
        cache = [None] * len(self.dec.layers)
        logp = F.log_softmax(self._dec([self.bos] + prev, 0, cache, cross), dim=-1)
        best = logp.argmax(-1).tolist()
        keep = 0
        while keep < len(prev) and keep < max_new and best[keep] == prev[keep]:
            keep += 1
        toks = prev[:keep]; probs = [math.exp(logp[i, t]) for i, t in enumerate(toks)]
        cache = [(k[:, :, : keep + 1], v[:, :, : keep + 1]) for k, v in cache]
        row = logp[keep]
        while len(toks) < max_new:
            t = int(row.argmax())
            if t == self.eos:
                break
            toks.append(t); probs.append(math.exp(row[t]))
            if len(toks) >= max_new:
                break
            row = F.log_softmax(self._dec([t], len(toks), cache, cross)[-1], dim=-1)
        self.hyp, self.probs = toks, probs
        self.guess_toks, self.guess_probs = [], []
        if self.n_guess and len(toks) < max_new:
            self._guess(cache, cross, len(toks) + 1)

    def _guess(self, cache, cross, pos):
        g, p, words, t = [], [], 0, self.eos
        for _ in range(4 * self.n_guess + 2):
            row = F.log_softmax(self._dec([t], pos, cache, cross)[-1], dim=-1); pos += 1
            t = int(row.argmax())
            if t == self.eos:
                break
            if self._is_start(t):
                words += 1
                if words > self.n_guess:
                    break
            g.append(t); p.append(math.exp(row[t]))
        self.guess_toks, self.guess_probs = g, p

    @torch.no_grad()
    def _step(self, chunk):
        self.n_samples += len(chunk)
        n_fin, prov = self._encode(self._embed(chunk))
        fin_all = self.fin[-1]
        if n_fin:
            kv = self._cross_kv(fin_all[:, -n_fin:], fin_all.shape[1] - n_fin)
            self.cross = [(torch.cat([c[0], k], 2), torch.cat([c[1], v], 2)) for c, (k, v) in zip(self.cross, kv)]
        t = self.n_samples / SR
        if t < self.min_s or fin_all.shape[1] + prov.shape[1] == 0:
            return
        cross = self.cross
        if prov.shape[1]:
            kv = self._cross_kv(prov, fin_all.shape[1])
            cross = [(torch.cat([c[0], k], 2), torch.cat([c[1], v], 2)) for c, (k, v) in zip(self.cross, kv)]
        self._decode(cross, int(t * self.tok_per_s) + 4)


# ---------------------------------------------------------------- command line

def _show(u):
    words = " ".join(w for w, _ in u.words)
    guess = " ".join(w for w, _ in u.guess)
    return f"{u.time:6.2f}s  {words}" + (f"  [{guess}]" if guess else "")


if __name__ == "__main__":
    import argparse, sys, time
    ap = argparse.ArgumentParser(description="Stream a 16 kHz file through Earlybird.")
    ap.add_argument("audio")
    ap.add_argument("--model", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--live", action="store_true", help="play it back at real-time speed")
    a = ap.parse_args()
    eb = Earlybird.from_pretrained(a.model)
    x = load_audio(a.audio)
    s = eb.stream()
    last = None
    for i in range(0, len(x), STEP):
        for u in s.feed(x[i: i + STEP]):
            line = _show(u)
            if line[9:] != last:
                print(line, flush=True); last = line[9:]
        if a.live:
            time.sleep(STEP / SR)
    u = s.finish()
    print("\n" + " ".join(f"{w} ({p:.0%})" for w, p in u.words))
