"""Incremental no-hold-back decoding of a Moonshine streaming checkpoint (PyTorch, CPU).

Gives the same text as re-encoding and greedily re-decoding the whole audio prefix at
every step (what sim_hf.py does), at a small, roughly constant cost per step:

- Front end and encoder: the convolutions are causal and the encoder has no positions,
  only attention windows of 16 frames back and at most 3 ahead per layer. A frame's
  output in a layer is final once its window lies inside audio that has arrived, so only
  the newest frames of each layer are recomputed (about 60 frame-layers per 80 ms step
  instead of 6 times the number of frames so far).
- Cross-attention keys and values are cached for final encoder frames.
- Decoder: the previous hypothesis is checked in one parallel pass. Greedy decoding from
  scratch would reproduce it up to the first position where the best token differs, so
  decoding resumes from there, token by token, with the pass's attention cache.

`step` returns the text; `self.last` holds the tokens, their probabilities and how many
tokens were re-decoded.

Options:
  commit=N  words more than N back from the end of the text are never rewritten.
  pad=True  append the three frames the convolutions produce from silence padding after
            the last real frame. The Transformers library does this whenever a batch is
            padded, so the released weights were presumably trained with it; they are
            earlier and steadier with it. The frames hold no future audio and are
            recomputed at every step.
  ghost=N   after end-of-text keep decoding N more words (`self.ghost`, with `self.ghost_p`,
            the probability of its first word, `self.ghost_word_p` each word's; `self.ghost_eos`
            when the guess itself ends in end-of-text, a claim that the whole utterance is
            known, with `self.ghost_done_p` = the guess's probability times end-of-text's). Only meaningful for checkpoints trained with
            ghost targets (train.py).
  gates=[..] also return, for each threshold, the text with trailing words hidden while
            their least certain token is below it; `step` then returns {name: text}."""
import json, math, os
import numpy as np, torch
import torch.nn.functional as F

SR = 16000


class _Remap:
    """The original tokenizer seen through a pruned vocabulary: id i of the model is
    original id keep[i]."""
    def __init__(self, tok, keep):
        self.t, self.keep = tok, keep

    def decode(self, ids, **kw):
        return self.t.decode([self.keep[i] for i in ids], **kw)

    def convert_ids_to_tokens(self, i):
        return self.t.convert_ids_to_tokens(self.keep[i])


class Engine:
    def __init__(self, model="UsefulSensors/moonshine-streaming-tiny", threads=1, tok_per_s=6.5,
                 min_ms=240, ghost=0, gates=(), pad=False, commit=0):
        self.pad, self.commit = pad, commit
        self.n_ghost, self.gates = ghost, list(gates)
        from transformers import AutoTokenizer, MoonshineStreamingForConditionalGeneration
        torch.set_num_threads(threads)
        m = MoonshineStreamingForConditionalGeneration.from_pretrained(model).eval()
        self.m, self.enc, self.dec = m, m.model.encoder, m.model.decoder
        self.tok = AutoTokenizer.from_pretrained(model)
        if os.path.exists(f"{model}/kept_ids.json"):   # a pruned vocabulary (scripts/prune_vocab.py)
            self.tok = _Remap(self.tok, json.load(open(f"{model}/kept_ids.json")))
        self.cfg = m.config
        ecfg = self.cfg.encoder_config
        self.frame = self.enc.embedder.frame_len
        # frames visible ahead of the query: the mask keeps -dist < right, so right - 1
        self.back = [w[0] for w in ecfg.sliding_windows]
        self.ahead = [max(0, w[1] - 1) for w in ecfg.sliding_windows]
        self.H = self.cfg.num_attention_heads
        self.tok_per_s, self.min_s = tok_per_s, min_ms / 1000
        self.bos, self.eos = self.cfg.decoder_start_token_id, self.cfg.eos_token_id
        self.start()

    # ---- state ----
    def start(self):
        d = self.cfg.encoder_config.hidden_size
        self.n_samples = 0
        self.rem = np.zeros(0, dtype=np.float32)      # samples short of a whole 5 ms frame
        self.c0 = torch.zeros(1, d, 0)                # linear output (conv1 input), tail kept
        self.c0_n = 0
        self.c1 = torch.zeros(1, 2 * d, 0)            # conv1 output (conv2 input), tail kept
        self.c1_n = 0
        nl = len(self.enc.layers)
        self.fin = [torch.zeros(1, 0, d) for _ in range(nl + 1)]   # final inputs of layer l; [nl] = final outputs
        self.cross = [(torch.zeros(1, self.H, 0, self.cfg.head_dim), torch.zeros(1, self.H, 0, self.cfg.head_dim))
                      for _ in self.dec.layers]
        self.hyp = []
        self.last = dict(tokens=[], probs=[], redecoded=0)
        self.ghost, self.ghost_p, self.ghost_word_p = "", 0.0, []
        self.ghost_eos, self.ghost_done_p = False, 0.0

    # ---- front end ----
    def _conv(self, conv, buf, n_seen, new):
        """Causal strided conv over a stream: `buf` holds the inputs still needed."""
        x = torch.cat([buf, new], dim=2)
        first = n_seen + new.shape[2] - x.shape[2]            # stream index of x[..., 0]
        k, s = conv.kernel_size[0], conv.stride[0]
        total = n_seen + new.shape[2]
        # output i reads inputs s*i - (k-1) .. s*i; outputs already produced: ceil(n_seen / s)
        i0 = -(-n_seen // s)
        i1 = (total - 1) // s + 1 if total else 0
        if i1 <= i0:
            return x, total, x[:, :0, :0]
        lo = s * i0 - (k - 1)
        seg = x[:, :, max(0, lo - first): s * (i1 - 1) - first + 1]
        if lo < first:  # only at the very start of the stream (left padding)
            seg = F.pad(seg, (first - lo, 0))
        y = F.conv1d(seg, conv.weight, conv.bias, stride=s)
        keep = s * i1 - (k - 1)                                # first input the next output needs
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

    # ---- encoder ----
    def _enc_layer(self, layer, x, q0, back, ahead):
        """Outputs of `layer` for frames q0.. of x (x starts at frame 0 of its own slice)."""
        a = layer.self_attn
        h = layer.input_layernorm(x)
        T, nq = h.shape[1], h.shape[1] - q0
        shp = lambda t: t.view(1, -1, self.H, a.head_dim).transpose(1, 2)
        q, k, v = shp(a.q_proj(h[:, q0:])), shp(a.k_proj(h)), shp(a.v_proj(h))
        qi = torch.arange(q0, T)[:, None]; ki = torch.arange(T)[None, :]
        dist = qi - ki
        mask = ((dist >= 0) & (dist < back)) | ((dist < 0) & (-dist <= ahead))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        o = a.o_proj(o.transpose(1, 2).reshape(1, nq, -1))
        y = x[:, q0:] + o
        return y + layer.mlp(layer.post_attention_layernorm(y))

    def _pad_frames(self):
        """The frames a zero-padded batch would add after the audio so far (see `pad`)."""
        e = self.enc.embedder
        d = self.c0.shape[1]
        k1, s1 = e.conv1.kernel_size[0], e.conv1.stride[0]
        i0 = -(-self.c0_n // s1); i1 = (self.c0_n + k1 - 2) // s1 + 1     # outputs that still touch real input
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

    def _encode(self, new):
        """Take new embedder frames; return (new final encoder frames, provisional frames)."""
        self.fin[0] = torch.cat([self.fin[0], new], dim=1)
        # provisional inputs of the current layer
        prov = self._pad_frames() if self.pad else torch.zeros(1, 0, new.shape[2])
        for l, layer in enumerate(self.enc.layers):
            n_fin_in = self.fin[l].shape[1]
            done = self.fin[l + 1].shape[1]           # final outputs already stored
            x = torch.cat([self.fin[l], prov], dim=1)
            lo = max(0, done - (self.back[l] - 1))
            y = self._enc_layer(layer, x[:, lo:], done - lo, self.back[l], self.ahead[l])
            n_new_fin = max(0, n_fin_in - self.ahead[l] - done)
            self.fin[l + 1] = torch.cat([self.fin[l + 1], y[:, :n_new_fin]], dim=1)
            prov = y[:, n_new_fin:]
        return n_new_fin, prov

    def _cross_kv(self, x, start):
        """Keys and values of every decoder layer for encoder frames x at positions start.."""
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
        """Decoder over `ids` at positions pos0.., extending the self-attention cache."""
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
                mask = torch.ones(L, P + L, dtype=torch.bool).tril(P)
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            else:
                o = F.scaled_dot_product_attention(q, k, v)
            x = x + a.o_proj(o.transpose(1, 2).reshape(1, L, -1))
            c = layer.encoder_attn
            q = shp(c.q_proj(layer.post_attention_layernorm(x)))
            o = F.scaled_dot_product_attention(q, cross[i][0], cross[i][1])
            x = x + c.o_proj(o.transpose(1, 2).reshape(1, L, -1))
            x = x + layer.mlp(layer.final_layernorm(x))
        return self.m.proj_out(self.dec.norm(x))[0]

    def _decode(self, cross, max_new, forced=()):
        prev = self.hyp
        if self.commit:   # freeze everything but the last `commit` words
            starts = [i for i, t in enumerate(prev) if self._is_word_start(t)]
            forced = prev[: starts[-self.commit]] if len(starts) > self.commit else []
        cache = [None] * len(self.dec.layers)
        logp = F.log_softmax(self._dec([self.bos] + prev, 0, cache, cross), dim=-1)
        best = logp.argmax(-1).tolist()
        keep = 0
        while keep < len(prev) and keep < max_new and (best[keep] == prev[keep] or keep < len(forced)):
            keep += 1
        toks = prev[:keep]; probs = [math.exp(logp[i, t]) for i, t in enumerate(toks)]
        cache = [(k[:, :, : keep + 1], v[:, :, : keep + 1]) for k, v in cache]
        row = logp[keep]; n_re = 0
        while len(toks) < max_new:
            t = int(row.argmax())
            if t == self.eos:
                break
            toks.append(t); probs.append(math.exp(row[t])); n_re += 1
            if len(toks) >= max_new:
                break
            row = F.log_softmax(self._dec([t], len(toks), cache, cross)[-1], dim=-1)
        self.hyp = toks
        self.last = dict(tokens=toks, probs=probs, redecoded=n_re)
        self.ghost, self.ghost_p, self.ghost_word_p = "", 0.0, []
        self.ghost_eos, self.ghost_done_p = False, 0.0
        if self.n_ghost and len(toks) < max_new:
            self._ghost(cache, cross, len(toks) + 1)
        return self._out()

    def _is_word_start(self, t):
        return self.tok.convert_ids_to_tokens(t).startswith("\u2581")

    def _ghost(self, cache, cross, pos):
        """Decode on past end-of-text for n_ghost words."""
        g, p, words, t = [], [], 0, self.eos
        self.ghost_eos, self.ghost_done_p = False, 0.0
        for _ in range(4 * self.n_ghost + 2):
            row = F.log_softmax(self._dec([t], pos, cache, cross)[-1], dim=-1); pos += 1
            t = int(row.argmax())
            if t == self.eos:   # the guess ends the utterance: a claim that the whole text is known
                self.ghost_eos, self.ghost_done_p = True, float(np.prod(p)) * math.exp(row[t]) if p else math.exp(row[t])
                break
            if self._is_word_start(t):
                words += 1
                if words > self.n_ghost:
                    break
            g.append(t); p.append(math.exp(row[t]))
        self.ghost = self.tok.decode(g, skip_special_tokens=True).strip()
        # probability of each ghost word: product over its tokens (ghost_p: the first word's)
        starts = [i for i in range(len(g)) if i == 0 or self._is_word_start(g[i])] + [len(g)]
        self.ghost_word_p = [float(np.prod(p[s:e])) for s, e in zip(starts, starts[1:])] if g else []
        self.ghost_p = self.ghost_word_p[0] if g else 0.0

    def _gated(self, thr):
        toks, probs = self.last["tokens"], self.last["probs"]
        n = len(toks)
        while n:
            s = n - 1
            while s > 0 and not self._is_word_start(toks[s]):
                s -= 1
            if min(probs[s:n]) >= thr:
                break
            n = s
        return self.tok.decode(toks[:n], skip_special_tokens=True).strip()

    def _out(self):
        text = self.tok.decode(self.hyp, skip_special_tokens=True).strip()
        if not self.gates:
            return text
        out = {"": text}
        for g in self.gates:
            out[f"gate{int(g * 100)}"] = self._gated(g)
        return out

    # ---- recognizer interface ----
    @torch.no_grad()
    def step(self, chunk):
        self.n_samples += len(chunk)
        new = self._embed(np.asarray(chunk, dtype=np.float32))
        n_fin, prov = self._encode(new)
        fin_all = self.fin[-1]
        if n_fin:
            kv = self._cross_kv(fin_all[:, -n_fin:], fin_all.shape[1] - n_fin)
            self.cross = [(torch.cat([c[0], k], 2), torch.cat([c[1], v], 2)) for c, (k, v) in zip(self.cross, kv)]
        t = self.n_samples / SR
        if t < self.min_s or fin_all.shape[1] + prov.shape[1] == 0:
            return self._out()
        cross = self.cross
        if prov.shape[1]:
            kv = self._cross_kv(prov, fin_all.shape[1])
            cross = [(torch.cat([c[0], k], 2), torch.cat([c[1], v], 2)) for c, (k, v) in zip(self.cross, kv)]
        return self._decode(cross, int(t * self.tok_per_s) + 4)

    def finish(self):
        text = self.tok.decode(self.hyp, skip_special_tokens=True).strip()
        return {k: text for k in [""] + [f"gate{int(g * 100)}" for g in self.gates]} if self.gates else text
