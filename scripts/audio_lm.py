"""Baseline after Yusuf et al. 2024 (speculative speech recognition by audio-prefixed language
models): a small language model guesses the next word from the transcript so far AND a summary
of the recent audio, so it can hear the start of the word being spoken.

  audio    the last WINDOW seconds before the cut through a frozen streaming encoder (our p29's,
           same front end as the recognizers here), its last FRAMES frames mapped by a trained
           linear adapter into the language model's embedding space: a soft prompt.
  text     the words fully spoken by the cut (the reference in training; at test time whatever
           the recognizer has on screen, ghost_demo.py --alm).
  target   the next word (its tokens, then a space-led token boundary is not needed: the word is
           decoded until the next word starts).
The language model (SmolLM2-135M by default, Apache-2.0) is fine-tuned in full with the adapter.

usage: audio_lm.py train OUT --data DIR [--data DIR ...] [--steps 8000] [--lm HuggingFaceTB/SmolLM2-135M]
       (inference: AudioLM(OUT).next_word(text, audio_16k) -> (word, probability))"""
import argparse, json, math, os, random, sys, time
import numpy as np, soundfile as sf, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from text import norm

SR = 16000
WINDOW, FRAMES = 4.0, 50


def encode_frames(enc_model, wav, n):
    from train import encode
    h, valid = encode(enc_model, wav, n)
    return h, valid


def last_frames(h, valid, k=FRAMES):
    """The last k real frames of each row (left-padded with zeros when shorter)."""
    out = torch.zeros(h.shape[0], k, h.shape[2], device=h.device, dtype=h.dtype)
    for i in range(h.shape[0]):
        n = int(valid[i].sum()); m = min(k, n)
        if m:
            out[i, k - m:] = h[i, n - m: n]
    return out


class AudioLM(torch.nn.Module):
    def __init__(self, path=None, lm="HuggingFaceTB/SmolLM2-135M", encoder="runs/p29_from_tiny/step10000", device="cpu", frames=FRAMES):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer, MoonshineStreamingForConditionalGeneration
        cfg = json.load(open(f"{path}/alm.json")) if path and os.path.exists(f"{path}/alm.json") else dict(lm=lm, encoder=encoder, frames=frames)
        self.frames = cfg.get("frames", FRAMES)
        if path and encoder != "runs/p29_from_tiny/step10000":
            cfg = dict(cfg, encoder=encoder)   # where the frozen encoder lives on this machine
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(path if path else cfg["lm"])
        self.lm = AutoModelForCausalLM.from_pretrained(path if path else cfg["lm"]).to(device)
        self.enc = MoonshineStreamingForConditionalGeneration.from_pretrained(cfg["encoder"]).to(device).eval().requires_grad_(False)
        d_a = self.enc.config.encoder_config.hidden_size; d_l = self.lm.config.hidden_size
        self.adapter = torch.nn.Sequential(torch.nn.Linear(d_a, d_l), torch.nn.GELU(), torch.nn.Linear(d_l, d_l)).to(device)
        if path and os.path.exists(f"{path}/adapter.pt"):
            self.adapter.load_state_dict(torch.load(f"{path}/adapter.pt", map_location=device))
        self.device = device

    def prefix(self, wav, n):
        with torch.no_grad():
            h, valid = encode_frames(self.enc, wav, n)
        return self.adapter(last_frames(h, valid, self.frames).float())

    def save(self, out):
        os.makedirs(out, exist_ok=True)
        self.lm.save_pretrained(out); self.tok.save_pretrained(out)
        torch.save(self.adapter.state_dict(), f"{out}/adapter.pt")
        json.dump(self.cfg, open(f"{out}/alm.json", "w"))

    @torch.no_grad()
    def next_word(self, text, audio, max_tokens=5):
        """Greedy next word after `text`, hearing the last WINDOW s of `audio` (16 kHz float)."""
        x = audio[-int(WINDOW * SR):]
        x = x[len(x) % 80:]   # whole 5 ms frames (the encoder's front end reads 80 samples at a time)
        wav = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None].to(self.device)
        pre = self.prefix(wav, torch.tensor([len(x)], device=self.device))
        ids = self.tok(" " + text if text else "", return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        bos = torch.tensor([[self.tok.bos_token_id or self.tok.eos_token_id]], device=self.device)
        emb = self.lm.get_input_embeddings()
        inp = torch.cat([emb(bos), pre.to(emb.weight.dtype), emb(ids)], 1)
        out, p, past = [], 1.0, None
        for _ in range(max_tokens):
            o = self.lm(inputs_embeds=inp, past_key_values=past, use_cache=True); past = o.past_key_values
            probs = o.logits[0, -1].float().softmax(-1)
            t = int(probs.argmax()); piece = self.tok.decode([t])
            if out and (piece[:1] in " \n" or not piece.strip("\"',.;:!?")):
                break
            if t == self.tok.eos_token_id or (not out and not norm(piece)):
                break
            out.append(t); p *= float(probs[t])
            inp = emb(torch.tensor([[t]], device=self.device))
        return self.tok.decode(out).strip(), p


def examples(rows, rng, n, inside=0.0):
    """n (audio window, words heard, next word) samples: a cut between two word ends."""
    out = []
    while len(out) < n:
        r = rng.choice(rows); ws = r["words"]
        if len(ws) < 2 or not all("end" in w for w in ws):
            continue
        pieces = r["text"].split() if len(r["text"].split()) == len(ws) else [w["w"] for w in ws]
        k = rng.randrange(0, len(ws) - 1)                       # words fully heard: ws[:k]; next: ws[k]
        lo = ws[k - 1]["end"] if k else 0.0
        if rng.random() < inside:                               # while the next word is being spoken: where hearing it helps
            t = rng.uniform(max(lo, ws[k]["start"]), ws[k]["end"])
        else:
            t = rng.uniform(lo, ws[k]["end"])                   # anywhere before the next word ends
        x, _ = sf.read(r["path"], dtype="float32", frames=int(t * SR))
        x = x[-int(WINDOW * SR):]; x = x[len(x) % 80:]
        if len(x) < 1600:
            continue
        out.append((x, " ".join(pieces[:k]), pieces[k]))
    return out


def train(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = AudioLM(lm=a.lm, encoder=a.encoder, device=dev, frames=a.frames)
    rows = []
    for d in a.data:
        rr = [json.loads(l) for l in open(f"{d}/manifest.jsonl")]
        for r in rr:
            r["path"] = f"{d}/{r['audio']}"
        rows += rr
    print(f"{len(rows)} utterances", flush=True)
    opt = torch.optim.AdamW([{"params": m.lm.parameters(), "lr": a.lr}, {"params": m.adapter.parameters(), "lr": a.lr * a.adapter_lr_mult}],
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 300) * max(0.05, 0.5 * (1 + math.cos(math.pi * s / a.steps))))
    rng = random.Random(0); emb = m.lm.get_input_embeddings(); t0 = time.time(); run = None
    bos_id = m.tok.bos_token_id or m.tok.eos_token_id
    for step in range(1, a.steps + 1):
        ex = examples(rows, rng, a.batch, a.inside)
        L = max(len(x) for x, _, _ in ex)
        wav = torch.zeros(len(ex), L); ns = torch.tensor([len(x) for x, _, _ in ex])
        for i, (x, _, _) in enumerate(ex):
            wav[i, : len(x)] = torch.from_numpy(x)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            pre = m.prefix(wav.to(dev), ns.to(dev))
            seqs, tgts = [], []
            for (_, heard, nxt) in ex:
                h = m.tok(" " + heard if heard else "", add_special_tokens=False).input_ids
                y = m.tok(" " + nxt, add_special_tokens=False).input_ids
                seqs.append(h + y); tgts.append([-100] * len(h) + y)
            S = max(len(s) for s in seqs)
            ids = torch.full((len(ex), S), m.tok.eos_token_id); lab = torch.full((len(ex), S), -100)
            for i, (s, y) in enumerate(zip(seqs, tgts)):
                ids[i, : len(s)] = torch.tensor(s); lab[i, : len(y)] = torch.tensor(y)
            ids, lab = ids.to(dev), lab.to(dev)
            inp = torch.cat([emb(torch.full((len(ex), 1), bos_id, device=dev)), pre.to(emb.weight.dtype), emb(ids)], 1)
            logits = m.lm(inputs_embeds=inp).logits[:, m.frames: -1]   # position 1 + frames - 1 + j predicts token j of ids
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), lab.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(m.lm.parameters()) + list(m.adapter.parameters()), 1.0)
        opt.step(); sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 100 == 0:
            print(f"step {step}  loss {run:.4f}  {time.time() - t0:.0f}s", flush=True)
        if step % 2000 == 0 or step == a.steps:
            m.save(f"{a.out}/step{step}")


@torch.no_grad()
def probe(a):
    """Does the model use the audio? Mean next-word loss on held-out cuts with the real audio,
    with silence, and with the audio of a different cut."""
    m = AudioLM(a.out, encoder=a.encoder, device="cpu").eval()
    rows = []
    for d in a.data:
        rr = [json.loads(l) for l in open(f"{d}/manifest.jsonl")]
        for r in rr:
            r["path"] = f"{d}/{r['audio']}"
        rows += rr
    ex = examples(rows, random.Random(1), a.batch)
    emb = m.lm.get_input_embeddings(); bos_id = m.tok.bos_token_id or m.tok.eos_token_id
    def nll(audio_list):
        tot = n = 0.0
        for (x, heard, nxt), xa in zip(ex, audio_list):
            wav = torch.from_numpy(np.ascontiguousarray(xa, dtype=np.float32))[None]
            pre = m.prefix(wav, torch.tensor([len(xa)]))
            h = m.tok(" " + heard if heard else "", add_special_tokens=False).input_ids; y = m.tok(" " + nxt, add_special_tokens=False).input_ids
            ids = torch.tensor([h + y])
            inp = torch.cat([emb(torch.tensor([[bos_id]])), pre.to(emb.weight.dtype), emb(ids)], 1)
            lp = m.lm(inputs_embeds=inp).logits[0, m.frames: m.frames + len(h) + len(y)].float().log_softmax(-1)
            tot -= sum(float(lp[len(h) + j, t]) for j, t in enumerate(y)); n += 1
        return tot / n
    real = [x for x, _, _ in ex]
    print(f"{len(ex)} held-out cuts; next-word loss (nats): real audio {nll(real):.3f} | silence {nll([np.zeros_like(x) for x in real]):.3f} "
          f"| another cut's audio {nll(real[1:] + real[:1]):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "probe"]); ap.add_argument("out")
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--lm", default="HuggingFaceTB/SmolLM2-135M"); ap.add_argument("--encoder", default="runs/p29_from_tiny/step10000")
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--frames", type=int, default=FRAMES, help="encoder frames (20 ms each) in the audio prompt")
    ap.add_argument("--inside", type=float, default=0.0, help="share of training cuts placed inside the next word")
    ap.add_argument("--adapter-lr-mult", type=float, default=10.0)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else probe(a)


if __name__ == "__main__":
    main()
