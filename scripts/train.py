"""Fine-tune a Moonshine streaming checkpoint to show words early and keep them.

Every training example is an utterance cut off at a random 80 ms step, exactly as the
streaming engine sees it (nothing after the cut, not even padding frames), with the target
from targets.py: the words that are due by then, end-of-text, then the next few words as
a ghost guess. Because the targets of successive cuts are prefixes of one another, the
model is taught not to print a guess it will have to take back.

usage: train.py OUT_DIR --data DIR[:weight] [--data ...] [--steps 20000] ...

Student (distillation): --student s10 --teacher runs/tiny_v1/stepN builds a smaller model
from the same Transformers classes (so Engine and the ONNX export load it unchanged) and
trains it on the same targets (or, with --plain, ordinary whole-utterance transcripts), with a KL term towards the teacher's distribution at every
target position. s10: width 160 (4 heads of 40), 6 encoder and 4 decoder layers, input and output
embeddings tied, 10M parameters; its token embedding starts from the top 160
principal directions of the teacher's.
Cut-down teacher: --init DIR --teacher MODEL starts the student from DIR (shrink.py: the
teacher's own weights, fewer layers, a smaller word table). Targets are made in the teacher's
token ids (a word using a dropped token is spelled with kept pieces, vocab.Restrict), the
teacher reads them as they are, and the student reads them mapped to its own ids; the
distillation term compares the two over the kept tokens only.
Quantization-aware: --qat BITS trains with every weight matrix rounded to BITS bits in groups
of 128 on the forward pass (straight-through gradient to the full-precision copy), except those
matching --qat-keep (the audio front end, which does not survive 4 bits); checkpoints hold the
rounded weights, and quantize.py then packs them (front end at --keep-bits).
Each DIR has manifest.jsonl and the audio it names. Writes OUT_DIR/step*/ (a checkpoint
Engine can load) and OUT_DIR/log.jsonl. Run again with the same OUT_DIR after an
interruption and it carries on from the last checkpoint."""
import argparse, json, math, os, random, sys, time
import numpy as np, soundfile as sf, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from targets import Utt
from vocab import Restrict
from quantize import fake_quant


class FakeQuant(torch.nn.Module):
    """Parametrization: the weight as a BITS-bit build sees it, gradient passed straight through."""

    def __init__(self, bits, group):
        super().__init__(); self.bits, self.group = bits, group

    def forward(self, w):
        return w + (fake_quant(w.detach(), self.bits, self.group)[0] - w.detach())


def add_qat(model, bits, keep, group=128):
    import re
    from torch.nn.utils import parametrize
    emb = model.get_input_embeddings(); n = 0
    for name, m in list(model.named_modules()):
        w = getattr(m, "weight", None)
        if m is model.proj_out or not isinstance(w, torch.nn.Parameter) or w.dim() < 2 or re.search(keep, f"{name}.weight"):
            continue
        parametrize.register_parametrization(m, "weight", FakeQuant(bits, group)); n += w.numel()
    if model.proj_out.weight is getattr(getattr(getattr(emb, "parametrizations", None), "weight", None), "original", None):   # tied: the output side reads the rounded table too
        model.proj_out.forward = lambda x: F.linear(x, emb.weight)
    return n


def plain_copy(model):
    """A fresh model without the QAT wrappers, holding the rounded weights (what save_pretrained
    writes). Not deepcopy + remove_parametrizations: the parametrized class is shared with the
    copy, so removing it there strips the live model too."""
    from torch.nn.utils import parametrize
    sd = {k: v for k, v in model.state_dict().items() if ".parametrizations." not in k}
    for name, m in model.named_modules():
        if parametrize.is_parametrized(m, "weight"):
            sd[f"{name}.weight"] = m.weight.detach()
    if model.config.tie_word_embeddings:
        sd["proj_out.weight"] = sd[next(k for k in sd if k.endswith("embed_tokens.weight"))]
    c = type(model)(model.config)
    c.load_state_dict(sd)
    c.tie_weights()
    return c

SR, CHUNK = 16000, 1280


def n_frames(n_samples, frame=80):
    n = n_samples // frame
    n = (n - 1) // 2 + 1
    return (n - 1) // 2 + 1


PAD = False   # --pad: keep the padding frames (Engine(pad=True)); set in main


def encode(model, wav, n_samples):
    """Encoder output for a zero-padded batch, identical to encoding each row alone.

    The library's own padded path lets three frames of padding through the convolutions;
    here a frame exists only if real audio reached it, unless PAD: then each row gets the
    frames its convolutions produce from silence after its audio, exactly as the library's
    padded path and Engine(pad=True) make them (they hold no audio past the cut)."""
    enc = model.model.encoder
    if PAD:
        wav = F.pad(wav, (0, 16 * enc.embedder.frame_len))   # room for the padding frames of the longest row
        smask = torch.arange(wav.shape[1], device=wav.device)[None, :] < n_samples[:, None]
        h, valid = enc.embedder(wav, padding_mask=smask)
        valid = valid.bool()
    else:
        h, _ = enc.embedder(wav)
        valid = torch.arange(h.shape[1], device=wav.device)[None, :] < n_frames(n_samples)[:, None]
    T = h.shape[1]
    idx = torch.arange(T, device=wav.device)
    dist = idx[:, None] - idx[None, :]
    neg = torch.finfo(h.dtype).min
    for layer, (back, ahead) in zip(enc.layers, enc.config.sliding_windows):
        win = ((dist >= 0) & (dist < back)) | ((dist < 0) & (-dist < ahead))
        ok = (win[None] & valid[:, None, :]) | (dist == 0)[None]
        mask = torch.zeros(ok.shape, dtype=h.dtype, device=h.device).masked_fill(~ok, neg)[:, None]
        h = layer(h, attention_mask=mask)
    return enc.final_norm(h), valid


def forward(model, wav, n_samples, dec_in, with_enc=False):
    h, valid = encode(model, wav, n_samples)
    # a copy: the library's decoder adds its per-frame position table to encoder_hidden_states
    # IN PLACE, which (until step 35000 of s10_from_tiny_v2) leaked into the h returned below,
    # so --enc-kd compared encoder output + positions, student's and teacher's
    out = model.model.decoder(input_ids=dec_in, encoder_hidden_states=h.clone(), encoder_attention_mask=valid.long())
    logits = model.proj_out(out.last_hidden_state)
    return (logits, h, valid) if with_enc else logits


class Data:
    def __init__(self, specs, tok, lead, max_dur, dev_n=200):
        self.sets = []; self.dev = []; cache = {}
        for spec in specs:
            d, _, w = spec.partition(":")
            rows = [json.loads(l) for l in open(f"{d}/manifest.jsonl") if l.endswith("\n")]
            utts = []
            for r in rows:
                if r["dur"] > max_dur:
                    continue
                u = Utt(r, tok, lead, cache)
                if u.ok:
                    u.path = f"{d}/{r['audio']}"; u.dur = r["dur"]; utts.append(u)
            random.Random(0).shuffle(utts)
            k = min(dev_n, len(utts) // 10)
            self.dev += utts[:k]
            if utts[k:]:
                self.sets.append((utts[k:], float(w or 1)))
            print(f"{d}: {len(utts)} of {len(rows)} utterances usable, {sum(u.dur for u in utts)/3600:.1f} h", flush=True)

    def sample(self, rng, n=1):
        """n utterances of one set (n > 1: to be joined into one long example)."""
        utts = rng.choices([s[0] for s in self.sets], weights=[s[1] for s in self.sets])[0]
        return rng.choice(utts) if n == 1 else [rng.choice(utts) for _ in range(n)]


def joined(rng, a, data):
    """--concat: back-to-back utterances of one set with 0.1-0.5 s between them, up to
    --concat-max seconds, so the model hears (and its per-frame positions are trained on)
    audio longer than any single training utterance. With --plain only."""
    us = data.sample(rng, 4)
    out, tot = [us[0]], us[0].dur
    for v in us[1:]:
        if tot + 0.5 + v.dur > a.concat_max:
            break
        out.append(v); tot += 0.5 + v.dur
    return out


def make_example(u, rng, a, bos, eos, augment=True, t=None):
    if isinstance(u, list):   # joined utterances (--plain): whole audio, whole transcripts
        parts = []
        for j, v in enumerate(u):
            y, sr = sf.read(v.path, dtype="float32"); assert sr == SR
            if j:
                parts.append(np.zeros(int(rng.uniform(0.1, 0.5) * SR), dtype=np.float32))
            parts.append(y)
        x = np.concatenate(parts)
        heard = [i for v in u for p in v.full for i in p]
    else:
        x, sr = sf.read(u.path, dtype="float32"); assert sr == SR
        heard = None
    shift = 0.0
    if augment:
        if rng.random() < 0.5:       # the stream does not always start right at the first word
            shift = rng.uniform(0, 0.5)
            x = np.concatenate([np.zeros(int(shift * SR), dtype=np.float32), x])
        x = x * 10 ** (rng.uniform(-12, 3) / 20)
        if rng.random() < a.noise_p:
            snr = rng.uniform(10, 40)
            p = float((x ** 2).mean()) + 1e-9
            x = x + np.random.randn(len(x)).astype(np.float32) * math.sqrt(p / 10 ** (snr / 10))
    total = len(x) + int(0.8 * SR)
    if t is None:
        # --end-p of the cuts (default 15%) are at the very end, so finishing a sentence is practised too
        n = rng.randrange(3, total // CHUNK + 1) if rng.random() >= a.end_p and not a.plain else total // CHUNK
    else:
        n = max(3, min(total // CHUNK, round((t + shift) * SR / CHUNK)))
    cut = n * CHUNK
    x = np.concatenate([x, np.zeros(max(0, cut - len(x)), dtype=np.float32)])[:cut]
    if heard is None:
        heard, ghost, k = u.target(cut / SR - shift, a.ghost)
    else:
        ghost = []
    ids = [bos] + heard + [eos] + ghost
    # after the guess: end-of-text when the guess reaches the end of the utterance (so the model
    # learns to claim "that is the whole command", --ghost-end), else nothing to learn
    ends = a.ghost and (not ghost or (a.ghost_end and k + a.ghost >= len(u.ps)))
    labels = heard + [eos] + ghost + ([eos] if ends else [-100])
    w = [1.0] * (len(heard) + 1) + [a.ghost_weight] * (len(labels) - len(heard) - 1)
    return x, ids, labels, w


def collate(exs, device):
    B = len(exs); L = max(len(e[0]) for e in exs); S = max(len(e[1]) for e in exs)
    wav = torch.zeros(B, L); ns = torch.zeros(B, dtype=torch.long)
    ids = torch.zeros(B, S, dtype=torch.long); lab = torch.full((B, S), -100); wt = torch.zeros(B, S)
    for i, (x, d, l, w) in enumerate(exs):
        wav[i, : len(x)] = torch.from_numpy(x); ns[i] = len(x)
        ids[i, : len(d)] = torch.tensor(d); lab[i, : len(l)] = torch.tensor(l); wt[i, : len(w)] = torch.tensor(w)
    wt[lab == -100] = 0
    return [t.to(device, non_blocking=True) for t in (wav, ns, ids, lab, wt)]


class Loader(torch.utils.data.IterableDataset):
    def __init__(self, data, a, bos, eos):
        self.data, self.a, self.bos, self.eos = data, a, bos, eos

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        seed = (self.a.seed * 1000 + (info.id if info else 0)) * 1000003 + self.a.first_step
        rng = random.Random(seed); np.random.seed(seed % 2 ** 32)
        while True:
            # a bucket of utterances, sorted by cut length so padding stays small
            exs = [make_example(joined(rng, self.a, self.data) if rng.random() < self.a.concat else self.data.sample(rng),
                                rng, self.a, self.bos, self.eos)
                   for _ in range(self.a.batch * 8)]
            exs.sort(key=lambda e: len(e[0]))
            if self.a.batch_secs:   # at most batch x batch_secs seconds of (padded) audio per batch
                batches, cur = [], []
                for e in exs:
                    if cur and (len(cur) + 1) * len(e[0]) > self.a.batch * self.a.batch_secs * SR:
                        batches.append(cur); cur = []
                    cur.append(e)
                batches.append(cur)
            else:
                batches = [exs[i: i + self.a.batch] for i in range(0, len(exs), self.a.batch)]
            rng.shuffle(batches)
            yield from batches


def _same(b):
    return b


STUDENTS = dict(   # overrides of the teacher's config
    s10=dict(hidden=160, inter=640, heads=4, enc_layers=6, dec_layers=4),
)


def build_student(teacher, name):
    import copy
    c = STUDENTS[name]; cfg = copy.deepcopy(teacher.config)
    e = cfg.encoder_config
    e.hidden_size, e.intermediate_size = c["hidden"], c["inter"]
    e.num_attention_heads = e.num_key_value_heads = c["heads"]; e.head_dim = c["hidden"] // c["heads"]
    e.num_hidden_layers = c["enc_layers"]; e.sliding_windows = e.sliding_windows[: c["enc_layers"]]
    cfg.encoder_hidden_size = cfg.hidden_size = c["hidden"]; cfg.intermediate_size = c["inter"]
    cfg.num_attention_heads = cfg.num_key_value_heads = c["heads"]; cfg.head_dim = c["hidden"] // c["heads"]
    cfg.num_hidden_layers = c["dec_layers"]; cfg.tie_word_embeddings = True
    student = type(teacher)(cfg)
    with torch.no_grad():   # token embedding: the teacher's, projected onto its top principal directions
        E = teacher.get_input_embeddings().weight.float()
        U, S, _ = torch.linalg.svd(E - E.mean(0), full_matrices=False)
        P = U[:, : c["hidden"]] * S[: c["hidden"]]
        student.get_input_embeddings().weight.copy_(P * (E.std() / P.std()))
    student.tie_weights()
    return student


def to_student(r, ids, lab):
    """Teacher ids -> the cut-down student's (no-op without a cut-down word table)."""
    if r is None:
        return ids, lab
    lut = r.lut.to(ids.device)
    return lut[ids], torch.where(lab == -100, lab, lut[lab.clamp(min=0)])


def kd_loss(logits, t_logits, wt):
    kl = F.kl_div(F.log_softmax(logits.float(), -1), F.log_softmax(t_logits.float(), -1), log_target=True,
                  reduction="none").sum(-1)
    return (kl * wt).sum() / wt.sum().clamp(min=1)


def enc_kd_loss(h, t_h, valid, proj):
    """Student encoder output, projected to the teacher's width, against the teacher's encoder
    output: squared error over real frames, relative to the teacher's mean square."""
    d = ((proj(h.float()) - t_h.float()) ** 2).sum(-1)
    v = valid.float()
    return (d * v).sum() / ((t_h.float() ** 2).sum(-1) * v).sum().clamp(min=1e-6)


def loss_fn(logits, lab, wt):
    ce = F.cross_entropy(logits.float().transpose(1, 2), lab, ignore_index=-100, reduction="none")
    return (ce * wt).sum() / wt.sum().clamp(min=1), ce


@torch.no_grad()
def evaluate(model, data, a, bos, eos, device, r=None):
    model.eval(); rng = random.Random(1)
    tot = dict(main=0.0, n_main=0, ghost=0.0, n_ghost=0, exact=0, n=0, g1=0, n_g1=0)
    exs = [make_example(u, rng, a, bos, eos, augment=False) for u in data.dev for _ in range(2)]
    exs.sort(key=lambda e: len(e[0]))
    for i in range(0, len(exs), a.batch):
        wav, ns, ids, lab, wt = collate(exs[i: i + a.batch], device)
        ids, lab = to_student(r, ids, lab)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = forward(model, wav, ns, ids)
        _, ce = loss_fn(logits, lab, wt)
        main = wt == 1.0; gh = (wt > 0) & ~main
        pred = logits.argmax(-1)
        tot["main"] += float(ce[main].sum()); tot["n_main"] += int(main.sum())
        tot["ghost"] += float(ce[gh].sum()); tot["n_ghost"] += int(gh.sum())
        tot["exact"] += int(((pred == lab) | ~main).all(1).sum()); tot["n"] += len(wav)
        first = gh & (gh.cumsum(1) == 1)   # first ghost token of each row
        tot["g1"] += int(((pred == lab) & first).sum()); tot["n_g1"] += int(first.sum())
    model.train()
    return dict(dev_main=tot["main"] / max(1, tot["n_main"]), dev_ghost=tot["ghost"] / max(1, tot["n_ghost"]),
                dev_exact=tot["exact"] / tot["n"], dev_ghost_tok1=tot["g1"] / max(1, tot["n_g1"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--model", default="UsefulSensors/moonshine-streaming-tiny")
    ap.add_argument("--lead", type=float, default=0.12, help="seconds before a word's end at which it is due")
    ap.add_argument("--ghost", type=int, default=2, help="words of continuation after end-of-text")
    ap.add_argument("--ghost-weight", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--max-dur", type=float, default=20.0); ap.add_argument("--noise-p", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=1000); ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--student", choices=STUDENTS, help="train a smaller model of this shape by distillation")
    ap.add_argument("--teacher", help="checkpoint to distil from (with --student)")
    ap.add_argument("--gpu-frac", type=float, default=0, help="cap this process at this share of GPU memory (other jobs share the GPU)")
    ap.add_argument("--plain", action="store_true", help="ordinary targets: whole utterance, whole transcript, no ghost words")
    ap.add_argument("--enc-kd", type=float, default=0, help="weight of matching the teacher's encoder output (student only); "
                    "gives the from-scratch encoder a direct signal from the teacher's")
    ap.add_argument("--concat", type=float, default=0, help="share of examples made of back-to-back utterances (--plain only)")
    ap.add_argument("--concat-max", type=float, default=30.0, help="longest joined example, seconds")
    ap.add_argument("--batch-secs", type=float, default=0, help="cap a batch at batch x this many seconds of audio (0: no cap)")
    ap.add_argument("--extend-pos", type=int, default=0, help="student, once on resume: refill the decoder's per-frame "
                    "position rows from this frame on (never trained) from the teacher's, through a linear map fitted on the rows before it")
    ap.add_argument("--init", help="start the student from this checkpoint (shrink.py) instead of --student's shape")
    ap.add_argument("--end-p", type=float, default=0.15, help="share of cuts at the very end of the utterance")
    ap.add_argument("--qat", type=int, default=0, help="train with weights rounded to this many bits (0: off)")
    ap.add_argument("--qat-keep", default=r"encoder\.embedder", help="weights left out of --qat (regex on the name)")
    ap.add_argument("--pad", action="store_true", help="train with the padding frames after the cut, as Engine(pad=True) decodes "
                    "(the released weights were trained with them; untrained, they make Tiny much earlier but jumpier)")
    ap.add_argument("--min-heard", type=float, default=None, help="a word is never due before this many seconds of it are heard "
                    "(targets.MIN_HEARD, default 0.10); lower lets --lead pull short words earlier")
    ap.add_argument("--ghost-end", action="store_true", help="end-of-text after the ghost words when they reach the end of the utterance")
    ap.add_argument("--kd-agree", action="store_true", help="distil only on heard-text positions where the teacher's best token is the target (anticipation runs)")
    ap.add_argument("--kd", type=float, default=0.5, help="weight of the distillation term (the rest is cross-entropy)")
    a = ap.parse_args()
    if a.plain:
        a.ghost = 0
    assert not a.concat or a.plain, "--concat needs --plain"
    if a.gpu_frac and a.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.gpu_frac)
    from transformers import AutoTokenizer, MoonshineStreamingForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(a.model)
    state = torch.load(f"{a.out}/state.pt", weights_only=True) if os.path.exists(f"{a.out}/state.pt") else None
    a.first_step = state["step"] if state else 0
    teacher = None
    if a.student or a.init:
        teacher = MoonshineStreamingForConditionalGeneration.from_pretrained(a.teacher).to(a.device).eval().requires_grad_(False)
        src = f"{a.out}/step{a.first_step}" if state else a.init
        model = (MoonshineStreamingForConditionalGeneration.from_pretrained(src) if src
                 else build_student(teacher, a.student)).to(a.device).train()
        print(f"student {a.student or a.init}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters", flush=True)
    else:
        model = MoonshineStreamingForConditionalGeneration.from_pretrained(
            f"{a.out}/step{a.first_step}" if state else a.model).to(a.device).train()
    pos_extended = bool(state and state.get("pos_extended"))
    if a.extend_pos and teacher is not None and not pos_extended:
        with torch.no_grad():
            K = a.extend_pos
            tp = teacher.model.decoder.pos_emb.weight.float(); sp = model.model.decoder.pos_emb.weight
            X = torch.cat([tp[:K], torch.ones(K, 1, device=tp.device)], 1)
            W = torch.linalg.lstsq(X.cpu(), sp[:K].float().cpu()).solution.to(tp.device)
            fit = torch.cat([tp, torch.ones(len(tp), 1, device=tp.device)], 1) @ W
            err = float((fit[:K] - sp[:K].float()).norm() / sp[:K].float().norm())
            sp[K:] = fit[K:].to(sp.dtype)
        pos_extended = True
        print(f"pos_emb rows {K}.. refilled from the teacher's (fit relative error {err:.3f} on rows ..{K})", flush=True)
    bos, eos = model.config.decoder_start_token_id, model.config.eos_token_id
    r = None; kept_src = f"{a.out}/step{a.first_step}" if state else a.init
    if kept_src and os.path.exists(f"{kept_src}/kept_ids.json"):
        r = Restrict(tok, json.load(open(f"{kept_src}/kept_ids.json")))
        bos, eos = teacher.config.decoder_start_token_id, teacher.config.eos_token_id   # targets in the teacher's ids
        print(f"cut-down word table: {len(r.ids)} tokens", flush=True)
    if a.pad:
        global PAD
        PAD = True
    if a.min_heard is not None:
        import targets
        targets.MIN_HEARD = a.min_heard
    data = Data(a.data, r or tok, a.lead, a.max_dur)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)
    if a.first_step >= a.steps:
        return
    dl = torch.utils.data.DataLoader(Loader(data, a, bos, eos), batch_size=None, num_workers=a.workers, collate_fn=_same,
                                     prefetch_factor=4 if a.workers else None)
    proj = None
    if a.qat:
        print(f"QAT: {add_qat(model, a.qat, a.qat_keep):,} weights trained as {a.qat}-bit", flush=True)
    params = list(model.parameters())
    if teacher is not None and a.enc_kd:   # a training-only projection, not part of the checkpoint
        if model.config.encoder_config.hidden_size == teacher.config.encoder_config.hidden_size:
            proj = torch.nn.Identity()   # same width (a cut-down teacher): match the teacher's encoder output directly
        else:
            proj = torch.nn.Linear(model.config.encoder_config.hidden_size, teacher.config.encoder_config.hidden_size).to(a.device)
            if state and "proj" in state:
                proj.load_state_dict(state["proj"])
            params += list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / a.warmup) * max(
        0.05, 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps)))))
    if state:
        opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"])
        print(f"resuming after step {a.first_step}", flush=True)
    log = open(f"{a.out}/log.jsonl", "a"); t0 = time.time(); run = state["run"] if state else 0.0

    def note(d):
        log.write(json.dumps(d) + "\n"); log.flush(); print(d, flush=True)

    def save(step):
        d = f"{a.out}/step{step}"
        (plain_copy(model) if a.qat else model).save_pretrained(d); tok.save_pretrained(d)
        if r is not None:
            json.dump(r.ids.tolist(), open(f"{d}/kept_ids.json", "w"))
        torch.save(dict(step=step, opt=opt.state_dict(), sched=sched.state_dict(), run=run, pos_extended=pos_extended,
                        **({"proj": proj.state_dict()} if proj is not None and not isinstance(proj, torch.nn.Identity) else {})), f"{a.out}/state.tmp")
        os.replace(f"{a.out}/state.tmp", f"{a.out}/state.pt")

    if not state:
        note(dict(step=0, **evaluate(model, data, a, bos, eos, a.device, r)))
    for step, exs in enumerate(dl, a.first_step + 1):
        wav, ns, ids, lab, wt = collate(exs, a.device)
        s_ids, s_lab = to_student(r, ids, lab)
        with torch.autocast(a.device, dtype=torch.bfloat16, enabled=a.device == "cuda"):
            logits, h, valid = forward(model, wav, ns, s_ids, with_enc=True)
            if teacher is not None:
                with torch.no_grad():
                    t_logits, t_h, _ = forward(teacher, wav, ns, ids, with_enc=True)
                    if r is not None:
                        t_logits = t_logits[..., r.ids.to(t_logits.device)]   # the kept tokens only
        loss, _ = loss_fn(logits, s_lab, wt)
        if teacher is not None:
            # --kd-agree: the teacher only knows ordinary transcription: distil it where the target is plainly
            # heard text (not the ghost words) and the teacher agrees with that target; where the
            # target runs ahead of the audio (--lead) the teacher would pull against anticipating
            kd_w = ((wt == 1.0) & (t_logits.argmax(-1) == s_lab)).float() if a.kd_agree else wt
            loss = (1 - a.kd) * loss + a.kd * kd_loss(logits, t_logits, kd_w)
        if proj is not None:
            ekd = enc_kd_loss(h, t_h, valid, proj); loss = loss + a.enc_kd * ekd
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run = loss.item() if step == 1 else 0.98 * run + 0.02 * loss.item()
        if step % 100 == 0:
            print(f"step {step}  loss {run:.4f}" + (f"  enc_kd {ekd.item():.3f}" if proj is not None else "") + f"  lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            note(dict(step=step, loss=round(run, 4), secs=round(time.time() - t0), **evaluate(model, data, a, bos, eos, a.device, r)))
        if step % a.save_every == 0 or step == a.steps:
            save(step)
        if step >= a.steps:
            break


if __name__ == "__main__":
    main()
