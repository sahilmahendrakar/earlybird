"""Ghost text: guess the next word while the speaker is still talking, and keep score.

The recognizer is the no-hold-back engine. Every time the text on screen changes, a guess
for the word after it is made and logged with the audio time, then judged against the
transcript. Two sources of guesses:

  --lm ID        a small language model continues the text on screen (text only)
  --own          the checkpoint's own ghost output (models trained by train.py)

A guess is only judged when the last word on screen is right and in place ("context ok");
otherwise the guess is logged as made on a wrong context.

Per utterance the log holds every guess: time, screen text, guess, its probability, the
word that actually came next, whether it matched, and how long before that word's
acoustic end the guess was on screen.

With --own the guess is --ghost words long, each with its own probability ("wp"), and an
event also records when screen + guess spell the whole utterance ("complete", with the
product of the guess words' probabilities), the time a voice agent could act on the full
command. Each utterance line carries its last word's end and, for SLURP, its intent.

--oracle (with --lm) replaces the recognizer by the reference transcript, each word appearing
at its acoustic end: what a text model could do on top of a perfect, instant recognizer.

--alm CKPT guesses with an audio-prefixed language model (audio_lm.py: the transcript plus the
last seconds of audio), the design of Yusuf et al. 2024.

usage: ghost_demo.py DATA_DIR NAME [--limit 300] [--lm Qwen/Qwen3-0.6B [--oracle] | --alm CKPT | --own [--ghost 2]] [--model CKPT]
Writes results/ghost/NAME.jsonl and prints a summary."""
import argparse, json, os, sys, time
import numpy as np, soundfile as sf, torch
sys.path.insert(0, os.path.dirname(__file__))
from engine import Engine, SR
from score import matched
from text import norm

CHUNK = 1280


class LM:
    def __init__(self, name, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        dt = torch.float16 if device != "cpu" else torch.float32
        self.m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dt).to(device).eval()
        self.device = device

    @torch.no_grad()
    def next_word(self, text, max_tokens=5):
        """Greedy next word after `text` and the probability of its tokens."""
        ids = self.tok(text or "The", return_tensors="pt").input_ids.to(self.device)
        out, p = [], 1.0
        past = None; inp = ids
        for _ in range(max_tokens):
            o = self.m(inp, past_key_values=past, use_cache=True); past = o.past_key_values
            probs = o.logits[0, -1].float().softmax(-1)
            t = int(probs.argmax()); piece = self.tok.decode([t])
            if out and (piece[:1] in " \n" or not piece.strip("\"',.;:!?")):
                break
            if not out and not norm(piece):   # punctuation first: skip over it
                inp = torch.tensor([[t]], device=self.device); continue
            out.append(t); p *= float(probs[t])
            inp = torch.tensor([[t]], device=self.device)
        return self.tok.decode(out).strip(), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("name")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--lm", default="Qwen/Qwen3-0.6B"); ap.add_argument("--own", action="store_true")
    ap.add_argument("--ghost", type=int, default=2, help="--own: words of guess to decode")
    ap.add_argument("--model", default="UsefulSensors/moonshine-streaming-tiny")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--pad", action="store_true", help="decode with the padding frames (Engine(pad=True))")
    ap.add_argument("--alm", help="audio-prefixed language model checkpoint (audio_lm.py): guesses hear the last seconds of audio too")
    ap.add_argument("--oracle", action="store_true", help="with --lm: the screen is the reference's words as each ends "
                    "(a perfect, instant transcript), the upper bound for any recognizer + text model")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    eng = None if a.oracle else Engine(a.model, threads=a.threads, ghost=a.ghost if a.own else 0, pad=a.pad)
    if a.alm:
        from audio_lm import AudioLM
        lm = AudioLM(a.alm, encoder=a.model, device=a.device)   # --model: the frozen encoder's checkpoint here
    else:
        lm = None if a.own else LM(a.lm, a.device)
    data = a.data.rstrip("/")
    rows = [json.loads(l) for l in open(f"{data}/manifest.jsonl")]
    rows = [r for r in rows if all("end" in w for w in r["words"])]
    rows = rows[:: max(1, len(rows) // a.limit)][: a.limit]
    os.makedirs("results/ghost", exist_ok=True)
    ev_all = []; lm_ms = []
    with open(f"results/ghost/{a.name}.jsonl", "w") as f:
        for n, r in enumerate(rows):
            x, _ = sf.read(f"{data}/{r['audio']}", dtype="float32")
            x = np.concatenate([x, np.zeros(SR, dtype=np.float32)])
            ref = [w["w"] for w in r["words"]]
            if eng:
                eng.start()
            screen = ""; events = []
            pieces = r["text"].split() if len(r["text"].split()) == len(ref) else ref
            for i in range(0, len(x), CHUNK):
                t = min(i + CHUNK, len(x)) / SR
                if a.oracle:
                    text = " ".join(pieces[: sum(1 for w in r["words"] if w["end"] <= t)])
                else:
                    text = eng.step(x[i: i + CHUNK])
                if text == screen and not a.own and not a.alm:   # an audio-prefixed model hears new audio every step: guess every step
                    continue
                changed = text != screen; screen = text
                if a.alm and not changed and events:
                    guess, p = lm.next_word(text, x[: i + CHUNK]); wp = [p] if guess else []; eos, done_p = False, 0.0
                    if events[-1]["guess"] == guess and abs(events[-1]["p"] - round(p, 4)) < 0.02:
                        continue
                elif a.own:
                    guess, p, wp = eng.ghost, eng.ghost_p, list(eng.ghost_word_p)
                    eos, done_p = eng.ghost_eos, eng.ghost_done_p
                    if not changed and events and events[-1]["guess"] == guess and events[-1]["wp"] == [round(v, 4) for v in wp]:
                        continue
                else:
                    t0 = time.perf_counter()
                    guess, p = lm.next_word(text, x[: i + CHUNK]) if a.alm else lm.next_word(text)
                    lm_ms.append((time.perf_counter() - t0) * 1e3)
                    wp = [p] if guess else []; eos, done_p = False, 0.0
                hyp = norm(text).split(); m = matched(hyp, ref)
                ctx_ok = (not hyp) or (bool(m) and hyp[-1] == ref[max(m)])
                nxt = (max(m) + 1 if m else 0) if ctx_ok else None
                g = norm(guess).split()
                ev = dict(t=round(t, 3), screen=text, guess=guess, p=round(p, 4), wp=[round(v, 4) for v in wp], ctx_ok=ctx_ok)
                if eos:   # the model claims screen + guess is the whole utterance
                    ev.update(eos=True, done_p=round(done_p, 4), done_ok=hyp + g == ref)
                if nxt is not None and nxt < len(ref):
                    w = r["words"][nxt]
                    ev.update(next=w["w"], ok=bool(g) and g[0] == w["w"], lead_ms=round((w["end"] - t) * 1e3),
                              before_start_ms=round((w["start"] - t) * 1e3))
                    # the whole utterance known: screen + guess spell the reference exactly
                    if hyp == ref[: len(hyp)] and hyp + g == ref:
                        ev.update(complete=True, complete_p=round(float(np.prod(wp[: len(g)])) if g else 1.0, 4))
                elif nxt is not None:
                    ev.update(next=None, ok=not g)   # nothing follows; right only if no guess was made
                events.append(ev)
            f.write(json.dumps(dict(id=r["id"], text=r["text"], end=r["words"][-1]["end"], dur=r["dur"],
                                    intent=r.get("intent"), events=events)) + "\n"); f.flush()
            ev_all += events
            if (n + 1) % 25 == 0:
                print(f"{n+1}/{len(rows)}", flush=True)
    summarize(ev_all, lm_ms)


def summarize(ev, lm_ms=()):
    judged = [e for e in ev if e["ctx_ok"] and e.get("next")]
    print(f"guesses: {len(ev)}; made on a correct context with a word still to come: {len(judged)}")
    if lm_ms:
        print(f"language model compute per guess: mean {np.mean(lm_ms):.0f} ms, p95 {np.percentile(lm_ms, 95):.0f} ms")
    print("| Show a guess when its probability is at least | Guesses shown % | Right % | Median time on screen before the word ends, ms |")
    print("|---:|---:|---:|---:|")
    for thr in (0.0, 0.2, 0.4, 0.6, 0.8):
        s = [e for e in judged if e["p"] >= thr and e["guess"]]
        if not s:
            continue
        ok = [e for e in s if e["ok"]]
        lead = np.median([e["lead_ms"] for e in ok]) if ok else float("nan")
        print(f"| {thr:.1f} | {100 * len(s) / len(judged):.0f} | {100 * len(ok) / len(s):.1f} | {lead:.0f} |")
    g = [e for e in judged if e["guess"]]
    if g:   # calibration of the first guessed word's probability: 10 equal-width bins
        P = np.array([e["p"] for e in g]); Y = np.array([e["ok"] for e in g], float)
        b = np.minimum((P * 10).astype(int), 9)
        ece = sum(abs(Y[b == k].mean() - P[b == k].mean()) * (b == k).sum() for k in range(10) if (b == k).any()) / len(P)
        print(f"calibration error (ECE, 10 bins) of the first guessed word: {100 * ece:.1f} points")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--summary":
        summarize([e for l in open(sys.argv[2]) for e in json.loads(l)["events"]])
    else:
        main()
