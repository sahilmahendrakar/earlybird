"""Synthesize sentences with Kokoro and save per-word timings.

Word times come from Kokoro's own predicted phoneme durations (KPipeline's token
start_ts/end_ts). Kokoro folds the pause after a word into that word's last phoneme, so a
word before a comma or at the end of the sentence gets an end time far past where the
speech stops. `end` is therefore pulled back to the last frame in the word that carries
energy; the untouched value is kept as `end_raw`.

usage: synth_kokoro.py OUT_DIR --split test --n 500 --voices af_heart,am_michael,...
Writes OUT_DIR/audio/*.wav (16 kHz mono int16) and OUT_DIR/manifest.jsonl. Resumable."""
import argparse, json, os, random, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, torch, torchaudio
sys.path.insert(0, os.path.dirname(__file__))
from text import norm, wikitext_sentences

SR_IN, SR = 24000, 16000


def speech_end(audio, start, end, thr, win=0.010, min_trim=0.04):
    """Last time in [start, end] where 10 ms RMS exceeds thr; `end` if the tail is short."""
    a, b = int(start * SR_IN), int(end * SR_IN)
    w = int(win * SR_IN)
    seg = audio[a:b]
    if len(seg) < 2 * w:
        return end
    n = len(seg) // w
    rms = np.sqrt((seg[: n * w].reshape(n, w) ** 2).mean(1))
    loud = np.nonzero(rms > thr)[0]
    if len(loud) == 0:
        return end
    t = start + (loud[-1] + 1) * win
    return t if end - t >= min_trim else end


def word_times(tokens, audio):
    """Group Kokoro tokens into whitespace-delimited words with start/end seconds."""
    peak = np.sqrt((audio ** 2).mean()) or 1.0
    thr = peak * 10 ** (-30 / 20)  # 30 dB under the utterance RMS
    words, cur = [], None
    for tk in tokens:
        w = norm(tk.text)
        if w and tk.start_ts is not None and tk.end_ts is not None:
            if cur is None:
                cur = dict(w=w, start=tk.start_ts, end_raw=tk.end_ts)
            else:
                cur["w"] += w
                cur["end_raw"] = tk.end_ts
        elif w:
            return None  # a spoken token without a time: skip the sentence
        if tk.whitespace and cur is not None:
            words.append(cur); cur = None
    if cur is not None:
        words.append(cur)
    for x in words:
        x["end"] = round(speech_end(audio, x["start"], x["end_raw"], thr), 4)
        x["start"], x["end_raw"] = round(x["start"], 4), round(x["end_raw"], 4)
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--skip", type=int, default=0, help="skip this many sentences of the split first")
    ap.add_argument("--voices", default="af_heart,af_bella,am_michael,bf_emma,bm_george")
    ap.add_argument("--text-file", help="one sentence per line, used instead of WikiText")
    ap.add_argument("--speeds", default="1.0", help="comma list; one is drawn per sentence")
    ap.add_argument("--flac", action="store_true")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-file", help="of the sentences, synthesize only those listed in this file (ids stay the same)")
    ap.add_argument("--start", type=int, default=0, help="begin at this sentence of the shuffled list (to split the work between machines)")
    a = ap.parse_args()

    from kokoro import KPipeline
    voices = a.voices.split(","); speeds = [float(s) for s in a.speeds.split(",")]
    rng = random.Random(a.seed)
    if a.text_file:
        sents = [l.strip() for l in open(a.text_file) if l.strip()]
    else:
        sents = []
        for i, s in enumerate(wikitext_sentences(a.split)):
            if i >= a.skip + a.n * 3:
                break
            if i >= a.skip:
                sents.append(s)
    rng.shuffle(sents)

    os.makedirs(f"{a.out}/audio", exist_ok=True)
    mpath = f"{a.out}/manifest.jsonl"
    done = {json.loads(l)["id"] for l in open(mpath)} if os.path.exists(mpath) else set()
    if a.device == "mps" and not torch.backends.mps.is_available():
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    pipes = {}
    def pipe_for(voice):
        lc = voice[0]
        if lc not in pipes:
            base = next(iter(pipes.values()), None)
            pipes[lc] = KPipeline(lang_code=lc, device=a.device, repo_id="hexgrad/Kokoro-82M",
                                  model=base.model if base else True)
        return pipes[lc]
    resample = torchaudio.transforms.Resample(SR_IN, SR)

    only = {l.strip() for l in open(a.only_file)} if a.only_file else None
    t0 = time.time(); secs = 0.0; kept = len(done)
    with open(mpath, "a") as mf:
        for i, text in enumerate(sents):
            if kept >= a.n:
                break
            uid = f"{a.split}_{a.skip + i:07d}"
            if i < a.start or uid in done or (only is not None and text not in only):
                continue
            voice = voices[i % len(voices)]; speed = rng.choice(speeds)
            res = list(pipe_for(voice)(text, voice=voice, speed=speed))
            if len(res) != 1 or res[0].audio is None or not res[0].tokens:
                continue
            audio = res[0].audio.cpu().float()
            words = word_times(res[0].tokens, audio.numpy())
            if not words or [x["w"] for x in words] != norm(text).split():
                continue
            ext = "flac" if a.flac else "wav"
            rel = f"audio/{uid}.{ext}"
            sf.write(f"{a.out}/{rel}", resample(audio).clamp(-1, 1).numpy(), SR, subtype="PCM_16")
            dur = len(audio) / SR_IN; secs += dur; kept += 1
            mf.write(json.dumps(dict(id=uid, audio=rel, text=text, voice=voice, speed=speed,
                                     dur=round(dur, 3), words=words)) + "\n"); mf.flush()
            if kept % 100 == 0:
                el = time.time() - t0
                print(f"{kept} utts  {secs/3600:.2f} h  {el:.0f}s  {secs/el:.1f}x RT", flush=True)
    print(f"done: {kept} utts, {secs/3600:.2f} h new audio")


if __name__ == "__main__":
    main()
