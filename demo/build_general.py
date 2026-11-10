"""Build the general-speech anticipation demo (demo/general.html + audio/) from ghost_demo logs.

Each lane replays one system's screen and guesses (time, screen text, guess, per-word
probabilities) in sync with the clip. Clips: a spread of short utterances from the general
test sets where the main system guessed several words right before they ended.
usage: python3 demo/build_general.py GHOST_DIR OUT_DIR --main antic_gen --lanes qwen-on-p29[,alm135-on-p29]
       [--sets librispeech_test_clean,tedlium_test,ps_test,kokoro_test] [--n 8] [--stats stats.json]"""
import argparse, json, os, re, subprocess

B = "s3://" + os.environ["EARLYBIRD_BUCKET"]
LABELS = {"qwen-on-p29": ("Text model", "29M transcript + Qwen3-0.6B guessing the next word"),
          "alm135-on-p29": ("Text + audio model", "29M transcript + SmolLM2-135M that also hears the last 4 s (Yusuf et al. 2024 design)"),
          "smollm135-oracle": ("Text model, perfect transcript", "SmolLM2-135M on the reference words, each appearing as it ends"),
          "qwen06-oracle": ("Text model, perfect transcript", "Qwen3-0.6B on the reference words, each appearing as it ends")}


STT = {"ref-tiny-nohold": ("Moonshine Tiny", "the released 44M streaming recognizer we started from: a normal live transcript, no guesses"),
       "moonshine-tiny": ("Moonshine Tiny", "the released 44M streaming model, through its official runtime"),
       "moonshine-small": ("Moonshine Small", "the released 123M streaming model, through its official runtime"),
       "whistle-live.live": ("Cactus Whistle", "the released 16.9 MB model in its own live mode (committed text plus its pending tail)"),
       "whisper-base.en.raw": ("Whisper base.en", "OpenAI's 74M model, re-run on all the audio so far every 0.48 s"),
       "parakeet-tdt": ("NVIDIA Parakeet", "Parakeet TDT 0.6B v2, re-run on all the audio so far every 0.24 s"),
       "ref-tiny-nohold-pad": ("Moonshine Tiny + padding", "the same 44M model given three frames of silence padding after the audio so far: its earliest setting, no guesses")}


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", s.lower()).split())


def load(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)} if os.path.exists(p) else {}


def early_words(u):
    """Reference words the system guessed right (any confidence) before they ended."""
    return sum(1 for e in u["events"] if e.get("ok") and e.get("next") and e.get("lead_ms", 0) > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ghost"); ap.add_argument("out"); ap.add_argument("--main", default="antic_gen")
    ap.add_argument("--lanes", default="qwen-on-p29"); ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--sets", default="librispeech_test_clean,tedlium_test,ps_test,kokoro_test"); ap.add_argument("--stats")
    ap.add_argument("--label", nargs="*", default=[], help="NAME=Title|subtitle for a lane (or for the main system)")
    ap.add_argument("--main-label", default="Earlybird|29M · writes ahead of the audio, every 80 ms")
    ap.add_argument("--title", help="replace the page title and heading")
    ap.add_argument("--lede", help="replace the intro paragraph (HTML allowed)")
    ap.add_argument("--stt", default="", help="comma list of plain streaming recognizers to add as lanes, by results/raw system name (e.g. ref-tiny-nohold)")
    a = ap.parse_args()
    lanes = [l for l in a.lanes.split(",") if l]
    for spec in a.label:
        n, v = spec.split("=", 1); t, _, sub = v.partition("|"); LABELS[n] = (t, sub)
    mt, _, ms = a.main_label.partition("|")
    os.makedirs(f"{a.out}/audio", exist_ok=True)
    clips = []
    for d in a.sets.split(","):
        man = f"{a.out}/.{d}.manifest.jsonl"
        if not os.path.exists(man):
            subprocess.run(["aws", "s3", "cp", f"{B}/data/{d}/manifest.jsonl", man, "--only-show-errors"], check=True)
        rows = {json.loads(l)["id"]: json.loads(l) for l in open(man)}
        own = load(f"{a.ghost}/{a.main}.{d}.jsonl"); others = {l: load(f"{a.ghost}/{l}.{d}.jsonl") for l in lanes}
        stt = {}
        for sname in [x for x in a.stt.split(",") if x]:
            stt[sname] = {}
            for fn in (d, d + "_full"):
                loc = f"{a.out}/.{sname}.{fn}.jsonl"
                if not os.path.exists(loc):
                    subprocess.run(["aws", "s3", "cp", f"{B}/results/raw/{sname}/{fn}.jsonl", loc, "--only-show-errors"])
                if os.path.exists(loc):
                    for l in open(loc):
                        x = json.loads(l); stt[sname].setdefault(x["id"], x)
        cand = []
        for i, u in own.items():
            r = rows.get(i)
            if not r or not (3.0 <= r["dur"] <= 9.0) or any(i not in o for o in others.values()) or any(i not in v for v in stt.values()):
                continue
            if not all("end" in w for w in r["words"]):
                continue
            cand.append((early_words(u) / max(1, len(r["words"])), i))
        cand.sort(reverse=True)
        per = max(1, a.n // len(a.sets.split(",")))
        for _, i in cand[:per]:
            r = rows[i]; src = f"{B}/data/{d}/{r['audio']}"; base = os.path.splitext(os.path.basename(r["audio"]))[0]
            tmp = f"{a.out}/.{base}{os.path.splitext(r['audio'])[1]}"; wav = f"{a.out}/audio/{base}.wav"
            subprocess.run(["aws", "s3", "cp", src, tmp, "--only-show-errors"], check=True)
            subprocess.run([os.path.expanduser("~/.local/bin/ffmpeg"), "-loglevel", "error", "-y", "-i", tmp, "-ar", "16000", "-ac", "1", wav], check=True)
            os.remove(tmp)
            lane = lambda u: [[e["t"], e["screen"], e["guess"], e.get("wp") or ([e["p"]] if e["guess"] else [])] for e in u["events"]]
            clips.append(dict(id=i, set=d, audio=f"audio/{base}.wav", dur=r["dur"], text=r["text"],
                              words=[[norm(w["w"]), w["start"], w["end"]] for w in r["words"]],
                              lanes=[dict(name=mt, sub=ms, ev=lane(own[i]), main=True)] +
                                    [dict(name=STT.get(sn, (sn, ""))[0], sub=STT.get(sn, (sn, ""))[1], ev=[[t, txt, "", []] for t, txt in stt[sn][i]["partials"]]) for sn in stt] +
                                    [dict(name=LABELS.get(l, (l, ""))[0], sub=LABELS.get(l, (l, ""))[1], ev=lane(others[l][i])) for l in lanes]))
    data = dict(clips=clips, stats=json.load(open(a.stats)) if a.stats else {})
    tpl = open(os.path.join(os.path.dirname(__file__), "general_template.html")).read()
    if a.title:
        tpl = tpl.replace("<title>Earlybird Writes Ahead</title>", f"<title>{a.title}</title>").replace("<h1>Earlybird Writes Ahead</h1>", f"<h1>{a.title}</h1>")
    if a.lede:
        i0 = tpl.index('<p class="lede">'); i1 = tpl.index("</p>", i0)
        tpl = tpl[:i0] + f'<p class="lede">{a.lede}' + tpl[i1:]
    open(f"{a.out}/index.html", "w").write(tpl.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"{len(clips)} clips -> {a.out}/index.html")


if __name__ == "__main__":
    main()
