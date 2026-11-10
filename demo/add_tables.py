"""Add the results tables (earliness sweep, noise, size) to a built demo page, from results/eval/*.md copied next to this script.
usage: python3 demo/add_tables.py DEMO_INDEX_HTML demo/general_template.html"""
import json, re, os, sys
D = os.path.dirname(os.path.abspath(__file__))
def parse(fn):
    out, cur = {}, None
    for l in open(f"{D}/{fn}"):
        if l.startswith("### "): cur = l[4:].strip(); continue
        if l.startswith("| ") and not l.startswith("| System") and cur:
            c = [x.strip() for x in l.strip().strip("|").split("|")]
            out.setdefault(cur, {})[c[0]] = dict(wer=c[2], stable=c[7], rev=c[9])
    return out
R = {}
for f in ("abl_lead50_mh05.md", "noisy.md", "gen2_q4.md", "gen2_q3fe8.md", "q4pad.md", "noisy_pad.md"):
    if os.path.exists(f"{D}/{f}"):
        for s, rows in parse(f).items(): R.setdefault(s, {}).update(rows)
ms = lambda v: (v.replace("-", "−") if v.startswith("-") else ("+" + v.lstrip("+") if v not in ("+0", "0") else "0")) + " ms"
def table(head, rows): return "<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table>"
SETS = [("librispeech_test_clean_full", "audiobooks"), ("librispeech_test_other_full", "harder audiobooks"), ("tedlium_test", "talks"), ("ps_test", "public speech")]
tabs = []
sweep = [("abl_lead30_mh05-step8000", "300 ms"), ("abl_lead40_mh05-step8000", "400 ms"), ("abl_lead50_mh05-step8000", "500 ms"), ("antic_gen2-step20000", "300 ms, trained 2.5x longer (Earlybird)")]
rows = []
for k, name in sweep:
    rows.append([name] + [ms(R[s][k]["stable"]) for s, _ in SETS] + [R["librispeech_test_clean_full"][k]["rev"], R["librispeech_test_clean_full"][k]["wer"]])
tabs.append(dict(title="How early can it be pushed?", html=table(["Trained to write a word this long before it ends"] + [f"Stays right from: {n}" for _, n in SETS] + ["Rewrites per word (audiobooks)", "WER (audiobooks)"], rows),
                 note="Without padding frames. Each row is the same recipe with one change: how early, before the end of a word, training asks for it. Earliness stops improving by 400 ms while rewrites keep climbing and accuracy is unchanged; Earlybird (300 ms, trained longer) is as early as 500 ms with the fewest rewrites. Medians over full test sets."))
NS = [("librispeech_test_clean_full", "none"), ("ls_noisy_white10", "white noise, 10 dB"), ("ls_noisy_white5", "white noise, 5 dB"), ("ls_noisy_babble10", "chatter, 10 dB"), ("ls_noisy_babble5", "chatter, 5 dB")]
SY = [("gen2_pad-step5000-pad", "Earlybird (29M)"), ("ref-tiny-nohold-pad", "Tiny + padding (44M)"), ("ref-tiny-nohold", "Tiny as released"), ("ref-small-nohold", "Moonshine Small (123M)")]
rows = []
for s, n in NS:
    if s not in R: continue
    rows.append([n] + [f"{ms(R[s][k]['stable'])} · {R[s][k]['wer']}%" for k, _ in SY])
tabs.append(dict(title="With background noise", html=table(["Noise added to audiobooks"] + [n for _, n in SY], rows),
                 note="Each cell: when a word stays right (median, vs its end) · word error rate. Earlybird is the earliest at every noise level and more accurate than Tiny in either setting, with about 40% fewer rewrites than Tiny + padding; Small, four times its size and running four times slower, is far more accurate in chatter."))
SZ = [("gen2_pad-step5000-pad", "Earlybird, full precision", "58 MB"), ("gen2_pad_q4fe8-pad", "Earlybird, 4-bit (front end 8-bit)", "16.7 MB"), ("ref-tiny-nohold-pad", "Moonshine Tiny + padding, full precision", "88 MB")]
rows = []
for k, n, size in SZ:
    if k not in R["tedlium_test"]: continue
    rows.append([n, size] + [f"{ms(R[s][k]['stable'])} · {R[s][k]['wer']}%" for s, _ in SETS])
W = R["librispeech_test_clean"]
rows.append(["Whistle (Cactus), live mode", "16.9 MB", f"{ms(W['whistle-live.live']['stable'])} · {W['whistle-live.live']['wer']}% ‡", f"– · {R['librispeech_test_other']['whistle.offline']['wer']}% §", f"– · {R['tedlium_test']['whistle.offline']['wer']}% §", f"– · {R['ps_test']['whistle.offline']['wer']}% §"])
tabs.append(dict(title="Smaller files", html=table(["Model", "File"] + [n for _, n in SETS], rows),
                 note="Each cell: when a word stays right (median, vs its end) · WER. 4-bit by quantization-aware training, 5,000 steps; all 4-bit (15.7 MB) costs 0.5–2 more WER points and 3-bit (13.3 MB) far more (measured without padding). ‡ Whistle numbers on the 462-clip audiobook subset. § Whistle offline WER, for reference; its live mode commits words about 0.4 s after they end."))
h = sys.argv[1]; s = open(h).read()
i = s.index("const DATA = ") + len("const DATA = "); j = s.index(";\n", i)
d = json.loads(s[i:j]); d["stats"]["tables"] = [t for t in d["stats"].get("tables", []) if t["title"] not in {x["title"] for x in tabs}] + tabs
tpl = open(sys.argv[2]).read()
old = 'if (f.table) $("tables").innerHTML = `<h3>${esc(f.table.title)}</h3><div class="tbl">${f.table.html}</div><p>${esc(f.table.note || "")}</p>`;'
if old in s:
    a = tpl.index("  const ts = "); b = tpl.index("\n", tpl.index('$("tables").innerHTML', a))
    s = s.replace("  " + old, tpl[a:b])
s = s[:i] + json.dumps(d, separators=(",", ":")) + s[j:]
open(h, "w").write(s); print(len(tabs), "tables")
