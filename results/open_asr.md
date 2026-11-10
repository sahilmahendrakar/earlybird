# Open ASR Leaderboard: English sets (2026-11-06)

Word error rate (%) on the eight English test sets of the Hugging Face Open ASR Leaderboard,
scored with the leaderboard's own text normalizer (github.com/huggingface/open_asr_leaderboard,
`normalizer.EnglishTextNormalizer`, commit a768a2c) and corpus WER as in its `run_eval.py`.
Audio comes from the leaderboard's copies (hf-audio/open-asr-leaderboard); TED-LIUM from our
identical 1,155-segment copy (the leaderboard repo no longer ships it). Runner: `scripts/open_asr.py`
(box script `scripts/open_asr_box.sh`). Word timings used elsewhere in our results come from
`scripts/align.py`.

Each system gives its final transcript of the clip followed by 1 s of silence:
- **Earlybird** (full precision, and the real 4-bit release weights): the streaming engine fed the
  whole clip at once, which gives exactly its streaming final text (checked: 100/100 clips identical).
- **Moonshine Tiny**: the official moonshine-voice runtime (quantized). Its published leaderboard
  entry (moonshine-streaming-tiny, full-precision Transformers) is 12.0 mean; this runtime scores worse.
- **Whisper base.en**: openai-whisper, greedy, one pass per <=30 s piece. Published leaderboard mean
  10.32; ours matches per set (TED 4.87 = 4.87, LS clean 4.10 vs 4.25, SPGI 4.11 vs 4.26).
- **Whistle** (Cactus, offline 5-beam, engine build of 2026-10-19 from the Hub history, since the
  wheels were removed on 2026-11-05): TED-LIUM in full, a fixed ~200-clip sample of every other set
  (it ran at ~1x real time per core here).

Sampling: Whisper base gets a fixed hash sample of GigaSpeech (10%) and SPGISpeech (5%); Whistle a
nested smaller sample of every set but TED-LIUM. The second table puts every system on exactly the
same clips.

Full sets (Whisper base and Whistle: a fixed sample of GigaSpeech and SPGISpeech, see SAMPLE):

| System | tedlium | voxpopuli | librispeech_clean | earnings22 | librispeech_other | ami | gigaspeech | spgispeech | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| earlybird-4bit | 6.06 | 13.80 | 4.37 | 21.91 | 13.18 | 21.23 | 13.66 | 7.30 | 12.69 |
| earlybird | 5.64 | 12.99 | 4.03 | 20.52 | 11.64 | 19.27 | 12.77 | 6.45 | 11.66 |
| moonshine-tiny | 7.87 | 15.71 | 5.50 | 23.40 | 15.33 | 24.73 | 15.01 | 8.18 | 14.47 |
| whisper-base.en | 4.87 | 9.30 | 4.10 | 14.94 | 10.20 | 19.53 | 11.18 | 4.11 | 9.78 |
| whistle | 7.65 | 10.77 | 4.11 | 17.87 | 11.29 | 24.01 | 14.59 | 7.05 | 12.17 |

Same utterances for every system (the ids all systems have, per set):

| System | tedlium (1155) | voxpopuli (215) | librispeech_clean (212) | earnings22 (200) | librispeech_other (201) | ami (188) | gigaspeech (182) | spgispeech (190) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| earlybird-4bit | 6.06 | 13.61 | 3.73 | 21.71 | 14.18 | 21.84 | 13.11 | 6.77 |
| earlybird | 5.64 | 12.94 | 3.42 | 20.04 | 12.01 | 19.43 | 11.69 | 6.07 |
| moonshine-tiny | 7.87 | 15.40 | 5.44 | 23.27 | 15.99 | 24.94 | 14.53 | 7.33 |
| whisper-base.en | 4.87 | 9.83 | 3.75 | 14.42 | 10.24 | 18.50 | 11.61 | 4.01 |
| whistle | 7.65 | 10.77 | 4.11 | 17.87 | 11.29 | 24.01 | 14.59 | 7.05 |

Mean over the eight sets on the same clips: Whisper base 9.65 · Earlybird 11.41 · Whistle 12.17 ·
Earlybird 4-bit 12.63 · Moonshine Tiny (official runtime) 14.35.

Speed (audio seconds per compute second, one CPU thread, c7i.8xlarge): Earlybird 10-11x,
Moonshine Tiny runtime 4-6x, Whisper base 1-3x (Whistle not comparable: its engine uses every core).

Cost: 5 on-demand c7i.8xlarge boxes, 7.3 box-hours, about $10.40.
