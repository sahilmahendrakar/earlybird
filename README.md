# Earlybird

A tiny streaming speech-to-text model that predicts words before the speaker finishes
saying them, with a confidence score.

- **16.8 MB** on disk: 29M parameters at 4 bits, fine-tuned from Moonshine Tiny.
- **Word latency −90 ms**: on LibriSpeech, the median word is predicted correctly (and never
  changed after) 90 ms before the speaker finishes it. Released streaming models: 90 to 400 ms after.
- **A confidence on every word**, including the one still being spoken. At 90% or more on an
  unfinished word, it is right 91 to 96% of the time.

Write-up and live demo: [sahilmahendrakar.com/writings/earlybird](https://sahilmahendrakar.com/writings/earlybird)
· Model: [huggingface.co/sahilmahendrakar/Earlybird-29M](https://huggingface.co/sahilmahendrakar/Earlybird-29M)

## Quickstart

The weights are in a packed 4-bit format, so they load through one file, `earlybird.py`
(PyTorch and Transformers, CPU is fine).

```sh
pip install torch "transformers>=5" soundfile huggingface_hub
hf download sahilmahendrakar/Earlybird-29M earlybird.py --local-dir .
```

```python
from earlybird import Earlybird

eb = Earlybird.from_pretrained("sahilmahendrakar/Earlybird-29M")   # or a local folder
print(eb.transcribe("speech.wav"))                              # any sample rate, resampled to 16 kHz

# Live: feed 16 kHz mono float32 audio in chunks of any size
stream = eb.stream()
for chunk in mic_chunks:
    for u in stream.feed(chunk):           # one update per 80 ms of audio
        print(u.text, u.words, u.guess)
print(stream.finish().text)
```

Each update has:

| Field | |
|---|---|
| `text` | the transcript so far, including the word being spoken |
| `words` | `[(word, confidence)]` for every word of `text`; the last may be unfinished |
| `guess` | `[(word, confidence)]` the next words Earlybird expects (up to 2) |
| `time` | seconds of audio heard |

From the command line, with this repo cloned (the weights are in `release/earlybird/`):

```sh
python release/earlybird/earlybird.py speech.wav          # every update, then each word's confidence
python release/earlybird/earlybird.py speech.wav --live   # the same at real-time speed
```

It runs about twice as fast as real time on one CPU thread (39 ms per 80 ms step on an Intel Xeon).

## What's here

| | |
|---|---|
| `release/earlybird/` | the released model: weights, config, tokenizer and `earlybird.py` |
| `scripts/` | data preparation, training, quantization and the streaming timing harness |
| `results/` | the tables behind the write-up (`open_asr.md`, `tables.md`, `timing.md`, …) |
| `demo/` | the live demo server (Modal) and the sample player |
| `paper/` | paper notes |

## Reproducing the results

### Setup

```sh
uv sync
```

Python 3.12. Models are downloaded on first use (Moonshine and Whisper automatically;
Nemotron by hand, see below). `data/`, `models/` and `results/raw/` are not in git.

### Test sets

Both end up in the same format: `DIR/audio/*.wav` (16 kHz mono) and `DIR/manifest.jsonl`
with, per utterance, the text and a list of words with start and end times in seconds.

```sh
# 500 held-out WikiText-103 (test split) sentences, 5 Kokoro voices, 100 each
uv run python scripts/synth_kokoro.py data/kokoro_test --split test --n 500

# LibriSpeech test-clean with Montreal Forced Aligner word times (seeded subset of 500,
# minus utterances the alignment does not cover word for word or that run past 25 s)
curl -L -o data/dl/ls_test_clean.parquet \
  https://huggingface.co/datasets/gilkeyio/librispeech-alignments/resolve/main/data/test_clean-00000-of-00001.parquet
uv run python scripts/prep_librispeech.py data/dl/ls_test_clean.parquet data/librispeech_test_clean 500

# compare word ends with an independent forced aligner
uv run python scripts/check_boundaries.py data/kokoro_test 60 end end_raw
```

### Harness

```sh
uv run python scripts/run_stream.py moonshine-tiny data/kokoro_test --workers 8
uv run python scripts/score.py --md results/tables.md
scripts/run_baselines.sh        # everything, then a single-process pass for compute
```

`run_stream.py` feeds audio in 80 ms chunks and logs each change of the on-screen text
with the audio time at which it appeared. `score.py` turns the logs into, per reference
word, the time it first appeared correctly relative to the word's acoustic end (negative
means ahead of the speaker), plus WER and revisions per word. The definitions are in the
docstrings of those two files and in `results/baselines.md`.

To add a recognizer, write a class with `start()`, `step(chunk)` and `finish()` in
`scripts/recognizers.py` and add it to `MODELS`.

Nemotron models for sherpa-onnx (464 MB each):

```sh
cd models
for c in 80ms 160ms 560ms; do
  f=sherpa-onnx-nemotron-speech-streaming-en-0.6b-$c-int8-2026-04-25.tar.bz2
  curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$f && tar xjf $f && rm $f
done
```

### Training corpus

Mostly real speech, with word times so every prefix of an utterance has a target:

| Set | Hours | Licence | Word times |
|---|---:|---|---|
| LibriSpeech train (clean 100, clean 360, other 500) | 960 | CC BY 4.0 | Montreal Forced Aligner (published) |
| People's Speech `clean`, a few hundred hours | ~300 | CC BY | `scripts/align.py` (wav2vec 2.0 CTC) |
| Kokoro: 15,000 templated spoken commands | ~14 | synthetic | Kokoro's phoneme durations |
| Kokoro: 6,000 WikiText sentences | 11 | synthetic | same; the ruler the aligner is checked against |

No non-commercial sets (GigaSpeech, SPGISpeech, TED-LIUM). The real sets have unpunctuated
transcripts, so a teacher (Moonshine Medium) writes each one and an utterance is kept only
when the teacher's words are exactly the reference words (about 80% of LibriSpeech, about
20% of People's Speech in small trials).

```sh
# real speech (GPU): one command per split; --shard i/n splits the files between machines
uv run python scripts/prep_real.py data/ls_clean_100 --source ls --split train_clean_100
uv run python scripts/prep_real.py data/ls_clean_360 --source ls --split train_clean_360
uv run python scripts/prep_real.py data/ls_other_500 --source ls --split train_other_500
uv run python scripts/prep_real.py data/ps_clean --source ps --split clean --hours 300

# Kokoro (Mac): text, then all of it or only the commands
uv run python scripts/build_train_text.py data/text/train.txt 50000 15000
caffeinate -i scripts/synth_train.sh 65000 data/text/train.commands.txt

# how far the aligner's word times are from Kokoro's and from MFA's
uv run python scripts/check_boundaries.py data/kokoro_train 150
uv run python scripts/check_boundaries.py data/librispeech_test_clean 150
```

Aligner against the two references (150 utterances each): word ends within a median of
4 to 8 ms of Kokoro's and 26 ms of MFA's, about 80% within 50 ms; word starts a steady
60 to 70 ms late. `align.py` removes both constants.

26 Kokoro voices are used; `am_michael` and `bf_emma` are left out of training so the test
set has two voices the model has never heard.

### Training

```sh
uv run python scripts/train.py runs/v1 --data data/ls_clean_100:1 --data data/ls_clean_360:3 \
  --data data/ls_other_500:3 --data data/ps_clean:3 --data data/kokoro_train:1 --steps 60000
EARLYBIRD_MODEL=runs/v1/step60000 uv run python scripts/run_stream.py earlybird data/kokoro_test --workers 3
```

The number after each set is its sampling weight. Started again with the same output
directory, `train.py` continues from the last checkpoint. `scripts/gpu_box.sh` copies the
repo to an EC2 machine and runs commands there. The AWS scripts read the bucket, account,
subnets and security group from `EARLYBIRD_BUCKET`, `AWS_ACCOUNT_ID`, `EARLYBIRD_SUBNETS` and
`EARLYBIRD_SG`.

### Notes

- `docs/moonshine-availability.md`: which Moonshine v2 weights and training code exist.
- `results/baselines.md`: baseline table and how to read it.

## License

MIT, see [LICENSE](LICENSE). Earlybird is fine-tuned from
[Moonshine Tiny](https://huggingface.co/moonshine-ai/moonshine-streaming-tiny) by Moonshine AI (MIT); its notice is in
LICENSE too.

## Citation

```bibtex
@misc{mahendrakar2026earlybird,
  title  = {Earlybird: predictive speech-to-text in 17 MB},
  author = {Mahendrakar, Sahil},
  year   = {2026},
  url    = {https://sahilmahendrakar.com/writings/earlybird}
}
```
