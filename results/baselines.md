# Baselines

Measured 2026-10-07 on a MacBook Pro (M4 Pro, 12 cores, 24 GB), CPU only.

## What is measured

Audio is fed to each recognizer in 80 ms chunks, followed by 1 s of silence. Every change
of the on-screen text is logged with the audio time at which it appeared.

- **First correct**: for each reference word, the first moment the screen shows it
  correctly in its right place, minus the time the speaker finished the word.
  Positive means the word appeared after the speaker finished it; negative means before.
- **Stable**: the same, but for the moment after which the word never changes again.
- **Ahead %**: share of words on screen before the speaker finished them.
- **Revisions / word**: words taken back from the screen, per reference word.
- **WER**: word error rate of the final transcript (Whisper's English normalizer).
- **Compute / step**: wall time per 80 ms chunk, one process, 20 utterances per set.
  **Compute / audio** is total compute divided by audio length (under 1 keeps up).

The latency columns use the audio clock only; compute time is not added. On a device a
word appears later by roughly the compute of the step that produced it.

Word end times: Kokoro's own phoneme durations for the synthetic set, Montreal Forced
Aligner for LibriSpeech (10 ms resolution).

## Systems

| Name | What it is |
|---|---|
| moonshine-tiny, moonshine-small | Moonshine v2 streaming (official runtime, quantized), re-decoded on every 80 ms chunk |
| moonshine-*-default | Same models at the runtime's own update pace (about every 0.5 s) |
| nemotron-80ms / 160ms / 560ms | NVIDIA Nemotron streaming 0.6B, int8, sherpa-onnx, greedy; the number is the encoder chunk |
| whisper-base.en.raw | Whisper base.en re-run on the whole buffer every 480 ms, latest hypothesis shown as is |
| whisper-base.en.la2 | Same runs, showing only words two consecutive hypotheses agree on (LocalAgreement-2) |

Moonshine ran on the full sets (500 and 462 utterances). Nemotron and Whisper are much
slower to evaluate and ran on an evenly spaced 150 utterances per set, so their numbers
are not on exactly the same utterances as Moonshine's.

## Kokoro test set (synthetic, 5 voices, WikiText sentences)

| System | Utts | WER % | First correct, median ms | p95 ms | Ahead % | Stable, median ms | Revisions / word | Compute / step, mean ms | p95 ms | Compute / audio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| whisper-base.en.raw | 150 | 4.8 | +85 | +410 | 34.3 | +85 | 1.01 | 29.8 | 193.6 | 0.40 |
| nemotron-80ms | 150 | 3.9 | +208 | +515 | 6.2 | +208 | 0.51 | 49.4 | 60.7 | 0.69 |
| nemotron-160ms | 150 | 3.8 | +215 | +530 | 6.0 | +215 | 0.38 | 28.0 | 69.4 | 0.39 |
| moonshine-tiny | 500 | 7.8 | +282 | +458 | 4.6 | +290 | 2.30 | 19.0 | 43.0 | 0.24 |
| moonshine-small | 500 | 5.0 | +295 | +503 | 5.5 | +303 | 2.31 | 72.3 | 158.8 | 0.90 |
| nemotron-560ms | 150 | 3.7 | +370 | +765 | 2.8 | +370 | 0.11 | 10.2 | 78.0 | 0.14 |
| moonshine-tiny-default | 500 | 7.6 | +510 | +840 | 0.9 | +510 | 0.50 | 4.6 | 39.7 | 0.06 |
| moonshine-small-default | 500 | 5.1 | +522 | +880 | 1.3 | +522 | 0.47 | 17.3 | 151.1 | 0.22 |
| whisper-base.en.la2 | 150 | 6.0 | +570 | +943 | 0.4 | +570 | 0.00 | 29.8 | 193.6 | 0.40 |

## LibriSpeech test-clean (real speech)

| System | Utts | WER % | First correct, median ms | p95 ms | Ahead % | Stable, median ms | Revisions / word | Compute / step, mean ms | p95 ms | Compute / audio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| whisper-base.en.raw | 150 | 3.3 | +100 | +480 | 31.9 | +100 | 0.77 | 32.9 | 216.0 | 0.44 |
| nemotron-160ms | 150 | 1.6 | +250 | +570 | 4.5 | +250 | 0.22 | 27.4 | 65.0 | 0.38 |
| nemotron-80ms | 150 | 1.9 | +250 | +560 | 4.7 | +250 | 0.29 | 50.8 | 59.9 | 0.71 |
| moonshine-tiny | 462 | 4.4 | +270 | +470 | 5.5 | +270 | 2.09 | 19.1 | 46.2 | 0.24 |
| moonshine-small | 462 | 2.4 | +290 | +500 | 4.9 | +290 | 1.47 | 71.0 | 176.8 | 0.89 |
| nemotron-560ms | 150 | 1.5 | +400 | +812 | 1.9 | +400 | 0.06 | 9.9 | 75.4 | 0.14 |
| moonshine-tiny-default | 462 | 4.3 | +500 | +850 | 1.5 | +500 | 0.52 | 5.3 | 45.1 | 0.07 |
| moonshine-small-default | 462 | 2.3 | +520 | +880 | 1.3 | +520 | 0.38 | 17.1 | 148.5 | 0.21 |
| whisper-base.en.la2 | 150 | 4.0 | +585 | +1020 | 0.4 | +585 | 0.00 | 32.9 | 216.0 | 0.44 |

## Moonshine Tiny with no hold-back

The official runtime only decodes encoder frames whose 320 ms of look-ahead has arrived.
`scripts/sim_hf.py` instead gives the released Tiny weights (float32, PyTorch) the audio
up to "now" and nothing else, re-encodes it and decodes greedily, at every 80 ms step.
No retraining. 100 utterances per set; the runtime row is the same 100 utterances.

| Set | System | WER % | First correct, median ms | p95 ms | Ahead % | Stable, median ms | p95 ms | Revisions / word |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Kokoro | Tiny, no hold-back | 5.0 | -85 | +50 | 80.7 | -75 | +78 | 2.69 |
| Kokoro | Tiny, runtime, every 80 ms | 5.2 | +280 | +438 | 4.5 | +285 | +464 | 1.9 |
| LibriSpeech | Tiny, no hold-back | 2.8 | -90 | +114 | 82.4 | -80 | +248 | 2.82 |
| LibriSpeech | Tiny, runtime, every 80 ms | 3.8 | +260 | +470 | 5.8 | +270 | +530 | 2.2 |

Dropping the hold-back alone moves the median word from about 270 ms after its end to
about 90 ms before it, with no loss in final accuracy (the final transcript sees the
whole utterance either way; the float32 weights are slightly better than the quantized
runtime). The cost is more flicker: 2.7 to 2.8 words taken back per word against about 2.
So the released model already anticipates; what training has to add is stability.

### The same, computed incrementally (full test sets)

`scripts/engine.py` produces the same text as the re-encoding script but keeps the
encoder state and re-decodes only the tokens that can change, so the compute column is
real. 500 Kokoro and 462 LibriSpeech utterances; compute is one thread, 20 utterances.

| Set | System | WER % | First correct, median ms | p95 ms | Ahead % | Stable, median ms | p95 ms | Revisions / word | Compute / step, mean ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Kokoro | tiny-nohold-pad | 6.9 | -83 | +67 | 80.8 | -75 | +143 | 3.11 | 14.7 |
| Kokoro | tiny-nohold (no padding frames) | 6.9 | -15 | +120 | 55.4 | -4 | +168 | 3.24 | 11.0 |
| Kokoro | moonshine-tiny (runtime, every 80 ms) | 7.8 | +282 | +458 | 4.6 | +290 | +482 | 2.30 | 19.0 |
| LibriSpeech | tiny-nohold-pad | 3.3 | -90 | +110 | 81.3 | -80 | +270 | 2.72 | 20.2 |
| LibriSpeech | tiny-nohold (no padding frames) | 3.3 | -30 | +130 | 61.0 | -20 | +240 | 2.61 | 14.6 |
| LibriSpeech | moonshine-tiny (runtime, every 80 ms) | 4.4 | +270 | +470 | 5.5 | +270 | +520 | 2.09 | 19.1 |

- The three frames the convolutions make out of silence padding are worth about 60 ms:
  with them the released weights guess the end of the word in progress.
- Waiting for confidence does not buy stability cheaply: showing a word only above 0.5
  probability (`.gate50`) cuts revisions from 3.1 to 2.4 per word and costs 20 ms of
  median and 300 ms of p95 stable latency; 0.8 costs far more.
- Freezing all but the last two words (`-commit2`, 150 utterances) halves the revisions
  (1.4 and 0.9 per word) at the same latency but locks in early mistakes: WER rises from
  6.9 to 9.7 and from 3.3 to 6.0.

So no decoding rule gets early, stable and accurate at once from the released weights.
That is the case for training on prefix targets.

## Reading the numbers

- No existing streaming model shows the typical word before the speaker finishes it. The
  best purpose-built streaming models (Nemotron at 80 or 160 ms, Moonshine re-decoded
  every chunk) sit at a median of 210 to 290 ms after the word ends, and only about 5%
  of words appear early.
- Nemotron's 80 ms and 160 ms settings give the same latency; 80 ms just costs twice the
  compute. Its delay comes from the model, not from the chunk size.
- Moonshine's delay matches the 320 ms of look-ahead built into its encoder. Re-decoding
  every chunk halves the delay of the default pace, at about 4 times the compute and 2
  revisions per word.
- Unfiltered Whisper is the earliest (+85 to +100 ms median, a third of words early)
  because it guesses the rest of a half-heard word, but one word is taken back for every
  word shown and each step costs up to about 200 ms of compute that is not in the latency
  column. LocalAgreement removes the revisions and costs almost 500 ms.
- Target for Earlybird: median first-correct at or below 0 ms with Tiny-class compute
  (about 20 ms per 80 ms step) and far fewer than Moonshine's 2 revisions per word.

## Caveats

- The compute columns were measured while an unrelated training job was using the
  machine's GPU and 11 GB of memory. Treat them as approximate, likely on the slow side.
- Nemotron's revisions partly count a word growing piece by piece ("anti" then
  "anticipate"), which the scorer sees as a take-back of the shorter form.
- The two test sets disagree on WER level (the synthetic WikiText sentences are full of
  rare names) but agree on latency within about 40 ms per system.
