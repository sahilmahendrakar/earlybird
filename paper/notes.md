# Paper notes (arXiv short paper)

Material kept for the paper, not the blog.

## Next-word guessing vs a text language model (cut from the blog 2026-11-06)

Another way to guess upcoming words is to run a separate text model on the transcript. We tried
Qwen3-1.7B, a language model 60 times Earlybird's size, and gave it the exact transcript so it never
had to deal with recognition errors.

- Its guesses come later. A text model can only guess a word once the word before it is finished.
  Earlybird hears the start of the word itself. Earlybird's right guesses land about 200 ms before
  the word ends; the text model's about 100 ms.
- Its confidence can't be trusted. When it says 80% sure, it is not right 80% of the time: its stated
  confidence is off by 13-16 points on average. Earlybird's is off by about 1.

Caveat to state in the paper: both guess right about equally often (LibriSpeech, confidence >= 0.5:
Qwen 13.8% of words, Earlybird 15.6%). The win is timing and calibration, not hit rate.
Sources: results/guess_baselines.txt (qwen17-oracle sections), run log 2026-10-20 22:54Z.

## Decisions so far
- Baselines: official runtimes only (Moonshine Tiny/Small official runtime, Whistle, Whisper base,
  Parakeet, Nemotron). No padded-Tiny comparison. The earliness claim vs official Moonshine Tiny stays.
- Standard-accuracy table: Open ASR Leaderboard English sets (scripts/open_asr.py), leaderboard normalizer.
