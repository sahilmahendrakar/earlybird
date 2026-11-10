### librispeech_test_clean: 67 utterances every system has

| System | WER % | First correct: median ms | Ahead of word end % | Stable: median ms | Stable p95 ms | Revisions / word |
|---|---:|---:|---:|---:|---:|---:|
| Earlybird (29M) | 2.3 | -100 | 86.4 | -90 | +80 | 1.29 |
| Earlybird 4-bit (16.7 MB) | 1.7 | -100 | 86.3 | -90 | +105 | 1.18 |
| Moonshine Tiny + padding (44M) | 2.1 | -90 | 83.0 | -80 | +240 | 1.82 |
| Earlybird, no padding | 2.0 | -60 | 72.9 | -50 | +126 | 1.10 |
| Moonshine Tiny, no hold-back | 2.4 | -30 | 62.3 | -20 | +210 | 2.39 |
| Moonshine Small, no hold-back (123M) | 0.6 | -30 | 57.5 | -20 | +230 | 1.58 |
| Whisper base.en, re-decoded every step | 2.3 | +90 | 33.7 | +90 | +480 | 0.39 |
| Nemotron streaming 80 ms (600M) | 1.4 | +230 | 5.4 | +230 | +550 | 0.30 |
| Nemotron streaming 160 ms | 0.7 | +240 | 5.3 | +240 | +560 | 0.22 |
| Moonshine Tiny, official runtime | 3.9 | +260 | 6.5 | +270 | +520 | 2.07 |
| Moonshine Small, official runtime | 1.4 | +280 | 4.8 | +290 | +530 | 1.39 |
| Whistle live (Cactus, 16.9 MB) | 3.1 | +400 | 12.1 | +400 | +925 | 0.16 |
| Whistle committed text | 3.1 | +1540 | 0.0 | +1540 | +2515 | 0.00 |

### kokoro_test: 67 utterances every system has

| System | WER % | First correct: median ms | Ahead of word end % | Stable: median ms | Stable p95 ms | Revisions / word |
|---|---:|---:|---:|---:|---:|---:|
| Moonshine Tiny + padding (44M) | 6.1 | -83 | 81.4 | -75 | +116 | 3.48 |
| Earlybird 4-bit (16.7 MB) | 5.9 | -78 | 81.4 | -75 | +100 | 1.70 |
| Earlybird (29M) | 5.6 | -80 | 81.4 | -73 | +81 | 1.86 |
| Earlybird, no padding | 5.1 | -35 | 63.8 | -30 | +125 | 1.64 |
| Moonshine Small, no hold-back (123M) | 4.2 | -37 | 60.5 | -20 | +198 | 3.49 |
| Moonshine Tiny, no hold-back | 5.7 | -15 | 55.0 | -5 | +155 | 3.46 |
| Whisper base.en, re-decoded every step | 4.8 | +90 | 32.1 | +90 | +417 | 1.32 |
| Nemotron streaming 80 ms (600M) | 3.5 | +200 | 5.7 | +200 | +503 | 0.50 |
| Nemotron streaming 160 ms | 3.4 | +213 | 6.2 | +213 | +517 | 0.37 |
| Moonshine Tiny, official runtime | 7.0 | +287 | 4.9 | +292 | +475 | 2.02 |
| Moonshine Small, official runtime | 3.9 | +295 | 6.1 | +303 | +520 | 2.20 |
| Whistle live (Cactus, 16.9 MB) | 7.5 | +422 | 9.4 | +424 | +906 | 0.33 |
| Whistle committed text | 7.5 | +1568 | 0.0 | +1568 | +3015 | 0.00 |
