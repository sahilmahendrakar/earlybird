CPU per-step compute, 1 thread, 50 LibriSpeech test-clean utterances, systems run one after another. Model name:                              Intel(R) Xeon(R) Platinum 8488C

| System | mean ms per 80 ms step | p95 ms | compute / audio |
|---|---:|---:|---:|
| earlybird | 38.8 | 64.1 | 0.49 |
| earlybird_nopad | 33.5 | 53.7 | 0.42 |
| small_pad | 211.8 | 577.9 | 2.65 |
| tiny | 46.9 | 79.0 | 0.59 |
| tiny_pad | 56.3 | 167.1 | 0.70 |
