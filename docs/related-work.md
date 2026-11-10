# Related work: anticipatory streaming ASR (Earlybird)

Literature review compiled 2026-10-15 for the Earlybird paper or blog post. The system under review is a 29M-parameter (16.7 MB at 4-bit) streaming recognizer distilled from Moonshine streaming Tiny. It has five properties:

1. Words are due about 120 ms before their acoustic end.
2. It emits 1 or 2 "ghost" words beyond the audio, each with a probability.
3. Those probabilities are calibrated (ECE about 1 to 2.5 points).
4. It makes about 3x fewer revisions than the teacher, because it was trained on random audio-prefix cut-offs with prefix-consistent targets.
5. At equal confidence, it predicts 2 to 3x more words correctly before they end than a small ASR followed by a text LM (Qwen3-0.6B).

A possible sixth property is an utterance-complete claim made before the last word ends.

Verification legend. **[V]** means I checked the title, authors and venue against arXiv, ACL Anthology, ISCA or the publisher page. **[V-abs]** means the same check was made and I also read the abstract or full text. **[U]** means I could not fully verify it, and the reason is given.

---

## 1. Anticipation and prediction of future words in streaming or incremental ASR

This is the area closest to our work, and it has **two direct precedents** (1a, 1b) that must be cited and distinguished.

**1a. Zink, Higuchi, Mullov, Waibel, Kobayashi. "Predictive Speech Recognition and End-of-Utterance Detection Towards Spoken Dialog Systems." arXiv:2409.19990, Sept 2024 (submitted to ICASSP 2025; I did not find the ICASSP proceedings version).** https://arxiv.org/abs/2409.19990 [V-abs, full text read]
- **What they did:**
  - Trained an ESPnet hybrid CTC/attention Conformer on LibriSpeech-100 and Switchboard.
  - During training, zeroed the last T_mask ∈ U(0, 500 ms) of audio before end-of-utterance (EOU), keeping positional encodings. This forces the decoder to predict the words in the masked span.
  - Evaluated with "FWER" (WER on future words only, given ground-truth past words) at T_mask = 100 to 500 ms. EOU time is predicted from cross-attention on the padded frames.
  - Reported that the model "can reasonably operate up to 300 ms before an utterance ends".
- **Results:** FWER stays high (above 70% at FWER@1, below 70% only with the oracle best of 5 beams).
- **What they did not do:**
  - The model is **offline** (full-context encoder-decoder, utterance-final masking only), not streaming.
  - There are no probabilities or calibration for predicted words.
  - There is no partial-transcript stability analysis.
  - There is no comparison with an external LM baseline at equal confidence.
  - Masking happens only at the utterance end, not at random prefixes.
- **Relation to us:** This is the closest "in-recognizer anticipation" precedent. Our differences are streaming operation, random-prefix (not only end-of-utterance) training, per-word calibrated probabilities, word-level "before it ends" timing, an on-device model size, and a direct head-to-head against ASR+LM.

**1b. Yusuf, Baskar, Rosenberg, Ramabhadran. "Speculative Speech Recognition by Audio-Prefixed Low-Rank Adaptation of Language Models." Interspeech 2024, pp. 792–796. arXiv:2407.04641.** https://arxiv.org/abs/2407.04641 [V-abs]
- **What they did:**
  - Defined "speculative speech recognition (SSR)", in which the recognizer "runs ahead of audio".
  - Built a 100M Conformer-Transducer plus a 100M Transformer LM. The LM takes the ASR hypothesis plus an audio-derived soft prompt and is LoRA-finetuned to complete it.
  - Truncated the final 1 s of audio (about 2 words).
  - Measured with suffix-oracle WER (SOWER, best of k=8).
  - Showed the audio-prefixed LM beats a hypothesis-only finetuned LM.
- **What they did not do:** calibration or stability.
- **Relation to us:** This is the canonical "ASR + separate LM" speculation design, so it is our baseline family. They already show that *audio conditioning helps the LM*. Our result goes further: anticipation *inside the recognizer* beats ASR + a much larger text LM (Qwen3-0.6B) by 2 to 3x at matched confidence. Note that their metric is oracle best-of-k, which is not the same as our calibrated single guesses.

**1c. Schwarz, He, Van Segbroeck, Hethnawi, Rastrow. "Personalized Predictive ASR for Latency Reduction in Voice Assistants." Interspeech 2023. arXiv:2305.13794.** https://arxiv.org/abs/2305.13794 [V-abs]
- **What they did:** Amazon work that predicts the *full utterance* from a partial one, so that responses can be prefetched. Prediction comes from personalized history and lists of popular utterances. They study the trade-off between latency gain and wasted prefetch cost. Data are internal plus SLURP.
- **Relation to us:** It motivates the use case (prefetch and speculative execution for agents) and the cost of wrong predictions. Their predictor works at utterance level and retrieval-like, not as acoustic word-level anticipation.

**1d. Chang, Li, Rybach, He, Li, Sainath, Strohman. "Low Latency Speech Recognition Using End-to-End Prefetching." Interspeech 2020, pp. 1962–1966.** https://www.isca-archive.org/interspeech_2020/chang20b_interspeech.html [V]
- **What they did:** Submit partial hypotheses downstream before finalization and use them if the final matches. This saves about 200 ms.
- **Relation to us:** This is the "speculate on the partial" systems idea. Its value depends on how stable the partials are, which connects to our stability result.

**1e. Baumann and Schlangen. "Predicting the Micro-Timing of User Input for an Incremental Spoken Dialogue System that Completes a User's Ongoing Turn." SIGDIAL 2011, pp. 120–129.** https://aclanthology.org/W11-2015/ [V]
- **What they did:** Predict the remaining duration of the current word and the duration of the next word while the user is speaking, so a system can complete the user's utterance in sync.
- **Relation to us:** This is an early precedent for "finish the word before it ends" and for completion timing. It is a classic incremental-dialogue reference.

**1f. Sakuma, Fujie, Kobayashi. "Response timing estimation for spoken dialog systems based on syntactic completeness prediction." SLT 2023, pp. 369–374.** [V via citation in 1a and 2606.13450. I did not open the paper itself.]
- **What they did:** Feed streaming-ASR text to an LM that forecasts future tokens and EOU.
- **Relation to us:** This is the ASR-then-LM design used for timing.

**1g. An, Zheng, Ou et al. "CUSIDE: Chunking, Simulating Future Context and Decoding for Streaming ASR." Interspeech 2022. arXiv:2203.16758.** CUSIDE-T is the transducer extension (arXiv:2407.10255). https://arxiv.org/abs/2203.16758 [V. The author list beyond "An, Zheng" was not re-checked.]
- **What they did:** Simulate *future acoustic frames* to replace real right-context.
- **Relation to us:** This is the nearest "predict the future inside the recognizer" idea, but it predicts *features*, to save lookahead latency. It does not emit future *words*, and it has no probabilities.

**1h. Udupa, Watanabe, Schwarz, Černocký. "Endpoint Anticipation for Low-Latency Spoken Dialogue." Interspeech 2026. arXiv:2606.13450.** https://arxiv.org/abs/2606.13450 [V-abs]
- **What they did:** A speech-only model forecasts end-of-turn up to 2.56 s ahead to trigger speculative LLM/TTS. Integrated with Unmute, it reduces latency by 505 ms on average at the cost of 28.4% more speculative compute.
- **Relation to us:** This is the endpoint-anticipation counterpart to our possible "utterance complete before the last word ends" claim. It forecasts timing, not words.

**Also relevant (systems):**
- Li and Grover. "PredGen: Accelerated Inference of Large Language Models through Input-Time Speculation for Real-Time Speech Interaction." arXiv:2506.15556, 2025 [V-abs]. Speculatively generates LLM responses while the user is still speaking. This is a downstream consumer of anticipation.
- "KAME" (Kuroki et al., 2025) and "Stream RAG" (Arora et al., 2025) are cited in 2606.13450 as speculative processing on partial transcripts. **[U] I did not verify the titles or venues.**

---

## 2. Partial-hypothesis stability, flicker, re-translation erasure

**2a. Arivazhagan, Cherry, Macherey, Foster. "Re-translation versus Streaming for Simultaneous Translation." IWSLT 2020.** https://aclanthology.org/2020.iwslt-1.27/ (arXiv:2004.03643) [V]
- **What they did:** Re-translating from scratch on each new source token, with stability heuristics, matches or beats streaming MT. Flicker is quantified with **Normalized Erasure (NE)**.
- **Relation to us:** This is the standard source for erasure as a flicker metric. Our "3x fewer rewrites" should be reported in NE (or an equivalent) for comparability.

**2b. Arivazhagan, Cherry, Te, Macherey, Baljekar, Foster. "Re-Translation Strategies for Long Form, Simultaneous, Spoken Language Translation." ICASSP 2020. arXiv:1912.03393.** https://arxiv.org/abs/1912.03393 [V. The author list beyond Arivazhagan and Cherry was not re-checked.]
- **What they did:** Introduced the erasure and stability metrics for re-translation of live captions, combining inference heuristics with **prefix training** (from Niehues et al. 2018).

**2c. Niehues, Pham, Ha, Sperber, Waibel. "Low-Latency Neural Speech Translation." Interspeech 2018.** https://www.isca-archive.org/interspeech_2018/niehues18_interspeech.html [V-abs]
- **What they did:** Trained on artificially generated *partial-sentence* source/target pairs, which **cut the number of displayed corrections by 45%** with no loss in quality.
- **Relation to us:** This is the **closest precedent for our stability mechanism** (training on prefixes with prefix-consistent targets) in the MT/SLT setting. Our contribution is to apply it to a streaming ASR student on random audio cut-offs, measured against its own teacher.

**2d. Shangguan, Knister, He, McGraw, Beaufays. "Analyzing the Quality and Stability of a Streaming End-to-End On-Device Speech Recognizer." Interspeech 2020. arXiv:2006.01416.** https://arxiv.org/abs/2006.01416 [V-abs]
- **What they did:** Introduced word- and segment-level instability metrics (for example the unstable partial word ratio) for an on-device RNN-T. They showed that some quality-improving techniques hurt stability, and they proposed mitigations.
- **Relation to us:** This is the ASR-specific stability metric reference, also on-device.

**2e. Baumann, Atterer, Schlangen. "Assessing and Improving the Performance of Speech Recognition for Incremental Systems." NAACL-HLT 2009.** https://aclanthology.org/N09-1043/ [V]
- **What they did:** Introduced the incremental-ASR metrics edit overhead, correction time and first/final decision times. Simple right-context smoothing cut spurious edits from 90% to 10% at 320 ms lag.
- **Relation to us:** This is the classic stability-timeliness trade-off. Our result improves stability *without* adding lag.

**2f. Flickering reduction by partial-hypothesis re-ranking.** "Flickering Reduction with Partial Hypothesis Reranking for Streaming ASR." IEEE SLT 2022 (IEEE doc 10023016). https://ieeexplore.ieee.org/document/10023016 [U: I confirmed the title and IEEE entry but not the authors or the exact venue.]
- **What they did:** Re-rank beam partials toward a stable prefix, which roughly halves flicker.
- **Relation to us:** This is an inference-time fix. Ours is a training-time fix.

**2g. Macháček, Dabre, Bojar. "Turning Whisper into Real-Time Transcription System." IJCNLP-AACL 2023 (demo). arXiv:2307.14743.** https://aclanthology.org/2023.ijcnlp-demo.3/ [V]
- **What they did:** The **LocalAgreement** policy commits the longest common prefix of consecutive chunk outputs. This trades latency (about 3.3 s) for stability.
- **Relation to us:** This is the de facto stabilizer for non-streaming models, including Whistle's "commit what two passes agree on".

**2h. Wang et al. "Simul-Whisper: Attention-Guided Streaming Whisper with Truncation Detection." Interspeech 2024. arXiv:2406.10052.** https://arxiv.org/abs/2406.10052 [V]
- **What they did:** Identified *truncated words at chunk boundaries* as a key error source and added an integrate-and-fire truncation detector to avoid emitting them.
- **Relation to us:** This is the opposite policy. They suppress partial words, while we deliberately learn to complete them.

**Also:**
- Madureira, Kahardipraja, Schlangen. "The Road to Quality is Paved with Good Revisions: A Detailed Evaluation Methodology for Revision Policies in Incremental Sequence Labelling." SIGDIAL 2023. arXiv:2307.15508 [V]. It formalizes edits and revisions and is useful for metric definitions.
- Selfridge et al. "Stability and Accuracy in Incremental Speech Recognition." SIGDIAL 2011 [U: I saw only the title, via ResearchGate].

---

## 3. Emission-latency reduction in streaming ASR

All of these papers move emission *toward* the acoustic evidence: they reduce delay to roughly zero relative to the word's acoustic occurrence. None trains an explicit target to emit a word *before* its acoustic end with a calibrated probability, and none emits words beyond the audio. "Negative latency" appears only as an occasional side effect (see TrimTail).

- **3a. Yu, Chiu, Li, Chang, Sainath, He, Narayanan, Han, Gulati, Wu, Pang. "FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization." ICASSP 2021. arXiv:2010.11148.** https://arxiv.org/abs/2010.11148 [V. The author tail is from memory and should be checked.] It adds an alignment-free transducer regularizer that favors non-blank emission. On LibriSpeech, P90 latency drops from 210 to 30 ms.
- **3b. Kim, Lu, Tripathi, Zhang, Sak. "Reducing Streaming ASR Model Delay with Self Alignment." Interspeech 2021. arXiv:2105.05005.** https://www.isca-archive.org/interspeech_2021/kim21j_interspeech.html [V] It uses the model's own Viterbi alignments to push toward lower-delay paths. Delay is 25% lower than FastEmit at similar WER.
- **3c. Kang, Yao, Kuang, Guo, Yang, Lin, Żelasko, Povey. "Delay-penalized transducer for low-latency streaming ASR." ICASSP 2023. arXiv:2211.00490.** https://arxiv.org/abs/2211.00490 [V] It is equivalent to penalizing average symbol delay, and it is in k2/icefall.
- **3d. Song, Wu, Wu, Zhang, Zhang, Peng, Li, Pan, Zhu. "TrimTail: Low-Latency Streaming ASR with Simple but Effective Spectrogram-Level Length Penalty." ICASSP 2023. arXiv:2211.00522.** https://arxiv.org/abs/2211.00522 [V] It trims trailing frames during training, which reduces latency by 100 to 200 ms. The paper notes that models can then predict trailing tokens "even before they were spoken". **This is the nearest latency-literature mention of emitting before the audio, and it is a side effect rather than a calibrated objective. Cite it when we claim "before the word ends".**
- **3e. Zeghidour, Kharitonov, Orsini, Volhejn, de Marmiesse, Grave, Pérez, Mazaré, Défossez. "Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling." arXiv:2509.08753, 2025 (Kyutai STT/TTS).** https://arxiv.org/abs/2509.08753 [V] ASR is a text stream delayed relative to aligned audio. The delay is a training-time constant. Our "due 120 ms before the end" is, in effect, a *negative* delay at the word boundary, and DSM is the natural framing to contrast with.
- **3f. Zheng. "Look Less, Hear Better: Jointly Rewarded GRPO for Streaming ASR." arXiv:2609.18333, Sept 2026.** [V-abs] It shows that structural delay poorly reflects user-perceived latency, proposes a word-level emission-delay metric (AWED), and applies GRPO with joint WER and latency rewards. It is relevant to how we *measure* "before the word ends".
- **3g. Lin, Fu, Wen, Liu, Qin, Gan, Wang. "X2Streaming-ASR: wait when uncertain, emit when ready for streaming ASR." arXiv:2609.08672, Sept 2026.** [V-abs] It separates *when* to commit from *what* to commit, using an RL-trained commit policy. Commit latency is 12 to 85 ms for English words. It is a confidence-gated emission policy but does not anticipate.
- **Other related papers** [V, titles only]: Tian et al.'s Peak-first CTC and Bayes-risk CTC; Shinohara and Watanabe, "Minimum Latency Training of Sequence Transducers" (arXiv:2211.02333); Mahadeokar et al., "Alignment Restricted Streaming RNN-T" (arXiv:2011.03072); Strimel et al., "Lookahead When It Matters: Adaptive Non-causal Transformers for Streaming Neural Transducers" (ICML 2023).

**How we differ:**
- Latency-regularization work minimizes delay *subject to acoustic evidence*, and its targets stay causal with respect to the word.
- Our targets deliberately place the word *before* its end, and they extend 1 to 2 words *past* the audio, with explicit probabilities.
- The trade-off we report is therefore precision against earliness at matched calibration, not WER against delay.

---

## 4. Anticipation in simultaneous translation

- **4a. Ma, Huang, Xiong, Zheng, Liu, Zheng, Zhang, He, Liu, Li, Wu, Wang. "STACL: Simultaneous Translation with Implicit Anticipation and Controllable Latency using Prefix-to-Prefix Framework." ACL 2019, pp. 3025–3036.** https://aclanthology.org/P19-1289/ [V. The author list is from memory and matches dblp's "MaHXZLZZHLLWW19" key.] wait-k prefix-to-prefix training makes the model *implicitly anticipate* target words whose source has not arrived (for example, German verbs). It also gives an anticipation-rate analysis. Prefix-to-prefix training is the MT analogue of our prefix-consistent training.
- **4b. Grissom II, He, Boyd-Graber, Morgan, Daumé III. "Don't Until the Final Verb Wait: Reinforcement Learning for Simultaneous Machine Translation." EMNLP 2014.** https://aclanthology.org/D14-1140/ [V] It *explicitly predicts the final verb* and learns with RL when to trust the prediction. This is the earliest explicit "predict unseen source words, act on confidence" work.
- **4c. Chen, Zheng, Kita, Ma, Huang. "Improving Simultaneous Translation by Incorporating Pseudo-References with Fewer Reorderings." EMNLP 2021.** https://aclanthology.org/2021.emnlp-main.473/ [V] It defines the **k-anticipation rate (k-AR)** and shows that excess anticipation in training causes hallucination. This is the counterpoint: anticipation is only useful when it is calibrated.
- **4d. Chang, Chuang, Lee. "Anticipation-Free Training for Simultaneous Machine Translation." IWSLT 2022.** https://aclanthology.org/2022.iwslt-1.5/ [V] It removes forced anticipation, which is the opposite design choice.
- **4e. Ouyang, Hrinchuk, Chen, Lavrukhin, Balam, Li, Ginsburg. "Anticipating Future with Large Language Model for Simultaneous Machine Translation" (TAF). NAACL 2025. arXiv:2410.22499.** https://aclanthology.org/2025.naacl-long.286/ [V] An LLM samples continuations of the *source prefix*. Each is translated, and the system commits only the prefix most samples agree on, gaining up to +5 BLEU at 3-word latency. This is the external-LM anticipation design, used here for MT, with agreement acting as a confidence proxy.

**Relation to us:** In SimulMT, "anticipation" means producing output that depends on input not yet received. That is what our ghost words are. This literature supplies (i) the anticipation-rate metric, (ii) evidence that uncalibrated anticipation leads to hallucination, and (iii) the external-LM-continuation baseline (TAF), which mirrors our ASR + Qwen3 baseline.

---

## 5. Next-word, end-of-turn and turn-taking prediction for spoken dialogue

- **5a. Ekstedt and Skantze. "TurnGPT: a Transformer-based Language Model for Predicting Turn-taking in Spoken Dialog." Findings of EMNLP 2020.** https://aclanthology.org/2020.findings-emnlp.268/ [V] GPT-2 over the incremental transcript predicts the probability of a turn shift after each word. It is text-only and depends on the ASR text.
- **5b. Ekstedt and Skantze. "Voice Activity Projection: Self-supervised Learning of Turn-taking Events." Interspeech 2022. arXiv:2205.09812.** https://www.isca-archive.org/interspeech_2022/ekstedt22_interspeech.html [V] It predicts future voice activity of both speakers from audio in discrete future bins, with a self-supervised objective. It anticipates *when*, not *what*.
- **5c. Skantze. "Turn-taking in conversational systems and human-robot interaction: A review." Computer Speech & Language 67:101178, 2021.** [V] This is the review to cite for prediction-based (rather than reactive) turn-taking.
- **5d. Chang, Li, Sainath, Zhang, Strohman, Liang, He. "Turn-Taking Prediction for Natural Conversational Speech." Interspeech 2022. arXiv:2208.13321.** https://www.isca-archive.org/interspeech_2022/chang22_interspeech.html [V] A turn-taking head joint with E2E ASR handles disfluent pauses. It reaches 97% recall and 85% precision at about 100 ms.
- **5e. Umair, Sarathy, de Ruiter. "Large Language Models Know What To Say But Not When To Speak." Findings of EMNLP 2024. arXiv:2410.16044.** https://aclanthology.org/2024.findings-emnlp.909/ [V] LLMs predict within-turn TRPs poorly. It supports the case that text-only LMs are weak at timing.
- **5f. Industry toolkits.** These are toolkit blogs and model cards, not papers, so cite them as URLs.
  - **LiveKit turn detector:** a fine-tuned SmolLM-v2 of about 135M parameters that reads the transcript (last 4 turns) and outputs P(end of utterance). https://blog.livekit.io/using-a-transformer-to-improve-end-of-turn-detection and https://huggingface.co/livekit/turn-detector [V].
  - **Pipecat Smart Turn v2:** an audio-only Whisper-based classifier. https://huggingface.co/pipecat-ai/smart-turn-v2 [V].
  - Both represent the "separate model after ASR" design.
- **5g. Recent joint ASR and turn-state work (2026):**
  - Fu et al. "X2-Turn: Frame-Synchronous Dual-Head Modeling for Joint Streaming ASR and Turn State Prediction." arXiv:2608.10878 [V-abs].
  - Li and Shi. "The Trade-off Was in the Labels: Causal Supervision for Turn-Aware Streaming ASR." arXiv:2609.04225 [V-abs]. **Highly relevant to our training design.** They argue that every streaming-decision label must be computable from the input up to the decision point, and that offline labels leak the future. Our prefix-consistent targets are the same principle applied to transcripts. The one deliberate exception is our anticipation targets, which is exactly why they need calibration.

**Relation to us:**
- Turn-taking work predicts *when* the turn ends. TurnGPT and LiveKit predict from text, while VAP and Smart Turn predict from audio.
- Our model predicts *which words* come next from audio, and could predict completeness (the possible utterance-complete claim).
- Zink et al. (1a) and Udupa et al. (1h) are the closest to an "utterance complete before the last word ends" claim. Zink predicts the EOU up to about 300 ms early.

---

## 6. Confidence estimation and calibration for E2E ASR

- **6a. Li, Qiu, Zhang, Li, He, Woodland, Cao, Strohman. "Confidence Estimation for Attention-based Sequence-to-sequence Models for Speech Recognition." ICASSP 2021. arXiv:2010.11428.** https://arxiv.org/abs/2010.11428 [V] It shows that E2E models are overconfident (they memorize training sequences) and proposes a lightweight Confidence Estimation Module (CEM).
- **6b. Qiu, Li, He, Zhang, Li, Cao, Prabhavalkar, Bhatia, Li, Hu, Sainath, McGraw. "Learning Word-Level Confidence for Subword End-to-End ASR." ICASSP 2021. arXiv:2103.06716.** https://arxiv.org/abs/2103.06716 [V] It estimates word-level (not wordpiece-level) confidence, which matches our per-word ghost probabilities.
- **6c. Laptev and Ginsburg. "Fast Entropy-Based Methods of Word-Level Confidence Estimation for End-to-End Automatic Speech Recognition." SLT 2022/2023. arXiv:2212.08703.** https://arxiv.org/abs/2212.08703 [V for title and arXiv. Venue SLT 2022 is my belief, not checked.] It aggregates normalized entropy for CTC and RNN-T word confidence (NVIDIA NeMo).
- **6d. Guo, Pleiss, Sun, Weinberger. "On Calibration of Modern Neural Networks." ICML 2017.** [Well known; not re-fetched.] It is the source for ECE and temperature scaling.
- **6e. "Identifying and Calibrating Overconfidence in Noisy Speech Recognition." arXiv:2509.07195, 2025.** [U: title and abstract only, authors not checked.] It reports Whisper token overconfidence in noise and uses selective temperature scaling.

**Relation to us:** Prior ASR confidence work calibrates confidence in words that were *heard*, usually to detect errors. We found no prior work that reports the **calibration of predictions about words not yet heard**. Neither Zink et al. nor Yusuf et al. report confidence or calibration; they use FWER and an oracle best-of-k. **This looks novel.**

---

## 7. Small on-device streaming ASR (context)

- **7a. Jeffries, King, Kudlur, Nicholson, Wang, Warden. "Moonshine: Speech Recognition for Live Transcription and Voice Commands." arXiv:2410.15608, 2024.** https://arxiv.org/abs/2410.15608 [V]
- **7b. Kudlur, King, Wang, Warden. "Moonshine v2: Ergodic Streaming Encoder ASR for Latency-Critical Speech Applications." arXiv:2602.12241, Feb 2026.** https://arxiv.org/abs/2602.12241 [V] It uses a sliding-window streaming encoder and reports TTFT. This is our teacher (streaming Tiny).
- **7c. King, Sabra, Kudlur, Wang, Warden. "Flavors of Moonshine: Tiny Specialized ASR Models for Edge Devices." arXiv:2509.02523, 2025.** [V] The models are about 27M-parameter monolingual Tiny models.
- **7d. Noroozi, Majumdar, Kumar, Balam, Ginsburg. "Stateful Conformer with Cache-based Inference for Streaming Automatic Speech Recognition." ICASSP 2024. arXiv:2312.17279.** [V] This is the cache-aware FastConformer behind NVIDIA's streaming Parakeet and Nemotron models.
- **7e. Banfic et al. "Pushing the Limits of On-Device Streaming ASR: A Compact, High-Accuracy English Model for Low-Latency Inference." arXiv:2604.14493, Apr 2026.** [V-abs] It surveys more than 50 configurations (Whisper, Nemotron, Parakeet TDT, Canary, Qwen3-ASR) and picks Nemotron Speech Streaming, quantized from 2.47 GB to 0.67 GB. Average streaming WER is 8.20%.
- **7f. Macháček et al. Whisper-Streaming (see 2g). Wang et al. Simul-Whisper (see 2h).**
- **7g. Cactus Compute. "Whistle: Speech Recognition for Tiny Devices" (Mroz, Ndubuaku, Mosoyan, Cylich, Kumar, Sandhu, Shemet, Lee), 2026, model card and repo.** https://huggingface.co/Cactus-Compute/whistle, code at https://github.com/cactus-compute/needle [V for the model card. Released around Oct 2 2026 according to press coverage.]
  - The model card has a misc BibTeX entry only; I found no arXiv paper.
  - It is a 16.9 MB file in 7 languages and gives per-word timestamps and probabilities.
  - Streaming works by committing the words two passes agree on, which is LocalAgreement-style rather than natively streaming.
  - **[U]** Third-party blogs report 55.15M params and "no streaming output / 30 s clip cap". The model card does not state the parameter count, so do not cite 55M without checking.
- **7h. Kyutai DSM STT (3e) is a streaming ASR with a configurable delay.**

---

## 8. Early intent and incremental SLU from partial utterances (brief, for a later paper)

- DeVault, Sagae, Traum. "Can I Finish? Learning When to Respond to Incremental Interpretation Results in Interactive Dialogue." SIGDIAL 2009. https://aclanthology.org/W09-3902/ [V] It predicts the *final* meaning from partial utterances and learns when that prediction is reliable enough to act on. **This is the conceptual ancestor of "act on a confident anticipation".**
- Potdar, Avila, Xing, Wang, Cao, Chen. "A Streaming End-to-End Framework For Spoken Language Understanding." IJCAI 2021. arXiv:2105.10042. [V] A CTC RNN detects intents as soon as there is enough evidence.
- Shivakumar, Kumar, Georgiou, Narayanan. "RNN Based Incremental Online Spoken Language Understanding." SLT 2021. [U: title verified via the USC PDF, author order not checked.]
- Arora, Dalmia, Chang, Yan, Black, Watanabe. "Two-Pass Low Latency End-to-End Spoken Language Understanding." Interspeech 2022. arXiv:2207.06670. [V]
- Schwarz et al. 2023 (1c), which uses SLURP for prefix-based full-utterance prediction.
- Chang et al. "Streaming Intended Query Detection using E2E Modeling for Continued Conversation." arXiv:2208.13322 (Google, 2022). [U: title only.]

---

## Overall assessment

### What overlaps with prior work (cite and differentiate)

| Our claim | Closest prior work | How it differs |
|---|---|---|
| ASR predicts words not yet spoken | **Zink et al. 2024** (masked-future training, FWER, up to 300 ms); **Yusuf et al. 2024** (speculative speech recognition, "runs ahead of audio") | Theirs are offline encoder-decoder or utterance-final settings, about 100M+ parameters, with best-of-k oracle metrics. Ours is streaming, random-prefix, about 29M parameters on-device, with single calibrated guesses evaluated as precision vs earliness. |
| ASR + external LM as the alternative | **Yusuf et al. 2024** (ASR→LM, and audio-prefixed LM beats hypothesis-only LM); Sakuma et al. 2023; TurnGPT; TAF (MT) | Yusuf already shows the LM benefits from audio. Our contribution is the head-to-head at matched confidence, showing in-recognizer anticipation gets 2 to 3x more early-correct words than a 20x-larger text LM. To be fair, mention that an audio-prefixed LM (Yusuf) is the stronger hybrid baseline, which we did not test. |
| Training on random prefixes to reduce flicker | **Niehues et al. 2018** (partial-sentence training cut corrections by 45% in SLT); Arivazhagan et al. 2020 (prefix training plus erasure); STACL prefix-to-prefix | The idea is known in MT/SLT. Applying it to a streaming ASR student (distillation with prefix-consistent targets from random audio cut-offs), and measuring about 3x fewer rewrites than the teacher, is a solid empirical contribution but not a new idea. Frame it as "bringing prefix training to streaming ASR distillation". |
| Emitting earlier | FastEmit, self-alignment, delay-penalized transducer, TrimTail (which notes tokens predicted "before they were spoken" as a side effect), Kyutai DSM delays | These minimize delay toward zero relative to the evidence. Ours sets *negative* delay targets (due about 120 ms before the end), plus explicit beyond-audio words. |
| Utterance-complete claim before the last word ends | Zink et al. (predictive EOU, up to about 300 ms); Udupa et al. 2026 (endpoint anticipation, up to 2.56 s); VAP; Chang et al. 2022; Li and Shi 2026 | This is a well-populated area. Only claim novelty if it falls out of the same calibrated anticipation head, and benchmark it against VAP or the LiveKit-style baselines. |

### What appears novel (I found no prior instance)

1. **Calibrated probabilities on anticipated (not-yet-heard) words** in ASR, with ECE reported. Neither predictive-ASR paper reports confidence or calibration. ASR calibration work covers only heard words.
2. **Streaming, on-device anticipatory ASR.** Both precedents are offline or large (about 100M ASR plus 100M LM). A roughly 17 MB streaming model that commits words before they end and shows ghost words is new as far as I can find.
3. **A unified evaluation of early-correct words at matched confidence**, as a precision-earliness curve, comparing in-recognizer anticipation with ASR plus LLM. Prior metrics are FWER (Zink), oracle best-of-k SOWER (Yusuf), and anticipation rate (SimulMT). A per-word "time before acoustic end against precision at confidence ≥ p" metric appears to be new.
4. **Anticipation and stability together.** Prior work treats anticipation (which tends to cause revisions and hallucination, per Chen et al. 2021) and stability as being in tension. Showing that anticipation can coexist with 3x fewer rewrites, because ghost words are marked separately with probabilities, is a useful framing.

### Suggested positioning

- Position the work as "making speculative or predictive ASR (Yusuf et al. 2024; Zink et al. 2024) practical: streaming, tiny, stable and calibrated".
- The distinctive scientific claims are (a) calibration of anticipations, which makes them actionable for voice agents, in the spirit of DeVault et al. 2009 and Grissom et al. 2014 "act when confident", and (b) the matched-confidence comparison against the ASR + LLM pipeline.
- Present the stability gain as prefix training (Niehues 2018; STACL) carried over to streaming ASR distillation, and use Normalized Erasure (Arivazhagan 2020) and/or Shangguan 2020's metrics so the numbers are comparable.
- Contrast "due 120 ms before end" with latency regularizers (FastEmit, delay-penalized, TrimTail) and with DSM's fixed delay. Those papers reduce lag. Ours has a negative target delay plus a confidence gate.
- Cite Li and Shi 2026 ("every label must be computable from input up to the decision point") as supporting prefix-consistent targets, and note that anticipation is the deliberate, calibrated exception.

### Flags and unverified items

- Zink et al. 2024: I found no ICASSP 2025 proceedings version. Cite it as an arXiv preprint unless one is found.
- The FastEmit, STACL and Arivazhagan 2019/2020 (ICASSP) author lists beyond the first authors were reproduced from memory and dblp keys, so check them against the bib.
- The authors of "Flickering Reduction with Partial Hypothesis Reranking", Selfridge 2011, the KAME and Stream RAG titles, the authors of arXiv:2509.07195, and the Google intended-query paper were **not** verified.
- Whistle's parameter count (55M) and streaming limits come only from third-party blogs.
- Many 2026 arXiv papers (2606.13450, 2608.10878, 2609.04225, 2609.08672, 2609.18333) are very recent preprints. They were verified as existing, but their venues may be unconfirmed.
- I found no paper on a **Moonshine-distilled** anticipatory model or on "ghost words" terminology. The term appears unused in the literature.
