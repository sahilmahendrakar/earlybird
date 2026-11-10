# Moonshine v2: what is available to start from

Checked 2026-10-07 on this Mac, with `moonshine-voice` 0.1.5 and `transformers` 5.19.0.

## Weights

| Model | Hub id | File | Parameters in the checkpoint | License |
|---|---|---|---:|---|
| Tiny ("34M") | `UsefulSensors/moonshine-streaming-tiny` | `model.safetensors`, 176 MB, float32 | 44.1M | MIT |
| Small ("123M") | `UsefulSensors/moonshine-streaming-small` | `model.safetensors`, 561 MB, float32 | 140.1M | MIT |
| Medium ("245M") | `UsefulSensors/moonshine-streaming-medium` | `model.safetensors`, 1064 MB, float32 | 265.9M | MIT |

The `UsefulSensors/...` ids redirect to `moonshine-ai/...`. None is gated.

The checkpoints hold more parameters than the advertised sizes because the token
embedding and the output layer are stored as two separate matrices
(`tie_word_embeddings: false`). For Tiny: encoder 9.5M, decoder 24.1M (of which the
token embedding is 10.5M), output layer 10.5M. The vocabulary is 32,768 tokens, so the
two vocabulary matrices are 21M of Tiny's 44M. A ~10M student has to shrink the
vocabulary or tie and factor these matrices; the transformer layers are the small part.

The deployment runtime uses different files: quantized ONNX Runtime graphs downloaded
from `download.moonshine.ai` (Tiny is 45 MB in total: `frontend`, `encoder`, `adapter`,
`cross_kv`, `decoder_kv` as `.ort`, plus `tokenizer.bin`).

## Architecture facts that matter for latency

From `config.json` of Tiny:

- 6 encoder layers and 6 decoder layers, width 320, 8 heads.
- 50 frames per second (20 ms per frame).
- Encoder attention windows per layer, as (frames back, frames ahead):
  (16, 4), (16, 4), (16, 0), (16, 0), (16, 4), (16, 4).
  Four layers look 4 frames ahead, so the encoder as a whole waits for
  **16 frames = 320 ms** of future audio before a frame is final.

That 320 ms is built into the pretrained encoder. It lines up with what the harness
measures: Tiny shows a word a median of 250 to 300 ms after the word ends.

## Training code

1. **Official fine-tuning code ships inside the pip package.**
   `moonshine_voice/lora/` (about 2,300 lines) trains a Moonshine Streaming checkpoint
   through Hugging Face Transformers. Install with `pip install 'moonshine-voice[finetune]'`.
   - `--adapt lora` (default) puts LoRA on self-attention in the decoder, encoder or
     both. `--adapt full` unfreezes the whole model.
   - Its own default learning rates: 1e-3 for decoder LoRA, 1e-4 when the encoder is
     touched, 1e-5 for full fine-tuning.
   - Data: `--train-manifest` takes JSONL/JSON/TSV rows of `{audio, text}`. It mixes in
     general-domain "replay" audio by default so the model does not forget.
   - Loss is plain teacher-forced cross-entropy on `[BOS] tokens [EOS]`. A comment in
     `train.py` warns that passing `labels=` shifted tokens twice in Transformers up to
     5.14; it passes `decoder_input_ids` and shifts by hand instead.
   - It selects CUDA or CPU only (no MPS path), with bf16/fp16 autocast on CUDA.
   - `python -m moonshine_voice.lora --export` writes the five ONNX graphs the runtime
     loads from a fine-tuned checkpoint. Quantizing to `.ort` uses
     `quantize-streaming-model.sh` from the GitHub repo.

2. **Transformers.** `MoonshineStreamingForConditionalGeneration` and `AutoProcessor`
   are in released Transformers. I checked on this Mac that Tiny loads, transcribes a
   LibriSpeech utterance correctly, gives a teacher-forced loss of 0.09 on it, and
   back-propagates into the encoder. So a custom training loop (changed targets, a
   distillation loss, a smaller student built from the same classes) is ordinary
   PyTorch. The model card notes that the Transformers path does not do efficient
   streaming: it re-encodes the whole clip. Fine for training, not for the demo.

3. **No from-scratch training recipe.** The GitHub repo (`moonshine-ai/moonshine`) has
   the C++ runtime, bindings, docs and scripts, but not the code or data that produced
   the released weights. The paper gives the settings: Schedule-Free optimizer, learning
   rate 2e-3, 400K steps, effective batch 512, 8 H100 GPUs. Training a model of this
   kind from random weights is out of reach for this sprint; fine-tuning and
   distillation from the released weights are not.

## What this means for the training path

- Fine-tune Tiny with our own loop on top of the Transformers classes, borrowing the
  loss alignment and the export step from `moonshine_voice.lora`. The official trainer
  cannot be used unchanged because our targets are not plain transcripts (the model has
  to learn to finish a word early and to guess the next one).
- The deployed model can stay on the official runtime: export, quantize, and the
  existing harness entry for Moonshine measures it with no changes.
- The encoder's 320 ms look-ahead is the main thing between the current model and
  showing a word before it ends. Either the decoder learns to predict past what the
  encoder has finalized, or the look-ahead windows are narrowed and the encoder is
  fine-tuned to cope. That is the first experiment to run.

## AWS

The AWS CLI is not installed on this Mac (`aws: command not found`) and there is no
`~/.aws` directory, so there are no credentials to test. Nothing was launched.
