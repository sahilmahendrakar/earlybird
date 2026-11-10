"""Streaming recognizers behind one interface.

    rec.start()                 begin an utterance
    rec.step(chunk) -> text     feed 80 ms of 16 kHz float32 audio, get what is on screen now
    rec.finish()    -> text     end of audio, get the final transcript

`step`/`finish` may instead return {variant_name: text}; the runner logs each variant
as its own system (used to get committed and uncommitted Whisper output from one run)."""
import glob, os
import numpy as np

SR = 16000
CHUNK = 1280  # 80 ms


class Moonshine:
    """Moonshine v2 streaming through the official moonshine-voice runtime (quantized .ort
    graphs). `force=True` re-decodes on every chunk; otherwise the runtime decides when it
    has enough new audio (about every 0.5 s)."""

    def __init__(self, arch="TINY_STREAMING", force=True, threads=None):
        from moonshine_voice import Transcriber, ModelArch, get_model_for_language
        path, arch = get_model_for_language("en", getattr(ModelArch, arch))
        # Without this every update copies the line's audio back into Python lists.
        self.tr = Transcriber(path, arch, options={"return_audio_data": "false"})
        self.flags = Transcriber.MOONSHINE_FLAG_FORCE_UPDATE if force else 0
        self.stream = None

    @staticmethod
    def _text(tx):
        return " ".join(l.text.strip() for l in tx.lines if l.text.strip())

    def start(self):
        self.stream = self.tr.create_stream(update_interval=1e9)
        self.stream.start()

    def step(self, chunk):
        self.stream.add_audio(chunk.tolist(), SR)
        return self._text(self.stream.update_transcription(self.flags))

    def finish(self):
        text = self._text(self.stream.stop())
        self.stream.close(); self.stream = None
        return text


class Sherpa:
    """A sherpa-onnx streaming transducer (used for NVIDIA Nemotron streaming 0.6B int8).
    The model's own chunk size is baked into the exported encoder."""

    def __init__(self, model_dir, threads=2):
        import sherpa_onnx
        g = lambda pat: sorted(glob.glob(os.path.join(model_dir, pat)))[0]
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=g("tokens.txt"), encoder=g("encoder*.onnx"), decoder=g("decoder*.onnx"),
            joiner=g("joiner*.onnx"), num_threads=threads, sample_rate=SR, feature_dim=80,
            decoding_method="greedy_search")

    def start(self):
        self.s = self.rec.create_stream()

    def _decode(self):
        while self.rec.is_ready(self.s):
            self.rec.decode_stream(self.s)
        return self.rec.get_result(self.s).strip()

    def step(self, chunk):
        self.s.accept_waveform(SR, chunk)
        return self._decode()

    def finish(self):
        self.s.accept_waveform(SR, np.zeros(int(0.8 * SR), dtype=np.float32))  # flush right context
        self.s.input_finished()
        return self._decode()


class ParakeetRedecode:
    """NVIDIA Parakeet TDT 0.6B v2 (int8, sherpa-onnx), an offline model, streamed the usual way:
    the whole buffer is re-transcribed every `interval` seconds and the latest text shown."""

    def __init__(self, model_dir, interval=0.24, threads=4):
        import sherpa_onnx
        g = lambda pat: sorted(glob.glob(os.path.join(model_dir, pat)))[0]
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=g("tokens.txt"), encoder=g("encoder*.onnx"), decoder=g("decoder*.onnx"),
            joiner=g("joiner*.onnx"), num_threads=threads, sample_rate=SR, feature_dim=80,
            decoding_method="greedy_search", model_type="nemo_transducer")
        self.every = max(1, round(interval * SR / CHUNK))

    def start(self):
        self.buf = []; self.n = 0; self.text = ""

    def _hyp(self):
        s = self.rec.create_stream()
        s.accept_waveform(SR, np.concatenate(self.buf))
        self.rec.decode_stream(s)
        return s.result.text.strip()

    def step(self, chunk):
        self.buf.append(np.asarray(chunk, dtype=np.float32)); self.n += 1
        if self.n % self.every == 0:
            self.text = self._hyp()
        return self.text

    def finish(self):
        self.text = self._hyp() if self.buf else ""
        return self.text


class WhisperLA:
    """Whisper re-run on the whole buffer every `interval` seconds.

    Two variants come out of each run:
      la2  LocalAgreement-2: show only the words two consecutive hypotheses agree on
      raw  show the latest hypothesis as is
    """

    def __init__(self, name="base.en", interval=0.48, device=None, threads=None):
        import torch, whisper
        if threads:
            torch.set_num_threads(threads)
        self.whisper, self.torch = whisper, torch
        self.device = device or "cpu"
        self.model = whisper.load_model(name, device=self.device)
        self.opts = whisper.DecodingOptions(language="en", without_timestamps=True, fp16=False,
                                            temperature=0.0)
        self.every = max(1, round(interval * SR / CHUNK))

    def start(self):
        self.buf = []; self.n = 0; self.committed = []; self.prev = []; self.raw = ""

    def _hyp(self):
        w = self.whisper
        audio = w.pad_or_trim(np.concatenate(self.buf)[-30 * SR:])
        mel = w.log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels).to(self.device)
        with self.torch.no_grad():
            r = w.decode(self.model, mel, self.opts)
        return "" if r.no_speech_prob > 0.8 else r.text.strip()

    @staticmethod
    def _key(word):
        return "".join(c for c in word.lower() if c.isalnum())

    def _update(self, final=False):
        self.raw = self._hyp()
        hyp = self.raw.split(); k = len(self.committed)
        new, old = hyp[k:], self.prev[k:]
        if final:
            self.committed += new
        else:
            for a, b in zip(new, old):
                if self._key(a) != self._key(b):
                    break
                self.committed.append(a)
        self.prev = hyp

    def _out(self):
        return {"la2": " ".join(self.committed), "raw": self.raw}

    def step(self, chunk):
        self.buf.append(chunk); self.n += 1
        if self.n % self.every == 0:
            self._update()
        return self._out()

    def finish(self):
        self._update(final=True)
        return self._out()


class Whistle:
    """Cactus Compute's Whistle (2026-10-02; 55M parameters, 2-4-bit, 16.9 MB; offline encoder-
    decoder with 5-beam search) through its own C++ engine (pip install cactus-needle), English forced.

    Three variants from one run:
      offline  nothing until the audio ends, then one transcription of the whole clip (the model
               as published: up to 30 s in one pass)
      commit   its own live mode (Whistle.stream: ~1 s chunks), the text it has committed
      live     the same, committed text plus its unconfirmed tail
    """

    def __init__(self, threads=None, chunk_s=0.96, live=True):
        os.environ.setdefault("NEEDLE_TELEMETRY", "0")   # the package reports usage unless told not to
        from needle.agent.whistle import Whistle as W, _lib
        self.w = W(); self.lib = _lib(); self.every = max(1, round(chunk_s * SR / CHUNK)); self.live = live

    def start(self):
        if self.live:
            try:   # end any stream left from the previous utterance
                self.lib.needle_stream_transcribe_stop(self.w._buffer, len(self.w._buffer))
            except Exception:
                pass
        self.buf = []; self.pend = []; self.done = []; self.tail = ""

    def _feed(self, x, final=False):
        w, lib = self.w, self.lib
        if x is not None and len(x):
            from needle.agent.whistle import _samples, _options
            s, n = _samples(x); lang, kw = _options("en", None)
            r = w._result(lib.needle_stream_transcribe_process(s, n, lang, kw, w._buffer, len(w._buffer)))
            if r.get("text"):
                self.done.append(r["text"].strip())
            self.tail = (r.get("pending") or "").strip()
        if final:
            r = w._result(lib.needle_stream_transcribe_stop(w._buffer, len(w._buffer)))
            if r.get("text"):
                self.done.append(r["text"].strip())
            self.tail = ""

    def _out(self, offline=""):
        if not self.live:
            return {"offline": offline}
        c = " ".join(t for t in self.done if t)
        return {"offline": offline, "commit": c, "live": (c + " " + self.tail).strip()}

    def step(self, chunk):
        self.buf.append(chunk); self.pend.append(chunk)
        if self.live and len(self.pend) >= self.every:
            self._feed(np.concatenate(self.pend)); self.pend = []
        return self._out()

    def finish(self):
        if self.live:
            self._feed(np.concatenate(self.pend) if self.pend else None, final=True)
        self.pend = []
        audio = np.concatenate(self.buf)   # over 30 s (a few test clips): equal pieces, each one pass
        k = -(-len(audio) // (30 * SR)); n = -(-len(audio) // k)
        text = " ".join(self.w.transcribe(audio[i:i + n], language="en")["text"].strip() for i in range(0, len(audio), n)).strip()
        return self._out(offline=text)


def _nemotron(chunk_ms):
    def make(threads=2):
        d = glob.glob(f"models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-{chunk_ms}ms-int8-*/")
        if not d:
            raise SystemExit(f"Nemotron {chunk_ms} ms model is not in models/ (see README)")
        return Sherpa(d[0], threads=threads)
    return make


def _engine(model=None, threads=1, **kw):
    """EARLYBIRD_MODEL names a checkpoint directory (default: the released Tiny);
    EARLYBIRD_GATES a comma list of confidence thresholds to log as extra systems;
    EARLYBIRD_COMMIT the number of trailing words that may still be rewritten (0: all)."""
    from engine import Engine
    gates = [float(g) for g in os.environ.get("EARLYBIRD_GATES", "").split(",") if g]
    kw.setdefault("commit", int(os.environ.get("EARLYBIRD_COMMIT", "0")))
    return Engine(model or os.environ.get("EARLYBIRD_MODEL", "UsefulSensors/moonshine-streaming-tiny"),
                  threads=threads or 1, gates=gates, **kw)


# name -> (constructor, description)
MODELS = {
    "moonshine-tiny": (lambda **k: Moonshine("TINY_STREAMING", True), "Moonshine v2 Tiny, re-decode every 80 ms"),
    "moonshine-small": (lambda **k: Moonshine("SMALL_STREAMING", True), "Moonshine v2 Small, re-decode every 80 ms"),
    "moonshine-tiny-default": (lambda **k: Moonshine("TINY_STREAMING", False), "Moonshine v2 Tiny, runtime's own update pace"),
    "moonshine-small-default": (lambda **k: Moonshine("SMALL_STREAMING", False), "Moonshine v2 Small, runtime's own update pace"),
    "nemotron-80ms": (_nemotron(80), "Nemotron streaming 0.6B int8, 80 ms chunks"),
    "nemotron-160ms": (_nemotron(160), "Nemotron streaming 0.6B int8, 160 ms chunks"),
    "nemotron-560ms": (_nemotron(560), "Nemotron streaming 0.6B int8, 560 ms chunks"),
    "parakeet-tdt": (lambda threads=4, **k: ParakeetRedecode(glob.glob("models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8*/")[0], threads=threads),
                     "NVIDIA Parakeet TDT 0.6B v2 int8 (offline), re-transcribed every 240 ms"),
    "whisper-base.en": (lambda **k: WhisperLA("base.en", **k), "Whisper base.en re-run every 480 ms"),
    "tiny-nohold": (lambda **k: _engine(**k), "Released Moonshine v2 Tiny, no hold-back, incremental PyTorch engine"),
    "tiny-nohold-pad": (lambda **k: _engine(pad=True, **k), "Same, with the three padding frames after the audio"),
    "tiny-nohold-pad-commit2": (lambda **k: _engine(pad=True, commit=2, **k), "Same, only the last two words may be rewritten"),
    "tiny-nohold-pad-commit4": (lambda **k: _engine(pad=True, commit=4, **k), "Same, only the last four words may be rewritten"),
    "whistle": (lambda **k: Whistle(live=False, **k), "Cactus Whistle, offline (one pass over the whole clip), English forced"),
    "whistle-live": (lambda **k: Whistle(**k), "Same, plus its own live mode (~1 s chunks): committed text, and with the pending tail"),
    "earlybird": (lambda **k: _engine(**k), "A fine-tuned checkpoint (EARLYBIRD_MODEL) on the same engine"),
    "earlybird-pad": (lambda **k: _engine(pad=True, **k), "Same, with the three padding frames after the audio"),
}
