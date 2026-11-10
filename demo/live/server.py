"""Live Earlybird over a WebSocket: the browser streams 16 kHz mono float32 audio, the server runs
the streaming engine and sends back, after each batch of audio, what Earlybird has on screen.

Message to the browser (JSON): {t, words: [[text, p], ...], ghost: [[text, p], ...]}
  words  the transcript, each word as written (cased, with punctuation) and its probability
  ghost  the guessed next words after the transcript, each with its probability
A session is one connection; at most MAX_SESSIONS at a time, each up to MAX_SECONDS of audio.

usage: server.py MODEL_DIR [--port 8000]   (serve behind a TLS proxy: browsers only allow the
microphone on https pages)"""
import argparse, asyncio, json, os, sys, time
import numpy as np
import torch
import websockets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from engine import Engine

CHUNK, SR = 1280, 16000
MAX_SESSIONS, MAX_SECONDS = int(os.environ.get("EB_SESSIONS", "2")), 120
torch.set_num_threads(int(os.environ.get("EB_THREADS", "1")))
GHOST_EVERY = int(os.environ.get("EB_GHOST_EVERY", "2"))
norm_w = lambda w: "".join(c for c in w.lower() if c.isalnum() or c == "'")


def words_with_p(eng):
    toks, probs = eng.last["tokens"], eng.last["probs"]
    out, s = [], 0
    for i in range(1, len(toks) + 1):
        if i == len(toks) or eng._is_word_start(toks[i]):
            w = eng.tok.decode(toks[s:i], skip_special_tokens=True).strip()
            if w:
                out.append([w, round(float(np.prod(probs[s:i])), 3)])
            s = i
    return out


def ghost_with_p(eng):
    return [[w, round(float(p), 3)] for w, p in zip(eng.ghost.split(), eng.ghost_word_p)]


class Server:
    def __init__(self, model):
        self.model = model; self.active = 0
        Engine(model, pad=True, ghost=2)   # warm the file cache and fail early if the model is bad

    async def handle(self, ws):
        if self.active >= MAX_SESSIONS:
            await ws.send(json.dumps({"busy": True})); await ws.close(); return
        self.active += 1
        try:
            loop = asyncio.get_running_loop()
            eng = await loop.run_in_executor(None, lambda: Engine(self.model, pad=True, ghost=int(os.environ.get("EB_GHOST", "2"))))
            eng.start(); buf = np.zeros(0, np.float32); n = 0
            # guesses are re-decoded every GHOST_EVERY steps (they cost as much as the transcript); in
            # between, a guessed word that came true is dropped from the front, a contradicted guess cleared
            full = eng.n_ghost; ghost, said = [], []
            await ws.send(json.dumps({"ready": True}))
            async for msg in ws:
                if isinstance(msg, str):
                    if msg == "reset":
                        eng.start(); buf = np.zeros(0, np.float32); n = 0; ghost, said = [], []
                        await ws.send(json.dumps({"t": 0, "words": [], "ghost": []}))
                    continue
                buf = np.concatenate([buf, np.frombuffer(msg, dtype=np.float32)])
                stepped = False
                while len(buf) >= CHUNK and n * CHUNK < MAX_SECONDS * SR:
                    chunk, buf = buf[:CHUNK], buf[CHUNK:]
                    t0 = time.perf_counter()
                    eng.n_ghost = full if n % GHOST_EVERY == 0 else 0
                    await loop.run_in_executor(None, eng.step, chunk)
                    n += 1; stepped = True; ms = (time.perf_counter() - t0) * 1e3
                    words = words_with_p(eng)
                    if eng.n_ghost:
                        ghost = ghost_with_p(eng)
                    else:
                        new = [w for w, _ in words][len(said):]
                        k = 0
                        while k < len(new) and k < len(ghost) and norm_w(new[k]) == norm_w(ghost[k][0]):
                            k += 1
                        ghost = ghost[k:] if k == len(new) else []
                    said = [w for w, _ in words]
                if stepped:
                    await ws.send(json.dumps({"t": round(n * CHUNK / SR, 2), "words": words,
                                              "ghost": ghost, "ms": round(ms, 1)}))
                if n * CHUNK >= MAX_SECONDS * SR:
                    await ws.send(json.dumps({"limit": True})); break
        except websockets.ConnectionClosed:
            pass
        finally:
            self.active -= 1


async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    srv = Server(a.model)
    async with websockets.serve(srv.handle, "127.0.0.1", a.port, max_size=2 ** 20):
        print(f"listening on 127.0.0.1:{a.port}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
