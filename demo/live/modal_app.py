"""The live demo (server.py + index.html) on Modal: CPU only, scales to zero, memory snapshot after
torch and the model are loaded, so a cold start is a restore (~1-2 s) instead of a full boot.

  modal deploy demo/live/modal_app.py      (from the repo root, with a Modal token)

The model is the real 4-bit release file in release/earlybird (16.8 MB, in git), unpacked to an
ordinary checkpoint when the image is built (scripts/pack.py unpack), so any checkout can deploy.

Each container serves at most EB_SESSIONS talkers (one engine thread each); more visitors get
more containers, up to MAX_CONTAINERS. Idle containers stop after SCALEDOWN seconds."""
import os
import modal

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SESSIONS, THREADS, MAX_CONTAINERS, SCALEDOWN = 2, 2, 3, 120

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("transformers==5.19.0", "tokenizers==0.23.2", "numpy==2.5.3", "websockets", "fastapi[standard]")
    .env({"EB_SESSIONS": str(SESSIONS), "EB_THREADS": str(THREADS), "EB_GHOST_EVERY": "3"})
    .add_local_dir(os.path.join(ROOT, "release/earlybird"), "/eb/release", copy=True)
    .add_local_file(os.path.join(ROOT, "scripts/pack.py"), "/eb/pack.py", copy=True)
    .run_commands("python /eb/pack.py unpack /eb/release /eb/model")
    .add_local_dir(os.path.join(ROOT, "scripts"), "/eb/scripts", ignore=["__pycache__"])
    .add_local_file(os.path.join(ROOT, "demo/live/server.py"), "/eb/demo/live/server.py")
    .add_local_file(os.path.join(ROOT, "demo/live/index.html"), "/eb/demo/live/index.html")
)
app = modal.App("earlybird-live", image=image)


class _WS:
    """A Starlette WebSocket seen through the slice of the `websockets` API that server.py uses."""
    def __init__(self, ws):
        self.ws = ws

    async def send(self, s):
        await self.ws.send_text(s)

    async def close(self):
        await self.ws.close()

    def __aiter__(self):
        return self

    async def __anext__(self):
        m = await self.ws.receive()
        if m["type"] == "websocket.disconnect":
            raise StopAsyncIteration
        return m["bytes"] if m.get("bytes") is not None else m.get("text")


@app.cls(cpu=SESSIONS * THREADS, memory=2048, enable_memory_snapshot=True, min_containers=0,
         max_containers=MAX_CONTAINERS, scaledown_window=SCALEDOWN, timeout=3600)
@modal.concurrent(max_inputs=SESSIONS)
class Live:
    @modal.enter(snap=True)
    def load(self):
        import sys
        sys.path.insert(0, "/eb/demo/live")
        import server   # imports torch, transformers and the engine
        self.server = server
        self.srv = server.Server("/eb/model")   # loads the model once (and fails early if it is bad)

    @modal.enter(snap=False)
    def threads(self):
        import torch
        torch.set_num_threads(THREADS)   # thread settings are not kept across a snapshot restore

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, WebSocket
        from fastapi.responses import HTMLResponse
        api = FastAPI()
        page = open("/eb/demo/live/index.html").read()

        @api.get("/")
        def index():
            return HTMLResponse(page)

        @api.websocket("/ws")
        async def ws(sock: WebSocket):
            await sock.accept()
            try:
                await self.srv.handle(_WS(sock))
            except Exception:   # the browser went away mid-send
                pass

        return api
