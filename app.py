"""Application web de surveillance CCTV.

En local :
    python app.py
    http://127.0.0.1:8000

En ligne : Render (render.yaml) ou Railway (railway.toml).
Le disque persistant garde les comptes et les photos.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from accounts import authenticate, get_user, register, sign_user, user_id_from_cookie
from alerts import AlertLog
from faces import FaceDetector
from session import MonitorSession

ROOT = Path(__file__).resolve().parent
COOKIE = "cctv_user"
MEDIA_NAME = re.compile(r"^[A-Za-z0-9._-]+$")

app = FastAPI(title="Surveillance CCTV")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

SHARED_FACES = FaceDetector()


def alerts_root() -> Path:
    configured = os.environ.get("ALERTS_DIR")
    path = Path(configured) if configured else Path(os.environ.get("DATA_DIR", ROOT)) / "alerts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def current_user_id(request: Request) -> str | None:
    return user_id_from_cookie(request.cookies.get(COOKIE))


def session_directory(user_id: str) -> Path:
    path = alerts_root() / user_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def cookie_secure(request: Request) -> bool:
    forwarded = request.headers.get("x-forwarded-proto", "")
    return request.url.scheme == "https" or forwarded.split(",")[0].strip() == "https"


def set_login_cookie(response, request: Request, user_id: str):
    response.set_cookie(
        COOKIE,
        sign_user(user_id),
        httponly=True,
        samesite="lax",
        secure=cookie_secure(request),
        max_age=60 * 60 * 24 * 30,
    )
    return response


@app.get("/")
def index(request: Request):
    if current_user_id(request) is None:
        return RedirectResponse("/login", status_code=303)
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/login")
def login_page(request: Request):
    if current_user_id(request):
        return RedirectResponse("/", status_code=303)
    return FileResponse(ROOT / "static" / "login.html")


@app.post("/api/register")
async def api_register(request: Request):
    data = await request.json()
    user, error = register(
        str(data.get("name", "")),
        str(data.get("email", "")),
        str(data.get("password", "")),
    )
    if error or user is None:
        return JSONResponse({"error": error or "Inscription impossible."}, status_code=400)
    response = JSONResponse({"name": user["name"], "email": user["email"]})
    return set_login_cookie(response, request, user["id"])


@app.post("/api/login")
async def api_login(request: Request):
    data = await request.json()
    user = authenticate(str(data.get("email", "")), str(data.get("password", "")))
    if user is None:
        return JSONResponse({"error": "E-mail ou mot de passe incorrect."}, status_code=401)
    response = JSONResponse({"name": user["name"], "email": user["email"]})
    return set_login_cookie(response, request, user["id"])


@app.post("/api/logout")
def api_logout():
    response = JSONResponse({"ok": True})
    response.delete_cookie(COOKIE)
    return response


@app.get("/api/me")
def api_me(request: Request):
    user_id = current_user_id(request)
    if user_id is None:
        raise HTTPException(status_code=401)
    user = get_user(user_id)
    if user is None:
        raise HTTPException(status_code=401)
    return {"name": user["name"], "email": user["email"]}


@app.get("/media/{name}")
def media(name: str, request: Request) -> FileResponse:
    user_id = current_user_id(request)
    if user_id is None or not MEDIA_NAME.fullmatch(name):
        raise HTTPException(status_code=404)
    path = session_directory(user_id) / name
    if not path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(path)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def demo_frame(index: int, width: int = 640, height: int = 360) -> np.ndarray:
    image = np.full((height, width, 3), 180, np.uint8)
    cv2.rectangle(image, (0, 0), (width - 1, height - 1), (160, 160, 160), 2)
    if index > 50:
        x = 40 + ((index - 50) * 6) % (width - 140)
        cv2.rectangle(image, (x, 70), (x + 90, 210), (245, 245, 245), -1)
    return image


def allowed_stream_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    return parsed.scheme in {"http", "https", "rtsp", "rtsps"} and bool(parsed.hostname)


def open_stream(url: str) -> cv2.VideoCapture | None:
    capture = cv2.VideoCapture()
    capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    if not capture.open(url):
        capture.release()
        return None
    return capture


async def run_demo(websocket: WebSocket, session: MonitorSession, send_lock: asyncio.Lock) -> None:
    index = 0
    try:
        while True:
            payload = await asyncio.to_thread(session.process_frame, demo_frame(index))
            index += 1
            async with send_lock:
                await websocket.send_json(payload)
            await asyncio.sleep(0.07)
    except asyncio.CancelledError:
        return


async def run_capture(
    websocket: WebSocket,
    session: MonitorSession,
    send_lock: asyncio.Lock,
    url: str,
) -> None:
    capture = await asyncio.to_thread(open_stream, url)
    if capture is None:
        async with send_lock:
            await websocket.send_json({"error": "Flux inaccessible"})
        return
    try:
        while True:
            ok, frame = await asyncio.to_thread(capture.read)
            if not ok or frame is None:
                async with send_lock:
                    await websocket.send_json({"error": "Flux interrompu"})
                break
            payload = await asyncio.to_thread(session.process_frame, frame)
            async with send_lock:
                await websocket.send_json(payload)
    except asyncio.CancelledError:
        return
    finally:
        capture.release()


@app.websocket("/ws")
async def stream(websocket: WebSocket) -> None:
    await websocket.accept()
    user_id = user_id_from_cookie(websocket.cookies.get(COOKIE))
    if user_id is None:
        await websocket.close(code=4401)
        return
    session = MonitorSession(AlertLog(session_directory(user_id)), SHARED_FACES)
    await websocket.send_json(
        {
            "status": "En attente",
            "history": session.history,
            "objects": 0,
            "persist": 0,
            "persist_max": 12,
            "zones": 0,
        }
    )
    send_lock = asyncio.Lock()
    source_task: asyncio.Task | None = None

    async def stop_source_task() -> None:
        nonlocal source_task
        if source_task is not None and not source_task.done():
            source_task.cancel()
            try:
                await source_task
            except asyncio.CancelledError:
                pass
        source_task = None

    try:
        while True:
            incoming = await websocket.receive()
            if incoming.get("type") == "websocket.disconnect":
                break
            if incoming.get("text"):
                data = json.loads(incoming["text"])
                command = session.apply_control(data)
                if command == "demo":
                    await stop_source_task()
                    source_task = asyncio.create_task(run_demo(websocket, session, send_lock))
                elif command == "stream":
                    url = str(data.get("url", "")).strip()
                    await stop_source_task()
                    if not allowed_stream_url(url):
                        await websocket.send_json({"error": "Adresse de flux invalide"})
                    else:
                        source_task = asyncio.create_task(
                            run_capture(websocket, session, send_lock, url)
                        )
                elif command == "stop":
                    await stop_source_task()
                continue
            payload_bytes = incoming.get("bytes")
            if not payload_bytes:
                continue
            await stop_source_task()
            frame = cv2.imdecode(np.frombuffer(payload_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            payload = await asyncio.to_thread(session.process_frame, frame)
            async with send_lock:
                await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
    finally:
        await stop_source_task()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"Surveillance CCTV : http://0.0.0.0:{port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
