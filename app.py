"""SAM3 Studio: interfaz web local para SAM3 (osam) en Apple Silicon.

Uso:
    .venv/bin/python app.py            # abre http://127.0.0.1:8765
    .venv/bin/python app.py --device webgpu --port 9000
"""

from __future__ import annotations

import argparse
import functools
import io
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import uvicorn
from PIL import Image
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

import osam

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"  # temporal: fotogramas del archivo abierto
EXPORTS = ROOT / "videos exportados"
MODEL = "sam3"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Etiquetas de objetos sin texto; la página las traduce.
BOX, POINT, SIMILAR = "@box", "@point", "@similar"

ARGS: argparse.Namespace
print = functools.partial(print, flush=True)  # noqa: A001


class Session:
    """Un solo archivo abierto a la vez (app de un usuario)."""

    def __init__(self) -> None:
        self.sid: str | None = None
        self.name = ""
        self.kind: str | None = None  # "image" | "video"
        self.frames: list[np.ndarray] = []
        self.frame_dir: Path | None = None
        self.fps: float = 0
        self.w = self.h = 0
        self.reset_results()
        # trabajo en curso
        self.busy = False
        self.job: str | None = None
        self.progress: dict = {}
        self.error: str | None = None
        self.last_ms: int | None = None
        self.export_path: str | None = None

    def reset_results(self) -> None:
        self.version = 0
        self.objects: dict[int, dict] = {}  # id -> {id, label, first, last}
        self.frame_objs: list[list[dict]] = [[] for _ in self.frames]
        self.labels: list[np.ndarray | None] = [None] * len(self.frames)
        # vídeo: historial de acciones de osam y último resultado
        self.actions: list[dict] = []
        self.base: str | None = None
        # imagen: anotaciones por id
        self.image_anns: dict[int, object] = {}
        self.image_prompts: dict[int, dict] = {}  # clics/caja de cada objeto
        self.next_id = 0

    def summary(self) -> dict:
        return {
            "sid": self.sid,
            "name": self.name,
            "kind": self.kind,
            "n_frames": len(self.frames),
            "fps": self.fps,
            "width": self.w,
            "height": self.h,
            "version": self.version,
            "objects": sorted(self.objects.values(), key=lambda o: o["id"]),
            "frame_objs": self.frame_objs,
            "busy": self.busy,
            "job": self.job,
            "progress": self.progress,
            "error": self.error,
            "last_ms": self.last_ms,
            "export_path": self.export_path,
            "n_actions": len(self.actions),
        }


S = Session()
LOCK = threading.Lock()
_client: osam.Client | None = None


def client() -> osam.Client:
    global _client
    if _client is None:
        _client = osam.Client(device=ARGS.device)
    return _client


def err(code: str, message: str, status: int = 400) -> JSONResponse:
    # code: la página lo traduce; message: texto en inglés de respaldo
    return JSONResponse({"code": code, "error": message}, status)


# ---------------------------------------------------------------- utilidades


def paint(label: np.ndarray, ann, oid: int) -> None:
    if ann.mask is None or ann.bounding_box is None:
        return
    bb = ann.bounding_box
    region = label[bb.ymin : bb.ymax + 1, bb.xmin : bb.xmax + 1]
    region[ann.mask] = oid + 1


def bbox_of(ann) -> list[int] | None:
    bb = ann.bounding_box
    return None if bb is None else [bb.xmin, bb.ymin, bb.xmax, bb.ymax]


def run_job(job: str, fn, *args) -> JSONResponse:
    with LOCK:
        if S.busy:
            return err("busy", "Another task is running", 409)
        S.busy = True
        S.job = job
        S.error = None
        S.progress = {"stage": "starting"}

    def worker():
        t0 = time.time()
        try:
            fn(*args)
            S.last_ms = int((time.time() - t0) * 1000)
        except osam.ResponseError as e:
            S.error = f"{e.code}: {e}"
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            S.error = str(e)
        finally:
            S.progress = {}
            S.busy = False

    threading.Thread(target=worker, daemon=True).start()
    return JSONResponse({"ok": True})


# ------------------------------------------------------------------- imagen


def image_segment(kind: str, data: dict) -> None:
    S.progress = {"stage": "segmenting"}
    img = S.frames[0]
    th = data.get("threshold")
    kw: dict = {"keep_alive": "30m"}
    new: list[tuple] = []
    if kind == "text":
        for t in [t.strip() for t in data["text"].split(",") if t.strip()]:
            # repetir una palabra sustituye sus objetos anteriores
            for oid in [i for i, o in S.objects.items() if o["label"] == t]:
                S.objects.pop(oid)
                S.image_anns.pop(oid, None)
            anns = client().segment(MODEL, img, texts=[t], score_threshold=th, **kw)
            new += [(a, t) for a in anns]
    elif kind == "box" and data.get("similar"):
        anns = client().segment(
            MODEL, img, example_boxes=[data["box"]], score_threshold=th, **kw
        )
        new = [(a, SIMILAR) for a in anns]
    elif kind == "box":
        anns = client().segment(MODEL, img, box=data["box"], **kw)
        new = [(a, BOX) for a in anns[:1]]
    elif kind == "point":
        anns = client().segment(
            MODEL, img, points=[data["point"]], point_labels=[1], **kw
        )
        new = [(a, POINT) for a in anns[:1]]
    elif kind == "refine":
        # más clics sobre un objeto: se vuelve a segmentar con todos sus puntos
        oid = data["id"]
        ann = S.image_anns[oid]
        pr = S.image_prompts.setdefault(oid, {"points": [], "labels": [], "box": None})
        if not pr["points"] and pr["box"] is None and ann.bounding_box is not None:
            bb = ann.bounding_box  # objetos de texto: partimos de su caja
            pr["box"] = [bb.xmin, bb.ymin, bb.xmax, bb.ymax]
        pr["points"].append(data["point"])
        pr["labels"].append(int(data["label"]))
        anns = client().segment(MODEL, img, points=pr["points"],
                                point_labels=pr["labels"], box=pr["box"], **kw)
        if anns:
            S.image_anns[oid] = anns[0]
        rebuild_image()
        return
    else:
        raise ValueError(kind)

    for ann, label in new:
        oid = S.next_id
        S.next_id += 1
        S.image_anns[oid] = ann
        if kind == "point":
            S.image_prompts[oid] = {"points": [data["point"]], "labels": [1], "box": None}
        elif kind == "box" and not data.get("similar"):
            S.image_prompts[oid] = {"points": [], "labels": [], "box": data["box"]}
        S.objects[oid] = {"id": oid, "label": label, "first": 0, "last": 0}
    rebuild_image()


def rebuild_image() -> None:
    label = np.zeros((S.h, S.w), np.uint16)
    objs = []
    for oid, ann in S.image_anns.items():
        paint(label, ann, oid)
        objs.append({"id": oid, "score": ann.score, "bbox": bbox_of(ann)})
    S.labels = [label]
    S.frame_objs = [objs]
    S.version += 1


# -------------------------------------------------------------------- vídeo


def video_track(new_actions: list[dict], label: str, reset: bool) -> None:
    if reset:
        S.reset_results()
    actions = S.actions + new_actions

    def progress(p):
        S.progress = {"stage": "tracking", "frame": p.frame, "total": p.total_frames}

    S.progress = {"stage": "preparing"}
    resp = client().track(MODEL, S.frames, actions, keep_alive="30m",
                          progress=progress)
    S.actions = actions
    S.base = resp.done.result

    objects = {}
    for o in resp.done.objects:
        prev = S.objects.get(o.object_id, {})
        objects[o.object_id] = {
            "id": o.object_id,
            "label": prev.get("label", label),
            "first": o.first_frame,
            "last": o.last_frame,
        }
    labels = [np.zeros((S.h, S.w), np.uint16) for _ in S.frames]
    frame_objs: list[list[dict]] = [[] for _ in S.frames]
    for fr in resp.frames:
        for ann in fr.annotations:
            oid = ann.object_id
            if oid is None:
                continue
            objects.setdefault(oid, {"id": oid, "label": label,
                                     "first": fr.frame, "last": fr.frame})
            paint(labels[fr.frame], ann, oid)
            frame_objs[fr.frame].append(
                {"id": oid, "score": ann.score, "bbox": bbox_of(ann)}
            )
    S.objects = objects
    S.labels = labels
    S.frame_objs = frame_objs
    S.version += 1


# ------------------------------------------------------------------ exportar

PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
]


EXPORT_KINDS = ("overlay", "original", "matte", "alpha", "perobject")

# Códecs: H.264 para ver; ProRes para editar (Final Cut, Premiere, DaVinci).
H264 = ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]
PRORES_MATTE = ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]
PRORES_ALPHA = ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
                "-alpha_bits", "16"]


class Encoder:
    """Envía fotogramas en crudo a ffmpeg (vídeo) o los guarda como PNG (imagen)."""

    def __init__(self, path: Path, mode: str, codec: list[str]):
        self.path, self.mode = path, mode  # mode: "RGB", "L" o "RGBA"
        self.proc = None
        if S.kind == "video":
            pix = {"RGB": "rgb24", "L": "gray", "RGBA": "rgba"}[mode]
            self.proc = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix,
                 "-s", f"{S.w}x{S.h}", "-r", str(S.fps), "-i", "-", *codec,
                 str(path)],
                stdin=subprocess.PIPE,
            )

    def write(self, arr: np.ndarray) -> None:
        if self.proc:
            self.proc.stdin.write(np.ascontiguousarray(arr).tobytes())
        else:
            Image.fromarray(arr, self.mode).save(self.path)

    def close(self) -> None:
        if self.proc:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed: {self.path.name}")


def export_job(hidden: set[int], kinds: list[str]) -> None:
    kinds = [k for k in EXPORT_KINDS if k in kinds] or ["overlay"]
    palette = np.array(PALETTE, np.float32)
    pal_u8 = np.array(PALETTE, np.uint8)
    visible = sorted(o for o in S.objects if o not in hidden)

    stem = re.sub(r"[^\w\-. ]", "_", S.name) or "sam3"
    folder = EXPORTS / f"{stem}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    folder.mkdir(parents=True)
    vid = S.kind == "video"
    ext_view, ext_edit = (".mp4", ".mov") if vid else (".png", ".png")

    enc: dict[str, Encoder] = {}
    if "overlay" in kinds:
        enc["overlay"] = Encoder(folder / f"{stem}_mascaras{ext_view}", "RGB", H264)
    if "original" in kinds:
        enc["original"] = Encoder(folder / f"{stem}_original{ext_view}", "RGB", H264)
    if "matte" in kinds:
        enc["matte"] = Encoder(folder / f"{stem}_mate_BN{ext_edit}", "L", PRORES_MATTE)
    if "alpha" in kinds:
        enc["alpha"] = Encoder(folder / f"{stem}_color_alfa{ext_edit}", "RGBA",
                               PRORES_ALPHA)
    if "perobject" in kinds:
        for oid in visible:
            label = re.sub(r"[^\w\-]", "_", S.objects[oid]["label"].lstrip("@"))
            enc[f"obj{oid}"] = Encoder(
                folder / f"{stem}_objeto{oid}_{label}{ext_edit}", "L", PRORES_MATTE)

    n = len(S.frames)
    try:
        for i, img in enumerate(S.frames):
            S.progress = {"stage": "exporting", "frame": i, "total": n}
            lab = S.labels[i] if i < len(S.labels) else None
            if lab is None:
                lab = np.zeros((S.h, S.w), np.uint16)
            if hidden:
                lab = np.where(np.isin(lab, [h + 1 for h in hidden]), 0, lab)
            m = lab > 0
            if "overlay" in enc:
                res = img.astype(np.float32)
                res[m] = res[m] * 0.45 + palette[(lab[m].astype(int) - 1) % len(palette)] * 0.55
                enc["overlay"].write(res.astype(np.uint8))
            if "original" in enc:
                enc["original"].write(img)
            if "matte" in enc:
                enc["matte"].write(m.astype(np.uint8) * 255)
            if "alpha" in enc:
                rgba = np.zeros((S.h, S.w, 4), np.uint8)
                rgba[m, :3] = pal_u8[(lab[m].astype(int) - 1) % len(pal_u8)]
                rgba[m, 3] = 255
                enc["alpha"].write(rgba)
            for oid in visible:
                if f"obj{oid}" in enc:
                    enc[f"obj{oid}"].write((lab == oid + 1).astype(np.uint8) * 255)
        S.progress = {"stage": "encoding"}
    finally:
        for e in enc.values():
            e.close()
    S.export_path = str(folder)


# ------------------------------------------------------------------- rutas


async def index(_: Request):
    return FileResponse(ROOT / "static" / "index.html")


async def state(_: Request):
    return JSONResponse(S.summary())


async def upload(request: Request):
    if S.busy:
        return err("busy", "Another task is running", 409)
    name = request.query_params.get("name", "file")
    fps = float(request.query_params.get("fps", "10"))
    max_w = int(request.query_params.get("max_width", "960"))
    max_frames = int(request.query_params.get("max_frames", "300"))
    ext = Path(name).suffix.lower()
    if ext not in VIDEO_EXT | IMAGE_EXT:
        return err("format", f"Unsupported format: {ext}")

    shutil.rmtree(WORK, ignore_errors=True)
    sid = uuid.uuid4().hex[:8]
    d = WORK / sid
    (d / "frames").mkdir(parents=True)
    src = d / f"input{ext}"
    with open(src, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)

    S.__init__()
    S.sid = sid
    S.name = Path(name).stem
    S.frame_dir = d / "frames"
    if ext in IMAGE_EXT:
        img = Image.open(src).convert("RGB")
        S.kind = "image"
        S.frames = [np.asarray(img)]
        img.save(S.frame_dir / "00000.jpg", quality=95)
    else:
        S.kind = "video"
        vf = f"fps={fps},scale='min({max_w},iw)':-2"
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(src), "-vf", vf,
             "-frames:v", str(max_frames), "-q:v", "2", "-start_number", "0",
             str(S.frame_dir / "%05d.jpg")],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return err("ffmpeg", "ffmpeg: " + r.stderr[-500:])
        files = sorted(S.frame_dir.glob("*.jpg"))
        if not files:
            return err("noframes", "No frames were extracted")
        S.frames = [np.asarray(Image.open(p).convert("RGB")) for p in files]
        S.fps = fps
    S.h, S.w = S.frames[0].shape[:2]
    S.reset_results()
    src.unlink()  # el original sigue donde estaba; aquí solo quedan fotogramas
    return JSONResponse(S.summary())


async def frame(request: Request):
    i = int(request.path_params["i"])
    p = S.frame_dir / f"{i:05d}.jpg" if S.frame_dir else None
    if p is None or not p.exists():
        return Response(status_code=404)
    return FileResponse(p, headers={"Cache-Control": "max-age=3600"})


async def mask(request: Request):
    i = int(request.path_params["i"])
    lab = S.labels[i] if i < len(S.labels) else None
    if lab is None:
        lab = np.zeros((S.h, S.w), np.uint16)
    # RGBA: R = byte bajo de id+1, G = byte alto, A = 255 si hay objeto
    rgba = np.zeros((S.h, S.w, 4), np.uint8)
    rgba[..., 0] = lab & 0xFF
    rgba[..., 1] = lab >> 8
    rgba[..., 3] = np.where(lab > 0, 255, 0)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", compress_level=1)
    return Response(buf.getvalue(), media_type="image/png",
                    headers={"Cache-Control": "max-age=3600"})


async def action(request: Request):
    if S.kind is None:
        return err("nofile", "Open an image or a video first")
    data = await request.json()
    kind = data["kind"]
    frame_i = int(data.get("frame", 0))

    if kind == "export":
        return run_job("export", export_job, set(data.get("hidden", [])),
                       list(data.get("kinds", ["overlay"])))

    if kind == "rename":
        o = S.objects.get(data["id"])
        name = str(data.get("name", "")).strip()[:40]
        if o and name:
            o["label"] = name
            S.version += 1
        return JSONResponse({"ok": True})

    if kind == "clear":
        if S.busy:
            return err("busy", "Another task is running", 409)
        S.reset_results()
        S.version = 1
        return JSONResponse({"ok": True})

    if S.kind == "image":
        if kind == "remove":
            S.image_anns.pop(data["id"], None)
            S.objects.pop(data["id"], None)
            rebuild_image()
            return JSONResponse({"ok": True})
        return run_job("segment", image_segment, kind, data)

    # vídeo
    if kind == "text":
        texts = [t.strip() for t in data["text"].split(",") if t.strip()]
        if len(texts) != 1:
            return err("oneclass", "In video osam tracks one class at a time")
        # en vídeo osam no acepta umbrales
        act = {"type": "prompt", "frame": frame_i, "texts": texts}
        return run_job("track", video_track, [act], texts[0], True)

    if kind in ("box", "point"):
        geom = ({"box": data["box"]} if kind == "box"
                else {"points": [data["point"]], "point_labels": [1]})
        label = BOX if kind == "box" else POINT
        if not S.actions:
            act = {"type": "prompt", "frame": frame_i, **geom}
            return run_job("track", video_track, [act], label, True)
        act = {"type": "add", "frame": frame_i, "base": S.base, **geom}
        return run_job("track", video_track, [act], label, False)

    if not S.actions:
        return err("noobjects", "No tracked objects")
    if kind == "remove":
        act = {"type": "remove", "object_id": data["id"], "base": S.base}
        return run_job("remove", video_track, [act], "", False)
    if kind == "refine":
        act = {"type": "refine", "frame": frame_i, "object_id": data["id"],
               "points": [data["point"]], "point_labels": [int(data["label"])],
               "base": S.base}
        return run_job("track", video_track, [act], "", False)
    return err("unknown", f"Unknown action: {kind}")


async def open_exports(_: Request):
    EXPORTS.mkdir(exist_ok=True)
    target = S.export_path if S.export_path and Path(S.export_path).exists() else None
    cmd = ["open", target] if target else ["open", str(EXPORTS)]
    subprocess.run(cmd, check=False)
    return JSONResponse({"ok": True})


async def palette(_: Request):
    return JSONResponse(PALETTE)


app = Starlette(routes=[
    Route("/", index),
    Route("/api/state", state),
    Route("/api/palette", palette),
    Route("/api/upload", upload, methods=["POST"]),
    Route("/api/frame/{sid}/{i:int}.jpg", frame),
    Route("/api/mask/{sid}/{v:int}/{i:int}.png", mask),
    Route("/api/action", action, methods=["POST"]),
    Route("/api/open_exports", open_exports, methods=["POST"]),
])


def main() -> None:
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="coreml", choices=["coreml", "webgpu", "cpu"])
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    ARGS = p.parse_args()

    print(f"Loading SAM3 on {ARGS.device} / Cargando SAM3 en {ARGS.device}...")
    client().load(MODEL, keep_alive="30m")
    url = f"http://127.0.0.1:{ARGS.port}"
    print(f"Ready / Listo: {url}")
    if not ARGS.no_browser:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    try:
        uvicorn.run(app, host="127.0.0.1", port=ARGS.port, log_level="warning")
    finally:
        if _client is not None:
            _client.close()


if __name__ == "__main__":
    main()
