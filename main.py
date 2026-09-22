"""Surveillance CCTV : mouvement, zone interdite, visage, sabotage.

Interface utilisateur :
    python app.py
    puis ouvrir http://127.0.0.1:8000

Fenêtres OpenCV :
    python main.py
    python main.py chemin/video.mp4

Touches :
    q quitter    r reapprendre le fond    o Otsu / seuil manuel
    + / - seuil  m median / gaussien      z dessiner une zone
    c effacer les zones
Clic gauche : sommet du polygone. Clic droit : fermer la zone.
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from alerts import AlertLog
from faces import FaceDetector
from pipeline import FrameResult, MotionPipeline, MovingObject
from zones import ZoneEditor

WINDOW = "Surveillance CCTV"
PERSIST_FRAMES = 12
PANEL_W = 640
PANEL_H = 360


class DisplayMap:
    def __init__(self) -> None:
        self.scale = 0.0
        self.ox = 0
        self.oy = 0
        self.frame_w = 1
        self.frame_h = 1

    def map_point(self, mx: int, my: int) -> tuple[int, int] | None:
        if self.scale <= 0 or mx >= PANEL_W or my >= PANEL_H:
            return None
        x = (mx - self.ox) / self.scale
        y = (my - self.oy) / self.scale
        if x < 0 or y < 0 or x >= self.frame_w or y >= self.frame_h:
            return None
        return int(x), int(y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Surveillance CCTV sans deep learning")
    parser.add_argument(
        "source",
        nargs="?",
        default="0",
        help="0 pour la webcam, ou le chemin d'une video",
    )
    return parser.parse_args()


def open_capture(source: str) -> tuple[cv2.VideoCapture, bool]:
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise SystemExit(f"Impossible d'ouvrir la source : {source}")
    return capture, not source.isdigit()


def letterbox(image: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, int, int]:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    resized_w = max(1, int(src_w * scale))
    resized_h = max(1, int(src_h * scale))
    resized = cv2.resize(image, (resized_w, resized_h))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    ox = (width - resized_w) // 2
    oy = (height - resized_h) // 2
    canvas[oy : oy + resized_h, ox : ox + resized_w] = resized
    return canvas, scale, ox, oy


def put_line(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.55,
) -> None:
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
    )


def draw_scene(
    frame: np.ndarray,
    result: FrameResult,
    zones: ZoneEditor,
    faces: list[tuple[int, int, int, int]],
    pipeline: MotionPipeline,
    persist: int,
    latched: bool,
    saw_face: bool,
    help_text: str | None = None,
) -> np.ndarray:
    annotated = zones.draw(frame.copy())
    for obj in result.objects:
        color = (0, 0, 255) if obj.has_face else (0, 165, 255)
        cv2.rectangle(
            annotated,
            (obj.x, obj.y),
            (obj.x + obj.w, obj.y + obj.h),
            color,
            2,
        )
        cv2.circle(annotated, (obj.cx, obj.cy), 3, color, -1)
    for x, y, w, h in faces:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        put_line(annotated, "visage", (x, max(16, y - 6)), (0, 0, 255))

    status, color = _status(result, latched, saw_face)
    bar_top = max(0, annotated.shape[0] - 100)
    cv2.rectangle(annotated, (0, bar_top), (annotated.shape[1], annotated.shape[0]), (0, 0, 0), -1)
    put_line(annotated, status, (8, bar_top + 18), color)
    seuil = "Otsu" if pipeline.use_otsu else str(pipeline.manual_threshold)
    put_line(
        annotated,
        (
            f"Objets: {len(result.objects)}  Seuil: {seuil}  "
            f"Presence: {persist}/{PERSIST_FRAMES}"
        ),
        (8, bar_top + 38),
        (255, 255, 255),
    )
    put_line(
        annotated,
        (
            f"Egalisation: {'oui' if result.equalized else 'non'}  "
            f"Median: {'oui' if pipeline.use_median else 'non'}"
        ),
        (8, bar_top + 56),
        (255, 255, 255),
    )
    if zones.drawing:
        zone_text = f"Dessin zone ({len(zones.draft)} points, clic droit pour fermer)"
    elif zones.polygons:
        zone_text = f"Zones: {len(zones.polygons)}"
    else:
        zone_text = "Zone: image entiere"
    put_line(annotated, zone_text, (8, bar_top + 74), (0, 255, 255))
    put_line(
        annotated,
        help_text or "q quitter   r fond   o otsu   +/- seuil   m median   z zone   c effacer",
        (8, bar_top + 90),
        (180, 180, 180),
        0.42,
    )
    return annotated


def _status(result: FrameResult, latched: bool, saw_face: bool) -> tuple[str, tuple[int, int, int]]:
    if result.learning:
        return "Apprentissage du fond", (0, 255, 255)
    if result.tamper == "obscurcissement":
        return "SABOTAGE : objectif masque", (0, 0, 255)
    if result.tamper == "deplacement":
        return "SABOTAGE : camera deplacee", (0, 0, 255)
    if latched and saw_face:
        return "ALERTE VISAGE", (0, 0, 255)
    if latched:
        return "ALERTE MOUVEMENT", (0, 140, 255)
    return "Surveillance", (0, 220, 0)


def compose(annotated: np.ndarray, result: FrameResult, display: DisplayMap) -> np.ndarray:
    panels = [
        (annotated, "Surveillance"),
        (result.background, "Fond"),
        (result.diff, "Difference"),
        (result.mask, "Masque"),
    ]
    tiles = []
    for index, (image, title) in enumerate(panels):
        tile, scale, ox, oy = letterbox(image, PANEL_W, PANEL_H)
        cv2.rectangle(tile, (0, 0), (PANEL_W, 28), (0, 0, 0), -1)
        put_line(tile, title, (8, 20), (255, 255, 255))
        tiles.append(tile)
        if index == 0:
            display.scale = scale
            display.ox = ox
            display.oy = oy
            display.frame_h, display.frame_w = annotated.shape[:2]
    top = np.hstack(tiles[:2])
    bottom = np.hstack(tiles[2:])
    return np.vstack([top, bottom])


def update_alarm(
    result: FrameResult,
    persist: int,
    saw_face: bool,
    latched: bool,
    tamper_latched: str | None,
) -> tuple[int, bool, bool, str | None, tuple[str, MovingObject | None] | None]:
    if result.tamper:
        event = None if result.tamper == tamper_latched else (result.tamper, None)
        return 0, False, False, result.tamper, event

    intruders = [obj for obj in result.objects if obj.in_zone]
    if result.learning or not intruders:
        return 0, False, False, None, None

    persist = min(persist + 1, PERSIST_FRAMES)
    if any(obj.has_face for obj in intruders):
        saw_face = True
    event = None
    if persist >= PERSIST_FRAMES and not latched:
        kind = "visage" if saw_face else "mouvement"
        event = (kind, max(intruders, key=lambda obj: obj.area))
        latched = True
    return persist, saw_face, latched, None, event


def handle_key(key: int, pipeline: MotionPipeline, zones: ZoneEditor) -> bool:
    if key in (ord("q"), 27):
        return True
    if key == ord("r"):
        pipeline.relearn()
    elif key == ord("o"):
        pipeline.toggle_otsu()
    elif key in (ord("+"), ord("=")):
        pipeline.adjust_threshold(5)
    elif key in (ord("-"), ord("_")):
        pipeline.adjust_threshold(-5)
    elif key == ord("m"):
        pipeline.toggle_median()
    elif key == ord("z"):
        zones.toggle_draw()
    elif key == ord("c"):
        zones.clear()
    return False


def main() -> None:
    args = parse_args()
    capture, is_file = open_capture(args.source)
    pipeline = MotionPipeline()
    zones = ZoneEditor()
    faces = FaceDetector()
    alerts = AlertLog()
    display = DisplayMap()
    persist = 0
    saw_face = False
    latched = False
    tamper_latched: str | None = None

    def on_mouse(event: int, mx: int, my: int, _flags: int, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            point = display.map_point(mx, my)
            if point is not None:
                zones.add_point(*point)
        elif event == cv2.EVENT_RBUTTONDOWN:
            zones.close()

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if is_file:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("Flux video interrompu.")
                break

            result = pipeline.process(frame)
            for obj in result.objects:
                obj.in_zone = zones.contains(obj.cx, obj.cy)
            found_faces = (
                []
                if result.learning or result.tamper
                else faces.detect(result.gray, result.objects)
            )
            persist, saw_face, latched, tamper_latched, event = update_alarm(
                result, persist, saw_face, latched, tamper_latched
            )
            annotated = draw_scene(
                frame, result, zones, found_faces, pipeline, persist, latched, saw_face
            )
            if event is not None:
                kind, obj = event
                alerts.record(annotated, kind, obj)

            cv2.imshow(WINDOW, compose(annotated, result, display))
            if handle_key(cv2.waitKey(20) & 0xFF, pipeline, zones):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) == 0:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
