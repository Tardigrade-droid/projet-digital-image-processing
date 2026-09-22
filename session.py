"""Session de surveillance partagée par la page web."""

from __future__ import annotations

import base64
import threading
from datetime import datetime

import cv2
import numpy as np

from alerts import AlertLog
from faces import FaceDetector
from main import PERSIST_FRAMES, _status, draw_scene, update_alarm
from pipeline import MotionPipeline
from zones import ZoneEditor

MAX_WIDTH = 640
WEB_HELP = "Cadre orange = mouvement    cadre rouge = visage"


def limit_width(frame: np.ndarray, max_width: int = MAX_WIDTH) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(
        frame,
        (max_width, max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _jpeg(image: np.ndarray, quality: int) -> str:
    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _thumbnail(image: np.ndarray, width: int = 320) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, src_w = image.shape[:2]
    if src_w <= width:
        return image
    scale = width / src_w
    return cv2.resize(
        image,
        (width, max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


class MonitorSession:
    def __init__(self, alerts: AlertLog | None = None, faces: FaceDetector | None = None) -> None:
        self.pipeline = MotionPipeline()
        self.zones = ZoneEditor()
        self.alerts = alerts or AlertLog()
        self.faces = faces or FaceDetector()
        self.persist = 0
        self.saw_face = False
        self.latched = False
        self.tamper_latched: str | None = None
        self.history: list[dict[str, str]] = self.alerts.history()
        self._lock = threading.Lock()

    def apply_control(self, data: dict) -> str | None:
        with self._lock:
            return self._apply_control(data)

    def process_frame(self, frame: np.ndarray) -> dict:
        frame = limit_width(frame)
        with self._lock:
            return self._process_frame(frame)

    def _apply_control(self, data: dict) -> str | None:
        if "otsu" in data:
            self.pipeline.use_otsu = bool(data["otsu"])
        if "threshold" in data:
            self.pipeline.manual_threshold = int(np.clip(int(data["threshold"]), 5, 80))
        if "median" in data and bool(data["median"]) != self.pipeline.use_median:
            self.pipeline.toggle_median()
            self._reset_alarm()
        if "polygons" in data:
            self.zones.replace_polygons(data["polygons"])
        command = data.get("cmd")
        if command == "relearn":
            self.pipeline.relearn()
            self._reset_alarm()
        elif command == "clear":
            self.zones.clear()
        return command if isinstance(command, str) else None

    def _reset_alarm(self) -> None:
        self.persist = 0
        self.saw_face = False
        self.latched = False
        self.tamper_latched = None

    def _process_frame(self, frame: np.ndarray) -> dict:
        result = self.pipeline.process(frame)
        for obj in result.objects:
            obj.in_zone = self.zones.contains(obj.cx, obj.cy)
        found_faces = (
            []
            if result.learning or result.tamper
            else self.faces.detect(result.gray, result.objects)
        )
        self.persist, self.saw_face, self.latched, self.tamper_latched, event = update_alarm(
            result, self.persist, self.saw_face, self.latched, self.tamper_latched
        )
        annotated = draw_scene(
            frame,
            result,
            self.zones,
            found_faces,
            self.pipeline,
            self.persist,
            self.latched,
            self.saw_face,
            WEB_HELP,
        )
        event_payload = None
        if event is not None:
            kind, obj = event
            image_path = self.alerts.record(annotated, kind, obj)
            item = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "kind": kind,
                "image": f"/media/{image_path.name}",
            }
            self.history.append(item)
            self.history = self.history[-20:]
            event_payload = item

        status, _color = _status(result, self.latched, self.saw_face)
        height, width = annotated.shape[:2]
        return {
            "annotated": _jpeg(annotated, 72),
            "background": _jpeg(_thumbnail(result.background), 60),
            "diff": _jpeg(_thumbnail(result.diff), 60),
            "mask": _jpeg(_thumbnail(result.mask), 60),
            "width": width,
            "height": height,
            "status": status,
            "learning": result.learning,
            "objects": len(result.objects),
            "persist": self.persist,
            "persist_max": PERSIST_FRAMES,
            "equalized": result.equalized,
            "median": self.pipeline.use_median,
            "otsu": self.pipeline.use_otsu,
            "threshold": self.pipeline.manual_threshold,
            "zones": len(self.zones.polygons),
            "event": event_payload,
            "history": list(self.history),
        }
