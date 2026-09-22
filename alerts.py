"""Journal CSV et photos d'alerte."""

from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pipeline import MovingObject


class AlertLog:
    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory or Path(__file__).resolve().parent / "alerts"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.csv_path = self.directory / "journal.csv"
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
                csv.writer(handle).writerow(
                    ["horodatage", "type", "x", "y", "largeur", "hauteur", "aire", "fichier"]
                )

    def record(
        self,
        frame: np.ndarray,
        kind: str,
        obj: MovingObject | None,
    ) -> Path:
        stamp = datetime.now()
        image_path = self.directory / f"{stamp.strftime('%Y%m%d_%H%M%S_%f')}_{kind}.jpg"
        with self._lock:
            cv2.imwrite(str(image_path), frame)
        row = [
            stamp.isoformat(timespec="seconds"),
            kind,
            "" if obj is None else obj.x,
            "" if obj is None else obj.y,
            "" if obj is None else obj.w,
            "" if obj is None else obj.h,
            "" if obj is None else int(obj.area),
            image_path.name,
        ]
        with self._lock:
            with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(row)
        return image_path

    def history(self) -> list[dict[str, str]]:
        if not self.csv_path.exists():
            return []
        items: list[dict[str, str]] = []
        with self._lock, self.csv_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                filename = row.get("fichier") or ""
                if not filename:
                    continue
                items.append(
                    {
                        "time": row.get("horodatage", "").replace("T", " "),
                        "kind": row.get("type", ""),
                        "image": f"/media/{filename}",
                    }
                )
        return items[-20:]
