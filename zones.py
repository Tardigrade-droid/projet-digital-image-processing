"""Zones interdites dessinées à la souris.

Sans polygone fermé, toute l'image est surveillée. Un centroïde est intrus
s'il tombe dans au moins un polygone (cv2.pointPolygonTest).
"""

from __future__ import annotations

import cv2
import numpy as np


class ZoneEditor:
    def __init__(self) -> None:
        self.polygons: list[np.ndarray] = []
        self.draft: list[tuple[int, int]] = []
        self.drawing = False

    def toggle_draw(self) -> None:
        if not self.drawing:
            self.drawing = True
            self.draft = []
            return
        if len(self.draft) >= 3:
            self._close()
        else:
            self.draft = []
        self.drawing = False

    def close(self) -> None:
        if len(self.draft) >= 3:
            self._close()
        else:
            self.draft = []
        self.drawing = False

    def clear(self) -> None:
        self.polygons.clear()
        self.draft = []
        self.drawing = False

    def replace_polygons(self, polygons: list) -> None:
        cleaned: list[np.ndarray] = []
        for points in polygons:
            try:
                array = np.array(points, dtype=np.int32)
            except (TypeError, ValueError):
                continue
            if array.ndim != 2 or array.shape[1] != 2 or len(array) < 3:
                continue
            cleaned.append(array.reshape(-1, 1, 2))
        self.polygons = cleaned
        self.draft = []
        self.drawing = False

    def add_point(self, x: int, y: int) -> None:
        if not self.drawing:
            return
        point = (int(x), int(y))
        if self.draft:
            last_x, last_y = self.draft[-1]
            if abs(last_x - point[0]) + abs(last_y - point[1]) < 8:
                return
        self.draft.append(point)

    def contains(self, x: int, y: int) -> bool:
        if not self.polygons:
            return True
        point = (float(x), float(y))
        return any(cv2.pointPolygonTest(poly, point, False) >= 0 for poly in self.polygons)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        output = frame
        if self.polygons:
            overlay = output.copy()
            cv2.fillPoly(overlay, self.polygons, (255, 200, 0))
            output = cv2.addWeighted(overlay, 0.25, output, 0.75, 0)
            cv2.polylines(output, self.polygons, True, (255, 220, 0), 2)

        if len(self.draft) >= 2:
            draft = np.array(self.draft, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(output, [draft], False, (0, 255, 255), 2)
        for x, y in self.draft:
            cv2.circle(output, (x, y), 4, (0, 255, 255), -1)
        return output

    def _close(self) -> None:
        polygon = np.array(self.draft, dtype=np.int32).reshape(-1, 1, 2)
        self.polygons.append(polygon)
        self.draft = []
