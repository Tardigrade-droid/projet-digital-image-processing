"""Soustraction de fond et segmentation du mouvement.

Chaîne du cours : niveaux de gris, flou, égalisation si la scène est sombre,
différence absolue, seuillage d'Otsu ou seuil manuel, ouverture 3x3,
contours externes, aire, boîte et centroïde.

Le fond est une moyenne glissante, pas un modèle appris :

    B_t = (1 - a) B_{t-1} + a I_t

Les pixels classés premier plan ne mettent pas le fond à jour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class MovingObject:
    x: int
    y: int
    w: int
    h: int
    area: float
    cx: int
    cy: int
    in_zone: bool = False
    has_face: bool = False


@dataclass
class FrameResult:
    gray: np.ndarray
    background: np.ndarray
    diff: np.ndarray
    mask: np.ndarray
    objects: list[MovingObject] = field(default_factory=list)
    learning: bool = False
    tamper: str | None = None
    equalized: bool = False
    raw_mean: float = 0.0


class MotionPipeline:
    def __init__(
        self,
        alpha: float = 0.02,
        learning_frames: int = 45,
        min_area: int = 0,
        manual_threshold: int = 25,
    ) -> None:
        self.alpha = alpha
        self.learning_total = learning_frames
        self.learning_left = learning_frames
        self.min_area = min_area
        self.manual_threshold = manual_threshold
        self.use_otsu = True
        self.use_median = False
        self.equalize_on = False
        self.background: np.ndarray | None = None
        self.brightness_ref: float | None = None
        self.global_count = 0
        self.kernel = np.ones((3, 3), np.uint8)

    def relearn(self) -> None:
        self.background = None
        self.learning_left = self.learning_total
        self.brightness_ref = None
        self.global_count = 0

    def toggle_otsu(self) -> None:
        self.use_otsu = not self.use_otsu

    def toggle_median(self) -> None:
        self.use_median = not self.use_median
        self.relearn()

    def adjust_threshold(self, delta: int) -> None:
        self.manual_threshold = int(np.clip(self.manual_threshold + delta, 5, 80))

    def process(self, frame: np.ndarray) -> FrameResult:
        gray, raw_mean, equalized = self._preprocess(frame)
        current = gray.astype(np.float32)

        if self.background is None:
            self.background = current.copy()

        background_u8 = np.clip(self.background, 0, 255).astype(np.uint8)
        diff = cv2.absdiff(gray, background_u8)

        if self.learning_left > 0:
            self.background = 0.8 * self.background + 0.2 * current
            self.learning_left -= 1
            if self.learning_left == 0:
                self.brightness_ref = raw_mean
            return FrameResult(
                gray=gray,
                background=np.clip(self.background, 0, 255).astype(np.uint8),
                diff=diff,
                mask=np.zeros_like(gray),
                learning=True,
                equalized=equalized,
                raw_mean=raw_mean,
            )

        mask = self._threshold(diff)
        if int(diff.max()) < 10:
            mask[:] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)

        tamper = self._tamper(raw_mean, diff)
        if tamper is None:
            alpha_map = np.full(gray.shape, self.alpha, dtype=np.float32)
            alpha_map[mask > 0] = 0.0
            self.background = (1.0 - alpha_map) * self.background + alpha_map * current
            if self.brightness_ref is None:
                self.brightness_ref = raw_mean
            else:
                self.brightness_ref = 0.98 * self.brightness_ref + 0.02 * raw_mean

        objects = [] if tamper else self._objects(mask)
        return FrameResult(
            gray=gray,
            background=np.clip(self.background, 0, 255).astype(np.uint8),
            diff=diff,
            mask=mask,
            objects=objects,
            learning=False,
            tamper=tamper,
            equalized=equalized,
            raw_mean=raw_mean,
        )

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, bool]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_mean = float(gray.mean())
        if not self._sudden_dark(raw_mean):
            self._update_equalize(raw_mean)

        if self.use_median:
            gray = cv2.medianBlur(gray, 5)
        else:
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.equalize_on:
            gray = cv2.equalizeHist(gray)
        return gray, raw_mean, self.equalize_on

    def _update_equalize(self, raw_mean: float) -> None:
        if self.equalize_on and raw_mean > 110:
            self.equalize_on = False
            self.relearn()
        elif not self.equalize_on and raw_mean < 70:
            self.equalize_on = True
            self.relearn()

    def _sudden_dark(self, raw_mean: float) -> bool:
        ref = self.brightness_ref
        if ref is None or ref < 25:
            return False
        return raw_mean < 28 and raw_mean < 0.45 * ref

    def _threshold(self, diff: np.ndarray) -> np.ndarray:
        if self.use_otsu:
            _, mask = cv2.threshold(
                diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            _, mask = cv2.threshold(
                diff, self.manual_threshold, 255, cv2.THRESH_BINARY
            )
        return mask

    def _objects(self, mask: np.ndarray) -> list[MovingObject]:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        height, width = mask.shape[:2]
        min_area = self.min_area if self.min_area > 0 else max(600, int(0.0015 * height * width))
        objects: list[MovingObject] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, box_w, box_h = cv2.boundingRect(contour)
            moments = cv2.moments(contour)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx = x + box_w // 2
                cy = y + box_h // 2
            objects.append(MovingObject(x, y, box_w, box_h, area, cx, cy))
        objects.sort(key=lambda obj: obj.area, reverse=True)
        return objects

    def _tamper(self, raw_mean: float, diff: np.ndarray) -> str | None:
        if self._sudden_dark(raw_mean):
            self.global_count = 0
            return "obscurcissement"

        # Avant l'ouverture : une personne ne couvre qu'une partie de l'image,
        # un déplacement de caméra change presque tous les pixels.
        changed = float(np.count_nonzero(diff > 20)) / float(diff.size)
        if changed > 0.60:
            self.global_count += 1
        else:
            self.global_count = 0
        if self.global_count >= 8:
            return "deplacement"
        return None
