"""Détection de visage par cascade de Haar (Viola-Jones, 2001).

Les descripteurs sont des features de Haar : des rectangles clairs et sombres
évalués sur une image intégrale. Ils sont inspirés des ondelettes de Haar,
mais ce module ne calcule pas une transformée en ondelettes.

Le fichier XML est un classifieur AdaBoost déjà entraîné. Ce n'est pas un
réseau de neurones. Le cours (slide 276) le range dans la colonne
« sans deep learning ». On ne ré-entraîne pas le modèle.

La cascade ne tourne que sur les boîtes en mouvement situées dans une zone.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from pipeline import MovingObject


class FaceDetector:
    def __init__(self) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.classifier = cv2.CascadeClassifier(str(cascade_path))
        if self.classifier.empty():
            raise SystemExit(f"Cascade de Haar introuvable : {cascade_path}")

    def detect(self, gray, objects: list[MovingObject]) -> list[tuple[int, int, int, int]]:
        height, width = gray.shape[:2]
        faces: list[tuple[int, int, int, int]] = []
        candidates = [obj for obj in objects if obj.in_zone][:5]
        for obj in candidates:
            pad_x = int(0.2 * obj.w)
            pad_y = int(0.2 * obj.h)
            x0 = max(0, obj.x - pad_x)
            y0 = max(0, obj.y - pad_y)
            x1 = min(width, obj.x + obj.w + pad_x)
            y1 = min(height, obj.y + obj.h + pad_y)
            if x1 - x0 < 40 or y1 - y0 < 40:
                continue
            crop = gray[y0:y1, x0:x1]
            found = self.classifier.detectMultiScale(
                crop,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )
            for fx, fy, fw, fh in found:
                face = (int(x0 + fx), int(y0 + fy), int(fw), int(fh))
                faces.append(face)
                obj.has_face = True
        return faces
