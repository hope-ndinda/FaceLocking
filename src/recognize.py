from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .haar_5pt import align_face_5pt
from .history import create_history_file, log_action
from .actions import detect_move, detect_blink, detect_smile

try:
    import mediapipe as mp
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e

# =========================
# FACE LOCKING CONFIG
# =========================
TARGET_IDENTITY = "Hope"
MAX_MISSED_FRAMES = 30

# -------------------------
# Data classes
# -------------------------

@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray

@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool

@dataclass
class FaceLockState:
    locked: bool = False
    name: Optional[str] = None
    bbox: Optional[Tuple[int,int,int,int]] = None
    kps: Optional[np.ndarray] = None
    missed_frames: int = 0
    history_file: Optional[Path] = None
    prev_nose_x: Optional[float] = None

# -------------------------
# Math helpers
# -------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.reshape(-1), b.reshape(-1)))

def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        return {}
    data = np.load(str(db_path), allow_pickle=True)
    return {k: data[k].astype(np.float32).reshape(-1) for k in data.files}

# -------------------------
# Embedder
# -------------------------

class ArcFaceEmbedderONNX:
    def __init__(self, model_path: str):
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def embed(self, img: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 128.0
        x = img[None, ...]
        emb = self.sess.run([self.out_name], {self.in_name: x})[0].reshape(-1)
        return emb / (np.linalg.norm(emb) + 1e-6)

# -------------------------
# Detector (UNCHANGED)
# -------------------------

class HaarFaceMesh5pt:
    def __init__(self):
        self.face = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True
        )

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face.detectMultiScale(gray, 1.1, 5)
        out = []

        for (x,y,w,h) in faces:
            roi = frame[y:y+h, x:x+w]
            res = self.mesh.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                continue

            lm = res.multi_face_landmarks[0].landmark
            idxs = [33, 263, 1, 61, 291]
            kps = np.array([[lm[i].x*w+x, lm[i].y*h+y] for i in idxs], dtype=np.float32)

            out.append(FaceDet(x, y, x+w, y+h, 1.0, kps))

        return out

# -------------------------
# Matcher
# -------------------------

class FaceDBMatcher:
    def __init__(self, db, thresh=0.34):
        self.db = db
        self.thresh = thresh

    def match(self, emb):
        best = ("", 0)
        for name, ref in self.db.items():
            sim = cosine_similarity(emb, ref)
            if sim > best[1]:
                best = (name, sim)
        dist = 1 - best[1]
        return MatchResult(best[0] if dist < self.thresh else None, dist, best[1], dist < self.thresh)

# -------------------------
# MAIN
# -------------------------

def main():
    det = HaarFaceMesh5pt()
    embedder = ArcFaceEmbedderONNX("models/embedder_arcface.onnx")
    matcher = FaceDBMatcher(load_db_npz(Path("data/db/face_db.npz")))
    lock = FaceLockState()

    cap = cv2.VideoCapture(0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = det.detect(frame)

        for f in faces:
            aligned,_ = align_face_5pt(frame, f.kps, out_size=(112,112))
            emb = embedder.embed(aligned)
            mr = matcher.match(emb)

            # LOCK ACQUIRE
            if not lock.locked and mr.accepted and mr.name == TARGET_IDENTITY:
                lock.locked = True
                lock.name = mr.name
                lock.bbox = (f.x1,f.y1,f.x2,f.y2)
                lock.history_file = create_history_file(lock.name)
                print("[LOCK]", lock.name)

            # LOCKED
            if lock.locked:
                cx = (f.x1 + f.x2) / 2
                lx = (lock.bbox[0] + lock.bbox[2]) / 2

                if abs(cx-lx) < 80:
                    lock.bbox = (f.x1,f.y1,f.x2,f.y2)
                    lock.missed_frames = 0

                    move, lock.prev_nose_x = detect_move(f.kps, lock.prev_nose_x)
                    if move: log_action(lock.history_file, *move)

                    b = detect_blink(f.kps)
                    if b: log_action(lock.history_file, *b)

                    s = detect_smile(f.kps)
                    if s: log_action(lock.history_file, *s)
                else:
                    lock.missed_frames += 1

        if lock.locked:
            x1,y1,x2,y2 = lock.bbox
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,255),3)
            cv2.putText(frame, f"LOCKED: {lock.name}", (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255),2)

        cv2.imshow("Face Locking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
