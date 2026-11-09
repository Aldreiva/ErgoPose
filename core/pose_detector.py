"""Ekstraksi 33 landmark MediaPipe Pose, seleksi sisi tubuh dominan, dan validasi pose."""
import math
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_full.task"

# Indeks landmark MediaPipe Pose (sama untuk API solutions maupun tasks)
SIDES = {
    "kiri": {"telinga": 7, "bahu": 11, "siku": 13, "pergelangan": 15, "pinggul": 23},
    "kanan": {"telinga": 8, "bahu": 12, "siku": 14, "pergelangan": 16, "pinggul": 24},
}
LUTUT = {"kiri": 25, "kanan": 26}

# Diukur dari foto duduk tampak samping: rasio bahu 0,04-0,07 dan sudut paha 4-14°.
# Tampak depan rasio bahu ~0,7-0,9; berdiri sudut paha ~80-90°. Ambang dipilih jauh dari keduanya.
BATAS_RASIO_BAHU = 0.45
BATAS_SUDUT_PAHA = 60


def _ensure_model():
    if MODEL_PATH.exists():
        return
    MODEL_PATH.parent.mkdir(exist_ok=True)
    tmp = MODEL_PATH.with_suffix(".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
    except OSError as e:
        raise RuntimeError(
            f"Gagal mengunduh model MediaPipe ({e}). Unduh manual dari {MODEL_URL} "
            f"lalu simpan sebagai {MODEL_PATH}."
        ) from e
    tmp.replace(MODEL_PATH)


class PoseDetector:
    """Fallback handler: pakai mp.solutions.pose bila ada, kalau tidak pakai PoseLandmarker (tasks)."""

    def __init__(self, min_detection_confidence=0.5):
        if hasattr(mp, "solutions"):  # mediapipe < 0.10.30
            self.backend = "solutions"
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=True, min_detection_confidence=min_detection_confidence
            )
        else:  # mediapipe >= 0.10.30: mp.solutions sudah dihapus
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions

            _ensure_model()
            self.backend = "tasks"
            self._pose = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                    min_pose_detection_confidence=min_detection_confidence,
                )
            )

    def detect(self, image_bgr):
        """Kembalikan list 33 tuple (x, y, z, visibility) ternormalisasi, atau None bila tak terdeteksi."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.backend == "solutions":
            res = self._pose.process(rgb)
            lms = res.pose_landmarks.landmark if res.pose_landmarks else None
        else:
            res = self._pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            lms = res.pose_landmarks[0] if res.pose_landmarks else None
        if not lms:
            return None
        return [(p.x, p.y, p.z, p.visibility or 0.0) for p in lms]


def pilih_sisi(landmarks, width, height):
    """Pilih sisi tubuh dengan total visibility terbesar.

    Kembalikan (nama_sisi, titik_px, visibility_min) dengan titik dalam koordinat piksel,
    agar sudut tidak terdistorsi oleh rasio aspek citra.
    """
    skor = {s: sum(landmarks[i][3] for i in idx.values()) for s, idx in SIDES.items()}
    sisi = max(skor, key=skor.get)
    idx = SIDES[sisi]
    titik = {n: (landmarks[i][0] * width, landmarks[i][1] * height) for n, i in idx.items()}
    vis_min = min(landmarks[i][3] for i in idx.values())
    return sisi, titik, vis_min


def validasi_pose(landmarks, width, height, sisi):
    """Kembalikan pesan penolakan bila foto bukan postur duduk tampak samping, atau None bila layak dinilai."""
    p = [(x * width, y * height) for x, y, _, _ in landmarks]
    tengah = lambda a, b: ((p[a][0] + p[b][0]) / 2, (p[a][1] + p[b][1]) / 2)
    badan = math.dist(tengah(11, 12), tengah(23, 24))
    if badan > 0 and abs(p[11][0] - p[12][0]) / badan > BATAS_RASIO_BAHU:
        return ("Foto terlihat dari depan atau belakang. Gunakan foto tampak samping agar sudut leher "
                "dan punggung dapat diukur.")

    lutut, pinggul = LUTUT[sisi], SIDES[sisi]["pinggul"]
    if landmarks[lutut][3] >= 0.5:  # lutut tertutup meja/tak terlihat: pengecekan duduk dilewati
        dx, dy = abs(p[lutut][0] - p[pinggul][0]), abs(p[lutut][1] - p[pinggul][1])
        if math.degrees(math.atan2(dy, dx)) > BATAS_SUDUT_PAHA:
            return ("Postur tidak terdeteksi sebagai posisi duduk (paha terlihat tegak). Metode ini "
                    "hanya untuk menilai postur duduk; gunakan foto saat sedang duduk.")
    return None
