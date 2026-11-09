"""Pra-pemrosesan kontras dan anotasi kerangka (skeletal overlay)."""
import cv2
from mediapipe.tasks.python.vision import PoseLandmarksConnections

from core.rula_evaluator import WARNA_HEX

WARNA_BGR = {k: tuple(int(v[i:i + 2], 16) for i in (5, 3, 1)) for k, v in WARNA_HEX.items()}
SEGMEN = [("telinga", "bahu"), ("bahu", "siku"), ("siku", "pergelangan"), ("bahu", "pinggul")]
LABEL_SUDUT = {  # kunci sudut: (sendi tempat label, teks label)
    "leher": ("telinga", "Leher"),
    "punggung": ("pinggul", "Punggung"),
    "lengan_atas": ("bahu", "Lengan atas"),
    "lengan_bawah": ("siku", "Siku"),
}


def tingkatkan_kontras(image_bgr):
    """CLAHE pada kanal L (ruang warna LAB) untuk citra dengan pencahayaan kurang."""
    l, a, b = cv2.split(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def gambar_anotasi(image_bgr, landmarks, titik, sudut, warna):
    """Kerangka penuh (abu-abu) + sisi terpilih berwarna sesuai risiko + label derajat sudut."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    tebal = max(2, round(min(h, w) / 200))
    skala = max(0.5, min(h, w) / 700)
    px = [(int(x * w), int(y * h)) for x, y, _, _ in landmarks]

    for c in PoseLandmarksConnections.POSE_LANDMARKS:
        cv2.line(out, px[c.start], px[c.end], (200, 200, 200), max(1, tebal // 2), cv2.LINE_AA)

    warna_bgr = WARNA_BGR[warna]
    ti = {n: (int(x), int(y)) for n, (x, y) in titik.items()}
    for a, b in SEGMEN:
        cv2.line(out, ti[a], ti[b], warna_bgr, tebal * 2, cv2.LINE_AA)
    for p in ti.values():
        cv2.circle(out, p, tebal * 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, p, tebal * 3, warna_bgr, tebal, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for nama, (sendi, label) in LABEL_SUDUT.items():
        teks = f"{label} {sudut[nama]:.0f}"
        (tw, th), _ = cv2.getTextSize(teks, font, skala, tebal)
        r = max(2, th // 5)  # lingkaran kecil sebagai simbol derajat (font Hershey tak punya "°")
        pad = tebal * 2
        lebar = tw + 3 * r + 2 * pad
        # geser ke dalam kanvas agar label tidak terpotong di tepi gambar
        x = max(pad, min(ti[sendi][0] + tebal * 5, w - lebar))
        y = max(th + 2 * pad, ti[sendi][1] + th // 2)
        # kotak gelap di belakang teks agar terbaca di latar terang maupun gelap
        cv2.rectangle(out, (x - pad, y - th - pad), (x + tw + 3 * r + pad, y + pad), (30, 30, 30), -1)
        cv2.putText(out, teks, (x, y), font, skala, (255, 255, 255), tebal, cv2.LINE_AA)
        cv2.circle(out, (x + tw + r + 2, y - th + r), r, (255, 255, 255), max(1, tebal // 2), cv2.LINE_AA)
    return out
