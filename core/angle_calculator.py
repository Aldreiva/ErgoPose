"""Perhitungan sudut biomekanik sendi dengan NumPy."""
import numpy as np


def hitung_sudut(a, b, c):
    """Sudut (derajat) di titik poros b, dibentuk vektor ba dan bc."""
    a, b, c = (np.asarray(p, dtype=float) for p in (a, b, c))
    ba, bc = a - b, c - b
    norma = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norma == 0:
        raise ValueError("Dua landmark berimpit sehingga sudut tidak dapat dihitung.")
    cos_sudut = np.dot(ba, bc) / norma
    return float(np.degrees(np.arccos(np.clip(cos_sudut, -1.0, 1.0))))


def hitung_semua_sudut(t):
    """t: dict titik piksel {telinga, bahu, siku, pergelangan, pinggul}."""
    # Referensi vertikal ke atas dari pinggul (sumbu y citra mengarah ke bawah); panjangnya tak berpengaruh.
    ref_vertikal = (t["pinggul"][0], t["pinggul"][1] - 1.0)
    return {
        "leher": abs(180 - hitung_sudut(t["telinga"], t["bahu"], t["pinggul"])),
        "punggung": hitung_sudut(t["bahu"], t["pinggul"], ref_vertikal),
        "lengan_atas": hitung_sudut(t["siku"], t["bahu"], t["pinggul"]),
        # Fleksi siku = 180° - sudut dalam (lengan lurus = 0°); rentang aman RULA 60-100° mengacu ke fleksi.
        "lengan_bawah": 180 - hitung_sudut(t["bahu"], t["siku"], t["pergelangan"]),
    }
