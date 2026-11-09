"""Self-check logika sudut & RULA. Jalankan: python tests/test_core.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.angle_calculator import hitung_sudut, hitung_semua_sudut
from core.rula_evaluator import evaluasi, skor_bagian


def test_hitung_sudut():
    assert round(hitung_sudut((1, 0), (0, 0), (0, 1))) == 90
    assert round(hitung_sudut((1, 0), (0, 0), (-1, 0))) == 180
    assert round(hitung_sudut((1, 0), (0, 0), (1, 1e-12))) == 0  # cos tepat 1, butuh clip


def test_postur_tegak():
    # koordinat piksel, y ke bawah: telinga di atas bahu, bahu di atas pinggul, lengan menggantung, siku 90°
    t = {"telinga": (100, 50), "bahu": (100, 100), "pinggul": (100, 300), "siku": (100, 200), "pergelangan": (200, 200)}
    s = hitung_semua_sudut(t)
    assert round(s["leher"]) == 0 and round(s["punggung"]) == 0
    assert round(s["lengan_atas"]) == 0 and round(s["lengan_bawah"]) == 90
    h = evaluasi(s)
    assert h["skor_akhir"] == 1 and h["warna"] == "green"
    assert "ergonomis" in h["rekomendasi"][0]
    # lengan lurus ke depan: sudut dalam 180° -> fleksi 0° -> di luar rentang aman
    t["pergelangan"] = (100, 300)
    assert round(hitung_semua_sudut(t)["lengan_bawah"]) == 0


def test_batas_skor():
    sk = skor_bagian({"leher": 10, "punggung": 61, "lengan_atas": 45.1, "lengan_bawah": 101})
    assert sk == {"leher": 1, "punggung": 4, "lengan_atas": 3, "lengan_bawah": 2}


def test_postur_buruk():
    h = evaluasi({"leher": 35, "punggung": 70, "lengan_atas": 100, "lengan_bawah": 130})
    # lengan atas 4, lengan bawah 2 -> A=4; leher 3, punggung 4 -> B=5; tabel C (A=4, B=5) = 5
    assert (h["grup_a"], h["grup_b"], h["skor_akhir"]) == (4, 5, 5)
    assert h["warna"] == "red" and len(h["rekomendasi"]) == 4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
