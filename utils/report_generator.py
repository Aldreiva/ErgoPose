"""Baris hasil evaluasi dan ekspor CSV."""
import pandas as pd


def baris_hasil(nama_berkas, sisi, sudut, hasil, detik=None):
    baris = {"nama_berkas": nama_berkas}
    if detik is not None:
        baris["detik"] = detik
    baris.update({
        "sisi_deteksi": sisi,
        "sudut_leher": round(sudut["leher"], 1),
        "sudut_punggung": round(sudut["punggung"], 1),
        "sudut_lengan_atas": round(sudut["lengan_atas"], 1),
        "sudut_lengan_bawah": round(sudut["lengan_bawah"], 1),
        "skor_grup_a": hasil["grup_a"],
        "skor_grup_b": hasil["grup_b"],
        "skor_akhir_rula": hasil["skor_akhir"],
        "kategori_risiko": hasil["kategori"],
        "rekomendasi": " | ".join(hasil["rekomendasi"]),
    })
    return baris


def ke_csv(baris):
    # utf-8-sig agar karakter seperti "°" terbaca benar saat dibuka di Excel
    return pd.DataFrame(baris).to_csv(index=False).encode("utf-8-sig")
