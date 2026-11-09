"""Ringkasan statistik, grafik distribusi, dan ekspor CSV."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from core.rula_evaluator import KATEGORI, WARNA_HEX


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


def grafik_distribusi(df):
    """Bar chart jumlah frame per kategori risiko (input video)."""
    label = [l for _, l, _ in KATEGORI]
    jumlah = df["kategori_risiko"].value_counts().reindex(label, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 2.4))
    bars = ax.barh(label, jumlah.values, color=[WARNA_HEX[w] for _, _, w in KATEGORI], height=0.55)
    ax.bar_label(bars, padding=4, color="#0F2A33")  # angka langsung di batang, tanpa perlu membaca sumbu
    ax.set_xlabel("Jumlah frame", color="#475569")
    ax.invert_yaxis()
    ax.tick_params(colors="#0F2A33", length=0)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#CFE3E9")
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    fig.tight_layout()
    return fig
