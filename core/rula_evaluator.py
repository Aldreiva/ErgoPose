"""Pemetaan sudut ke skor RULA (Grup A, Grup B, Skor Akhir), kategori risiko, dan rekomendasi."""

# Batas atas tiap tingkat skor; skor = 1 + jumlah batas yang dilampaui.
DEFAULT_THRESHOLDS = {
    "leher": [10, 20],
    "punggung": [10, 20, 60],
    "lengan_atas": [20, 45, 90],
    "lengan_bawah": [60, 100],  # rentang aman (skor 1); di luar rentang = skor 2
}

# shortcut: pergelangan tangan (skor 1, tanpa puntiran) dan kaki (skor 1, tertopang) diasumsikan
# karena tak terukur dari landmark tubuh; tabel A & B hanya menyimpan kolom itu. Simpan tabel penuh
# bila nanti pergelangan/kaki ikut diukur.
TABEL_A = [  # baris: lengan atas 1-6, kolom: lengan bawah 1-3
    [1, 2, 2], [2, 3, 3], [3, 3, 4], [4, 4, 4], [5, 5, 6], [7, 8, 9],
]
TABEL_B = [  # baris: leher 1-6, kolom: batang tubuh 1-6
    [1, 2, 3, 5, 6, 7], [2, 2, 4, 5, 6, 7], [3, 3, 4, 5, 6, 7],
    [5, 5, 6, 7, 7, 8], [7, 7, 7, 8, 8, 8], [8, 8, 8, 8, 9, 9],
]
TABEL_C = [  # baris: skor A 1-8+, kolom: skor B 1-7+
    [1, 2, 3, 3, 4, 5, 5], [2, 2, 3, 4, 4, 5, 5], [3, 3, 3, 4, 4, 5, 6], [3, 3, 3, 4, 5, 6, 6],
    [4, 4, 4, 5, 6, 7, 7], [4, 4, 5, 6, 6, 7, 7], [5, 5, 6, 6, 7, 7, 7], [5, 5, 6, 7, 7, 7, 7],
]

KATEGORI = [  # (skor maks, label, warna badge)
    (2, "Risiko Rendah / Ergonomis", "green"),
    (4, "Risiko Sedang / Perlu Perbaikan", "orange"),
    (7, "Risiko Tinggi / Intervensi Segera", "red"),
]
# Warna solid tiap kategori, dipakai bersama oleh anotasi gambar, grafik, dan UI
WARNA_HEX = {"green": "#059669", "orange": "#D97706", "red": "#DC2626"}


def skor_bagian(sudut, t=DEFAULT_THRESHOLDS):
    lo, hi = t["lengan_bawah"]
    return {
        "leher": 1 + sum(sudut["leher"] > b for b in t["leher"]),
        "punggung": 1 + sum(sudut["punggung"] > b for b in t["punggung"]),
        "lengan_atas": 1 + sum(sudut["lengan_atas"] > b for b in t["lengan_atas"]),
        "lengan_bawah": 1 if lo <= sudut["lengan_bawah"] <= hi else 2,
    }


def rekomendasi(skor, sudut, t=DEFAULT_THRESHOLDS):
    if all(s == 1 for s in skor.values()):
        return ["Postur duduk sudah ergonomis. Tetap lakukan peregangan ringan setiap 30-60 menit."]
    saran = []
    if sudut["leher"] > t["leher"][0]:
        saran.append("Leher: naikkan layar monitor sejajar mata dan tarik dagu agar kepala tidak menjorok ke depan.")
    if sudut["punggung"] > t["punggung"][0]:
        saran.append("Punggung: tempelkan punggung bawah (lumbar) pada sandaran kursi dan tegakkan torso.")
    if sudut["lengan_atas"] > t["lengan_atas"][0]:
        saran.append("Lengan atas: dekatkan keyboard/mouse ke tubuh dan sesuaikan tinggi sandaran lengan kursi.")
    if skor["lengan_bawah"] > 1:
        saran.append("Lengan bawah: atur tinggi meja/kursi agar siku membentuk sudut sekitar 90°.")
    return saran


def evaluasi(sudut, t=DEFAULT_THRESHOLDS):
    skor = skor_bagian(sudut, t)
    grup_a = TABEL_A[min(skor["lengan_atas"], 6) - 1][min(skor["lengan_bawah"], 3) - 1]
    grup_b = TABEL_B[min(skor["leher"], 6) - 1][min(skor["punggung"], 6) - 1]
    akhir = TABEL_C[min(grup_a, 8) - 1][min(grup_b, 7) - 1]
    label, warna = next((l, w) for maks, l, w in KATEGORI if akhir <= maks)
    return {
        "skor": skor,
        "grup_a": grup_a,
        "grup_b": grup_b,
        "skor_akhir": akhir,
        "kategori": label,
        "warna": warna,
        "rekomendasi": rekomendasi(skor, sudut, t),
    }
