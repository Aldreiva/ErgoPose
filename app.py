"""ErgoPose RULA Analyzer — antarmuka Streamlit."""
import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from core.angle_calculator import hitung_semua_sudut
from core.pose_detector import PoseDetector, pilih_sisi
from core.rula_evaluator import DEFAULT_THRESHOLDS, KATEGORI, WARNA_HEX, evaluasi
from utils.image_processor import gambar_anotasi, tingkatkan_kontras
from utils.report_generator import baris_hasil, grafik_distribusi, ke_csv

st.set_page_config(page_title="ErgoPose RULA Analyzer", page_icon=":material/accessibility_new:", layout="wide")

# ---------------- Gaya visual ----------------
st.html("""
<style>
:root{
  --ep-primary:#0E7490; --ep-primary-dark:#155E75; --ep-line:#CFE3E9; --ep-muted:#475569;
  --ep-green:#047857; --ep-green-bg:#E3F4EE; --ep-orange:#B45309; --ep-orange-bg:#FBEFDF;
  --ep-red:#B91C1C; --ep-red-bg:#FBE6E6;
}
.ep-hero{padding:28px 32px;border-radius:16px;color:#fff;
  background:linear-gradient(135deg,var(--ep-primary) 0%,var(--ep-primary-dark) 100%)}
.ep-hero .eyebrow{font-size:.8rem;letter-spacing:.08em;text-transform:uppercase;opacity:.9}
.ep-hero h1{color:#fff;font-size:2rem;line-height:1.2;margin:6px 0 8px;padding:0}
.ep-hero p{margin:0;max-width:64ch;opacity:.95;line-height:1.6}
.ep-steps{display:flex;flex-wrap:wrap;gap:10px;margin-top:18px}
.ep-step{display:flex;align-items:center;gap:8px;padding:6px 14px 6px 6px;border-radius:999px;
  background:rgba(255,255,255,.14);font-size:.9rem}
.ep-step b{display:grid;place-items:center;width:24px;height:24px;border-radius:50%;
  background:#fff;color:var(--ep-primary);font-size:.8rem}
.ep-card{background:#fff;border:1px solid var(--ep-line);border-radius:14px;padding:18px 20px;height:100%}
.ep-card h4{margin:0 0 10px;padding:0;font-size:1rem}
.ep-card ul{margin:0;padding-left:1.1rem;line-height:1.7;color:var(--ep-muted)}
.ep-legend{display:flex;align-items:center;gap:12px;margin:8px 0}
.ep-legend .rng{width:56px;font-weight:700;font-variant-numeric:tabular-nums}
.ep-chip{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;
  font-size:.82rem;font-weight:700;white-space:nowrap}
.ep-chip svg{width:14px;height:14px;flex:none}
.ep-green{color:var(--ep-green);background:var(--ep-green-bg);border-color:var(--ep-green)}
.ep-orange{color:var(--ep-orange);background:var(--ep-orange-bg);border-color:var(--ep-orange)}
.ep-red{color:var(--ep-red);background:var(--ep-red-bg);border-color:var(--ep-red)}
.ep-score{display:flex;align-items:center;gap:22px;padding:20px 24px;border-radius:16px;border:1px solid}
.ep-score .num{flex:none;display:grid;place-items:center;width:88px;height:88px;border-radius:50%;
  background:#fff;font-size:2.8rem;font-weight:700;line-height:1;border:3px solid}
.ep-score .cap{font-size:.8rem;text-transform:uppercase;letter-spacing:.06em;opacity:.85}
.ep-score .lbl{font-size:1.3rem;font-weight:700;margin:2px 0 4px}
.ep-score .desc{color:#0F2A33;margin:0;line-height:1.5}
.ep-scale{display:flex;gap:4px;margin-top:12px;max-width:340px}
.ep-scale span{flex:1;height:8px;border-radius:4px;opacity:.22}
.ep-scale span.on{opacity:1;outline:2px solid #0F2A33;outline-offset:1px}
.ep-scale-lbl{display:flex;justify-content:space-between;max-width:340px;font-size:.75rem;color:var(--ep-muted);margin-top:4px}
.ep-joint{background:#fff;border:1px solid var(--ep-line);border-radius:14px;padding:16px 18px;height:100%}
.ep-joint .name{font-size:.9rem;color:var(--ep-muted)}
.ep-joint .val{font-size:2rem;font-weight:700;margin:2px 0 8px;font-variant-numeric:tabular-nums;color:#0F2A33}
.ep-joint .hint{font-size:.8rem;color:var(--ep-muted);margin-top:10px}
@media (max-width:640px){.ep-hero{padding:20px}.ep-hero h1{font-size:1.5rem}.ep-score{flex-direction:column;align-items:flex-start}}
</style>
""")

IKON = {  # ikon SVG (Lucide) agar status tidak hanya dibedakan lewat warna
    "green": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>',
    "orange": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>',
    "red": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><path d="m15 9-6 6M9 9l6 6"/></svg>',
}
STATUS = {1: ("green", "Aman"), 2: ("orange", "Waspada")}
DESKRIPSI = {
    "green": "Postur dapat diterima. Pertahankan, dan tetap lakukan peregangan berkala.",
    "orange": "Perlu evaluasi lebih lanjut dan perubahan posisi duduk dalam waktu dekat.",
    "red": "Postur berbahaya bagi otot dan sendi. Segera lakukan koreksi.",
}


def chip(warna, teks):
    return f'<span class="ep-chip ep-{warna}">{IKON[warna]}{teks}</span>'


def status_bagian(skor):
    return STATUS.get(skor, ("red", "Berisiko"))


@st.cache_resource
def get_detector(conf):
    return PoseDetector(min_detection_confidence=conf)


def analisis(image_bgr, detector, kontras, thresholds):
    """Kembalikan dict hasil, atau string pesan peringatan bila gagal."""
    if kontras:
        image_bgr = tingkatkan_kontras(image_bgr)
    landmarks = detector.detect(image_bgr)
    if landmarks is None:
        return "Tubuh tidak terdeteksi. Pastikan seluruh tubuh bagian atas terlihat dari samping dan tidak tertutup objek."
    h, w = image_bgr.shape[:2]
    sisi, titik, vis_min = pilih_sisi(landmarks, w, h)
    try:
        sudut = hitung_semua_sudut(titik)
    except ValueError as e:
        return str(e)
    hasil = evaluasi(sudut, thresholds)
    return {
        "sisi": sisi,
        "sudut": sudut,
        "hasil": hasil,
        "vis_min": vis_min,
        "anotasi": gambar_anotasi(image_bgr, landmarks, titik, sudut, hasil["warna"]),
    }


def kartu_skor(h):
    w, akhir = h["warna"], h["skor_akhir"]
    warna_skala = [WARNA_HEX[next(c for m, _, c in KATEGORI if i <= m)] for i in range(1, 8)]
    skala = "".join(
        f'<span class="{"on" if i == akhir else ""}" style="background:{c}"></span>'
        for i, c in enumerate(warna_skala, start=1)
    )
    st.html(f"""
    <div class="ep-score ep-{w}" role="status">
      <div class="num ep-{w}" aria-label="Skor akhir RULA {akhir} dari 7">{akhir}</div>
      <div>
        <div class="cap">Skor Akhir RULA · skala 1–7</div>
        <div class="lbl">{chip(w, h["kategori"])}</div>
        <p class="desc">{DESKRIPSI[w]}</p>
        <div class="ep-scale" aria-hidden="true">{skala}</div>
        <div class="ep-scale-lbl" aria-hidden="true"><span>1 · aman</span><span>3–4 · waspada</span><span>7 · bahaya</span></div>
      </div>
    </div>""")


def kartu_sendi(kolom, nama, sudut, skor, hint):
    w, teks = status_bagian(skor)
    kolom.html(f"""
    <div class="ep-joint">
      <div class="name">{nama}</div>
      <div class="val">{sudut:.1f}°</div>
      {chip(w, f"{teks} · skor {skor}")}
      <div class="hint">{hint}</div>
    </div>""")


def tampilkan_dashboard(r, t):
    s, h, sk = r["sudut"], r["hasil"], r["hasil"]["skor"]
    if r["vis_min"] < 0.5:
        st.warning(
            f"Sebagian titik sisi {r['sisi']} kurang jelas (visibility {r['vis_min']:.2f}); hasil mungkin kurang akurat.",
            icon=":material/visibility_off:",
        )

    st.subheader("Hasil Penilaian", anchor=False)
    kartu_skor(h)
    st.write("")
    c = st.columns(4)
    kartu_sendi(c[0], "Leher", s["leher"], sk["leher"], f"Aman ≤ {t['leher'][0]}°")
    kartu_sendi(c[1], "Punggung", s["punggung"], sk["punggung"], f"Aman ≤ {t['punggung'][0]}°")
    kartu_sendi(c[2], "Lengan atas", s["lengan_atas"], sk["lengan_atas"], f"Aman ≤ {t['lengan_atas'][0]}°")
    kartu_sendi(c[3], "Tekukan siku", s["lengan_bawah"], sk["lengan_bawah"],
                f"Aman {t['lengan_bawah'][0]}–{t['lengan_bawah'][1]}°")

    st.write("")
    with st.container(border=True):
        st.markdown("##### :material/tips_and_updates: Rekomendasi perbaikan")
        tampil, ikon = {
            "green": (st.success, ":material/check_circle:"),
            "orange": (st.warning, ":material/warning:"),
            "red": (st.error, ":material/error:"),
        }[h["warna"]]
        for saran in h["rekomendasi"]:
            tampil(saran, icon=ikon)
    with st.container(border=True):
        st.markdown("##### :material/table_chart: Rincian skor per bagian tubuh")
        A, B = "A · Lengan", "B · Leher & badan"
        baris = [
            [A, "Lengan atas", f"{s['lengan_atas']:.1f}°", sk["lengan_atas"], status_bagian(sk["lengan_atas"])[1]],
            [A, "Tekukan siku", f"{s['lengan_bawah']:.1f}°", sk["lengan_bawah"], status_bagian(sk["lengan_bawah"])[1]],
            [A, "Pergelangan tangan", "tidak diukur", 1, "Dianggap aman"],
            [A, "Skor Grup A", "", h["grup_a"], ""],
            [B, "Leher", f"{s['leher']:.1f}°", sk["leher"], status_bagian(sk["leher"])[1]],
            [B, "Punggung", f"{s['punggung']:.1f}°", sk["punggung"], status_bagian(sk["punggung"])[1]],
            [B, "Kaki", "tidak diukur", 1, "Dianggap tertopang"],
            [B, "Skor Grup B", "", h["grup_b"], ""],
        ]
        df = pd.DataFrame(baris, columns=["Kelompok", "Bagian tubuh", "Sudut", "Skor", "Status"])
        # warna teks tua (kontras ≥ 4,5:1); dataframe digambar di canvas sehingga variabel CSS tidak berlaku
        warna_status = {"Aman": "#047857", "Waspada": "#B45309", "Berisiko": "#B91C1C"}
        gaya = (
            df.style
            .map(lambda v: f"color:{warna_status[v]};font-weight:700" if v in warna_status else "",
                 subset=["Status"])
            .apply(lambda row: ["font-weight:700"] * len(row) if row["Bagian tubuh"].startswith("Skor") else [""] * len(row),
                   axis=1)
        )
        st.dataframe(gaya, hide_index=True, width="stretch")
        st.caption(
            "RULA menilai tubuh dalam dua kelompok: **Grup A** (lengan & pergelangan) dan **Grup B** "
            "(leher, badan & kaki). Skor tiap bagian (1 = aman) digabung menjadi Skor Grup A dan B, "
            f"lalu keduanya digabung menjadi Skor Akhir RULA. Sisi tubuh yang dianalisis: **{r['sisi']}**."
        )


def png_bytes(image_bgr):
    return cv2.imencode(".png", image_bgr)[1].tobytes()


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## :material/accessibility_new: ErgoPose")
    st.caption("Analisis postur duduk dengan MediaPipe Pose & metode RULA.")
    st.divider()
    mode = st.radio(
        "Jenis input",
        ["Foto", "Video sesi kerja"],
        captions=["JPG / PNG, satu postur", "MP4 / AVI / MOV, dianalisis 1 frame per detik"],
    )
    st.markdown("**Pengaturan deteksi**")
    conf = st.slider("Minimum detection confidence", 0.1, 0.9, 0.5, 0.05,
                     help="Makin tinggi, makin ketat syarat tubuh dianggap terdeteksi.")
    kontras = st.toggle("Peningkatan kontras otomatis (CLAHE)",
                        help="Aktifkan bila foto gelap atau pencahayaannya kurang merata.")
    with st.expander("Kalibrasi batas sudut (°)", icon=":material/tune:"):
        d = DEFAULT_THRESHOLDS
        thresholds = {
            "leher": [st.number_input("Leher: batas skor 1", 0, 90, d["leher"][0]),
                      st.number_input("Leher: batas skor 2", 0, 90, d["leher"][1])],
            "punggung": [st.number_input("Punggung: batas skor 1", 0, 90, d["punggung"][0]),
                         st.number_input("Punggung: batas skor 2", 0, 90, d["punggung"][1]),
                         st.number_input("Punggung: batas skor 3", 0, 120, d["punggung"][2])],
            "lengan_atas": [st.number_input("Lengan atas: batas skor 1", 0, 90, d["lengan_atas"][0]),
                            st.number_input("Lengan atas: batas skor 2", 0, 120, d["lengan_atas"][1]),
                            st.number_input("Lengan atas: batas skor 3", 0, 180, d["lengan_atas"][2])],
            "lengan_bawah": [st.number_input("Fleksi siku: aman minimum", 0, 180, d["lengan_bawah"][0]),
                             st.number_input("Fleksi siku: aman maksimum", 0, 180, d["lengan_bawah"][1])],
        }

# ---------------- Halaman utama ----------------
st.html("""
<section class="ep-hero">
  <div class="eyebrow">Analisis ergonomi · Rapid Upper Limb Assessment</div>
  <h1>Seberapa aman postur dudukmu?</h1>
  <p>Unggah foto atau video tampak samping. ErgoPose mendeteksi kerangka tubuh, mengukur sudut sendi,
     lalu memberi skor risiko RULA beserta saran perbaikan per bagian tubuh.</p>
  <div class="ep-steps" aria-label="Langkah penggunaan">
    <span class="ep-step"><b>1</b>Unggah foto/video</span>
    <span class="ep-step"><b>2</b>Deteksi kerangka tubuh</span>
    <span class="ep-step"><b>3</b>Skor risiko & saran</span>
  </div>
</section>
""")

is_video = mode == "Video sesi kerja"
berkas = st.file_uploader(
    "Unggah video sesi kerja" if is_video else "Unggah foto postur duduk (tampak samping)",
    type=["mp4", "avi", "mov"] if is_video else ["jpg", "jpeg", "png"],
)
if berkas is None:
    kiri, kanan = st.columns(2, gap="large")
    kiri.html("""
    <div class="ep-card">
      <h4>Tips agar hasil akurat</h4>
      <ul>
        <li>Ambil gambar <b>tepat dari samping</b>, kamera setinggi bahu.</li>
        <li>Pastikan <b>telinga, bahu, siku, pergelangan, dan pinggul</b> terlihat jelas.</li>
        <li>Duduk dalam posisi kerja biasa, misalnya sedang mengetik.</li>
        <li>Gunakan pencahayaan cukup; aktifkan <i>peningkatan kontras</i> bila gelap.</li>
      </ul>
    </div>""")
    kanan.html(f"""
    <div class="ep-card">
      <h4>Arti skor akhir RULA</h4>
      <div class="ep-legend"><span class="rng">1–2</span>{chip("green", "Aman")}<span>Postur dapat diterima</span></div>
      <div class="ep-legend"><span class="rng">3–4</span>{chip("orange", "Waspada")}<span>Perlu perbaikan</span></div>
      <div class="ep-legend"><span class="rng">5–7</span>{chip("red", "Bahaya")}<span>Segera dikoreksi</span></div>
    </div>""")
    st.stop()

try:
    detector = get_detector(conf)
except RuntimeError as e:
    st.error(str(e), icon=":material/error:")
    st.stop()

if not is_video:
    image = cv2.imdecode(np.frombuffer(berkas.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        st.error("Berkas citra tidak dapat dibaca.", icon=":material/broken_image:")
        st.stop()
    with st.spinner("Mendeteksi kerangka tubuh…"):
        r = analisis(image, detector, kontras, thresholds)
    kiri, kanan = st.columns(2, gap="medium")
    with kiri.container(border=True):
        st.markdown("##### :material/image: Citra asli")
        st.image(image, channels="BGR", width="stretch")
    if isinstance(r, str):
        st.warning(r, icon=":material/person_off:")
        st.stop()
    with kanan.container(border=True):
        st.markdown("##### :material/accessibility_new: Hasil deteksi kerangka")
        st.image(r["anotasi"], channels="BGR", width="stretch")
        st.caption("Angka = sudut sendi (derajat). Warna garis = tingkat risiko.")
    tampilkan_dashboard(r, thresholds)
    baris = [baris_hasil(berkas.name, r["sisi"], r["sudut"], r["hasil"])]
    anotasi_terpilih = r["anotasi"]
else:
    # Hasil analisis disimpan per sesi agar menggeser slider detik tidak menganalisis ulang seluruh video.
    kunci = (berkas.file_id, conf, kontras, str(thresholds))
    if st.session_state.get("video_kunci") != kunci:
        suffix = os.path.splitext(berkas.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(berkas.getvalue())
        cap = cv2.VideoCapture(f.name)
        try:
            if not cap.isOpened():
                st.error("Berkas video tidak dapat dibaca.", icon=":material/error:")
                st.stop()
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            langkah = max(1, round(fps))  # 1 frame per detik
            progress = st.progress(0.0, "Mengekstraksi frame…")
            hasil_frame, gagal, i = [], 0, 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % langkah == 0:
                    # perkecil frame besar agar memori server tetap aman; sudut tidak terpengaruh skala
                    skala = 1280 / max(frame.shape[:2])
                    if skala < 1:
                        frame = cv2.resize(frame, None, fx=skala, fy=skala, interpolation=cv2.INTER_AREA)
                    r = analisis(frame, detector, kontras, thresholds)
                    if isinstance(r, str):
                        gagal += 1
                    else:
                        r["detik"] = round(i / fps, 1)
                        r["anotasi"] = cv2.imencode(".jpg", r["anotasi"], [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
                        hasil_frame.append(r)
                    progress.progress(min(i / total, 1.0), f"Menganalisis detik ke-{i / fps:.0f}…")
                i += 1
            progress.empty()
        finally:
            cap.release()
            os.remove(f.name)
        st.session_state.video_kunci = kunci
        st.session_state.video_hasil = (hasil_frame, gagal)
    hasil_frame, gagal = st.session_state.video_hasil

    if gagal:
        st.warning(f"{gagal} frame dilewati karena tubuh tidak terdeteksi.", icon=":material/person_off:")
    if not hasil_frame:
        st.error("Tidak ada frame dengan tubuh terdeteksi.", icon=":material/error:")
        st.stop()

    def tingkat_bahaya(r):
        # Skor akhir dulu; bila seri, skor Grup A+B, lalu total sudut leher+punggung+lengan atas.
        h, s = r["hasil"], r["sudut"]
        return h["skor_akhir"], h["grup_a"] + h["grup_b"], s["leher"] + s["punggung"] + s["lengan_atas"]

    terburuk = max(hasil_frame, key=tingkat_bahaya)
    per_detik = {r["detik"]: r for r in hasil_frame}
    kiri, kanan = st.columns(2, gap="medium")
    with kiri.container(border=True):
        st.markdown("##### :material/movie: Video asli")
        st.video(berkas.getvalue())
    with kanan.container(border=True):
        if len(per_detik) > 1:
            detik = st.select_slider(
                "Pilih detik yang ingin dilihat",
                options=list(per_detik),
                value=terburuk["detik"],
                format_func=lambda d: f"{d:g} dtk" + (" (terburuk)" if d == terburuk["detik"] else ""),
                help="Awalnya menampilkan frame paling berisiko, ditandai (terburuk).",
            )
        else:
            detik = terburuk["detik"]
        dipilih = per_detik[detik]
        label = "Frame paling berisiko" if dipilih is terburuk else "Frame terpilih"
        st.markdown(f"##### :material/accessibility_new: {label} · detik ke-{detik:g}")
        st.image(dipilih["anotasi"], width="stretch")
        st.caption("Angka = sudut sendi (derajat). Warna garis = tingkat risiko.")
    tampilkan_dashboard(dipilih, thresholds)

    baris = [baris_hasil(berkas.name, r["sisi"], r["sudut"], r["hasil"], r["detik"]) for r in hasil_frame]
    df = pd.DataFrame(baris)
    st.subheader("Ringkasan Video", anchor=False)
    m1, m2, m3 = st.columns(3)
    m1.metric("Frame dianalisis", len(df), border=True)
    m2.metric("Rata-rata skor RULA", f"{df['skor_akhir_rula'].mean():.1f}", border=True)
    m3.metric("Frame risiko tinggi", f"{(df['skor_akhir_rula'] >= 5).mean():.0%}", border=True)
    with st.container(border=True):
        st.markdown("##### :material/bar_chart: Distribusi kategori risiko per frame")
        st.pyplot(grafik_distribusi(df), width="stretch")
    with st.expander("Tabel hasil per frame", icon=":material/table_rows:"):
        st.dataframe(df.drop(columns=["nama_berkas", "rekomendasi"]), hide_index=True, width="stretch")
    anotasi_terpilih = cv2.imdecode(np.frombuffer(dipilih["anotasi"], np.uint8), cv2.IMREAD_COLOR)

st.subheader("Ekspor Data", anchor=False)
nama = os.path.splitext(berkas.name)[0]
with st.container(border=True):
    e1, e2 = st.columns(2)
    e1.download_button("Unduh hasil evaluasi (CSV)", ke_csv(baris), f"{nama}_rula.csv", "text/csv",
                       type="primary", icon=":material/download:", width="stretch")
    e2.download_button("Unduh gambar anotasi (PNG)", png_bytes(anotasi_terpilih), f"{nama}_anotasi.png",
                       "image/png", icon=":material/image:", width="stretch")
