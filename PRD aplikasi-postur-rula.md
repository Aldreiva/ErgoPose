# **Product Requirements Document (PRD): Aplikasi Analisis Postur Duduk Ergonomis Berbasis MediaPipe Pose dan Metode RULA**

## **1\. Ikhtisar Proyek (Project Overview)**

* **Nama Produk:** ErgoPose RULA Analyzer  
* **Pengembang:** Aldreiva Prasetya (2304111010089) – Teknik Komputer, Universitas Syiah Kuala  
* **Latar Belakang Masalah:** Postur duduk yang buruk dalam durasi lama memicu gangguan otot rangka atau *musculoskeletal disorders* (MSDs) pada pengguna komputer dan pekerja kantoran. Penilaian ergonomi konvensional menggunakan metode observasi manual memerlukan pengamat terlatih sehingga kurang praktis untuk evaluasi mandiri.  
* **Tujuan Sistem:** Membangun aplikasi berbasis *computer vision* (non-*real-time*) yang menerima input citra statis atau rekaman video sesi kerja untuk mengekstraksi titik kunci (*landmark*) tubuh menggunakan MediaPipe Pose, menghitung sudut biomekanik sendi, memetakan tingkat risiko ergonomi berdasarkan standar *Rapid Upper Limb Assessment* (RULA), serta menyajikan rekomendasi perbaikan postur per bagian tubuh.

## **2\. Spesifikasi Teknis & Pustaka (Tech Stack)**

| Komponen | Teknologi / Pustaka | Fungsi Utama |
| :---- | :---- | :---- |
| **Bahasa Pemrograman** | Python ($\geq 3.10$) | Bahasa pemrograman utama pemrosesan citra dan logika RULA. |
| **Framework Antarmuka (UI)** | Streamlit | Membangun antarmuka aplikasi web interaktif untuk unggah berkas, visualisasi hasil, dan unduh laporan. |
| **Deteksi Kerangka Tubuh** | MediaPipe Pose | Mengekstraksi 33 *keypoints/landmarks* anatomis tubuh manusia dari citra/video. |
| **Pemrosesan Citra & Video** | OpenCV (opencv-python) | Membaca berkas citra/video, konversi ruang warna RGB/BGR, serta menggambar anotasi kerangka (*skeletal overlay*). |
| **Komputasi Matematis** | NumPy | Operasi aljabar linier vektor dan perhitungan sudut trigonometri ($\arccos$) antar-titik sendi. |
| **Pengolahan Data & Grafik** | Pandas & Matplotlib | Menyusun tabel hasil evaluasi per *frame*, grafik distribusi postur pada input video, serta ekspor data ke format CSV. |

**Catatan Teknis Dependensi untuk Claude Code:** Pada lingkungan Python terbaru, pustaka mediapipe versi $\geq 0.10.30$ hingga $1.1.1$ memerlukan penanganan kompatibilitas modul mp.solutions.pose atau penggunaan mediapipe.tasks.python.vision.PoseLandmarker agar tidak terjadi galat AttributeError: module 'mediapipe' has no attribute 'solutions'. Sistem wajib menyertakan pengecekan otomatis (*fallback handler*) saat inisialisasi model.

## **3\. Arsitektur Direktori Proyek**

Struktur direktori modular yang harus dibuat oleh Claude Code:

Plaintext  
ergopose-rula-app/  
├── app.py                      \# Halaman utama aplikasi Streamlit (UI & alur interaksi)  
├── requirements.txt            \# Daftar dependensi pustaka Python  
├── PRD.md                      \# Dokumen spesifikasi produk  
├── core/  
│   ├── \_\_init\_\_.py  
│   ├── pose\_detector.py        \# Modul ekstraksi 33 landmark MediaPipe Pose & seleksi sisi tubuh  
│   ├── angle\_calculator.py     \# Modul perhitungan vektor sudut geometri (NumPy)  
│   └── rula\_evaluator.py       \# Modul pemetaan tabel skor Grup A, Grup B, Skor Akhir RULA, & rekomendasi  
└── utils/  
    ├── \_\_init\_\_.py  
    ├── image\_processor.py      \# Fungsi anotasi garis kerangka (skeletal overlay) & pra-pemrosesan kontras  
    └── report\_generator.py     \# Fungsi pembuatan ringkasan statistik & ekspor CSV

## **4\. Kebutuhan Fungsional (Functional Requirements)**

### **FR-01: Modul Akuisisi Input Media**

> 1. Sistem menyediakan komponen pengunggah berkas (*file uploader*) yang mendukung:  
   * **Mode Citra Statis:** Format .jpg, .jpeg, .png.  
   * **Mode Video Rekaman Sesi Kerja:** Format .mp4, .avi, .mov (diekstraksi secara berkala, misalnya 1 *frame* per detik untuk efisiensi komputasi).  
> 2. Sistem menyediakan opsi pra-pemrosesan peningkatan kontras citra (CLAHE / *brightness-contrast adjustment*) apabila citra input memiliki pencahayaan kurang optimal.

### **FR-02: Modul Deteksi Landmark & Seleksi Sisi Dominan**

> 1. Sistem mengekstraksi koordinat ternormalisasi $(x,y,z)$ dan nilai *visibility* dari titik kunci tubuh bagian atas: Telinga (*Ear*), Bahu (*Shoulder*), Siku (*Elbow*), Pergelangan Tangan (*Wrist*), dan Pinggul (*Hip*).  
> 2. Sistem secara otomatis membandingkan nilai *visibility* antara sisi kiri dan sisi kanan tubuh, lalu memilih sisi yang paling jelas menghadap kamera sebagai basis perhitungan sudut.  
> 3. Apabila *landmark* tidak terdeteksi akibat oklusi berat, sistem menampilkan pesan peringatan yang informatif tanpa menghentikan aplikasi (*graceful error handling*).

### **FR-03: Modul Perhitungan Sudut Biomekanik**

> 1. Perhitungan sudut menggunakan operasi *dot product* antara dua vektor dengan titik tengah sebagai poros sendi:  
>    \$\$\\theta \= \\arccos\\left(\\frac{\\vec{ba} \\cdot \\vec{bc}}{\\\|\\vec{ba}\\\| \\\|\\vec{bc}\\\|}\\right)\$\$  
> 2. Fungsi wajib menyertakan pembatasan nilai np.clip(cos\_sudut, \-1.0, 1.0) sebelum konversi np.degrees(np.arccos(...)) untuk menghindari galat numerik.  
> 3. Parameter sudut yang dihitung meliputi:  
   * **Sudut Fleksi Leher (*Neck Flexion*):** Deviasi sudut garis telinga–bahu terhadap sumbu batang tubuh (bahu–pinggul), dihitung dengan rumus \$\\vert{}180^\\circ \- \\text{hitung\\\_sudut}(\\text{telinga}, \\text{bahu}, \\text{pinggul})\\vert{}\$.  
   * **Sudut Fleksi Batang Tubuh (*Trunk Flexion*):** Deviasi sudut garis bahu–pinggul terhadap vektor referensi vertikal gravitasi (ref\_vertikal \= \[pinggul\_x, pinggul\_y \- 0.3\]).  
   * **Sudut Lengan Atas (*Upper Arm*):** Sudut antara garis siku–bahu terhadap garis bahu–pinggul.  
   * **Sudut Lengan Bawah (*Lower Arm*):** Sudut fleksi siku yang dibentuk oleh titik bahu, siku, dan pergelangan tangan.

### **FR-04: Modul Penilaian Standar RULA & Klasifikasi Risiko**

> 1. Sistem memetakan derajat sudut ke dalam skor bagian tubuh sesuai parameter RULA:

| Bagian Tubuh | Grup RULA | Rentang Sudut Aman (Skor 1\) | Rentang Sudut Waspada (Skor 2\) | Rentang Sudut Berisiko (Skor 3+) |
| :---- | :---- | :---- | :---- | :---- |
| **Fleksi Leher** | Grup B | ${0}^{\circ }-1{0}^{\circ }$ | $>1{0}^{\circ }-2{0}^{\circ }$ | $>2{0}^{\circ }$ |
| **Batang Tubuh (*Trunk*)** | Grup B | ${0}^{\circ }-1{0}^{\circ }$ (Tegak) | $>1{0}^{\circ }-2{0}^{\circ }$ | $>2{0}^{\circ }-6{0}^{\circ }$ (Skor 3), $>6{0}^{\circ }$ (Skor 4\) |
| **Lengan Atas** | Grup A | ${0}^{\circ }-2{0}^{\circ }$ | $>2{0}^{\circ }-4{5}^{\circ }$ | $>4{5}^{\circ }-9{0}^{\circ }$ (Skor 3), $>9{0}^{\circ }$ (Skor 4\) |
| **Lengan Bawah (Siku)** | Grup A | $6{0}^{\circ }-10{0}^{\circ }$ | $<6{0}^{\circ }$ atau $>10{0}^{\circ }$ | — |

> 2. Sistem menghitung kombinasi skor Grup A (Lengan Atas, Lengan Bawah, Pergelangan Tangan) dan Grup B (Leher, Batang Tubuh, Kaki diasumsikan tertopang seimbang \= Skor 1\) untuk menghasilkan **Skor Akhir RULA (Rentang 1–7)**.  
> 3. Sistem mengklasifikasikan tingkat risiko postur menjadi tiga kategori utama:  
   * **Risiko Rendah / Ergonomis (Skor 1–2):** Indikator Hijau – Postur dapat diterima/aman.  
   * **Risiko Sedang / Perlu Perbaikan (Skor 3–4):** Indikator Kuning/Oranye – Diperlukan evaluasi lebih lanjut dan perubahan posisi duduk.  
   * **Risiko Tinggi / Intervensi Segera (Skor 5–7):** Indikator Merah – Postur berbahaya dan wajib segera dikoreksi.

### **FR-05: Modul Rekomendasi Koreksi Postur Spesifik (*Diagnostic Feedback*)**

Sistem wajib menghasilkan saran koreksi spesifik berdasarkan sendi yang melampaui ambang batas aman:

* **Jika Semua Sendi Skor 1:** Menampilkan pesan konfirmasi bahwa postur duduk telah ergonomis beserta anjuran peregangan berkala.  
* **Jika Sudut Leher $>1{0}^{\circ }$:** Menampilkan rekomendasi untuk menaikkan posisi layar monitor sejajar mata dan menarik dagu agar kepala tidak menjorok ke depan.  
* **Jika Sudut Punggung $>1{0}^{\circ }$:** Menampilkan rekomendasi untuk menempelkan punggung bawah (*lumbar*) pada sandaran kursi dan menegakkan torso.  
* **Jika Sudut Lengan Atas $>2{0}^{\circ }$:** Menampilkan rekomendasi untuk mendekatkan posisi *keyboard/mouse* ke arah tubuh dan menyesuaikan tinggi sandaran lengan kursi.

### **FR-06: Visualisasi & Ekspor Laporan**

> 1. **Visualisasi Citra:** Menampilkan perbandingan berdampingan (*side-by-side*) antara citra asli dan citra hasil anotasi kerangka (*skeletal overlay*) beserta label derajat sudut langsung pada kanvas gambar.  
> 2. **Kartu Metrik (*Metric Cards*):** Menampilkan nilai derajat tiap sendi, Skor Grup A, Skor Grup B, Skor Akhir RULA, dan lencana (*badge*) kategori risiko.  
> 3. **Fitur Unduh Laporan:** Menyediakan tombol unduh hasil analisis dalam format berkas .csv (berisi nama berkas, sisi deteksi, sudut leher, sudut punggung, sudut lengan atas, sudut lengan bawah, skor Grup A, skor Grup B, skor akhir RULA, kategori risiko, dan teks rekomendasi) serta tombol unduh citra hasil anotasi.

## **5\. Rancangan Tata Letak Antarmuka (UI Layout Streamlit)**

* **Sidebar (Panel Kiri):**  
  * Judul dan informasi singkat aplikasi.  
  * Pemilih mode input: Analisis Citra (Foto) atau Analisis Video Sesi Kerja.  
  * Pengaturan parameter opsional: *Slider* Minimum Detection Confidence (default: 0.5), *Checkbox* Peningkatan Kontras Otomatis, dan kalibrasi batas sudut (*threshold*).  
* **Halaman Utama (Panel Tengah):**  
  * **Bagian 1 (Area Unggah):** Komponen *drag-and-drop file uploader*.  
  * **Bagian 2 (Visualisasi Dual-Column):** Kolom kiri menampilkan citra/video input asli; kolom kanan menampilkan visualisasi kerangka MediaPipe Pose dengan kode warna sesuai tingkat risiko.  
  * **Bagian 3 (Dashboard Skor RULA):** 4 kartu metrik utama (Sudut Leher, Sudut Punggung, Sudut Lengan Atas, dan Skor Akhir RULA \+ Status Risiko).  
  * **Bagian 4 (Tabel Rincian & Rekomendasi Ergonomi):** Tabel rincian skor per komponen sendi beserta kotak peringatan/rekomendasi tindak lanjut yang harus dilakukan pengguna.  
  * **Bagian 5 (Ekspor Data):** Tombol Unduh Hasil Evaluasi (CSV) dan Unduh Gambar Anotasi (PNG).

## **6\. Kriteria Penerimaan Pengujian (Acceptance Criteria)**

> 1. Aplikasi dapat dijalankan secara lokal melalui perintah streamlit run app.py tanpa galat impor modul mediapipe.  
> 2. Ketika diunggah citra postur duduk tampak samping, aplikasi mampu mengekstraksi titik koordinat, menggambar garis kerangka, dan menampilkan derajat sudut dalam waktu kurang dari 3 detik per citra.  
> 3. Hasil perhitungan sudut pada aplikasi konsisten dengan rumus fungsi hitung\_sudut yang telah diverifikasi pada Google Colab.  
> 4. Aplikasi menampilkan rekomendasi koreksi yang sesuai dengan bagian sendi yang melebihi batas sudut aman.

Dokumen PRD di atas dapat disimpan langsung sebagai berkas PRD.md di dalam folder proyek lokal agar dapat dibaca otomatis oleh Claude Code. Apakah diperlukan teks *prompt* perintah awal yang siap ditempelkan langsung ke terminal Claude Code untuk mengeksekusi pembuatan aplikasi ini?