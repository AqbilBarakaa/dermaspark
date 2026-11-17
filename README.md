# ⚡ DERMASPARK
### Platform Evaluasi & Hasil Pengujian Model ConvNeXtTiny-MHSA
**Program Hibah Penelitian BIMA 2026 — Universitas Trunojoyo Madura**  

**DERMASPARK** (*Dermatology × Apache Spark*) adalah platform web interaktif berbasis Streamlit untuk menyajikan evaluasi mendalam dan ringkasan data uji terisolasi (*n* = 3.291 citra) pada model klasifikasi kanker kulit **ConvNeXtTiny-MHSA**, berstandar **AntiSlop** (fokus saintifik, metrik riil, bebas bias *Zero Patient Leakage*).

---

## 🚀 Fitur Utama Dashboard

1. **Ikhtisar Metrik Uji Utama**:
   - Akurasi Agregat: **85,26%** (CI 95%: 84,01% – 86,43%)
   - ROC-AUC: **0,9174**
   - Sensitivitas: **81,10%** | Spesifisitas: **87,66%**
2. **Simulator Ambang Keputusan (*Live Cutoff Simulator*)**:
   - Penggeser ambang (0,00 – 1,00) yang mengkalkulasi ulang Matriks Konfusi (TP, TN, FP, FN) dan metrik klinis secara instan dari data probabilitas riil.
3. **Tabel Evaluasi Resmi per Dataset Uji**:
   - Rincian kinerja pada HAM10000 (90,85%), ISIC 2017 (86,89%), dan ISIC 2019 (82,53%).
4. **Bukti Validasi Grafis & Studi Ablasi**:
   - Visualisasi resmi Kurva ROC, Matriks Konfusi, dan justifikasi praproses *ROI Otsu + Shades of Gray ($p=6$)*.
5. **Penjelajah Sampel Uji (*Test Sample Explorer*)**:
   - Pencarian transparan per ID citra, filter status diagnosa (Ganas/Jinak), dan ekspor data ke format CSV.

---

## 💻 Panduan Menjalankan Secara Lokal

1. Masuk ke direktori aplikasi:
   ```bash
   cd "C:\Users\acer\Penelitian Riset\app_riset"
   ```

2. Buat virtual environment (opsional/disarankan) & pasang dependensi:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```
   Aplikasi akan terbuka otomatis di browser Anda pada alamat `http://localhost:8501`.

---

## ☁️ Panduan Deploy ke Streamlit Community Cloud

Aplikasi ini dirancang sangat ringan (<6 MB) sehingga tidak terkendala batasan ukuran berkas GitHub.

1. Inisialisasi repositori Git baru di folder `app_riset`:
   ```bash
   cd "C:\Users\acer\Penelitian Riset\app_riset"
   git init
   git add .
   git commit -m "feat: initial commit DERMASPARK evaluation platform"
   ```

2. Buat repositori baru di GitHub (misal: `dermaspark`), lalu hubungkan dan push:
   ```bash
   git branch -M main
   git remote add origin https://github.com/<username-anda>/dermaspark.git
   git push -u origin main
   ```

3. Kunjungi [share.streamlit.io](https://share.streamlit.io):
   - Klik **"Create app"** / **"New app"**.
   - Pilih repositori yang baru Anda push.
   - Set **Main file path** ke: `app.py`.
   - Klik **"Deploy!"**.
