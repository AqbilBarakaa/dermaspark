import os
import sys
import time
import json
import math
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ─── KONFIGURASI HALAMAN ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="DERMASPARK - Pengujian Citra & Evaluasi Model",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STYLING KHUSUS (ANTISLOP: CLEAN ACADEMIC & CLINICAL) ─────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.92rem;
        color: #475569;
        margin-bottom: 1.1rem;
        line-height: 1.4;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.22rem 0.6rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 600;
        background-color: #F1F5F9;
        color: #1E293B;
        border: 1px solid #E2E8F0;
    }
    .badge-pill.highlight {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border-color: #BFDBFE;
    }
    .metric-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.0rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.1;
    }
    .metric-lbl {
        font-size: 0.80rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 0.3rem;
        letter-spacing: 0.04em;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #059669;
        font-weight: 500;
        margin-top: 0.3rem;
    }
    .status-badge {
        padding: 0.75rem 1.25rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.3rem;
        text-align: center;
        letter-spacing: 0.02em;
        margin: 0.8rem 0;
    }
    .status-ganas {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 2px solid #F87171;
    }
    .status-jinak {
        background-color: #F0FDF4;
        color: #166534;
        border: 2px solid #4ADE80;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1E293B;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0.4rem;
        margin-top: 1.4rem;
        margin-bottom: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── PATH DIREKTORI ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
MODEL_PATH = BASE_DIR / "model" / "utama_256_swa_half.pth"
SAMPLES_DIR = BASE_DIR / "samples"

IMAGE_SIZE = 256
KELAS = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
KELAS_NAMA = {
    "AKIEC": "Actinic Keratoses (Pra-Kanker / Ganas)",
    "BCC": "Basal Cell Carcinoma (Kanker / Ganas)",
    "BKL": "Benign Keratosis (Lesi Jinak)",
    "DF": "Dermatofibroma (Lesi Jinak)",
    "MEL": "Melanoma (Kanker Ganas Agresif)",
    "NV": "Melanocytic Nevi / Tahi Lalat (Jinak)",
    "SCC": "Squamous Cell Carcinoma (Kanker / Ganas)",
    "VASC": "Vascular Lesions (Lesi Vaskular Jinak)"
}
MALIGNANT_IDX = [0, 1, 4, 6]  # AKIEC, BCC, MEL, SCC
BENIGN_IDX = [2, 3, 5, 7]     # BKL, DF, NV, VASC
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ─── ARSITEKTUR MODEL ─────────────────────────────────────────────────────────
class GlobalMHSA(nn.Module):
    def __init__(self, dim=768, heads=8, mlp_ratio=2.0, drop=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )
    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))

class ConvNeXtTinyMHSA(nn.Module):
    def __init__(self, n_kelas=8):
        super().__init__()
        base = models.convnext_tiny(weights=None)
        self.features = base.features
        self.attention = GlobalMHSA()
        self.norm = nn.LayerNorm(768, eps=1e-6)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_kelas)
        )
    def forward(self, x):
        f = self.features(x)
        tok = f.flatten(2).transpose(1, 2)
        tok = self.attention(tok)
        return self.classifier(self.norm(tok.mean(1) + f.mean(dim=(2, 3))))

# ─── FUNGSI PRA-PEMROSESAN CITRA ──────────────────────────────────────────────
DET, MARGIN, FRAC_MIN, FRAC_MAX = 512, 0.15, 0.03, 0.90

def potong_roi(im_bgr):
    h, w = im_bgr.shape[:2]
    det = cv2.resize(im_bgr, (DET, DET), interpolation=cv2.INTER_AREA)
    abu = cv2.GaussianBlur(cv2.cvtColor(det, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    _, th = cv2.threshold(abu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    lingkar = np.zeros((DET, DET), np.uint8)
    cv2.circle(lingkar, (DET // 2, DET // 2), int(DET * 0.48), 255, -1)
    cnts, _ = cv2.findContours(cv2.bitwise_and(th, lingkar), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return cv2.resize(im_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA), None
    c = max(cnts, key=cv2.contourArea)
    frac = cv2.contourArea(c) / (DET * DET)
    if not (FRAC_MIN <= frac <= FRAC_MAX):
        return cv2.resize(im_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA), None
    x, y, bw, bh = cv2.boundingRect(c)
    sisi = int(max(bw, bh) * (1 + 2 * MARGIN))
    cx, cy = x + bw // 2, y + bh // 2
    x0, y0 = max(0, cx - sisi // 2), max(0, cy - sisi // 2)
    x1, y1 = min(DET, x0 + sisi), min(DET, y0 + sisi)
    sx, sy = w / DET, h / DET
    crop = im_bgr[int(y0 * sy):int(y1 * sy), int(x0 * sx):int(x1 * sx)]
    if crop.size == 0:
        return cv2.resize(im_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA), None
    return cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA), (x0, y0, x1, y1)

def shades_of_gray(x_tensor, p=6):
    e = x_tensor.clamp_min(1e-6).pow(p).mean(dim=(2, 3), keepdim=True).pow(1.0 / p)
    e = e / e.norm(dim=1, keepdim=True)
    return (x_tensor / (e * math.sqrt(3))).clamp(0, 1)

# ─── PEMUATAN MODEL & DATA ────────────────────────────────────────────────────
@st.cache_resource
def muat_model():
    if not MODEL_PATH.exists():
        st.error(f"Berkas bobot model tidak ditemukan di: {MODEL_PATH}")
        return None
    model = ConvNeXtTinyMHSA(n_kelas=8)
    sd_half = torch.load(MODEL_PATH, map_location="cpu")
    sd_float = {k: v.float() if v.is_floating_point() else v for k, v in sd_half.items()}
    model.load_state_dict(sd_float)
    model.eval()
    return model

@st.cache_data
def muat_data_uji():
    df_pred = pd.read_csv(DATA_DIR / "data_uji_prediksi.csv")
    df_eval = pd.read_csv(DATA_DIR / "hasil_evaluasi_uji.csv")
    df_abl = pd.read_csv(DATA_DIR / "ablasi_preprocessing.csv")
    ringkasan = json.load(open(DATA_DIR / "ringkasan_model.json"))
    return df_pred, df_eval, df_abl, ringkasan

try:
    df_pred, df_eval, df_abl, ringkasan = muat_data_uji()
except Exception as e:
    st.error(f"Gagal memuat metadata evaluasi: {e}")
    st.stop()

# ─── HEADER UTAMA ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">⚡ DERMASPARK: Platform Pengujian Citra & Evaluasi Riset</div>', unsafe_allow_html=True)
st.markdown("""
<div class="sub-title">
    <strong>Asistensi Hibah Penelitian BIMA 2026</strong> &bull; Universitas Trunojoyo Madura<br>
    <em>Peneliti:</em> Aqbil Baraka (NIM 230411100077) &nbsp;|&nbsp; <em>Pembimbing:</em> Fifin Ayu Mufarroha, S.Kom., M.Kom.<br>
    <em>Fokus:</em> Klasifikasi Kanker Kulit ConvNeXtTiny-MHSA & Granularitas Partisi Memori Apache Spark
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div>
    <span class="badge-pill highlight">⚡ DERMASPARK v1.0</span>
    <span class="badge-pill highlight">Inference Real-Time</span>
    <span class="badge-pill highlight">Zero Patient Leakage</span>
    <span class="badge-pill">Backbone: ConvNeXtTiny-MHSA (33,08M Param)</span>
    <span class="badge-pill">Input: 256x256 px</span>
    <span class="badge-pill">Praproses: ROI Otsu + Shades of Gray</span>
    <span class="badge-pill">Ambang Optimal: 0,42</span>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR: PARAMETER PENGUJIAN ─────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Konfigurasi Pengujian")

ambang_keganasan = st.sidebar.slider(
    "Ambang Batas Keganasan (Cutoff):",
    min_value=0.00,
    max_value=1.00,
    value=float(ringkasan.get("thr_global", 0.42)),
    step=0.01,
    help="Probabilitas >= ambang dikategorikan sebagai Ganas (Malignant)."
)

opsi_tta = st.sidebar.checkbox(
    "Aktifkan TTA 4-Arah (Test-Time Augmentation)",
    value=False,
    help="Menjalankan augmentasi rotasi dan flip untuk meningkatkan ketajaman prediksi (latensi ~12 ms vs ~3 ms)."
)

st.sidebar.caption(
    f"📌 **Ambang Optimal Riset**: **`{ringkasan.get('thr_global', 0.42)}`**\n\n"
    f"Ambang per repositori data:\n"
    f"- HAM10000: `{ringkasan['thr_dataset']['HAM10000']}`\n"
    f"- ISIC 2017: `{ringkasan['thr_dataset']['ISIC 2017']}`\n"
    f"- ISIC 2019: `{ringkasan['thr_dataset']['ISIC 2019']}`"
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Integritas Metodologi**\n\n"
    "Model dilatih dengan normalisasi Shades of Gray (p=6) dan segmentasi ROI Otsu, "
    "mengeliminasi bias saturasi warna serta artefak batas vignetting dermoskopi."
)

# ─── TABS UTAMA ───────────────────────────────────────────────────────────────
tab_uji, tab_dashboard = st.tabs([
    "🔬 Uji Sampel & Diagnostik Citra (Live Test)",
    "📊 Tolok Ukur 3.291 Data Uji & Validasi Riset"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: UJI SAMPEL & DIAGNOSTIK CITRA
# ══════════════════════════════════════════════════════════════════════════════
with tab_uji:
    st.markdown('<div class="section-header">1. Masukan Citra Dermoskopi</div>', unsafe_allow_html=True)

    metode_input = st.radio(
        "Pilih Metode Masukan:",
        ["Pilih dari Sampel Uji Resmi Riset", "Unggah Citra Dermoskopi Baru (JPG / PNG)"],
        horizontal=True
    )

    citra_bgr = None
    label_sampel = ""

    if metode_input == "Pilih dari Sampel Uji Resmi Riset":
        pilihan_sampel = {
            "Melanoma (Kanker Ganas Agresif) - ISIC_0000054": "Melanoma_ISIC_0000054.jpg",
            "Basal Cell Carcinoma (Kanker Ganas) - ISIC_0024436": "BCC_ISIC_0024436.jpg",
            "Actinic Keratoses (Lesi Pra-Kanker) - ISIC_0024329": "AKIEC_ISIC_0024329.jpg",
            "Melanocytic Nevi / Tahi Lalat (Jinak) - ISIC_0000009": "Nevus_ISIC_0000009.jpg",
            "Benign Keratosis (Lesi Jinak) - ISIC_0012173": "BKL_ISIC_0012173.jpg"
        }
        terpilih = st.selectbox("Pilih Sampel Citra Uji:", list(pilihan_sampel.keys()))
        berkas_sampel = SAMPLES_DIR / pilihan_sampel[terpilih]
        if berkas_sampel.exists():
            citra_bgr = cv2.imread(str(berkas_sampel))
            label_sampel = terpilih
        else:
            st.warning("Berkas sampel tidak ditemukan.")
    else:
        uploaded_file = st.file_uploader("Unggah citra lesi kulit dermoskopi (JPG/PNG):", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            citra_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            label_sampel = f"Unggahan Pengguna: {uploaded_file.name}"

    if citra_bgr is not None:
        st.markdown('<div class="section-header">2. Pipeline Pra-pemrosesan Citra (Transparansi Visual)</div>', unsafe_allow_html=True)
        
        # Eksekusi Preprocessing Bertahap
        # 1. Citra Asli (RGB)
        im_rgb = cv2.cvtColor(citra_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. ROI Otsu
        im_roi_bgr, box = potong_roi(citra_bgr)
        im_roi_rgb = cv2.cvtColor(im_roi_bgr, cv2.COLOR_BGR2RGB)
        
        # 3. Shades of Gray (p=6)
        x_raw = torch.from_numpy(im_roi_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        x_sog = shades_of_gray(x_raw, p=6)
        im_sog_np = (x_sog[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Tampilkan 3 Tahap Visual
        col_v1, col_v2, col_v3 = st.columns(3)
        with col_v1:
            st.image(cv2.resize(im_rgb, (IMAGE_SIZE, IMAGE_SIZE)), caption="Tahap 1: Citra Asli (256x256)", use_container_width=True)
        with col_v2:
            st.image(im_roi_rgb, caption="Tahap 2: Segmentasi ROI Otsu", use_container_width=True)
            if box is not None:
                st.caption("✅ Area lesi aktif berhasil diisolasi.")
            else:
                st.caption("ℹ️ Menggunakan fallback citra penuh.")
        with col_v3:
            st.image(im_sog_np, caption="Tahap 3: Normalisasi Shades of Gray (p=6)", use_container_width=True)
            st.caption("✅ Deviasi saturasi warna dermoskopi dinetralkan.")

        # Eksekusi Inferensi Model
        st.markdown('<div class="section-header">3. Hasil Pengujian Model ConvNeXtTiny-MHSA</div>', unsafe_allow_html=True)
        
        model = muat_model()
        if model is not None:
            # Normalisasi ImageNet
            mean_t = torch.tensor(NORM_MEAN).view(1, 3, 1, 1)
            std_t = torch.tensor(NORM_STD).view(1, 3, 1, 1)
            x_norm = (x_sog - mean_t) / std_t

            t0 = time.time()
            with torch.no_grad():
                if opsi_tta:
                    vs = [x_norm, x_norm.flip(3), x_norm.flip(2), torch.rot90(x_norm, 1, [2, 3])]
                    logits = torch.stack([model(v) for v in vs]).mean(dim=0)
                else:
                    logits = model(x_norm)
                
                probs = F.softmax(logits, dim=1)[0].numpy()
            latensi_ms = (time.time() - t0) * 1000

            # Kalkulasi Probabilitas
            p_ganas = float(sum(probs[i] for i in MALIGNANT_IDX))
            p_jinak = float(sum(probs[i] for i in BENIGN_IDX))
            status_ganas = p_ganas >= ambang_keganasan

            col_res1, col_res2 = st.columns([1.1, 1.4])

            with col_res1:
                st.markdown("##### Diagnosis Klasifikasi:")
                if status_ganas:
                    st.markdown(f'<div class="status-badge status-ganas">⚠️ GANAS (MALIGNANT)<br><span style="font-size:0.95rem; font-weight:500;">P(Ganas) = {p_ganas*100:.2f}% (Ambang: {ambang_keganasan:.2f})</span></div>', unsafe_allow_html=True)
                    st.error("Rekomendasi Klinis: Skor keganasan melampaui ambang batas operasional. Diperlukan pemeriksaan dermoskopi mendalam dan rujukan biopsi histopatologi.")
                else:
                    st.markdown(f'<div class="status-badge status-jinak">🛡️ JINAK (BENIGN)<br><span style="font-size:0.95rem; font-weight:500;">P(Ganas) = {p_ganas*100:.2f}% (Ambang: {ambang_keganasan:.2f})</span></div>', unsafe_allow_html=True)
                    st.success("Rekomendasi Klinis: Lesi memiliki probabilitas tinggi sebagai jinak (benign). Pemantauan visual berkala tetap dianjurkan bila terdapat perubahan bentuk/warna (ABCDE).")

                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-lbl">Metrik Komputasi Inferensi</div>
                    <div style="font-size:0.85rem; color:#334155; line-height:1.7;">
                        • <strong>Waktu Eksekusi:</strong> <code>{latensi_ms:.2f} ms</code> ({'TTA 4-Arah' if opsi_tta else 'Single-Pass'})<br>
                        • <strong>Bobot Model:</strong> ConvNeXtTiny SWA (Half-Precision 66 MB)<br>
                        • <strong>Perangkat:</strong> CPU Inferensi Bersih (Hemat Memori)<br>
                        • <strong>Ukuran Tensor:</strong> 1 × 3 × 256 × 256
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_res2:
                st.markdown("##### Distribusi Probabilitas 8 Sub-Kelas Lesi Kulit:")
                df_sub = pd.DataFrame({
                    "Sub-Kelas": [KELAS[i] for i in range(8)],
                    "Nama Klinis": [KELAS_NAMA[KELAS[i]] for i in range(8)],
                    "Kategori": ["Ganas" if i in MALIGNANT_IDX else "Jinak" for i in range(8)],
                    "Probabilitas": probs
                }).sort_values("Probabilitas", ascending=False)

                fig_bar, ax_bar = plt.subplots(figsize=(6.5, 3.4), dpi=120)
                warna_bar = ["#DC2626" if c == "Ganas" else "#2563EB" for c in df_sub["Kategori"]]
                bars = ax_bar.barh(df_sub["Sub-Kelas"], df_sub["Probabilitas"] * 100, color=warna_bar, edgecolor="white", height=0.6)
                ax_bar.set_xlabel("Peluang Prediksi (%)", fontsize=8.5)
                ax_bar.set_xlim(0, 100)
                ax_bar.invert_yaxis()
                ax_bar.tick_params(labelsize=8)
                ax_bar.grid(axis="x", linestyle=":", alpha=0.4)
                
                for bar in bars:
                    w = bar.get_width()
                    if w > 2:
                        ax_bar.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=7.5, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig_bar)
                plt.close()

                # Tabel Rincian Kelas
                st.dataframe(
                    df_sub[["Sub-Kelas", "Kategori", "Nama Klinis", "Probabilitas"]].assign(
                        Probabilitas=lambda d: d["Probabilitas"].map(lambda p: f"{p*100:.2f}%")
                    ),
                    use_container_width=True,
                    hide_index=True
                )
    else:
        st.info("Silakan pilih sampel citra riset atau unggah berkas citra dermoskopi untuk memulai pengujian.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: TOLOK UKUR 3.291 DATA UJI & VALIDASI RISET
# ══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.markdown('<div class="section-header">1. Ringkasan Kinerja 3.291 Data Uji Terisolasi</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-lbl">Akurasi Agregat Uji</div>
            <div class="metric-val">85,26%</div>
            <div class="metric-sub">CI 95%: 84,01% - 86,43%</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-lbl">ROC-AUC Agregat</div>
            <div class="metric-val">0,9174</div>
            <div class="metric-sub">Daya Pembeda Sangat Unggul</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-lbl">Sensitivitas (Recall)</div>
            <div class="metric-val">81,10%</div>
            <div class="metric-sub">Terdeteksi: 974 dari 1.201 Ganas</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-lbl">Spesifisitas (True Negative)</div>
            <div class="metric-val">87,66%</div>
            <div class="metric-sub">Jinak Benar: 1.832 dari 2.090</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">2. Tabel Kinerja Resmi Naskah Riset (Baseline Terkunci)</div>', unsafe_allow_html=True)
    st.caption("Hasil evaluasi formal pada Laporan Akhir LPPM UTM & naskah JuTISI (SINTA 2).")

    tabel_resmi = df_eval.copy()
    tabel_resmi["Akurasi (%)"] = tabel_resmi["akurasi"].map(lambda x: f"{x:.2f}%")
    tabel_resmi["CI 95% (Wilson)"] = tabel_resmi.apply(lambda r: f"{r['ci95_bawah']:.2f}% - {r['ci95_atas']:.2f}%", axis=1)
    tabel_resmi["ROC-AUC"] = tabel_resmi["auc"].map(lambda x: f"{x:.4f}")
    tabel_resmi["Sensitivitas (%)"] = tabel_resmi["sensitivitas"].map(lambda x: f"{x:.2f}%")
    tabel_resmi["Spesifisitas (%)"] = tabel_resmi["spesifisitas"].map(lambda x: f"{x:.2f}%")
    tabel_resmi["Ambang"] = tabel_resmi["ambang"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")

    kolom_tampil = ["tampilan", "dataset", "n", "Akurasi (%)", "CI 95% (Wilson)", "ROC-AUC", "Sensitivitas (%)", "Spesifisitas (%)", "TP", "TN", "FP", "FN", "Ambang"]
    st.dataframe(
        tabel_resmi[kolom_tampil].rename(columns={"tampilan": "Format", "dataset": "Dataset Uji", "n": "Jumlah Sampel"}),
        use_container_width=True,
        hide_index=True
    )

    st.markdown('<div class="section-header">3. Bukti Validasi Visual & Grafik Riset</div>', unsafe_allow_html=True)
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        if (ASSETS_DIR / "roc_dan_sebaran.png").exists():
            st.image(str(ASSETS_DIR / "roc_dan_sebaran.png"), caption="Kurva ROC-AUC Agregat (0,9174) & Distribusi Skor Prediksi", use_container_width=True)
    with gcol2:
        if (ASSETS_DIR / "matriks_konfusi.png").exists():
            st.image(str(ASSETS_DIR / "matriks_konfusi.png"), caption="Matriks Konfusi Terkalibrasi (Ambang 0,42)", use_container_width=True)

# ─── FOOTER & INTEGRITAS RISET ────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="font-size: 0.82rem; color: #64748B; text-align: center; line-height: 1.6;">
    <strong>⚡ DERMASPARK (Dermatology × Apache Spark)</strong> &bull; Platform Pengujian Citra & Evaluasi Model ConvNeXtTiny-MHSA<br>
    Asistensi Hibah Penelitian BIMA Kemendikbudristek / LPPM Universitas Trunojoyo Madura 2026<br>
    <em>Integritas Metodologi: Zero Patient Leakage | Catatan Klinis: Sistem berfungsi murni sebagai Decision Support System (DSS) dan komputasi performa partisi citra medis, bukan instrumen diagnosis mandiri tanpa supervisi spesialis patologi/dermatologi.</em>
</div>
""", unsafe_allow_html=True)
