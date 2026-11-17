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
    page_title="DERMASPARK - Sistem Skrining Lesi Kulit",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── STYLING PROFESIONAL MEDIS ALAMI (BEBAS KESAN SCI-FI / AI GIMMICK) ────────
st.markdown("""
<style>
    /* Tipografi Bersih & Elegan */
    .app-header {
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 1.1rem;
        margin-bottom: 1.6rem;
    }
    .inst-tag {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #2563EB;
        letter-spacing: 0.04em;
        margin-bottom: 0.25rem;
    }
    .app-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
        line-height: 1.2;
    }
    .app-desc {
        font-size: 0.92rem;
        color: #64748B;
        margin-top: 0.35rem;
        line-height: 1.5;
    }

    /* Kartu & Wadah Konten */
    .med-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.2rem;
    }
    .med-card-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1E293B;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 0.5rem;
        margin-bottom: 1.0rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Hasil Diagnosa Sederhana & Bertanggung Jawab */
    .report-box {
        border-radius: 8px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1.0rem;
    }
    .report-benign {
        background-color: #F0FDF4;
        border: 1px solid #86EFAC;
        color: #166534;
    }
    .report-malignant {
        background-color: #FEF2F2;
        border: 1px solid #FCA5A5;
        color: #991B1B;
    }
    .report-status {
        font-size: 0.80rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.25rem;
    }
    .report-heading {
        font-size: 1.45rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }
    .report-text {
        font-size: 0.88rem;
        line-height: 1.5;
        color: #334155;
    }

    /* Tabel Nilai Ringkas */
    .metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin-bottom: 1.0rem;
    }
    .metric-item {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.75rem;
    }
    .metric-item-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748B;
    }
    .metric-item-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.2rem;
    }

    /* Footer Akademik */
    .app-footer {
        border-top: 1px solid #E2E8F0;
        padding-top: 1.2rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
        font-size: 0.80rem;
        color: #64748B;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ─── PATH DIREKTORI & KONSTANTA ───────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "model" / "utama_256_swa_half.pth"
SAMPLES_DIR = BASE_DIR / "samples"

IMAGE_SIZE = 256
KELAS = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
KELAS_NAMA = {
    "AKIEC": "Actinic Keratosis (Pra-Kanker)",
    "BCC": "Basal Cell Carcinoma (Kanker Kulit)",
    "BKL": "Benign Keratosis (Lesi Jinak)",
    "DF": "Dermatofibroma (Lesi Jinak)",
    "MEL": "Melanoma (Kanker Ganas)",
    "NV": "Melanocytic Nevus (Tahi Lalat Jinak)",
    "SCC": "Squamous Cell Carcinoma (Kanker Kulit)",
    "VASC": "Vascular Lesion (Lesi Pembuluh Darah)"
}
KELAS_KATEGORI = {
    "AKIEC": "Ganas / Berisiko",
    "BCC": "Ganas",
    "BKL": "Jinak",
    "DF": "Jinak",
    "MEL": "Ganas",
    "NV": "Jinak",
    "SCC": "Ganas",
    "VASC": "Jinak"
}
MALIGNANT_IDX = [0, 1, 4, 6]
BENIGN_IDX = [2, 3, 5, 7]
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

# ─── PRA-PEMROSESAN CITRA SAINTIFIK ───────────────────────────────────────────
DET, MARGIN, FRAC_MIN, FRAC_MAX = 512, 0.15, 0.03, 0.90

def proses_citra(im_bgr):
    h, w = im_bgr.shape[:2]
    det = cv2.resize(im_bgr, (DET, DET), interpolation=cv2.INTER_AREA)
    abu = cv2.GaussianBlur(cv2.cvtColor(det, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    _, th = cv2.threshold(abu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    lingkar = np.zeros((DET, DET), np.uint8)
    cv2.circle(lingkar, (DET // 2, DET // 2), int(DET * 0.48), 255, -1)
    masker = cv2.bitwise_and(th, lingkar)
    cnts, _ = cv2.findContours(masker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    crop = im_bgr
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        frac = cv2.contourArea(c) / (DET * DET)
        if FRAC_MIN <= frac <= FRAC_MAX:
            x, y, bw, bh = cv2.boundingRect(c)
            sisi = int(max(bw, bh) * (1 + 2 * MARGIN))
            cx, cy = x + bw // 2, y + bh // 2
            x0, y0 = max(0, cx - sisi // 2), max(0, cy - sisi // 2)
            x1, y1 = min(DET, x0 + sisi), min(DET, y0 + sisi)
            sx, sy = w / DET, h / DET
            crop_cand = im_bgr[int(y0 * sy):int(y1 * sy), int(x0 * sx):int(x1 * sx)]
            if crop_cand.size > 0:
                crop = crop_cand

    im_crop_rgb = cv2.cvtColor(cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    
    # Shades of gray (p=6)
    x_raw = torch.from_numpy(im_crop_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    e = x_raw.clamp_min(1e-6).pow(6).mean(dim=(2, 3), keepdim=True).pow(1.0 / 6)
    e = e / e.norm(dim=1, keepdim=True)
    x_sog = (x_raw / (e * math.sqrt(3))).clamp(0, 1)
    im_sog_rgb = (x_sog[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    return {
        "asli_rgb": cv2.cvtColor(cv2.resize(im_bgr, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2RGB),
        "masker_rgb": cv2.cvtColor(cv2.resize(masker, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_GRAY2RGB),
        "roi_rgb": im_crop_rgb,
        "sog_rgb": im_sog_rgb,
        "tensor_sog": x_sog
    }

# ─── PEMUATAN MODEL ───────────────────────────────────────────────────────────
@st.cache_resource
def muat_model():
    if not MODEL_PATH.exists():
        st.error(f"Berkas model tidak ditemukan: {MODEL_PATH}")
        return None
    net = ConvNeXtTinyMHSA(n_kelas=8)
    sd = torch.load(MODEL_PATH, map_location="cpu")
    sd_f32 = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
    net.load_state_dict(sd_f32)
    net.eval()
    return net

# ─── HEADER HALAMAN ───────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="inst-tag">Universitas Trunojoyo Madura &bull; Hibah Penelitian BIMA Kemendikbudristek</div>
    <h1 class="app-title">DERMASPARK: Sistem Skrining Citra Lesi Kulit</h1>
    <div class="app-desc">
        Aplikasi pendukung keputusan klinis berbasis model <strong>ConvNeXtTiny-MHSA</strong> dengan pra-pemrosesan ROI Otsu dan Shades of Gray.<br>
        <em>Peneliti: Aqbil Baraka (NIM 230411100077) &bull; Pembimbing: Fifin Ayu Mufarroha, S.Kom., M.Kom.</em>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── WORKSPACE DUA KOLOM ──────────────────────────────────────────────────────
col_input, col_report = st.columns([1, 1.25], gap="large")

with col_input:
    st.markdown("""
    <div class="med-card">
        <div class="med-card-title">📷 1. Masukan Citra Dermoskopi</div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Pilih Metode Masukan:",
        ["Gunakan Sampel Data Riset", "Unggah Foto Lesi Kulit"],
        horizontal=True
    )

    citra_bgr = None
    keterangan_sampel = ""

    if mode == "Gunakan Sampel Data Riset":
        pilihan_sampel = {
            "Melanoma (Kanker Ganas Agresif) - ISIC_0000054": "Melanoma_ISIC_0000054.jpg",
            "Basal Cell Carcinoma (Kanker Kulit) - ISIC_0024436": "BCC_ISIC_0024436.jpg",
            "Actinic Keratoses (Pra-Kanker) - ISIC_0024329": "AKIEC_ISIC_0024329.jpg",
            "Melanocytic Nevus (Tahi Lalat Jinak) - ISIC_0000009": "Nevus_ISIC_0000009.jpg",
            "Benign Keratosis (Lesi Jinak) - ISIC_0012173": "BKL_ISIC_0012173.jpg"
        }
        terpilih = st.selectbox("Pilih Sampel Uji:", list(pilihan_sampel.keys()))
        berkas = SAMPLES_DIR / pilihan_sampel[terpilih]
        if berkas.exists():
            citra_bgr = cv2.imread(str(berkas))
            keterangan_sampel = terpilih
    else:
        file_up = st.file_uploader("Unggah berkas foto dermoskopi (JPG / PNG):", type=["jpg", "jpeg", "png"])
        if file_up is not None:
            buf = np.asarray(bytearray(file_up.read()), dtype=np.uint8)
            citra_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            keterangan_sampel = f"Unggahan: {file_up.name}"

    if citra_bgr is not None:
        tahap = proses_citra(citra_bgr)
        st.image(tahap["asli_rgb"], caption="Pratinjau Citra Lesi", use_container_width=True)

    with st.expander("Pengaturan Ambang Keputusan"):
        ambang = st.slider("Batas Ambang Keganasan:", 0.0, 1.0, 0.42, 0.01)
        st.caption("Batas ambang default hasil validasi riset: **0,42** (Sensitivitas 81,10% | Spesifisitas 87,66%).")

    st.markdown("</div>", unsafe_allow_html=True)

with col_report:
    st.markdown("""
    <div class="med-card">
        <div class="med-card-title">📋 2. Lembar Hasil Skrining</div>
    """, unsafe_allow_html=True)

    if citra_bgr is not None:
        model = muat_model()
        if model is not None:
            # Normalisasi
            mean_t = torch.tensor(NORM_MEAN).view(1, 3, 1, 1)
            std_t = torch.tensor(NORM_STD).view(1, 3, 1, 1)
            x_norm = (tahap["tensor_sog"] - mean_t) / std_t

            t0 = time.time()
            with torch.no_grad():
                logits = model(x_norm)
                probs = F.softmax(logits, dim=1)[0].numpy()
            durasi_ms = (time.time() - t0) * 1000

            top_idx = int(np.argmax(probs))
            top_nama = KELAS_NAMA[KELAS[top_idx]]
            top_prob = float(probs[top_idx])

            prob_ganas = float(sum(probs[i] for i in MALIGNANT_IDX))
            is_ganas = prob_ganas >= ambang

            # Kotak Laporan Bersih
            if is_ganas:
                st.markdown(f"""
                <div class="report-box report-malignant">
                    <div class="report-status">Hasil Skrining: Terindikasi Ganas (Malignant)</div>
                    <div class="report-heading">{top_nama}</div>
                    <div class="report-text">
                        Skor peluang keganasan kumulatif tercatat sebesar <strong>{prob_ganas*100:.1f}%</strong> (melebihi ambang batas {ambang:.2f}).<br>
                        <em>Tindakan medis:</em> Disarankan untuk melakukan konsultasi langsung dengan dokter spesialis dermatologi untuk pemeriksaan dermoskopi lanjutan dan biopsi jaringan.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="report-box report-benign">
                    <div class="report-status">Hasil Skrining: Terindikasi Jinak (Benign)</div>
                    <div class="report-heading">{top_nama}</div>
                    <div class="report-text">
                        Skor peluang keganasan berada pada tingkat rendah, yaitu <strong>{prob_ganas*100:.1f}%</strong> (di bawah ambang batas {ambang:.2f}).<br>
                        <em>Tindakan medis:</em> Lesi saat ini terindikasi jinak. Tetap lakukan observasi mandiri berkala jika terdapat perubahan bentuk, ukuran, atau warna (aturan ABCDE).
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Baris Metrik Utama
            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-item">
                    <div class="metric-item-label">Probabilitas Keganasan</div>
                    <div class="metric-item-value" style="color:{'#B91C1C' if is_ganas else '#15803D'};">{prob_ganas*100:.2f}%</div>
                </div>
                <div class="metric-item">
                    <div class="metric-item-label">Keyakinan Klasifikasi ({KELAS[top_idx]})</div>
                    <div class="metric-item-value" style="color:#1E40AF;">{top_prob*100:.2f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Rincian Probabilitas 8 Kelas
            st.markdown("##### Rincian Probabilitas per Kategori Lesi:")
            
            df_res = pd.DataFrame({
                "Nama Lesi": [KELAS_NAMA[k] for k in KELAS],
                "Kategori": [KELAS_KATEGORI[k] for k in KELAS],
                "Probabilitas": probs
            }).sort_values("Probabilitas", ascending=False)

            fig, ax = plt.subplots(figsize=(6, 2.6), dpi=120)
            warna_bars = ["#EF4444" if k == "Ganas" or "Ganas" in k else "#3B82F6" for k in df_res["Kategori"]]
            bars = ax.barh(df_res["Nama Lesi"], df_res["Probabilitas"] * 100, color=warna_bars, height=0.6)
            ax.set_xlim(0, 100)
            ax.invert_yaxis()
            ax.set_xlabel("Peluang (%)", fontsize=8)
            ax.tick_params(labelsize=7.5)
            ax.grid(axis="x", linestyle="--", alpha=0.3)
            
            for bar in bars:
                w = bar.get_width()
                if w > 1.5:
                    ax.text(w + 1.0, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=7, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.caption(f"Waktu komputasi model: {durasi_ms:.1f} ms | Resolusi masukan: 256×256 piksel")
    else:
        st.info("Pilih sampel lesi kulit di sebelah kiri atau unggah berkas foto untuk melihat hasil skrining.")

    st.markdown("</div>", unsafe_allow_html=True)

# ─── TAHAPAN PRA-PEMROSESAN CITRA ─────────────────────────────────────────────
if citra_bgr is not None:
    st.markdown("""
    <div class="med-card">
        <div class="med-card-title">🔬 3. Tahapan Pra-pemrosesan Citra (Metodologi Riset)</div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.image(tahap["asli_rgb"], caption="1. Citra Asli (256×256)", use_container_width=True)
    with c2:
        st.image(tahap["masker_rgb"], caption="2. Masker Lingkaran & Otsu", use_container_width=True)
    with c3:
        st.image(tahap["roi_rgb"], caption="3. Segmentasi ROI (+15% Margin)", use_container_width=True)
    with c4:
        st.image(tahap["sog_rgb"], caption="4. Normalisasi Warna (p=6)", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ─── FOOTER RESMI ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <strong>DERMASPARK</strong> &bull; Program Asistensi Hibah Penelitian BIMA Kemendikbudristek 2026<br>
    Teknik Informatika, Fakultas Teknik, Universitas Trunojoyo Madura<br>
    <em>Pernyataan: Aplikasi ini merupakan perangkat lunak pendukung keputusan (decision support) untuk kebutuhan akademik riset dan bukan instrumen pengganti diagnosa dokter spesialis patologi/dermatologi.</em>
</div>
""", unsafe_allow_html=True)
