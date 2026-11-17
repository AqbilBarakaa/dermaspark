import os
import sys
import time
import json
import math
import base64
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ─── KONFIGURASI HALAMAN ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="DERMASPARK - Dermatology × Apache Spark",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── PATH DIREKTORI & KONSTANTA RISET ─────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "model" / "utama_256_swa_half.pth"
SAMPLES_DIR = BASE_DIR / "samples"

IMAGE_SIZE = 256
KELAS = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
KELAS_NAMA = {
    "AKIEC": "Actinic Keratoses",
    "BCC": "Basal Cell Carcinoma",
    "BKL": "Benign Keratosis",
    "DF": "Dermatofibroma",
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevi",
    "SCC": "Squamous Cell Carcinoma",
    "VASC": "Vascular Lesions"
}
KELAS_TIPE = {
    "AKIEC": "Malignant",
    "BCC": "Malignant",
    "BKL": "Benign",
    "DF": "Benign",
    "MEL": "Malignant",
    "NV": "Benign",
    "SCC": "Malignant",
    "VASC": "Benign"
}
KELAS_KETERANGAN = {
    "AKIEC": "Lesi pra-kanker epidermal akibat radiasi UV kronis.",
    "BCC": "Kanker sel basal, keganasan lokal paling sering ditemui.",
    "BKL": "Lesi keratosis seboroik jinak, non-malignant.",
    "DF": "Proliferasi jinak jaringan ikat/fibroblas kutan.",
    "MEL": "Keganasan melanosit invasif dengan risiko metastasis tinggi.",
    "NV": "Proliferasi jinak sel melanositik (tahi lalat tipikal).",
    "SCC": "Karsinoma sel skuamosa, memiliki potensi penyebaran nodus limfa.",
    "VASC": "Malformasi vaskular jinak atau hemangioma kutis."
}
MALIGNANT_IDX = [0, 1, 4, 6]
BENIGN_IDX = [2, 3, 5, 7]
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ─── TEMA DESAIN KHUSUS DERMASPARK (DEEP NAVY & SPARK GOLD) ───────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #e2e8f0;
    }

    .stApp {
        background-color: #080d1a;
        background-image: 
            radial-gradient(circle at 15% 10%, rgba(245, 158, 11, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 85% 15%, rgba(56, 189, 248, 0.09) 0%, transparent 45%),
            linear-gradient(rgba(30, 41, 59, 0.25) 1px, transparent 1px),
            linear-gradient(90deg, rgba(30, 41, 59, 0.25) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 28px 28px, 28px 28px;
    }

    /* Header Navigasi Khusus */
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(17, 24, 39, 0.9) 100%);
        border: 1px solid rgba(245, 158, 11, 0.22);
        border-radius: 16px;
        padding: 1.1rem 1.6rem;
        margin-bottom: 1.6rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
    }
    .hero-left {
        display: flex;
        align-items: center;
        gap: 1.1rem;
    }
    .hero-spark-badge {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 50%, #b45309 100%);
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.6rem;
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.4);
    }
    .hero-brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .hero-sub {
        font-size: 0.84rem;
        color: #94a3b8;
        margin-top: 0.15rem;
    }
    .hero-chips {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .hero-chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        font-weight: 600;
        padding: 0.3rem 0.75rem;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(15, 23, 42, 0.7);
        color: #cbd5e1;
    }
    .chip-spark {
        border-color: rgba(245, 158, 11, 0.35);
        color: #fbbf24;
        background: rgba(245, 158, 11, 0.10);
    }
    .chip-ai {
        border-color: rgba(56, 189, 248, 0.35);
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.10);
    }

    /* Container Kartu Kaca */
    .card-shell {
        background: rgba(15, 23, 42, 0.72);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.25rem;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.2rem;
    }
    .card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.88rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.9rem;
    }
    .card-title-highlight {
        color: #f59e0b;
    }

    /* Previewer & Viewport */
    .viewport-box {
        position: relative;
        background: #020617;
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 0.8rem;
    }
    .viewport-box img {
        width: 100%;
        display: block;
        border-radius: 9px;
    }
    .viewport-hud {
        position: absolute;
        bottom: 8px;
        left: 8px;
        right: 8px;
        display: flex;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.70rem;
        background: rgba(2, 6, 23, 0.75);
        backdrop-filter: blur(8px);
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        color: #cbd5e1;
    }

    /* Vonis Diagnostik */
    .verdict-box {
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.1rem;
        text-align: left;
    }
    .verdict-malignant {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.16) 0%, rgba(153, 27, 27, 0.22) 100%);
        border: 1px solid #ef4444;
    }
    .verdict-benign {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.14) 0%, rgba(6, 95, 70, 0.20) 100%);
        border: 1px solid #10b981;
    }
    .verdict-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
    }
    .verdict-tag.mal { color: #f87171; }
    .verdict-tag.ben { color: #34d399; }
    .verdict-main {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.2;
    }
    .verdict-sub {
        font-size: 0.85rem;
        color: #cbd5e1;
        margin-top: 0.45rem;
        line-height: 1.45;
    }

    /* Gauge / Skor Probabilitas */
    .metric-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin-bottom: 1.1rem;
    }
    .metric-cell {
        background: rgba(2, 6, 23, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
    .metric-cell-lbl {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.70rem;
        color: #64748b;
        text-transform: uppercase;
    }
    .metric-cell-val {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 0.15rem;
    }

    /* Bar Distribusi Kategori */
    .cat-bar-wrap {
        margin-bottom: 0.6rem;
    }
    .cat-bar-meta {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .cat-track {
        height: 7px;
        background: rgba(255, 255, 255, 0.07);
        border-radius: 9999px;
        overflow: hidden;
    }
    .cat-fill {
        height: 100%;
        border-radius: 9999px;
    }

    /* Pipeline Multi-Tahap */
    .pipe-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.7rem;
        margin-top: 0.5rem;
    }
    .pipe-step-card {
        background: rgba(2, 6, 23, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.6rem;
        text-align: center;
    }
    .pipe-step-card img {
        width: 100%;
        height: 95px;
        object-fit: cover;
        border-radius: 6px;
        display: block;
        margin-bottom: 0.4rem;
    }
    .pipe-step-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.70rem;
        font-weight: 600;
        color: #cbd5e1;
    }
    .pipe-step-desc {
        font-size: 0.65rem;
        color: #64748b;
        margin-top: 0.15rem;
    }

    /* Footer */
    .footer-bar {
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 1.2rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
        font-size: 0.78rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)

# ─── ARSITEKTUR JARINGAN CONVNEXTTINY-MHSA ────────────────────────────────────
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

# ─── ALUR PRAPEMROSESAN SAINTIFIK ─────────────────────────────────────────────
DET, MARGIN, FRAC_MIN, FRAC_MAX = 512, 0.15, 0.03, 0.90

def eksekusi_praproses(im_bgr):
    h, w = im_bgr.shape[:2]
    det = cv2.resize(im_bgr, (DET, DET), interpolation=cv2.INTER_AREA)
    abu = cv2.GaussianBlur(cv2.cvtColor(det, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    _, th = cv2.threshold(abu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    lingkar = np.zeros((DET, DET), np.uint8)
    cv2.circle(lingkar, (DET // 2, DET // 2), int(DET * 0.48), 255, -1)
    masker = cv2.bitwise_and(th, lingkar)
    cnts, _ = cv2.findContours(masker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    crop = im_bgr
    box_coords = None
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
                box_coords = (x0, y0, x1, y1)

    im_crop = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    
    # Shades of gray (p=6)
    x_raw = torch.from_numpy(cv2.cvtColor(im_crop, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    e = x_raw.clamp_min(1e-6).pow(6).mean(dim=(2, 3), keepdim=True).pow(1.0 / 6)
    e = e / e.norm(dim=1, keepdim=True)
    x_sog = (x_raw / (e * math.sqrt(3))).clamp(0, 1)
    im_sog_np = (x_sog[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    im_sog_bgr = cv2.cvtColor(im_sog_np, cv2.COLOR_RGB2BGR)

    return {
        "asli": cv2.resize(im_bgr, (IMAGE_SIZE, IMAGE_SIZE)),
        "masker": cv2.cvtColor(cv2.resize(masker, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_GRAY2BGR),
        "roi": im_crop,
        "sog": im_sog_bgr,
        "tensor_sog": x_sog,
        "ada_roi": box_coords is not None
    }

def enc_b64(im_bgr):
    _, b = cv2.imencode(".jpg", im_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(b).decode("utf-8")

# ─── MODEL LOADER ─────────────────────────────────────────────────────────────
@st.cache_resource
def ambil_model():
    if not MODEL_PATH.exists():
        st.error("Model ConvNeXtTiny SWA belum siap.")
        return None
    net = ConvNeXtTinyMHSA(n_kelas=8)
    sd = torch.load(MODEL_PATH, map_location="cpu")
    sd_f32 = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
    net.load_state_dict(sd_f32)
    net.eval()
    return net

# ─── HERO HEADER BANNER ───────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-left">
        <div class="hero-spark-badge">⚡</div>
        <div>
            <div class="hero-brand">
                DERMASPARK
                <span style="font-size:0.72rem; font-weight:600; padding:0.2rem 0.5rem; background:rgba(245,158,11,0.2); border:1px solid #f59e0b; border-radius:6px; color:#fbbf24;">PROTOTIPE V1.0</span>
            </div>
            <div class="hero-sub">Platform Skrining Lesi Kanker Kulit Medis &bull; Asistensi Hibah BIMA UTM 2026</div>
        </div>
    </div>
    <div class="hero-chips">
        <span class="hero-chip chip-spark">Spark Partition: 4-Core Optimal</span>
        <span class="hero-chip chip-ai">ConvNeXtTiny-MHSA (85,26%)</span>
        <span class="hero-chip">Ambang Klinis: 0,42</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── WORKSPACE DUA KOLOM ──────────────────────────────────────────────────────
col_input, col_result = st.columns([1, 1.25], gap="large")

with col_input:
    st.markdown("""
    <div class="card-shell">
        <div class="card-title"><span class="card-title-highlight">01.</span> MASUKAN CITRA DERMATOSKOPI</div>
    """, unsafe_allow_html=True)

    mode_masukan = st.radio(
        "Sumber Data:",
        ["Sampel Kurasi Validasi Riset", "Unggah Citra Dermoskopi Bebas"],
        horizontal=True,
        label_visibility="collapsed"
    )

    citra_input_bgr = None
    judul_citra = ""

    if mode_masukan == "Sampel Kurasi Validasi Riset":
        koleksi = {
            "🔴 Melanoma (Kanker Ganas Agresif) - ISIC_0000054": ("Melanoma_ISIC_0000054.jpg", "Malignant"),
            "🔴 Basal Cell Carcinoma (Kanker Kulit) - ISIC_0024436": ("BCC_ISIC_0024436.jpg", "Malignant"),
            "🔴 Actinic Keratoses (Pra-Kanker) - ISIC_0024329": ("AKIEC_ISIC_0024329.jpg", "Malignant"),
            "🟢 Melanocytic Nevi (Tahi Lalat Jinak) - ISIC_0000009": ("Nevus_ISIC_0000009.jpg", "Benign"),
            "🟢 Benign Keratosis (Lesi Jinak) - ISIC_0012173": ("BKL_ISIC_0012173.jpg", "Benign")
        }
        pilih_item = st.selectbox("Pilih Sampel Riil:", list(koleksi.keys()))
        berkas, sifat = koleksi[pilih_item]
        target_f = SAMPLES_DIR / berkas
        if target_f.exists():
            citra_input_bgr = cv2.imread(str(target_f))
            judul_citra = pilih_item
    else:
        up = st.file_uploader("Unggah foto lesi kulit (JPG / PNG):", type=["jpg", "jpeg", "png"])
        if up is not None:
            buf = np.asarray(bytearray(up.read()), dtype=np.uint8)
            citra_input_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            judul_citra = f"Citra Unggahan: {up.name}"

    if citra_input_bgr is not None:
        t0_prep = time.time()
        tahap_data = eksekusi_praproses(citra_input_bgr)
        durasi_prep_ms = (time.time() - t0_prep) * 1000

        b64_asli = enc_b64(tahap_data["asli"])
        st.markdown(f"""
        <div class="viewport-box">
            <img src="data:image/jpeg;base64,{b64_asli}" alt="Preview" />
            <div class="viewport-hud">
                <span>STATUS: CITRA TERSIAPKAN</span>
                <span>DIMENSI: 256×256 PX</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("⚙️ Parameter Ambang & Toleransi Klinis"):
        ambang_slider = st.slider("Batas Ambang Keganasan ($P_{ganas}$):", 0.0, 1.0, 0.42, 0.01)
        st.caption("Ambang optimal riset: **0,42** (Sensitivitas 81,10% | Spesifisitas 87,66%).")

    st.markdown("</div>", unsafe_allow_html=True)

with col_result:
    st.markdown("""
    <div class="card-shell">
        <div class="card-title"><span class="card-title-highlight">02.</span> HASIL SKRINING & INFERENSI DIAGNOSTIK</div>
    """, unsafe_allow_html=True)

    if citra_input_bgr is not None:
        model = ambil_model()
        if model is not None:
            # Normalisasi ImageNet
            mean_t = torch.tensor(NORM_MEAN).view(1, 3, 1, 1)
            std_t = torch.tensor(NORM_STD).view(1, 3, 1, 1)
            x_in = (tahap_data["tensor_sog"] - mean_t) / std_t

            t0_inf = time.time()
            with torch.no_grad():
                lg = model(x_in)
                probs = F.softmax(lg, dim=1)[0].numpy()
            durasi_inf_ms = (time.time() - t0_inf) * 1000

            top_idx = int(np.argmax(probs))
            top_kelas = KELAS[top_idx]
            top_nama = KELAS_NAMA[top_kelas]
            top_prob = float(probs[top_idx])

            prob_ganas = float(sum(probs[i] for i in MALIGNANT_IDX))
            prob_jinak = float(sum(probs[i] for i in BENIGN_IDX))
            status_ganas = prob_ganas >= ambang_slider

            # Vonis Panel
            if status_ganas:
                st.markdown(f"""
                <div class="verdict-box verdict-malignant">
                    <div class="verdict-tag mal">⚠️ INDIKASI KEGANASAN (MALIGNANT)</div>
                    <div class="verdict-main">{top_nama}</div>
                    <div class="verdict-sub">
                        Peluang Keganasan Kumulatif: <strong>{prob_ganas*100:.1f}%</strong> (Ambang Batas: {ambang_slider:.2f}).<br>
                        <em>Rekomendasi:</em> Lesi menunjukkan karakteristik atipikal. Diperlukan konfirmasi biopsi histopatologi serta konsultasi dokter spesialis dermatologi.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-box verdict-benign">
                    <div class="verdict-tag ben">🛡️ INDIKASI JINAK (BENIGN)</div>
                    <div class="verdict-main">{top_nama}</div>
                    <div class="verdict-sub">
                        Peluang Keganasan Kumulatif: <strong>{prob_ganas*100:.1f}%</strong> (Di Bawah Ambang: {ambang_slider:.2f}).<br>
                        <em>Rekomendasi:</em> Lesi terindikasi jinak / non-kanker. Tetap lakukan observasi klinis berkala jika terjadi evolusi asimetri, batas, atau warna.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Baris Metrik Utama
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-cell">
                    <div class="metric-cell-lbl">Peluang Ganas P(Ganas)</div>
                    <div class="metric-cell-val" style="color:{'#f87171' if status_ganas else '#34d399'};">{prob_ganas*100:.2f}%</div>
                </div>
                <div class="metric-cell">
                    <div class="metric-cell-lbl">Keyakinan Sub-Kelas ({top_kelas})</div>
                    <div class="metric-cell-val" style="color:#38bdf8;">{top_prob*100:.2f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Rincian 8 Sub-Kelas Lesi Kulit
            st.markdown('<div style="font-size:0.76rem; font-weight:700; color:#94a3b8; text-transform:uppercase; margin-bottom:0.5rem; letter-spacing:0.04em;">DISTRIBUSI PROBABILITAS KELAS LESI</div>', unsafe_allow_html=True)

            urutan = np.argsort(probs)[::-1]
            for idx in urutan[:5]:  # 5 teratas
                k = KELAS[idx]
                p = probs[idx]
                nama_sub = KELAS_NAMA[k]
                tipe = KELAS_TIPE[k]
                warna_grad = "linear-gradient(90deg, #f59e0b, #ef4444)" if tipe == "Malignant" else "linear-gradient(90deg, #38bdf8, #10b981)"
                st.markdown(f"""
                <div class="cat-bar-wrap">
                    <div class="cat-bar-meta">
                        <span style="color:#e2e8f0;">{nama_sub} <small style="color:#64748b;">({tipe})</small></span>
                        <span style="font-family:'JetBrains Mono', monospace; color:#cbd5e1;">{p*100:.2f}%</span>
                    </div>
                    <div class="cat-track">
                        <div class="cat-fill" style="width: {p*100}%; background: {warna_grad};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Telemetri Komputasi Singkat
            st.caption(f"⚡ Komputasi: Prep `{durasi_prep_ms:.1f} ms` | Inferensi `{durasi_inf_ms:.1f} ms` | Arsitektur: `ConvNeXtTiny-MHSA SWA`")
    else:
        st.info("Pilih sampel riset di kolom kiri atau unggah citra dermoskopi untuk memicu diagnosis.")

    st.markdown("</div>", unsafe_allow_html=True)

# ─── ZONE 03: ALUR PRAPEMROSESAN SAINTIFIK ────────────────────────────────────
if citra_input_bgr is not None and "tahap_data" in locals():
    st.markdown("""
    <div class="card-shell" style="margin-top:0.4rem;">
        <div class="card-title"><span class="card-title-highlight">03.</span> ALUR PRAPEMROSESAN & NORMALISASI CITRA (TESIS RISET BIMA)</div>
        <div class="pipe-container">
            <div class="pipe-step-card">
                <img src="data:image/jpeg;base64,{}" alt="Tahap 1" />
                <div class="pipe-step-title">1 &bull; Citra Asli</div>
                <div class="pipe-step-desc">Dermoskopi standar (256x256 px)</div>
            </div>
            <div class="pipe-step-card">
                <img src="data:image/jpeg;base64,{}" alt="Tahap 2" />
                <div class="pipe-step-title">2 &bull; Masking Otsu</div>
                <div class="pipe-step-desc">Eliminasi batas vignetting lensa</div>
            </div>
            <div class="pipe-step-card">
                <img src="data:image/jpeg;base64,{}" alt="Tahap 3" />
                <div class="pipe-step-title">3 &bull; Segmentasi ROI</div>
                <div class="pipe-step-desc">Isolasi lesi + margin 15%</div>
            </div>
            <div class="pipe-step-card">
                <img src="data:image/jpeg;base64,{}" alt="Tahap 4" />
                <div class="pipe-step-title">4 &bull; Shades of Gray (p=6)</div>
                <div class="pipe-step-desc">Penyelarasan iluminasi & warna</div>
            </div>
        </div>
    </div>
    """.format(
        enc_b64(tahap_data["asli"]),
        enc_b64(tahap_data["masker"]),
        enc_b64(tahap_data["roi"]),
        enc_b64(tahap_data["sog"])
    ), unsafe_allow_html=True)

# ─── FOOTER RESMI ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-bar">
    <strong>DERMASPARK</strong> &bull; Analisis Granularitas Partisi Memori Apache Spark & Klasifikasi Kanker Kulit ConvNeXtTiny-MHSA<br>
    Asistensi Program Hibah Penelitian BIMA Kemendikbudristek 2026 &bull; Teknik Informatika Universitas Trunojoyo Madura<br>
    <em>Peneliti: Aqbil Baraka (NIM 230411100077) &bull; Pembimbing: Fifin Ayu Mufarroha, S.Kom., M.Kom.</em><br>
    <small style="color:#475569;">Pernyataan Etik: Sistem ini dirancang sebagai Decision Support System (DSS) berbasis bukti ilmiah riset dan bukan instrumen diagnosis mandiri tanpa supervisi spesialis patologi anatomi/dermatologi.</small>
</div>
""", unsafe_allow_html=True)
