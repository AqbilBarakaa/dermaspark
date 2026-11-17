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
    page_title="DERMASPARK - Deteksi Lesi Kanker Kulit",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── PATH DIREKTORI ───────────────────────────────────────────────────────────
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
KELAS_DESKRIPSI = {
    "AKIEC": "Lesi bercak pra-kanker pada lapisan epidermis yang berpotensi berkembang menjadi karsinoma.",
    "BCC": "Kanker kulit non-melanoma paling umum, bertumbuh lambat namun invasif secara lokal.",
    "BKL": "Lesi seboroik jinak non-kanker yang sering muncul seiring penuaan kulit.",
    "DF": "Nodul jinak jaringan fibrosa kulit yang umumnya tidak berbahaya.",
    "MEL": "Kanker kulit ganas paling agresif yang berkembang dari melanosit. Memerlukan penanganan segera.",
    "NV": "Proliferasi melanosit jinak umum (tahi lalat). Mayoritas stabil dan tidak ganas.",
    "SCC": "Kanker kulit ganas sel skuamosa dengan risiko metastatik lebih tinggi dibanding BCC.",
    "VASC": "Kelainan vaskular jinak pada pembuluh darah kulit (angioma/hemangioma)."
}
MALIGNANT_IDX = [0, 1, 4, 6]  # AKIEC, BCC, MEL, SCC
BENIGN_IDX = [2, 3, 5, 7]     # BKL, DF, NV, VASC
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ─── TEMA VISUAL CYBER-MEDICAL GLASSMORPHISM (DERMATOAI STYLE) ────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        color: #f1f5f9;
    }

    .stApp {
        background-color: #070b14;
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(56, 189, 248, 0.10) 0%, transparent 65%),
            linear-gradient(rgba(56, 189, 248, 0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(56, 189, 248, 0.025) 1px, transparent 1px);
        background-size: 100% 100%, 32px 32px, 32px 32px;
    }

    /* Topbar Header */
    .topbar-wrap {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(13, 20, 36, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(56, 189, 248, 0.16);
        border-radius: 16px;
        padding: 0.85rem 1.4rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .topbar-left {
        display: flex;
        align-items: center;
        gap: 0.9rem;
    }
    .topbar-logo {
        width: 44px;
        height: 44px;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #6366f1 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.4);
    }
    .topbar-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.35rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .topbar-sub {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 0.15rem;
    }
    .topbar-right {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        flex-wrap: wrap;
    }
    .chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        font-weight: 600;
        padding: 0.28rem 0.75rem;
        border-radius: 9999px;
        border: 1px solid rgba(56, 189, 248, 0.25);
        background: rgba(56, 189, 248, 0.08);
        color: #38bdf8;
    }
    .chip-green {
        border-color: rgba(16, 185, 129, 0.3);
        background: rgba(16, 185, 129, 0.08);
        color: #34d399;
    }

    /* Section Labels */
    .sec-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        color: #38bdf8;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }
    .sec-num {
        font-family: 'JetBrains Mono', monospace;
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 6px;
        padding: 0.1rem 0.4rem;
        font-size: 0.75rem;
    }

    /* Viewfinder / Scanner Frame */
    .scan-wrap {
        position: relative;
        background: rgba(13, 20, 36, 0.65);
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 14px;
        padding: 1.1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.2rem;
    }
    .scan-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.8rem;
    }
    .scan-status {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        color: #38bdf8;
        font-weight: 600;
    }
    .scan-dim {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        background: rgba(255, 255, 255, 0.05);
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        color: #94a3b8;
    }
    .scan-frame {
        position: relative;
        width: 100%;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(56, 189, 248, 0.25);
        display: flex;
        align-items: center;
        justify-content: center;
        background: #020617;
    }
    .scan-frame img {
        width: 100%;
        height: auto;
        display: block;
        border-radius: 8px;
    }
    /* Corner Brackets */
    .scan-corner {
        position: absolute;
        width: 16px;
        height: 16px;
        border: 2px solid #38bdf8;
        pointer-events: none;
        z-index: 2;
    }
    .scan-corner.tl { top: 6px; left: 6px; border-right: none; border-bottom: none; }
    .scan-corner.tr { top: 6px; right: 6px; border-left: none; border-bottom: none; }
    .scan-corner.bl { bottom: 6px; left: 6px; border-right: none; border-top: none; }
    .scan-corner.br { bottom: 6px; right: 6px; border-left: none; border-top: none; }

    /* Hero Diagnosis Card */
    .res-card {
        background: rgba(13, 20, 36, 0.78);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(56, 189, 248, 0.20);
        border-radius: 16px;
        padding: 1.35rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
        margin-bottom: 1.2rem;
    }
    .res-top {
        display: flex;
        align-items: center;
        gap: 1.2rem;
    }
    .gauge-wrap {
        width: 90px;
        height: 90px;
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .gauge-svg {
        transform: rotate(-90deg);
        width: 90px;
        height: 90px;
    }
    .gauge-val {
        position: absolute;
        text-align: center;
    }
    .gauge-num {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1;
    }
    .gauge-lbl {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.58rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 2px;
    }
    .res-details {
        flex-grow: 1;
    }
    .res-overline {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .res-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.55rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.15;
        margin: 0.15rem 0 0.35rem 0;
    }
    .res-badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        font-weight: 700;
        padding: 0.22rem 0.65rem;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .res-badge.malignant {
        background: rgba(239, 68, 68, 0.18);
        border: 1px solid #ef4444;
        color: #f87171;
    }
    .res-badge.benign {
        background: rgba(16, 185, 129, 0.18);
        border: 1px solid #10b981;
        color: #34d399;
    }
    .res-desc {
        font-size: 0.83rem;
        color: #94a3b8;
        line-height: 1.45;
        margin-top: 0.55rem;
        padding-top: 0.55rem;
        border-top: 1px solid rgba(255, 255, 255, 0.07);
    }

    /* Risk Spectrum Bar */
    .risk-wrap {
        background: rgba(13, 20, 36, 0.60);
        border: 1px solid rgba(56, 189, 248, 0.14);
        border-radius: 12px;
        padding: 0.95rem 1.1rem;
        margin-bottom: 1.2rem;
    }
    .risk-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        margin-bottom: 0.55rem;
    }
    .risk-title {
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .risk-score {
        font-weight: 700;
        color: #f8fafc;
    }
    .risk-bar-bg {
        position: relative;
        height: 10px;
        border-radius: 9999px;
        background: linear-gradient(90deg, #10b981 0%, #eab308 50%, #ef4444 100%);
        margin-bottom: 0.35rem;
    }
    .risk-cursor {
        position: absolute;
        top: -4px;
        width: 4px;
        height: 18px;
        background: #ffffff;
        border-radius: 2px;
        box-shadow: 0 0 8px #ffffff;
        transform: translateX(-50%);
        transition: left 0.4s ease;
    }
    .risk-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.70rem;
        color: #64748b;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Probability Rows */
    .prow {
        margin-bottom: 0.55rem;
    }
    .prow-meta {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        margin-bottom: 0.2rem;
    }
    .prow-name {
        font-weight: 600;
        color: #cbd5e1;
    }
    .prow-val {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: #94a3b8;
    }
    .prow-bar-bg {
        height: 6px;
        border-radius: 9999px;
        background: rgba(255, 255, 255, 0.08);
        overflow: hidden;
    }
    .prow-bar-fill {
        height: 100%;
        border-radius: 9999px;
    }

    /* Telemetry Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.55rem;
        margin-top: 1.1rem;
    }
    .stat-card {
        background: rgba(13, 20, 36, 0.55);
        border: 1px solid rgba(56, 189, 248, 0.12);
        border-radius: 8px;
        padding: 0.55rem;
        text-align: center;
    }
    .stat-lbl {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 0.15rem;
    }

    /* Pipeline Stages Flow */
    .pipe-wrap {
        background: rgba(13, 20, 36, 0.65);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 14px;
        padding: 1.1rem;
        margin-top: 1.6rem;
    }
    .pipe-flow {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.45rem;
        overflow-x: auto;
    }
    .pipe-node {
        flex: 1;
        min-width: 130px;
        background: rgba(2, 6, 23, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 10px;
        padding: 0.55rem;
        text-align: center;
    }
    .pipe-node img {
        width: 100%;
        height: 100px;
        object-fit: cover;
        border-radius: 6px;
        display: block;
        margin-bottom: 0.4rem;
    }
    .pipe-lbl {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 600;
        color: #94a3b8;
    }
    .parr {
        font-size: 1.3rem;
        color: #38bdf8;
        opacity: 0.6;
        user-select: none;
    }

    /* Footer Note */
    .footer-note {
        text-align: center;
        font-size: 0.76rem;
        color: #64748b;
        margin-top: 2rem;
        margin-bottom: 1.2rem;
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

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

def praproses_tahap(im_bgr):
    """Menghasilkan tahap-tahap citra untuk alur visualisasi pipeline."""
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

    im_crop_256 = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    
    # Shades of gray (p=6)
    x_raw = torch.from_numpy(cv2.cvtColor(im_crop_256, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    e = x_raw.clamp_min(1e-6).pow(6).mean(dim=(2, 3), keepdim=True).pow(1.0 / 6)
    e = e / e.norm(dim=1, keepdim=True)
    x_sog = (x_raw / (e * math.sqrt(3))).clamp(0, 1)
    im_sog_np = (x_sog[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    im_sog_bgr = cv2.cvtColor(im_sog_np, cv2.COLOR_RGB2BGR)

    return {
        "asli": cv2.resize(im_bgr, (IMAGE_SIZE, IMAGE_SIZE)),
        "abu": cv2.cvtColor(cv2.resize(abu, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_GRAY2BGR),
        "masker": cv2.cvtColor(cv2.resize(masker, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_GRAY2BGR),
        "roi": im_crop_256,
        "sog": im_sog_bgr,
        "x_sog_tensor": x_sog
    }

def ke_base64(im_bgr):
    _, buf = cv2.imencode(".jpg", im_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode("utf-8")

# ─── PEMUATAN MODEL & BOBOT SWA ───────────────────────────────────────────────
@st.cache_resource
def muat_model():
    if not MODEL_PATH.exists():
        st.error(f"Bobot model tidak ditemukan di: {MODEL_PATH}")
        return None
    model = ConvNeXtTinyMHSA(n_kelas=8)
    sd_half = torch.load(MODEL_PATH, map_location="cpu")
    sd_float = {k: v.float() if v.is_floating_point() else v for k, v in sd_half.items()}
    model.load_state_dict(sd_float)
    model.eval()
    return model

# ─── TOPBAR HEADER ────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar-wrap">
    <div class="topbar-left">
        <div class="topbar-logo">🔬</div>
        <div>
            <div class="topbar-title">DERMASPARK</div>
            <div class="topbar-sub">Deteksi Kanker Kulit Medis &bull; BIMA 2026 UTM</div>
        </div>
    </div>
    <div class="topbar-right">
        <span class="chip">ConvNeXtTiny-MHSA (85,26%)</span>
        <span class="chip chip-green">● ROI Otsu + SoG (p=6)</span>
        <span class="chip">Ambang 0,42</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── DUAL-COLUMN LAYOUT (4.5 : 5.5) ───────────────────────────────────────────
col_input, col_output = st.columns([4.5, 5.5], gap="large")

with col_input:
    st.markdown('<div class="sec-label"><span class="sec-num">01</span> INPUT CITRA DERMOSKOPI</div>', unsafe_allow_html=True)

    tab_demo, tab_upload = st.tabs(["Sampel Demo Riset", "Unggah Sendiri"])

    citra_bgr = None
    info_sampel = ""

    with tab_demo:
        daftar_demo = {
            "🔴 Melanoma (Ganas Agresif)": "Melanoma_ISIC_0000054.jpg",
            "🔴 Basal Cell Carcinoma (Kanker)": "BCC_ISIC_0024436.jpg",
            "🔴 Actinic Keratoses (Pra-Kanker)": "AKIEC_ISIC_0024329.jpg",
            "🟢 Melanocytic Nevi (Tahi Lalat Jinak)": "Nevus_ISIC_0000009.jpg",
            "🟢 Benign Keratosis (Lesi Jinak)": "BKL_ISIC_0012173.jpg"
        }
        pilihan = st.selectbox("Pilih Sampel Kurasi Riset:", list(daftar_demo.keys()), label_visibility="collapsed")
        file_path = SAMPLES_DIR / daftar_demo[pilihan]
        if file_path.exists():
            citra_bgr = cv2.imread(str(file_path))
            info_sampel = pilihan

    with tab_upload:
        up_file = st.file_uploader("Pilih berkas foto dermoskopi (JPG / PNG):", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if up_file is not None:
            bytes_data = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
            citra_bgr = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
            info_sampel = f"Unggahan: {up_file.name}"

    with st.expander("⚙️ Pengaturan Model & Prapemrosesan"):
        ambang = st.slider("Batas Ambang Keganasan (Threshold):", 0.0, 1.0, 0.42, 0.01, help="Ambang optimal riset: 0,42")
        opsi_tta = st.checkbox("Aktifkan TTA 4-Arah (Rotasi & Flip)", value=False)
        margin_pct = st.slider("Margin Kulit Sehat (%):", 0, 30, 15, 1)

    if citra_bgr is not None:
        t0_prep = time.time()
        tahapan = praproses_tahap(citra_bgr)
        waktu_prep = (time.time() - t0_prep) * 1000

        b64_asli = ke_base64(tahapan["asli"])
        st.markdown(f"""
        <div class="scan-wrap">
            <div class="scan-header">
                <div class="scan-status"><span style="color:#10b981;">●</span> CITRA TERPILIH</div>
                <div class="scan-dim">256×256px</div>
            </div>
            <div class="scan-frame">
                <div class="scan-corner tl"></div>
                <div class="scan-corner tr"></div>
                <div class="scan-corner bl"></div>
                <div class="scan-corner br"></div>
                <img src="data:image/jpeg;base64,{b64_asli}" alt="Preview" />
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_output:
    st.markdown('<div class="sec-label"><span class="sec-num">02</span> HASIL DIAGNOSIS MODEL</div>', unsafe_allow_html=True)

    if citra_bgr is not None:
        model = muat_model()
        if model is not None:
            # Normalisasi ImageNet
            mean_t = torch.tensor(NORM_MEAN).view(1, 3, 1, 1)
            std_t = torch.tensor(NORM_STD).view(1, 3, 1, 1)
            x_norm = (tahapan["x_sog_tensor"] - mean_t) / std_t

            t0_inf = time.time()
            with torch.no_grad():
                if opsi_tta:
                    vs = [x_norm, x_norm.flip(3), x_norm.flip(2), torch.rot90(x_norm, 1, [2, 3])]
                    logits = torch.stack([model(v) for v in vs]).mean(dim=0)
                else:
                    logits = model(x_norm)
                probs = F.softmax(logits, dim=1)[0].numpy()
            waktu_inf = (time.time() - t0_inf) * 1000

            # Hitung Kelas Teratas & Keganasan
            top_idx = int(np.argmax(probs))
            top_kelas = KELAS[top_idx]
            top_nama = KELAS_NAMA[top_kelas]
            top_prob = float(probs[top_idx])

            p_ganas = float(sum(probs[i] for i in MALIGNANT_IDX))
            is_ganas = p_ganas >= ambang

            # Circular Gauge Math (Radius 36, Perimeter = 2*pi*36 ~ 226)
            circ = 2 * math.pi * 36
            dashoffset = circ * (1.0 - top_prob)
            gauge_color = "#ef4444" if top_idx in MALIGNANT_IDX else "#10b981"

            badge_html = (
                '<span class="res-badge malignant">Ganas &bull; Malignant</span>'
                if is_ganas else
                '<span class="res-badge benign">Jinak &bull; Benign</span>'
            )

            # Hero Card
            st.markdown(f"""
            <div class="res-card">
                <div class="res-top">
                    <div class="gauge-wrap">
                        <svg class="gauge-svg" viewBox="0 0 90 90">
                            <circle cx="45" cy="45" r="36" fill="transparent" stroke="rgba(255,255,255,0.08)" stroke-width="7" />
                            <circle cx="45" cy="45" r="36" fill="transparent" stroke="{gauge_color}" stroke-width="7"
                                    stroke-dasharray="{circ:.1f}" stroke-dashoffset="{dashoffset:.1f}" stroke-linecap="round" />
                        </svg>
                        <div class="gauge-val">
                            <div class="gauge-num">{top_prob*100:.1f}%</div>
                            <div class="gauge-lbl">Keyakinan</div>
                        </div>
                    </div>
                    <div class="res-details">
                        <div class="res-overline">DIAGNOSIS TERDETEKSI</div>
                        <div class="res-name">{top_nama}</div>
                        {badge_html}
                    </div>
                </div>
                <div class="res-desc">
                    {KELAS_DESKRIPSI[top_kelas]}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Risk Spectrum Bar
            p_ganas_pct = min(100.0, max(0.0, p_ganas * 100))
            st.markdown(f"""
            <div class="risk-wrap">
                <div class="risk-header">
                    <span class="risk-title">SPEKTRUM PROBABILITAS KEGANASAN</span>
                    <span class="risk-score">{p_ganas*100:.1f}%</span>
                </div>
                <div class="risk-bar-bg">
                    <div class="risk-cursor" style="left: {p_ganas_pct}%;"></div>
                </div>
                <div class="risk-labels">
                    <span>Rendah (Jinak)</span>
                    <span style="color:#f59e0b;">Batas ({ambang*100:.0f}%)</span>
                    <span>Tinggi (Ganas)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability Breakdown Bars
            st.markdown('<div style="font-family:\'Space Grotesk\',sans-serif; font-size:0.82rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.6rem;">DISTRIBUSI KELAS LESI</div>', unsafe_allow_html=True)

            # Urutkan berdasarkan probabilitas
            urutan_kelas = np.argsort(probs)[::-1]
            for idx in urutan_kelas[:4]:  # Tampilkan 4 probabilitas teratas
                k = KELAS[idx]
                p = probs[idx]
                nama_singkat = KELAS_NAMA[k]
                bar_gradient = (
                    "linear-gradient(90deg, #f97316, #ef4444)" if idx in MALIGNANT_IDX else
                    "linear-gradient(90deg, #06b6d4, #10b981)"
                )
                st.markdown(f"""
                <div class="prow">
                    <div class="prow-meta">
                        <span class="prow-name">{nama_singkat}</span>
                        <span class="prow-val">{p*100:.1f}%</span>
                    </div>
                    <div class="prow-bar-bg">
                        <div class="prow-bar-fill" style="width: {p*100}%; background: {bar_gradient};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Telemetry Grid
            st.markdown(f"""
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-lbl">PREP</div>
                    <div class="stat-val">{waktu_prep:.0f} ms</div>
                </div>
                <div class="stat-card">
                    <div class="stat-lbl">INFERENSI</div>
                    <div class="stat-val">{waktu_inf:.0f} ms</div>
                </div>
                <div class="stat-card">
                    <div class="stat-lbl">BACKBONE</div>
                    <div class="stat-val">ConvNeXt</div>
                </div>
                <div class="stat-card">
                    <div class="stat-lbl">DEVICE</div>
                    <div class="stat-val">CPU</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Pilih sampel demo di sebelah kiri atau unggah foto dermoskopi untuk memicu diagnosis.")

# ─── SECTION 03: PIPELINE PRAPEMROSESAN & EKSTRAKSI FITUR ─────────────────────
if citra_bgr is not None and "tahapan" in locals():
    st.markdown("""
    <div class="pipe-wrap">
        <div class="sec-label" style="margin-bottom:0.85rem;"><span class="sec-num">03</span> ALUR PRAPEMROSESAN CITRA MEDIS & EKSTRAKSI FITUR</div>
        <div class="pipe-flow">
            <div class="pipe-node">
                <img src="data:image/jpeg;base64,{}" alt="1" />
                <div class="pipe-lbl">1 &bull; Citra Asli</div>
            </div>
            <div class="parr">&rsaquo;</div>
            <div class="pipe-node">
                <img src="data:image/jpeg;base64,{}" alt="2" />
                <div class="pipe-lbl">2 &bull; Gaussian Blur</div>
            </div>
            <div class="parr">&rsaquo;</div>
            <div class="pipe-node">
                <img src="data:image/jpeg;base64,{}" alt="3" />
                <div class="pipe-lbl">3 &bull; Masker Otsu</div>
            </div>
            <div class="parr">&rsaquo;</div>
            <div class="pipe-node">
                <img src="data:image/jpeg;base64,{}" alt="4" />
                <div class="pipe-lbl">4 &bull; ROI +15% Margin</div>
            </div>
            <div class="parr">&rsaquo;</div>
            <div class="pipe-node">
                <img src="data:image/jpeg;base64,{}" alt="5" />
                <div class="pipe-lbl">5 &bull; Shades of Gray</div>
            </div>
        </div>
    </div>
    """.format(
        ke_base64(tahapan["asli"]),
        ke_base64(tahapan["abu"]),
        ke_base64(tahapan["masker"]),
        ke_base64(tahapan["roi"]),
        ke_base64(tahapan["sog"])
    ), unsafe_allow_html=True)

# ─── FOOTER MINIMALIS ─────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-note">
    Alat bantu riset akademik (Hibah BIMA UTM 2026). Bukan pengganti diagnosis dokter spesialis patologi.
</div>
""", unsafe_allow_html=True)
