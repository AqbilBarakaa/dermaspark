import os
import sys
import time
import json
import math
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
    page_title="DERMASPARK - Sistem Skrining & Triase Lesi Kulit",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── STYLING PROFESIONAL MODERN (CLEAN, TIDAK BERANTAKAN) ────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    /* Global Base */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #0F172A;
    }

    .stApp {
        background-color: #F8FAFC;
    }

    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1220px !important;
    }

    /* Container Border Card Styling */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.04) !important;
        padding: 1.35rem !important;
    }

    /* Segmented Control Full-Width Horizontal */
    div[data-testid="stSegmentedControl"] {
        width: 100% !important;
        margin-bottom: 0.6rem !important;
    }
    div[data-testid="stSegmentedControl"] > div {
        display: flex !important;
        flex-direction: row !important;
        width: 100% !important;
    }
    div[data-testid="stSegmentedControl"] button {
        flex: 1 1 0 !important;
        text-align: center !important;
        padding: 0.40rem 0.5rem !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
    }

    /* Top Brand Bar */
    .brand-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 0.95rem 1.4rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.04);
    }
    .brand-left {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }
    .brand-icon-box {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.35rem;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.22);
    }
    .brand-name {
        font-size: 1.32rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #0F172A;
        line-height: 1.15;
    }
    .brand-tagline {
        font-size: 0.80rem;
        color: #64748B;
        font-weight: 500;
    }
    .brand-badges {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .chip-status {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: #F0FDF4;
        color: #166534;
        border: 1px solid #BBF7D0;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .dot-live {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #22C55E;
    }
    .chip-meta {
        display: inline-flex;
        align-items: center;
        background: #F1F5F9;
        color: #475569;
        border: 1px solid #E2E8F0;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Section Title */
    .section-title {
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        color: #475569;
        text-transform: uppercase;
        margin-bottom: 1.0rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #F1F5F9;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Hero Triage Card */
    .triage-hero {
        display: flex;
        align-items: center;
        gap: 1.25rem;
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 1.2rem;
    }
    .triage-hero.malignant {
        background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%);
        border: 1px solid #FECDD3;
        color: #881337;
    }
    .triage-hero.benign {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 1px solid #BBF7D0;
        color: #14532D;
    }
    .triage-left {
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .triage-right {
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
    }
    .triage-badge {
        display: inline-flex;
        align-items: center;
        width: fit-content;
        gap: 0.35rem;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
    }
    .triage-hero.malignant .triage-badge {
        background: #E11D48;
        color: #FFFFFF;
    }
    .triage-hero.benign .triage-badge {
        background: #059669;
        color: #FFFFFF;
    }
    .triage-title {
        font-size: 1.40rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }
    .triage-urgency {
        font-size: 0.82rem;
        font-weight: 700;
        margin-top: 0.1rem;
    }
    .triage-hero.malignant .triage-urgency {
        color: #BE123C;
    }
    .triage-hero.benign .triage-urgency {
        color: #047857;
    }
    .triage-desc {
        font-size: 0.86rem;
        line-height: 1.5;
        opacity: 0.95;
    }

    /* Spectrum Gauge */
    .spectrum-container {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.95rem 1.1rem;
        margin-bottom: 1.2rem;
    }
    .spectrum-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.6rem;
    }
    .spectrum-label {
        font-size: 0.80rem;
        font-weight: 700;
        color: #334155;
    }
    .spectrum-val {
        font-size: 0.86rem;
        font-weight: 800;
    }
    .spectrum-track {
        position: relative;
        height: 12px;
        background: linear-gradient(90deg, #10B981 0%, #F59E0B 42%, #EF4444 100%);
        border-radius: 9999px;
        margin: 0.4rem 0 0.8rem 0;
    }
    .spectrum-threshold-line {
        position: absolute;
        top: -4px;
        bottom: -4px;
        width: 2px;
        background: #0F172A;
        transform: translateX(-50%);
    }
    .threshold-tag {
        position: absolute;
        top: -18px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 0.64rem;
        font-weight: 700;
        color: #0F172A;
        white-space: nowrap;
        background: #FFFFFF;
        padding: 0.1rem 0.35rem;
        border-radius: 4px;
        border: 1px solid #CBD5E1;
    }
    .spectrum-pointer {
        position: absolute;
        top: -5px;
        width: 22px;
        height: 22px;
        background: #FFFFFF;
        border: 3.5px solid #0F172A;
        border-radius: 50%;
        transform: translateX(-50%);
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
        transition: left 0.3s ease;
    }
    .spectrum-footer {
        display: flex;
        justify-content: space-between;
        font-size: 0.70rem;
        color: #64748B;
        font-weight: 600;
    }

    /* Native Class Progress Rows */
    .prob-container {
        display: flex;
        flex-direction: column;
        gap: 0.45rem;
        margin-bottom: 0.9rem;
    }
    .prob-row {
        background: #FFFFFF;
        border: 1px solid #F1F5F9;
        border-radius: 9px;
        padding: 0.55rem 0.80rem;
        transition: all 0.15s ease;
    }
    .prob-row.active {
        border-color: #93C5FD;
        background: #F8FAFC;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.07);
    }
    .prob-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.30rem;
    }
    .prob-title-wrap {
        display: flex;
        align-items: center;
        gap: 0.40rem;
    }
    .prob-name {
        font-size: 0.82rem;
        font-weight: 700;
        color: #1E293B;
    }
    .prob-pill {
        font-size: 0.66rem;
        font-weight: 700;
        padding: 0.12rem 0.42rem;
        border-radius: 4px;
    }
    .pill-benign {
        background: #DCFCE7;
        color: #15803D;
    }
    .pill-malignant {
        background: #FEE2E2;
        color: #B91C1C;
    }
    .prob-value {
        font-size: 0.84rem;
        font-weight: 800;
        color: #0F172A;
    }
    .prob-track {
        width: 100%;
        height: 6px;
        background: #F1F5F9;
        border-radius: 9999px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 9999px;
    }

    /* 4 Bottom Micro-Stats */
    .quad-stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.65rem;
        margin-top: 1.1rem;
    }
    .quad-stat-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.65rem 0.75rem;
        text-align: center;
    }
    .quad-stat-label {
        font-size: 0.68rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .quad-stat-val {
        font-size: 1.05rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 0.15rem;
    }

    /* Minimal Modern Footer */
    .app-footer-clean {
        text-align: center;
        padding: 1.6rem 0;
        margin-top: 1.6rem;
        border-top: 1px solid #E2E8F0;
        color: #94A3B8;
        font-size: 0.80rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ─── PATH & KONSTANTA ─────────────────────────────────────────────────────────
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
    "MEL": "Melanoma (Kanker Ganas Agresif)",
    "NV": "Melanocytic Nevus (Tahi Lalat Jinak)",
    "SCC": "Squamous Cell Carcinoma (Kanker Kulit)",
    "VASC": "Vascular Lesion (Lesi Pembuluh Darah)"
}
KELAS_KATEGORI = {
    "AKIEC": "Ganas / Pra-Kanker",
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

# ─── PRA-PEMROSESAN CITRA ─────────────────────────────────────────────────────
DET, MARGIN, FRAC_MIN, FRAC_MAX = 512, 0.15, 0.03, 0.90

def proses_citra(im_bgr):
    t_start = time.time()
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
    prep_ms = (time.time() - t_start) * 1000

    return {
        "asli_rgb": cv2.cvtColor(cv2.resize(im_bgr, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2RGB),
        "masker_rgb": cv2.cvtColor(cv2.resize(masker, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_GRAY2RGB),
        "roi_rgb": im_crop_rgb,
        "sog_rgb": im_sog_rgb,
        "tensor_sog": x_sog,
        "prep_ms": prep_ms
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

def render_svg_donut(pct, is_ganas):
    color = "#E11D48" if is_ganas else "#059669"
    return (
        f'<svg width="96" height="96" viewBox="0 0 36 36">'
        f'<path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#E2E8F0" stroke-width="3.5" />'
        f'<path stroke-dasharray="{pct:.1f}, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="{color}" stroke-width="3.5" stroke-linecap="round" />'
        f'<text x="18" y="17" text-anchor="middle" font-size="7.5" font-weight="800" fill="#0F172A">{pct:.1f}%</text>'
        f'<text x="18" y="23" text-anchor="middle" font-size="3.2" font-weight="700" fill="#64748B">KEYAKINAN</text>'
        f'</svg>'
    )

def render_prob_bars(probs, top_idx):
    order = np.argsort(probs)[::-1]
    rows = []
    for idx in order:
        k = KELAS[idx]
        nama = KELAS_NAMA[k]
        kat = KELAS_KATEGORI[k]
        p = float(probs[idx])
        pct = p * 100
        is_active = (idx == top_idx)
        is_mal = (kat == "Ganas" or "Ganas" in kat)
        
        pill_cls = "pill-malignant" if is_mal else "pill-benign"
        bar_color = "#E11D48" if is_mal else "#2563EB"
        active_cls = " active" if is_active else ""
        
        row_str = (
            f'<div class="prob-row{active_cls}">'
            f'<div class="prob-header">'
            f'<div class="prob-title-wrap">'
            f'<span class="prob-name">{nama}</span>'
            f'<span class="prob-pill {pill_cls}">{kat}</span>'
            f'</div>'
            f'<div class="prob-value">{pct:.2f}%</div>'
            f'</div>'
            f'<div class="prob-track">'
            f'<div class="prob-fill" style="width:{pct:.2f}%;background-color:{bar_color};"></div>'
            f'</div>'
            f'</div>'
        )
        rows.append(row_str)
    return '<div class="prob-container">' + "".join(rows) + '</div>'

# ─── HEADER / BRAND BAR ───────────────────────────────────────────────────────
st.markdown("""
<div class="brand-bar">
    <div class="brand-left">
        <div class="brand-icon-box">🩺</div>
        <div>
            <div class="brand-name">DERMASPARK</div>
            <div class="brand-tagline">Sistem Skrining & Triase Citra Lesi Kulit</div>
        </div>
    </div>
    <div class="brand-badges">
        <span class="chip-status"><span class="dot-live"></span> Model Siap</span>
        <span class="chip-meta">ConvNeXtTiny-MHSA (85,26%)</span>
        <span class="chip-meta">8 Kelas ISIC</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── WORKSPACE DUA KOLOM ──────────────────────────────────────────────────────
col_input, col_report = st.columns([1, 1.25], gap="large")

with col_input:
    with st.container(border=True):
        st.markdown(
            '<div class="section-title">01 &bull; MASUKAN CITRA LESI</div>',
            unsafe_allow_html=True
        )

        metode_input = st.segmented_control(
            "Metode Masukan",
            options=["Sampel Demo", "Unggah Citra"],
            default="Sampel Demo",
            label_visibility="collapsed"
        )

        citra_bgr = None
        label_sumber = ""

        if metode_input == "Sampel Demo":
            pilihan_sampel = {
                "Melanoma (Kanker Ganas Agresif) - ISIC_0000054": "Melanoma_ISIC_0000054.jpg",
                "Basal Cell Carcinoma (Kanker Kulit) - ISIC_0024436": "BCC_ISIC_0024436.jpg",
                "Actinic Keratoses (Pra-Kanker) - ISIC_0024329": "AKIEC_ISIC_0024329.jpg",
                "Melanocytic Nevus (Tahi Lalat Jinak) - ISIC_0000009": "Nevus_ISIC_0000009.jpg",
                "Benign Keratosis (Lesi Jinak) - ISIC_0012173": "BKL_ISIC_0012173.jpg"
            }
            terpilih = st.selectbox(
                "Pilih Sampel Citra ISIC:",
                list(pilihan_sampel.keys()),
                help="Pilih sampel lesi kulit dari dataset standar ISIC."
            )
            berkas = SAMPLES_DIR / pilihan_sampel[terpilih]
            if berkas.exists():
                citra_bgr = cv2.imread(str(berkas))
                label_sumber = terpilih
        else:
            file_up = st.file_uploader(
                "Unggah foto dermoskopi (JPG / PNG):",
                type=["jpg", "jpeg", "png"],
                help="Mendukung citra dermoskopi resolusi tinggi."
            )
            if file_up is not None:
                buf = np.asarray(bytearray(file_up.read()), dtype=np.uint8)
                citra_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                label_sumber = f"File Unggahan: {file_up.name}"

        # Pratinjau Citra Terpilih
        if citra_bgr is not None:
            tahap = proses_citra(citra_bgr)
            st.image(tahap["asli_rgb"], caption=f"Pratinjau: {label_sumber}", use_container_width=True)

        # Pengaturan Ambang Batas Klinis
        with st.expander("⚙️ Pengaturan Ambang Batas Klinis (Threshold)", expanded=False):
            ambang = st.slider("Ambang Batas Keganasan:", 0.0, 1.0, 0.42, 0.01)
            st.caption("Batas ambang standar optimal: **0,42** (Sensitivitas 81,10% | Spesifisitas 87,66%). Nilai di atas ambang dikategorikan berisiko ganas.")

with col_report:
    with st.container(border=True):
        st.markdown(
            '<div class="section-title">02 &bull; HASIL SKRINING & TRIASE KLINIS</div>',
            unsafe_allow_html=True
        )

        if citra_bgr is not None:
            model = muat_model()
            if model is not None:
                mean_t = torch.tensor(NORM_MEAN).view(1, 3, 1, 1)
                std_t = torch.tensor(NORM_STD).view(1, 3, 1, 1)
                x_norm = (tahap["tensor_sog"] - mean_t) / std_t

                t0 = time.time()
                with torch.no_grad():
                    logits = model(x_norm)
                    probs = F.softmax(logits, dim=1)[0].numpy()
                durasi_inf_ms = (time.time() - t0) * 1000

                top_idx = int(np.argmax(probs))
                top_nama = KELAS_NAMA[KELAS[top_idx]]
                top_prob = float(probs[top_idx])

                prob_ganas = float(sum(probs[i] for i in MALIGNANT_IDX))
                is_ganas = prob_ganas >= ambang

                # 1. KARTU TRIASE HERO
                hero_cls = "malignant" if is_ganas else "benign"
                badge_text = "⚠️ Terindikasi Ganas (Malignant)" if is_ganas else "🛡️ Terindikasi Jinak (Benign)"
                urgency_text = "Tingkat Urgensi: TINGGI • Disarankan Rujukan Spesialis & Biopsi Jaringan" if is_ganas else "Tingkat Urgensi: RENDAH • Lesi Non-Malignan, Pemantauan Rutin (ABCDE)"
                advice_text = (
                    f"Model mendeteksi akumulasi potensi keganasan sebesar <strong>{prob_ganas*100:.1f}%</strong> "
                    f"(melebihi ambang batas {ambang:.2f}). Diperlukan evaluasi klinis langsung dengan dokter spesialis dermatologi."
                    if is_ganas else
                    f"Model mendeteksi akumulasi potensi keganasan rendah sebesar <strong>{prob_ganas*100:.1f}%</strong> "
                    f"(di bawah ambang batas {ambang:.2f}). Pasien disarankan melakukan observasi mandiri jika ada perubahan bentuk/warna."
                )

                svg_donut = render_svg_donut(top_prob * 100, is_ganas)

                st.markdown(
                    f'<div class="triage-hero {hero_cls}">'
                    f'<div class="triage-left">{svg_donut}</div>'
                    f'<div class="triage-right">'
                    f'<span class="triage-badge">{badge_text}</span>'
                    f'<div class="triage-title">{top_nama}</div>'
                    f'<div class="triage-urgency">{urgency_text}</div>'
                    f'<div class="triage-desc">{advice_text}</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # 2. PENGUKUR RISIKO KEGANASAN KOMPREHENSIF (OMNI-MALIGNANCY SPECTRUM GAUGE)
                tingkat_label = "Kritis" if prob_ganas >= 0.70 else ("Tinggi (Di Atas Ambang)" if prob_ganas >= ambang else ("Sedang" if prob_ganas >= 0.20 else "Rendah"))
                color_risk = "#E11D48" if is_ganas else "#059669"
                pointer_pct = min(max(prob_ganas * 100, 1.5), 98.5)
                thresh_pct = ambang * 100

                st.markdown(
                    f'<div class="spectrum-container">'
                    f'<div class="spectrum-header">'
                    f'<span class="spectrum-label">Spektrum Risiko Keganasan</span>'
                    f'<span class="spectrum-val" style="color:{color_risk};">{prob_ganas*100:.1f}% ({tingkat_label})</span>'
                    f'</div>'
                    f'<div class="spectrum-track">'
                    f'<div class="spectrum-threshold-line" style="left:{thresh_pct}%;"></div>'
                    f'<div class="spectrum-pointer" style="left:{pointer_pct}%;"></div>'
                    f'</div>'
                    f'<div class="spectrum-footer">'
                    f'<span>0% (Aman / Jinak)</span>'
                    f'<span style="color:#2563EB; font-weight:700;">▲ Ambang Klinis ({thresh_pct:.0f}%)</span>'
                    f'<span>100% (Malignan)</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # 3. RINCIAN 8 KELAS
                st.markdown("<div style='font-size:0.84rem; font-weight:700; color:#334155; margin-bottom:0.6rem;'>Distribusi Probabilitas Spektrum 8 Kelas:</div>", unsafe_allow_html=True)
                st.markdown(render_prob_bars(probs, top_idx), unsafe_allow_html=True)

                # 4. 4 KOTAK STAT MIKRO OPERASIONAL
                st.markdown(
                    f'<div class="quad-stat-grid">'
                    f'<div class="quad-stat-card"><div class="quad-stat-label">Pra-proses</div><div class="quad-stat-val">{tahap["prep_ms"]:.0f} ms</div></div>'
                    f'<div class="quad-stat-card"><div class="quad-stat-label">Inferensi</div><div class="quad-stat-val">{durasi_inf_ms:.0f} ms</div></div>'
                    f'<div class="quad-stat-card"><div class="quad-stat-label">Segmentasi</div><div class="quad-stat-val">Otsu ROI</div></div>'
                    f'<div class="quad-stat-card"><div class="quad-stat-label">Perangkat</div><div class="quad-stat-val">CPU</div></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("Pilih sampel lesi kulit di sebelah kiri atau unggah berkas foto untuk melihat hasil skrining.")

# ─── PIPELINE PRA-PEMROSESAN CITRA (CONTAINER TERSATUKAN) ─────────────────────
if citra_bgr is not None:
    with st.container(border=True):
        st.markdown(
            '<div class="section-title">03 &bull; TAHAPAN PRA-PEMROSESAN CITRA (ROI OTSU & NORMALISASI WARNA)</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.image(tahap["asli_rgb"], caption="1. Citra Asli (256×256)", use_container_width=True)
        with c2:
            st.image(tahap["masker_rgb"], caption="2. Masker Lingkaran & Otsu", use_container_width=True)
        with c3:
            st.image(tahap["roi_rgb"], caption="3. Segmentasi ROI (+15%)", use_container_width=True)
        with c4:
            st.image(tahap["sog_rgb"], caption="4. Normalisasi Warna (p=6)", use_container_width=True)

# ─── FOOTER BERSIH ────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer-clean">
    DERMASPARK &bull; Sistem Skrining & Triase Citra Lesi Kulit
</div>
""", unsafe_allow_html=True)
