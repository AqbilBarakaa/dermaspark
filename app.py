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
    page_title="DERMASPARK - Sistem Skrining Lesi Kulit",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── STYLING MODERN CLEAN MEDTECH UI ──────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #0F172A;
    }

    .stApp {
        background-color: #F8FAFC;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1200px !important;
    }

    /* Top Navigation / Brand Bar */
    .nav-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.0rem 1.4rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px -3px rgba(15, 23, 42, 0.04);
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }
    .nav-icon {
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
    .brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #0F172A;
        line-height: 1.1;
    }
    .brand-sub {
        font-size: 0.82rem;
        color: #64748B;
        font-weight: 500;
        margin-top: 0.15rem;
    }
    .nav-badges {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .status-badge {
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
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #22C55E;
    }
    .info-badge {
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

    /* Modern Card */
    .modern-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.35rem;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04);
        margin-bottom: 1.25rem;
    }
    .card-header-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.1rem;
        padding-bottom: 0.65rem;
        border-bottom: 1px solid #F1F5F9;
    }
    .card-title-group {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .card-title-text {
        font-size: 0.98rem;
        font-weight: 700;
        color: #1E293B;
    }
    .card-badge {
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        background: #EFF6FF;
        color: #2563EB;
    }

    /* Diagnosis Alert Box */
    .diag-box {
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 1.1rem;
        display: flex;
        flex-direction: column;
        gap: 0.45rem;
    }
    .diag-box.malignant {
        background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%);
        border: 1px solid #FECDD3;
        color: #9F1239;
    }
    .diag-box.benign {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 1px solid #BBF7D0;
        color: #166534;
    }
    .diag-badge {
        display: inline-flex;
        align-items: center;
        width: fit-content;
        gap: 0.35rem;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
    }
    .diag-box.malignant .diag-badge {
        background: #F43F5E;
        color: #FFFFFF;
    }
    .diag-box.benign .diag-badge {
        background: #10B981;
        color: #FFFFFF;
    }
    .diag-title {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }
    .diag-desc {
        font-size: 0.88rem;
        line-height: 1.55;
        opacity: 0.95;
    }

    /* Stat Grid */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
        margin-bottom: 1.2rem;
    }
    .stat-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.85rem;
        text-align: center;
    }
    .stat-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .stat-val {
        font-size: 1.25rem;
        font-weight: 800;
        margin-top: 0.25rem;
        color: #0F172A;
    }

    /* Native Probability Distribution */
    .prob-container {
        display: flex;
        flex-direction: column;
        gap: 0.55rem;
    }
    .prob-row {
        background: #FFFFFF;
        border: 1px solid #F1F5F9;
        border-radius: 10px;
        padding: 0.65rem 0.85rem;
        transition: all 0.15s ease;
    }
    .prob-row.active {
        border-color: #93C5FD;
        background: #F8FAFC;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.08);
    }
    .prob-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.35rem;
    }
    .prob-title-wrap {
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }
    .prob-name {
        font-size: 0.84rem;
        font-weight: 700;
        color: #1E293B;
    }
    .prob-pill {
        font-size: 0.68rem;
        font-weight: 600;
        padding: 0.15rem 0.45rem;
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
        font-size: 0.85rem;
        font-weight: 800;
        color: #0F172A;
    }
    .prob-track {
        width: 100%;
        height: 7px;
        background: #F1F5F9;
        border-radius: 9999px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 9999px;
    }

    /* Minimal Modern Footer */
    .modern-footer {
        text-align: center;
        padding: 1.5rem 0;
        margin-top: 1.5rem;
        border-top: 1px solid #E2E8F0;
        color: #94A3B8;
        font-size: 0.80rem;
        font-weight: 500;
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

# ─── PRA-PEMROSESAN CITRA ─────────────────────────────────────────────────────
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

def format_prob_rows(probs, top_idx):
    order = np.argsort(probs)[::-1]
    rows_html = []
    for idx in order:
        k = KELAS[idx]
        nama = KELAS_NAMA[k]
        kat = KELAS_KATEGORI[k]
        p = float(probs[idx])
        pct = p * 100
        is_active = (idx == top_idx)
        is_mal = (kat == "Ganas" or "Ganas" in kat)
        
        pill_cls = "pill-malignant" if is_mal else "pill-benign"
        bar_color = "#F43F5E" if is_mal else "#3B82F6"
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
        rows_html.append(row_str)
    return '<div class="prob-container">' + "".join(rows_html) + '</div>'

# ─── NAVBAR / HEADER MODERN ───────────────────────────────────────────────────
st.markdown("""
<div class="nav-bar">
    <div class="nav-brand">
        <div class="nav-icon">🩺</div>
        <div>
            <div class="brand-title">DERMASPARK</div>
            <div class="brand-sub">Sistem Skrining Citra Lesi Kulit</div>
        </div>
    </div>
    <div class="nav-badges">
        <span class="status-badge"><span class="status-dot"></span> Model Siap</span>
        <span class="info-badge">ConvNeXtTiny-MHSA</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── WORKSPACE DUA KOLOM ──────────────────────────────────────────────────────
col_input, col_report = st.columns([1, 1.25], gap="large")

with col_input:
    st.markdown(
        '<div class="modern-card">'
        '<div class="card-header-bar">'
        '<div class="card-title-group"><span class="card-title-text">Masukan Citra Dermoskopi</span></div>'
        '<span class="card-badge">Input</span>'
        '</div>',
        unsafe_allow_html=True
    )

    mode = st.radio(
        "Pilih Metode Masukan:",
        ["Gunakan Contoh Sampel", "Unggah Foto Lesi Kulit"],
        horizontal=True
    )

    citra_bgr = None
    keterangan_sampel = ""

    if mode == "Gunakan Contoh Sampel":
        pilihan_sampel = {
            "Melanoma (Kanker Ganas Agresif) - ISIC_0000054": "Melanoma_ISIC_0000054.jpg",
            "Basal Cell Carcinoma (Kanker Kulit) - ISIC_0024436": "BCC_ISIC_0024436.jpg",
            "Actinic Keratoses (Pra-Kanker) - ISIC_0024329": "AKIEC_ISIC_0024329.jpg",
            "Melanocytic Nevus (Tahi Lalat Jinak) - ISIC_0000009": "Nevus_ISIC_0000009.jpg",
            "Benign Keratosis (Lesi Jinak) - ISIC_0012173": "BKL_ISIC_0012173.jpg"
        }
        terpilih = st.selectbox("Pilih Sampel:", list(pilihan_sampel.keys()))
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

    with st.expander("⚙️ Ambang Batas Keputusan (Threshold)", expanded=False):
        ambang = st.slider("Batas Ambang Keganasan:", 0.0, 1.0, 0.42, 0.01)
        st.caption("Batas ambang standar: **0,42** (Sensitivitas 81,10% | Spesifisitas 87,66%).")

    st.markdown("</div>", unsafe_allow_html=True)

with col_report:
    st.markdown(
        '<div class="modern-card">'
        '<div class="card-header-bar">'
        '<div class="card-title-group"><span class="card-title-text">Hasil Skrining & Diagnosis</span></div>'
        '<span class="card-badge">Analysis</span>'
        '</div>',
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
            durasi_ms = (time.time() - t0) * 1000

            top_idx = int(np.argmax(probs))
            top_nama = KELAS_NAMA[KELAS[top_idx]]
            top_prob = float(probs[top_idx])

            prob_ganas = float(sum(probs[i] for i in MALIGNANT_IDX))
            is_ganas = prob_ganas >= ambang

            # Kotak Diagnosa Modern
            if is_ganas:
                st.markdown(
                    f'<div class="diag-box malignant">'
                    f'<div class="diag-badge">⚠️ Terindikasi Ganas (Malignant)</div>'
                    f'<div class="diag-title">{top_nama}</div>'
                    f'<div class="diag-desc">'
                    f'Skor akumulasi peluang keganasan sebesar <strong>{prob_ganas*100:.1f}%</strong> (melebihi ambang batas {ambang:.2f}). '
                    f'Disarankan untuk melakukan pemeriksaan lanjutan bersama dokter spesialis dermatologi.'
                    f'</div></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="diag-box benign">'
                    f'<div class="diag-badge">✓ Terindikasi Jinak (Benign)</div>'
                    f'<div class="diag-title">{top_nama}</div>'
                    f'<div class="diag-desc">'
                    f'Skor akumulasi peluang keganasan berada pada tingkat rendah <strong>{prob_ganas*100:.1f}%</strong> (di bawah ambang batas {ambang:.2f}). '
                    f'Lesi saat ini diprediksi non-malignan. Lakukan pemantauan berkala terhadap perubahan bentuk atau warna.'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

            # Baris 3 Stat Modern
            val_color = '#E11D48' if is_ganas else '#059669'
            st.markdown(
                f'<div class="stat-grid">'
                f'<div class="stat-card"><div class="stat-label">Probabilitas Ganas</div><div class="stat-val" style="color:{val_color};">{prob_ganas*100:.1f}%</div></div>'
                f'<div class="stat-card"><div class="stat-label">Keyakinan Utama</div><div class="stat-val" style="color:#2563EB;">{top_prob*100:.1f}%</div></div>'
                f'<div class="stat-card"><div class="stat-label">Latensi Model</div><div class="stat-val" style="color:#475569;">{durasi_ms:.0f} ms</div></div>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Rincian Probabilitas 8 Kelas (Native HTML/CSS Baris Modern)
            st.markdown("<div style='font-size:0.86rem; font-weight:700; color:#334155; margin-bottom:0.6rem;'>Distribusi Probabilitas per Kelas:</div>", unsafe_allow_html=True)
            st.markdown(format_prob_rows(probs, top_idx), unsafe_allow_html=True)
    else:
        st.info("Pilih sampel lesi kulit di sebelah kiri atau unggah berkas foto untuk melihat hasil skrining.")

    st.markdown("</div>", unsafe_allow_html=True)

# ─── TAHAPAN PRA-PEMROSESAN CITRA ─────────────────────────────────────────────
if citra_bgr is not None:
    st.markdown(
        '<div class="modern-card">'
        '<div class="card-header-bar">'
        '<div class="card-title-group"><span class="card-title-text">Tahapan Pra-pemrosesan Citra</span></div>'
        '<span class="card-badge">Pipeline</span>'
        '</div>',
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

    st.markdown("</div>", unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="modern-footer">
    DERMASPARK &bull; Platform Skrining Citra Lesi Kulit
</div>
""", unsafe_allow_html=True)
