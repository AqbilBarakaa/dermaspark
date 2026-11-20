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
import base64

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ─── KONFIGURASI HALAMAN ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="DERMASPARK - Sistem Skrining Citra Lesi Kulit",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── STYLING MODERN CLINICAL CLEAN ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #0F172A;
}

.stApp {
    background-color: #F8FAFC;
}

.block-container {
    padding-top: 4.2rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 2.0rem !important;
    padding-right: 2.0rem !important;
    max-width: 1360px !important;
    margin: 0 auto !important;
}

/* Header Aplikasi Medis */
.ds-header {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.15rem 1.6rem;
    margin-top: 0.25rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}
.ds-header-left {
    display: flex;
    align-items: center;
    gap: 0.95rem;
}
.ds-header-icon {
    font-size: 1.75rem;
    background: #EFF6FF;
    border: 1px solid #DBEAFE;
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 12px;
}
.ds-header-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.01em;
    line-height: 1.35;
}
.ds-header-subtitle {
    font-size: 0.82rem;
    color: #64748B;
    margin-top: 0.15rem;
}
.ds-header-badge {
    font-size: 0.76rem;
    font-weight: 600;
    color: #334155;
    background: #F1F5F9;
    border: 1px solid #E2E8F0;
    padding: 0.35rem 0.80rem;
    border-radius: 6px;
}

/* Penataan Kartu Container Streamlit */
[data-testid="stVerticalBlockBorderWrapper"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
    padding: 1.35rem 1.5rem !important;
    margin-bottom: 1.25rem !important;
}

/* Header Section per Tingkat */
.ds-section-header {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 0.88rem;
    font-weight: 700;
    color: #1E293B;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding-bottom: 0.65rem;
    margin-bottom: 1.1rem;
    border-bottom: 1px solid #F1F5F9;
}
.ds-step-num {
    background: #2563EB;
    color: #FFFFFF;
    font-size: 0.72rem;
    width: 22px;
    height: 22px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    font-weight: 800;
}

/* Frame Pratinjau Citra */
[data-testid="stImage"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
    text-align: center !important;
    margin: 0 auto !important;
}
[data-testid="stImage"] > img, [data-testid="stImage"] img {
    margin-left: auto !important;
    margin-right: auto !important;
    display: block !important;
    max-height: 230px !important;
    width: auto !important;
    max-width: 100% !important;
    object-fit: contain !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}
.ds-image-caption {
    font-size: 0.78rem;
    color: #64748B;
    margin-top: 0.45rem;
    font-weight: 500;
    text-align: center;
}

/* Tingkat 2: Banner Diagnosis Utama */
.ds-diagnosis-card {
    border-radius: 12px;
    padding: 1.35rem 1.5rem;
    margin-bottom: 1.15rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1.0rem;
}
.ds-diagnosis-card.malignant {
    background: #FFF1F2;
    border: 1px solid #FECDD3;
    border-left: 6px solid #E11D48;
}
.ds-diagnosis-card.benign {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-left: 6px solid #10B981;
}

.ds-diag-badge {
    display: inline-block;
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    margin-bottom: 0.45rem;
}
.ds-diag-badge.malignant {
    background: #FFE4E6;
    color: #9F1239;
}
.ds-diag-badge.benign {
    background: #DCFCE7;
    color: #166534;
}

.ds-diag-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.25;
    margin-bottom: 0.35rem;
}
.ds-diag-urgency {
    font-size: 0.85rem;
    font-weight: 600;
}
.ds-diag-urgency.malignant {
    color: #BE123C;
}
.ds-diag-urgency.benign {
    color: #15803D;
}

.ds-score-box {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 0.80rem 1.15rem;
    text-align: right;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    border: 1px solid #E2E8F0;
    min-width: 160px;
}
.ds-score-label {
    font-size: 0.72rem;
    color: #64748B;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.ds-score-value {
    font-size: 1.60rem;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.1;
    margin: 0.15rem 0;
}
.ds-score-tag {
    font-size: 0.70rem;
    font-weight: 700;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    display: inline-block;
}

/* Tingkat 2: Visual Risk Meter */
.ds-risk-meter-container {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.15rem 1.35rem;
}
.ds-meter-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.60rem;
}
.ds-meter-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #1E293B;
}
.ds-meter-stat {
    font-size: 0.82rem;
    font-weight: 700;
}
.ds-meter-track {
    position: relative;
    height: 16px;
    background: #E2E8F0;
    border-radius: 8px;
    overflow: hidden;
    display: flex;
}
.ds-meter-zone-safe {
    width: 42%;
    background: #DCFCE7;
    height: 100%;
    border-right: 2px dashed #64748B;
}
.ds-meter-zone-alert {
    width: 58%;
    background: #FEE2E2;
    height: 100%;
}
.ds-meter-fill {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    border-radius: 8px;
    transition: width 0.4s ease;
}
.ds-meter-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: #64748B;
    margin-top: 0.40rem;
    font-weight: 600;
}
.ds-meter-explanation {
    font-size: 0.82rem;
    color: #475569;
    margin-top: 0.60rem;
    line-height: 1.5;
}

/* Tingkat 3: Kartu Distribusi & Info Klinis */
.ds-dist-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.70rem 0.90rem;
    margin-bottom: 0.55rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.ds-dist-card.top1 {
    background: #EFF6FF;
    border-color: #BFDBFE;
}
.ds-dist-name {
    font-size: 0.84rem;
    font-weight: 600;
    color: #1E293B;
}
.ds-dist-meta {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.ds-dist-pct {
    font-size: 0.88rem;
    font-weight: 700;
    color: #0F172A;
    min-width: 54px;
    text-align: right;
}

/* Badge Kategori */
.badge-ganas {
    background: #FEE2E2;
    color: #991B1B;
    font-size: 0.70rem;
    font-weight: 700;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
}
.badge-jinak {
    background: #DCFCE7;
    color: #166534;
    font-size: 0.70rem;
    font-weight: 700;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
}

/* Detail Klinis */
.ds-clinical-block {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.05rem;
    margin-bottom: 0.85rem;
}
.ds-clinical-heading {
    font-size: 0.88rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 0.55rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid #E2E8F0;
}
.ds-clinical-p {
    font-size: 0.82rem;
    color: #334155;
    line-height: 1.55;
    margin-bottom: 0.50rem;
}
.ds-clinical-label {
    font-weight: 700;
    color: #0F172A;
}

/* Kotak ABCDE */
.ds-abcde-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.45rem;
    margin-top: 0.60rem;
}
.ds-abcde-item {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 0.50rem 0.40rem;
    text-align: center;
}
.ds-abcde-letter {
    font-size: 1.0rem;
    font-weight: 800;
    color: #2563EB;
    line-height: 1.0;
    margin-bottom: 0.15rem;
}
.ds-abcde-title {
    font-size: 0.70rem;
    font-weight: 700;
    color: #0F172A;
}
.ds-abcde-desc {
    font-size: 0.64rem;
    color: #64748B;
    line-height: 1.3;
    margin-top: 0.15rem;
}

/* Preprocessing Inspector Styling */
.ds-prep-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.75rem;
    height: 100%;
    display: flex;
    flex-direction: column;
}
.ds-prep-step-badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    background: #EFF6FF;
    color: #1D4ED8;
    margin-bottom: 0.35rem;
}
.ds-prep-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 0.35rem;
}
.ds-prep-metric-pill {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 0.40rem 0.55rem;
    font-size: 0.72rem;
    color: #334155;
    margin-top: 0.45rem;
    line-height: 1.45;
}
.ds-prep-metric-pill strong {
    color: #0F172A;
}
.ds-prep-desc {
    font-size: 0.70rem;
    color: #64748B;
    line-height: 1.35;
    margin-top: 0.35rem;
}

/* Tabel Ablasi Preprocessing */
.ds-ablation-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78rem;
    margin-top: 0.5rem;
}
.ds-ablation-table th {
    background: #F8FAFC;
    border-bottom: 1px solid #E2E8F0;
    padding: 0.45rem 0.60rem;
    text-align: left;
    font-weight: 700;
    color: #475569;
}
.ds-ablation-table td {
    border-bottom: 1px solid #F1F5F9;
    padding: 0.45rem 0.60rem;
    color: #334155;
}
.ds-ablation-table tr.active-row {
    background: #EFF6FF;
    font-weight: 700;
    color: #1E40AF;
}

/* Disclaimer Medis Klinis */
.ds-disclaimer-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 0.70rem 0.95rem;
    font-size: 0.75rem;
    color: #475569;
    line-height: 1.5;
    margin-top: 1.0rem;
    display: flex;
    align-items: center;
    gap: 0.65rem;
}

/* Footer Aplikasi */
.ds-footer {
    text-align: center;
    padding-top: 1.5rem;
    margin-top: 1.75rem;
    font-size: 0.78rem;
    color: #94A3B8;
    border-top: 1px solid #E2E8F0;
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
    "AKIEC": "Actinic Keratosis (Lesi Pra-Kanker)",
    "BCC": "Basal Cell Carcinoma (Kanker Kulit)",
    "BKL": "Benign Keratosis (Lesi Jinak)",
    "DF": "Dermatofibroma (Lesi Jinak)",
    "MEL": "Melanoma (Kanker Ganas Agresif)",
    "NV": "Melanocytic Nevus (Tahi Lalat Jinak)",
    "SCC": "Squamous Cell Carcinoma (Kanker Kulit)",
    "VASC": "Vascular Lesion (Lesi Pembuluh Darah)"
}
KELAS_KATEGORI = {
    "AKIEC": "Ganas",
    "BCC": "Ganas",
    "BKL": "Jinak",
    "DF": "Jinak",
    "MEL": "Ganas",
    "NV": "Jinak",
    "SCC": "Ganas",
    "VASC": "Jinak"
}

PENJELASAN_MEDIS = {
    "MEL": {
        "nama_lengkap": "Melanoma Maligna (Kanker Kulit Ganas)",
        "definisi": "Kanker kulit paling agresif dan berbahaya yang berkembang dari sel melanosit (penghasil pigmen). Memiliki potensi metastasis (penyebaran) cepat ke kelenjar getah bening dan organ tubuh lain jika tidak ditangani sejak dini.",
        "ciri_klinis": "Bentuk asimetris, batas tepi bergerigi/tidak teratur, variasi warna tidak merata (hitam, cokelat tua, kebiruan), diameter seringkali >6 mm, dan terus mengalami perubahan (kriteria ABCDE).",
        "panduan_tindakan": "Segera lakukan rujukan ke dokter spesialis dermatologi untuk pemeriksaan dermoskopi digital lanjutan, biopsi eksisi diagnostik, dan penentuan kedalaman invasi Breslow."
    },
    "BCC": {
        "nama_lengkap": "Basal Cell Carcinoma (Kanker Sel Basal)",
        "definisi": "Kanker kulit non-melanoma paling umum di dunia, berkembang lambat dari lapisan basal epidermis akibat akumulasi radiasi ultraviolet (UV) kronis.",
        "ciri_klinis": "Papula mengkilap seperti mutiara (pearly papule), pelebaran pembuluh darah halus (telangiektasia), tepi menggulung, atau ulserasi sentral yang mudah berdarah dan sulit sembuh.",
        "panduan_tindakan": "Jarang bermetastasis jauh namun bersifat destruktif lokal terhadap jaringan lunak dan tulang di sekitarnya. Dianjurkan konsultasi untuk tindakan bedah eksisi atau bedah mikrografi Mohs."
    },
    "SCC": {
        "nama_lengkap": "Squamous Cell Carcinoma (Kanker Sel Skuamosa)",
        "definisi": "Kanker kulit ganas kedua paling umum, berasal dari keratinosit lapisan epidermis. Berkembang akibat paparan sinar matahari jangka panjang atau transformasi dari lesi pra-kanker sebelumnya.",
        "ciri_klinis": "Plak kemerahan menebal bersisik tebal, nodul keras yang mudah berdarah saat tergores, atau luka berkeropeng kronis pada area kulit yang sering terpapar sinar matahari.",
        "panduan_tindakan": "Memiliki risiko penyebaran 2-5% ke kelenjar getah bening regional. Memerlukan pemeriksaan histopatologi melalui biopsi serta tindakan eksisi bedah kuratif."
    },
    "AKIEC": {
        "nama_lengkap": "Actinic Keratosis / Bowen's Disease (Lesi Pra-Kanker)",
        "definisi": "Lesi pra-kanker (pre-malignant) pada epidermis akibat kerusakan DNA sel oleh radiasi UV matahari. Sekitar 10-20% lesi ini berpotensi bermutasi menjadi Kanker Sel Skuamosa (SCC) invasif bila tidak ditangani.",
        "ciri_klinis": "Bercak kasar berpasir saat diraba, kering, bersisik, berwarna kemerahan atau kecokelatan, sering timbul di wajah, dahi, daun telinga, bibir, atau punggung tangan.",
        "panduan_tindakan": "Dianjurkan konsultasi dermatologi untuk terapi ablasi preventif sebelum berkembang menjadi karsinoma invasif (krioterapi nitrogen cair, krim topikal 5-fluorouracil, atau terapi fotodinamik)."
    },
    "NV": {
        "nama_lengkap": "Melanocytic Nevus (Tahi Lalat Jinak)",
        "definisi": "Proliferasi jinak dari melanosit. Merupakan tumor jinak kulit yang sangat lazim dan non-karsinogenik yang dimiliki oleh hampir setiap individu.",
        "ciri_klinis": "Bentuk simetris melingkar atau oval, batas tepi tegas dan teratur, pigmentasi seragam (cokelat muda, cokelat tua, atau hitam merata), permukaan halus atau sedikit menonjol.",
        "panduan_tindakan": "Kondisi jinak murni dan tidak berbahaya. Cukup lakukan pemantauan mandiri berkala (aturan ABCDE). Segera periksakan jika tahi lalat membesar cepat, berubah warna, gatal, atau berdarah spontan."
    },
    "BKL": {
        "nama_lengkap": "Benign Keratosis / Seborrheic Keratosis (Lesi Jinak)",
        "definisi": "Pertumbuhan tumor jinak keratinosit epidermis yang sangat umum terjadi seiring pertambahan usia dan bersifat herediter.",
        "ciri_klinis": "Tampak seperti 'menempel' pada kulit (stuck-on appearance), permukaan berlilin, kasar atau berkeropeng, dengan warna bervariasi dari cokelat kekuningan hingga hitam pekat.",
        "panduan_tindakan": "Kondisi sepenuhnya jinak dan tidak memiliki potensi keganasan. Pengangkatan dengan krioterapi atau kuretase hanya dilakukan jika lesi teriritasi oleh gesekan pakaian atau untuk pertimbangan estetika."
    },
    "DF": {
        "nama_lengkap": "Dermatofibroma (Lesi Jaringan Ikat Jinak)",
        "definisi": "Nodul kutan jinak fibrositik pada dermis yang sering terbentuk sebagai reaksi respons terhadap mikrotrauma atau gigitan serangga.",
        "ciri_klinis": "Nodul keras padat berukuran 0.5 - 1 cm berwarna cokelat kemerahan yang memperlihatkan lekukan ke dalam saat ditekan atau dijepit dari kedua sisi (dimple sign / pinch sign positif).",
        "panduan_tindakan": "Bersifat jinak dan tidak memerlukan penanganan khusus kecuali jika menimbulkan nyeri atau rasa mengganjal."
    },
    "VASC": {
        "nama_lengkap": "Vascular Lesion / Cherry Angioma (Lesi Pembuluh Darah)",
        "definisi": "Proliferasi jinak endotel pembuluh darah kapiler superfisial, lazim ditemukan pada orang dewasa dan dapat bertambah seiring usia.",
        "ciri_klinis": "Papula kecil berwarna merah cerah atau keunguan, berbatas tegas, halus, dan memudar sesaat jika ditekan (blanching).",
        "panduan_tindakan": "Sangat jinak dan aman. Tindakan elektrokauter atau terapi laser hanya diperlukan jika mudah berdarah akibat trauma berulang atau alasan estetika."
    }
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

# ─── HITUNG GRAD-CAM (VISUAL EXPLAINABILITY) ──────────────────────────────────
def hitung_gradcam(model, x_norm, top_idx, im_roi_rgb):
    gradients = []
    activations = []

    def hook_act(module, input, output):
        activations.append(output)

    def hook_grad(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_act = model.features.register_forward_hook(hook_act)
    handle_grad = model.features.register_full_backward_hook(hook_grad)

    x_clone = x_norm.clone().detach().requires_grad_(True)
    logits = model(x_clone)

    model.zero_grad()
    loss = logits[0, top_idx]
    loss.backward()

    handle_act.remove()
    handle_grad.remove()

    if not activations or not gradients:
        return None, None

    act = activations[0].detach()
    grad = gradients[0].detach()
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * act, dim=1).squeeze(0)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam_np = cam.cpu().numpy()

    cam_resized = cv2.resize((cam_np * 255).astype(np.uint8), (IMAGE_SIZE, IMAGE_SIZE))
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(im_roi_rgb, 0.65, heatmap_rgb, 0.35, 0)
    return heatmap_rgb, overlay

# ─── PIPELINE PRA-PEMROSESAN CITRA ────────────────────────────────────────────
def proses_citra(im_bgr):
    t_start = time.time()
    h, w = im_bgr.shape[:2]
    DET = 256
    im_det = cv2.resize(im_bgr, (DET, DET), interpolation=cv2.INTER_AREA)

    # Deteksi lingkaran dermoskopi
    gray_det = cv2.cvtColor(im_det, cv2.COLOR_BGR2GRAY)
    _, mask_circle = cv2.threshold(gray_det, 25, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(mask_circle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    lingkaran_ada = False
    radius_val = 0.0
    if cnts:
        c_max = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c_max)
        if area > 0.35 * DET * DET:
            (cx, cy), radius = cv2.minEnclosingCircle(c_max)
            if radius > 0.35 * DET:
                lingkaran_ada = True
                radius_val = float(radius)

    # Segmentasi Otsu
    gray = cv2.cvtColor(im_det, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_val, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if lingkaran_ada:
        mask_lingkaran = np.zeros((DET, DET), dtype=np.uint8)
        cv2.circle(mask_lingkaran, (int(cx), int(cy)), int(radius * 0.92), 255, -1)
        thresh = cv2.bitwise_and(thresh, mask_lingkaran)

    masker = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts, _ = cv2.findContours(masker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crop = im_bgr
    roi_box = (0, 0, w, h)
    noise_reduced_pct = 0.0

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 0.01 * DET * DET:
            x, y, bw, bh = cv2.boundingRect(c)
            MARGIN = 0.15
            sisi = int(max(bw, bh) * (1 + 2 * MARGIN))
            cx_box, cy_box = x + bw // 2, y + bh // 2
            x0, y0 = max(0, cx_box - sisi // 2), max(0, cy_box - sisi // 2)
            x1, y1 = min(DET, x0 + sisi), min(DET, y0 + sisi)
            sx, sy = w / DET, h / DET
            crop_cand = im_bgr[int(y0 * sy):int(y1 * sy), int(x0 * sx):int(x1 * sx)]
            if crop_cand.size > 0:
                crop = crop_cand
                roi_box = (int(x0 * sx), int(y0 * sy), int((x1 - x0) * sx), int((y1 - y0) * sy))
                crop_area = crop.shape[0] * crop.shape[1]
                orig_area = h * w
                noise_reduced_pct = max(0.0, (1.0 - (crop_area / orig_area)) * 100)

    im_crop_rgb = cv2.cvtColor(cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    
    # Shades of gray (p=6)
    x_raw = torch.from_numpy(im_crop_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    e = x_raw.clamp_min(1e-6).pow(6).mean(dim=(2, 3), keepdim=True).pow(1.0 / 6)
    e = e / e.norm(dim=1, keepdim=True)
    x_sog = (x_raw / (e * math.sqrt(3))).clamp(0, 1)
    im_sog_rgb = (x_sog[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    prep_ms = (time.time() - t_start) * 1000

    e_vec = [float(e[0, 0, 0, 0]), float(e[0, 1, 0, 0]), float(e[0, 2, 0, 0])]

    # Kualitas Citra Input (Pre-flight Check)
    laplacian_var = float(cv2.Laplacian(gray_det, cv2.CV_64F).var())
    contrast_std = float(gray_det.std())
    lulus_kualitas = (laplacian_var >= 35.0) and (contrast_std >= 12.0)
    pesan_kualitas = "Kualitas Citra Memenuhi Standar Dermoskopi (Fokus & Kontras Optimal)" if lulus_kualitas else "Peringatan: Potensi Citra Kurang Fokus / Kontras Rendah"
    kualitas_dict = {
        "lulus": lulus_kualitas,
        "pesan": pesan_kualitas,
        "skor_fokus": laplacian_var,
        "skor_kontras": contrast_std
    }

    return {
        "asli_rgb": cv2.cvtColor(cv2.resize(im_bgr, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2RGB),
        "orig_w": w,
        "orig_h": h,
        "masker_rgb": cv2.cvtColor(cv2.resize(masker, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_GRAY2RGB),
        "roi_rgb": im_crop_rgb,
        "sog_rgb": im_sog_rgb,
        "tensor_sog": x_sog,
        "prep_ms": prep_ms,
        "otsu_val": float(otsu_val),
        "lingkaran_ada": lingkaran_ada,
        "radius_val": radius_val,
        "roi_box": roi_box,
        "noise_reduced_pct": noise_reduced_pct,
        "e_vec": e_vec,
        "kualitas": kualitas_dict
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

# ─── HEADER UTAMA ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="ds-header">
    <div class="ds-header-left">
        <div class="ds-header-icon" title="Analisis Histopatologi & AI">🔬</div>
        <div>
            <div class="ds-header-title">DERMASPARK</div>
            <div class="ds-header-subtitle">Sistem Skrining & Triase Citra Lesi Kulit berbasis Deep Learning</div>
        </div>
    </div>
    <div class="ds-header-badge">ConvNeXtTiny-MHSA &bull; 8 Kelas ISIC</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TINGKAT 1: MASUKAN CITRA & PENGATURAN KLINIS (ATAS)
# ═══════════════════════════════════════════════════════════════════════════════
citra_bgr = None
label_sumber = ""

with st.container(border=True):
    st.markdown("""
    <div class="ds-section-header">
        <span class="ds-step-num">1</span> Masukan Citra Dermoskopi & Pengaturan
    </div>
    """, unsafe_allow_html=True)

    col_ctrl, col_prev = st.columns([1.08, 1.0], gap="large")

    with col_ctrl:
        mode = st.radio(
            "Pilih Sumber Citra Lesi:",
            ["Pilih Sampel Klinis Terverifikasi", "Unggah Berkas Citra Baru"],
            horizontal=True
        )

        if mode == "Pilih Sampel Klinis Terverifikasi":
            pilihan_sampel = {
                "Melanoma (Kanker Ganas Agresif)": "Melanoma_ISIC_0024310.jpg",
                "Basal Cell Carcinoma (Kanker Kulit)": "BCC_ISIC_0024436.jpg",
                "Actinic Keratosis (Lesi Pra-Kanker)": "AKIEC_ISIC_0024329.jpg",
                "Kasus Borderline / Ambigu (Uji Ambang ~51.5%)": "BCC_Borderline_ISIC_0011099.jpg",
                "Melanocytic Nevus (Tahi Lalat Jinak)": "Nevus_ISIC_0000009.jpg",
                "Benign Keratosis (Lesi Jinak)": "BKL_ISIC_0012173.jpg"
            }
            terpilih = st.selectbox("Pilih Sampel Lesi Uji:", list(pilihan_sampel.keys()))
            berkas = SAMPLES_DIR / pilihan_sampel[terpilih]
            if berkas.exists():
                citra_bgr = cv2.imread(str(berkas))
                label_sumber = terpilih
        else:
            file_up = st.file_uploader("Unggah berkas foto lesi kulit (JPG / PNG):", type=["jpg", "jpeg", "png"])
            if file_up is not None:
                buf = np.asarray(bytearray(file_up.read()), dtype=np.uint8)
                citra_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                label_sumber = f"Berkas: {file_up.name}"

        with st.expander("⚙️ Ambang Batas Triase Klinis (Threshold)", expanded=False):
            if "ambang_keganasan" not in st.session_state:
                st.session_state.ambang_keganasan = 0.42

            def reset_ambang_callback():
                st.session_state.ambang_keganasan = 0.42

            col_sl, col_rst = st.columns([3, 1])
            with col_sl:
                ambang = st.slider(
                    "Batas Keganasan Minimum:",
                    min_value=0.01,
                    max_value=0.99,
                    step=0.01,
                    key="ambang_keganasan",
                    help="Ambang batas probabilitas keganasan untuk memicu alarm rujukan spesialis."
                )
            with col_rst:
                st.write("")
                st.button(
                    "Reset (0.42)",
                    on_click=reset_ambang_callback,
                    help="Kembalikan ke ambang batas optimal terkalibrasi riset (0.42)"
                )

            # Indikator Mode Operasional Triase
            if ambang <= 0.30:
                mode_label = "🎯 Mode Sensitivitas Tinggi (Skrining Ketat — Minimalkan Lolos Kanker)"
            elif ambang <= 0.55:
                mode_label = "⚖️ Mode Optimal Terkalibrasi Riset (0,42 — Keseimbangan F1-Score)"
            else:
                mode_label = "🛡️ Mode Spesifisitas Tinggi (Konservatif — Minimalkan Alarm Palsu)"

            st.caption(
                f"**{mode_label}**<br>"
                f"Ambang rujukan aktif: **{ambang:.2f}** ({ambang*100:.0f}%). "
                f"Alarm rujukan triase aktif jika potensi keganasan $\ge {ambang:.2f}$.<br>"
                f"<span style='color: #475569;'>💡 <em>Catatan: Ambang ini mengontrol kebijakan rujukan faskes (Sensitivitas vs Spesifisitas), bukan mengubah hasil diagnosa biologis model AI. Uji sampel 'Kasus Borderline / Ambigu' untuk melihat peralihan alarm di sekitar ambang 0,51.</em></span>",
                unsafe_allow_html=True
            )

    with col_prev:
        if citra_bgr is not None:
            tahap = proses_citra(citra_bgr)
            _, buf_img = cv2.imencode('.jpg', citra_bgr)
            img_b64 = base64.b64encode(buf_img).decode('utf-8')
            
            orig_w = tahap["orig_w"]
            orig_h = tahap["orig_h"]
            ar = orig_w / max(orig_h, 1)

            # Dimensi pratinjau klinis optimal: proporsional penuh diperbesar sejajar tinggi panel saat ambang dibuka
            target_h = 390
            target_w = int(round(target_h * ar))
            if target_w > 525:
                target_w = 525
                target_h = int(round(target_w / ar))

            # Pra-pemeriksaan Kualitas Citra (Pre-flight Quality Check)
            q = tahap["kualitas"]
            q_bg = "#ECFDF5" if q["lulus"] else "#FFFBEB"
            q_border = "#A7F3D0" if q["lulus"] else "#FDE68A"
            q_color = "#065F46" if q["lulus"] else "#92400E"
            q_icon = "✓" if q["lulus"] else "⚠️"
            
            st.markdown(f"""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; width: 100%; padding: 0.2rem 0 1.25rem 0;">
                <div style="display: inline-block; padding: 6px; background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 12px; box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08); max-width: 100%;">
                    <img src="data:image/jpeg;base64,{img_b64}" style="width: {target_w}px; max-width: 100%; height: auto; aspect-ratio: {orig_w} / {orig_h}; display: block; border-radius: 8px; object-fit: contain;" />
                </div>
                <div style="font-size: 0.80rem; color: #64748B; font-weight: 600; margin-top: 0.55rem; text-align: center;">
                    Pratinjau: <strong>{label_sumber}</strong> ({tahap["orig_w"]}×{tahap["orig_h"]} px)
                </div>
                <div style="background: {q_bg}; border: 1px solid {q_border}; border-radius: 9999px; padding: 0.38rem 0.95rem; margin-top: 0.45rem; font-size: 0.74rem; color: {q_color}; font-weight: 600; display: inline-flex; align-items: center; justify-content: center; gap: 0.45rem; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);">
                    <span>{q_icon}</span>
                    <span>{q["pesan"]} (Skor Fokus: {q['skor_fokus']:.1f})</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 12px; padding: 2.8rem 1.0rem; color: #94A3B8; text-align: center; max-width: 320px; margin: 0 auto;">
                <div style="font-size: 2.2rem; margin-bottom: 0.4rem;">📷</div>
                <div style="font-size: 0.86rem; font-weight: 600; color: #64748B;">Belum Ada Citra Terpilih</div>
                <div style="font-size: 0.76rem; color: #94A3B8;">Pilih sampel klinis atau unggah berkas lesi baru</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# INFERENSI MODEL & EVALUASI
# ═══════════════════════════════════════════════════════════════════════════════
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
        top_k = KELAS[top_idx]
        top_nama = KELAS_NAMA[top_k]
        top_prob = float(probs[top_idx])
        top_kat = KELAS_KATEGORI[top_k]

        prob_ganas = float(sum(probs[i] for i in MALIGNANT_IDX))
        prob_jinak = float(sum(probs[i] for i in BENIGN_IDX))
        is_ganas = prob_ganas >= ambang

        total_latensi = tahap["prep_ms"] + durasi_inf_ms

        # Hitung Grad-CAM Visual Explainability
        cam_heatmap_rgb, cam_overlay_rgb = hitung_gradcam(model, x_norm, top_idx, tahap["roi_rgb"])

        # ═══════════════════════════════════════════════════════════════════════
        # TINGKAT 2: HASIL DIAGNOSIS & INDIKATOR RISIKO UTAMA (TENGAH)
        # ═══════════════════════════════════════════════════════════════════════
        with st.container(border=True):
            st.markdown("""
            <div class="ds-section-header">
                <span class="ds-step-num">2</span> Hasil Diagnosis & Indikator Risiko Triase
            </div>
            """, unsafe_allow_html=True)

            # Pemisahan Status Biologis Lesi vs Status Alarm Triase
            is_top_malignant = ("Ganas" in top_kat)
            is_akiec = (top_k == "AKIEC")
            pct_ganas = prob_ganas * 100
            exceeds_threshold = (prob_ganas >= ambang)

            # 1. Kategori Biologis Lesi (Terkunci pada hasil model, tidak terdistorsi slider)
            if is_akiec:
                tag_text = "Pra-Kanker"
                tag_bg = "#FEF3C7"
                tag_color = "#92400E"
            elif is_top_malignant:
                tag_text = "Ganas (Kanker)"
                tag_bg = "#FEE2E2"
                tag_color = "#991B1B"
            else:
                tag_text = "Jinak (Benign)"
                tag_bg = "#DCFCE7"
                tag_color = "#166534"

            # 2. Status Keputusan Triase Rujukan (Dinamis merespons Ambang Batas Slider)
            selisih_ambang = abs(pct_ganas - (ambang * 100))
            if exceeds_threshold:
                card_class = "malignant"
                badge_text = f"🚨 TRIASE: MEMICU ALARM RUJUKAN ({pct_ganas:.1f}% ≥ {ambang*100:.1f}%)"
                badge_class = "malignant"
                urgency_text = "Tingkat Urgensi: Tinggi &bull; Rujukan Spesialis Dermatologi & Evaluasi Biopsi Jaringan"
                urgency_class = "malignant"
                meter_desc = f"Akumulasi potensi keganasan tercatat sebesar <strong>{pct_ganas:.2f}%</strong>, <strong>melampaui batas ambang triase ({ambang * 100:.1f}%)</strong> sebesar <strong>+{selisih_ambang:.1f}%</strong>. Kasus ini diprioritaskan untuk rujukan spesialis dan pemeriksaan histopatologi lanjutan."
            else:
                card_class = "benign"
                badge_text = f"🟢 TRIASE: DI BAWAH AMBANG RUJUKAN ({pct_ganas:.1f}% < {ambang*100:.1f}%)"
                badge_class = "benign"
                if is_top_malignant:
                    urgency_text = f"Tingkat Urgensi: Ditangguhkan &bull; Di Bawah Ambang Toleransi ({ambang*100:.1f}%), namun Lesi Teridentifikasi {tag_text}"
                    meter_desc = f"Akumulasi potensi keganasan tercatat sebesar <strong>{pct_ganas:.2f}%</strong>, berada <strong>di bawah batas ambang rujukan ({ambang * 100:.1f}%)</strong> dengan selisih <strong>-{selisih_ambang:.1f}%</strong>. Alarm rujukan otomatis dinonaktifkan pada ambang ini, namun konsultasi spesialis tetap disarankan karena profil lesi mengarah ke <strong>{top_nama}</strong>."
                else:
                    urgency_text = "Tingkat Urgensi: Rendah &bull; Pemantauan Mandiri Rutin (Aturan ABCDE)"
                    meter_desc = f"Akumulasi potensi keganasan berada pada tingkat rendah, yaitu <strong>{pct_ganas:.2f}%</strong> (di bawah batas ambang rujukan {ambang * 100:.1f}% dengan selisih <strong>-{selisih_ambang:.1f}%</strong>). Lesi terindikasi jinak/non-kanker."
                urgency_class = "benign"

            # 3. Label Tingkat Risiko Absolut pada Meter
            if pct_ganas >= 50.0:
                risk_level_label = "POTENSI TINGGI"
                meter_stat_color = "#BE123C"
                meter_fill_color = "#E11D48"
            elif pct_ganas >= 20.0:
                risk_level_label = "POTENSI SEDANG"
                meter_stat_color = "#B45309"
                meter_fill_color = "#F59E0B"
            else:
                risk_level_label = "POTENSI RENDAH"
                meter_stat_color = "#15803D"
                meter_fill_color = "#10B981"

            threshold_status = f"Melampaui Ambang ({ambang * 100:.1f}%)" if exceeds_threshold else f"Di Bawah Ambang ({ambang * 100:.1f}%)"

            st.markdown(f"""
            <div class="ds-diagnosis-card {card_class}">
                <div style="flex: 1; min-width: 260px;">
                    <span class="ds-diag-badge {badge_class}">{badge_text}</span>
                    <div class="ds-diag-title">{top_nama}</div>
                    <div class="ds-diag-urgency {urgency_class}">{urgency_text}</div>
                </div>
                <div class="ds-score-box">
                    <div class="ds-score-label">Keyakinan Model</div>
                    <div class="ds-score-value">{top_prob * 100:.2f}%</div>
                    <span class="ds-score-tag" style="background:{tag_bg}; color:{tag_color};">{tag_text}</span>
                </div>
            </div>

            <div class="ds-risk-meter-container">
                <div class="ds-meter-header">
                    <span class="ds-meter-title">Meter Estimasi Potensi Keganasan (Cancer Risk Meter)</span>
                    <span class="ds-meter-stat" style="color: {meter_stat_color};">
                        {pct_ganas:.2f}% ({risk_level_label} &bull; {threshold_status})
                    </span>
                </div>
                <div class="ds-meter-track">
                    <div class="ds-meter-zone-safe" style="width: {ambang * 100:.1f}%;"></div>
                    <div class="ds-meter-zone-alert" style="width: {100 - ambang * 100:.1f}%;"></div>
                    <div class="ds-meter-fill" style="width: {min(100.0, pct_ganas):.2f}%; background: {meter_fill_color}; opacity: 0.88;"></div>
                </div>
                <div class="ds-meter-labels">
                    <span>0% (Zona Rendah)</span>
                    <span style="color: #475569; font-weight: 700;">| Ambang Batas Triase: {ambang * 100:.1f}%</span>
                    <span>100% (Zona Tinggi)</span>
                </div>
                <div class="ds-meter-explanation">{meter_desc}</div>
            </div>
            """, unsafe_allow_html=True)

            # Grad-CAM Visual Explainability (Visual Saliency Map)
            if cam_overlay_rgb is not None:
                with st.expander("👁️ Peta Perhatian Visual Model (Grad-CAM Explainability)", expanded=False):
                    st.markdown('<div style="font-size:0.84rem; font-weight:700; color:#1E293B; margin-bottom:0.6rem;">Visualisasi Area Aktivasi Fitur Konvolusional & Attention (ConvNeXtTiny-MHSA)</div>', unsafe_allow_html=True)
                    g1, g2, g3 = st.columns(3)
                    with g1:
                        st.image(tahap["roi_rgb"], caption="1. Citra Masukan Lesi (ROI)", use_container_width=True)
                    with g2:
                        st.image(cam_heatmap_rgb, caption="2. Heatmap Aktivasi Grad-CAM", use_container_width=True)
                    with g3:
                        st.image(cam_overlay_rgb, caption="3. Tumpang Tindih (Overlay Heatmap)", use_container_width=True)
                    st.caption(f"Area berspektrum merah/hangat menandai pola dermoskopi spesifik pada lesi yang paling dominan memicu model mengklasifikasikannya sebagai **{top_nama}**.")

            # Catatan Disclaimer Klinis
            st.markdown("""
            <div class="ds-disclaimer-card">
                <div style="font-size: 1.15rem;">⚕️</div>
                <div>
                    <strong>Instrumen Penunjang Keputusan Klinis (Clinical Decision Support):</strong>
                    Sistem triase ini dirancang sebagai alat bantu penapisan awal berbasis kecerdasan buatan. Hasil skrining bukan diagnosis medis final dan tidak menggantikan pemeriksaan histopatologi / biopsi jaringan definitif oleh dokter spesialis dermatologi.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ═══════════════════════════════════════════════════════════════════════
        # TINGKAT 3: DISTRIBUSI PROBABILITAS & PANDUAN KLINIS (BAWAH)
        # ═══════════════════════════════════════════════════════════════════════
        with st.container(border=True):
            st.markdown("""
            <div class="ds-section-header">
                <span class="ds-step-num">3</span> Distribusi Probabilitas & Panduan Evaluasi Klinis
            </div>
            """, unsafe_allow_html=True)

            col_dist, col_guide = st.columns([1, 1], gap="large")

            with col_dist:
                st.markdown('<div class="ds-clinical-heading">📊 Distribusi Probabilitas Model (Top-3)</div>', unsafe_allow_html=True)
                
                order = np.argsort(probs)[::-1]
                for i, idx in enumerate(order[:3]):
                    k = KELAS[idx]
                    nama = KELAS_NAMA[k]
                    kat = KELAS_KATEGORI[k]
                    p = float(probs[idx])
                    pct = p * 100
                    is_top1 = (i == 0)
                    card_cls = "ds-dist-card top1" if is_top1 else "ds-dist-card"
                    badge_cls = "badge-ganas" if "Ganas" in kat else "badge-jinak"

                    st.markdown(f"""
                    <div class="{card_cls}">
                        <div>
                            <div class="ds-dist-name">{i+1}. {nama}</div>
                            <span class="{badge_cls}">{kat}</span>
                        </div>
                        <div class="ds-dist-meta">
                            <div class="ds-dist-pct">{pct:.2f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with st.expander("Lihat Rincian Lengkap Seluruh 8 Kelas ISIC", expanded=False):
                    rows_html = []
                    for idx in order:
                        k = KELAS[idx]
                        nama = KELAS_NAMA[k]
                        kat = KELAS_KATEGORI[k]
                        pct = float(probs[idx]) * 100
                        b_cls = "badge-ganas" if "Ganas" in kat else "badge-jinak"
                        rows_html.append(
                            f"<tr>"
                            f"<td style='padding:0.45rem 0.6rem; border-bottom:1px solid #F1F5F9; font-size:0.80rem;'>{nama}</td>"
                            f"<td style='padding:0.45rem 0.6rem; border-bottom:1px solid #F1F5F9;'><span class='{b_cls}'>{kat}</span></td>"
                            f"<td style='padding:0.45rem 0.6rem; border-bottom:1px solid #F1F5F9; text-align:right; font-weight:700; font-size:0.80rem;'>{pct:.2f}%</td>"
                            f"</tr>"
                        )
                    st.markdown(f"""
                    <table style="width:100%; border-collapse:collapse; margin-top:0.5rem;">
                        <thead>
                            <tr style="background:#F8FAFC; border-bottom:1px solid #E2E8F0; font-size:0.75rem; color:#64748B;">
                                <th style="padding:0.4rem 0.6rem; text-align:left;">Kelas Lesi</th>
                                <th style="padding:0.4rem 0.6rem; text-align:left;">Kategori</th>
                                <th style="padding:0.4rem 0.6rem; text-align:right;">Peluang</th>
                            </tr>
                        </thead>
                        <tbody>{''.join(rows_html)}</tbody>
                    </table>
                    """, unsafe_allow_html=True)

                st.caption(f"⚡ **Waktu Komputasi Total: {total_latensi:.0f} ms** (Praproses {tahap['prep_ms']:.0f}ms + Inferensi {durasi_inf_ms:.0f}ms) &bull; Resolusi 256×256 px")

            with col_guide:
                info_med = PENJELASAN_MEDIS.get(top_k, {})
                if info_med:
                    st.markdown(f"""
                    <div class="ds-clinical-block">
                        <div class="ds-clinical-heading">📖 Informasi Patologi: {info_med['nama_lengkap']}</div>
                        <div class="ds-clinical-p"><span class="ds-clinical-label">Definisi:</span> {info_med['definisi']}</div>
                        <div class="ds-clinical-p"><span class="ds-clinical-label">Karakteristik Dermoskopi:</span> {info_med['ciri_klinis']}</div>
                        <div class="ds-clinical-p"><span class="ds-clinical-label">Rekomendasi Tindakan:</span> {info_med['panduan_tindakan']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:0.85rem;">
                    <div style="font-size:0.82rem; font-weight:700; color:#1E293B; margin-bottom:0.25rem;">
                        🩺 Panduan Pemeriksaan Mandiri (Aturan ABCDE)
                    </div>
                    <div class="ds-abcde-grid">
                        <div class="ds-abcde-item">
                            <div class="ds-abcde-letter">A</div>
                            <div class="ds-abcde-title">Asymmetry</div>
                            <div class="ds-abcde-desc">Bentuk sisi tidak simetris</div>
                        </div>
                        <div class="ds-abcde-item">
                            <div class="ds-abcde-letter">B</div>
                            <div class="ds-abcde-title">Border</div>
                            <div class="ds-abcde-desc">Tepi tidak rata / berlekuk</div>
                        </div>
                        <div class="ds-abcde-item">
                            <div class="ds-abcde-letter">C</div>
                            <div class="ds-abcde-title">Color</div>
                            <div class="ds-abcde-desc">Variasi >1 corak warna</div>
                        </div>
                        <div class="ds-abcde-item">
                            <div class="ds-abcde-letter">D</div>
                            <div class="ds-abcde-title">Diameter</div>
                            <div class="ds-abcde-desc">Ukuran lesi > 6 milimeter</div>
                        </div>
                        <div class="ds-abcde-item">
                            <div class="ds-abcde-letter">E</div>
                            <div class="ds-abcde-title">Evolution</div>
                            <div class="ds-abcde-desc">Perubahan bentuk / rasa</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ═══════════════════════════════════════════════════════════════════════
        # PRA-PEMROSESAN CITRA: SCIENTIFIC PREPROCESSING INSPECTOR
        # ═══════════════════════════════════════════════════════════════════════
        with st.expander("🔬 Inspeksi Alur Pra-pemrosesan Citra Digital (ROI Otsu & Shades of Gray, p=6)", expanded=False):
            st.markdown('<div style="font-size:0.88rem; font-weight:700; color:#1E293B; margin-bottom:0.75rem;">Pipeline Transformasi Citra Dermoskopi Real-Time</div>', unsafe_allow_html=True)

            cp1, cp2, cp3, cp4 = st.columns(4)
            
            # 1. Citra Asli
            with cp1:
                st.markdown("""
                <div class="ds-prep-card">
                    <span class="ds-prep-step-badge">Tahap 1</span>
                    <div class="ds-prep-title">Citra Asli Dermoskopi</div>
                """, unsafe_allow_html=True)
                st.image(tahap["asli_rgb"], use_container_width=True)
                vignette_status = "Terdeteksi (Dimasker)" if tahap["lingkaran_ada"] else "Tidak Terdeteksi"
                st.markdown(f"""
                    <div class="ds-prep-metric-pill">
                        <div>Resolusi: <strong>{tahap['orig_w']}×{tahap['orig_h']} px</strong></div>
                        <div>Vignette Lensa: <strong>{vignette_status}</strong></div>
                    </div>
                    <div class="ds-prep-desc">Input citra dermoskopi mentah sebelum dilakukan eliminasi artefak lensa melingkar.</div>
                </div>
                """, unsafe_allow_html=True)

            # 2. Masker Otsu
            with cp2:
                st.markdown("""
                <div class="ds-prep-card">
                    <span class="ds-prep-step-badge">Tahap 2</span>
                    <div class="ds-prep-title">Segmentasi Otsu & Masking</div>
                """, unsafe_allow_html=True)
                st.image(tahap["masker_rgb"], use_container_width=True)
                st.markdown(f"""
                    <div class="ds-prep-metric-pill">
                        <div>Ambang Otsu (T): <strong>{tahap['otsu_val']:.0f} / 255</strong></div>
                        <div>Morfologi: <strong>Buka (Opening 5×5)</strong></div>
                    </div>
                    <div class="ds-prep-desc">Pemisahan otomatis bimodal intensitas lesi dari kulit normal bebas subjektivitas manual.</div>
                </div>
                """, unsafe_allow_html=True)

            # 3. Segmentasi ROI
            with cp3:
                st.markdown("""
                <div class="ds-prep-card">
                    <span class="ds-prep-step-badge">Tahap 3</span>
                    <div class="ds-prep-title">Ekstraksi ROI (+15% Margin)</div>
                """, unsafe_allow_html=True)
                st.image(tahap["roi_rgb"], use_container_width=True)
                bx, by, bw, bh = tahap["roi_box"]
                st.markdown(f"""
                    <div class="ds-prep-metric-pill">
                        <div>Kotak ROI: <strong>{bw}×{bh} px</strong></div>
                        <div>Reduksi Noise: <strong>{tahap['noise_reduced_pct']:.1f}% latar</strong></div>
                    </div>
                    <div class="ds-prep-desc">Margin dinamis +15% menjaga keutuhan tepian lesi (krusial bagi evaluasi kriteria Border).</div>
                </div>
                """, unsafe_allow_html=True)

            # 4. Normalisasi Shades of Gray
            with cp4:
                st.markdown("""
                <div class="ds-prep-card">
                    <span class="ds-prep-step-badge">Tahap 4</span>
                    <div class="ds-prep-title">Shades of Gray (p=6)</div>
                """, unsafe_allow_html=True)
                st.image(tahap["sog_rgb"], use_container_width=True)
                er, eg, eb = tahap["e_vec"]
                st.markdown(f"""
                    <div class="ds-prep-metric-pill">
                        <div>Vektor Cahaya: <strong>[{er:.2f}, {eg:.2f}, {eb:.2f}]</strong></div>
                        <div>Waktu Praproses: <strong>{tahap['prep_ms']:.0f} ms</strong></div>
                    </div>
                    <div class="ds-prep-desc">Koreksi ketidakseimbangan pencahayaan lampu dermatoskop untuk mencapai color constancy.</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<hr style="margin: 1.25rem 0 1.0rem 0; border: none; border-top: 1px solid #E2E8F0;">', unsafe_allow_html=True)

            col_ablasi, col_math = st.columns([1.1, 0.9], gap="large")

            with col_ablasi:
                st.markdown('<div style="font-size:0.84rem; font-weight:700; color:#0F172A; margin-bottom:0.25rem;">📊 Bukti Empiris Ablasi Riset Model</div>', unsafe_allow_html=True)
                st.markdown("""
                <table class="ds-ablation-table">
                    <thead>
                        <tr>
                            <th>Metode Masukan</th>
                            <th style="text-align:right;">Val Accuracy</th>
                            <th style="text-align:right;">Val ROC-AUC</th>
                            <th style="text-align:center;">Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Citra Utuh (Raw Baseline)</td>
                            <td style="text-align:right;">81.34%</td>
                            <td style="text-align:right;">0.8949</td>
                            <td style="text-align:center;"><span style="color:#64748B;">Baseline</span></td>
                        </tr>
                        <tr>
                            <td>Segmentasi ROI Saja</td>
                            <td style="text-align:right;">80.74%</td>
                            <td style="text-align:right;">0.8845</td>
                            <td style="text-align:center;"><span style="color:#64748B;">Tanpa Koreksi Warna</span></td>
                        </tr>
                        <tr class="active-row">
                            <td>ROI + Shades of Gray (p=6)</td>
                            <td style="text-align:right;">81.68%</td>
                            <td style="text-align:right;">0.8907</td>
                            <td style="text-align:center;"><span class="badge-jinak" style="background:#DBEAFE; color:#1D4ED8;">Pipeline Aktif (+0.94%)</span></td>
                        </tr>
                    </tbody>
                </table>
                """, unsafe_allow_html=True)
                st.caption("Hasil evaluasi empiris: Penambahan Shades of Gray (p=6) setelah ekstraksi ROI berhasil mengatasi degradasi warna dan mendongkrak akurasi validasi model ke level tertinggi.")

            with col_math:
                st.markdown('<div style="font-size:0.84rem; font-weight:700; color:#0F172A; margin-bottom:0.25rem;">📐 Rasional Matematis & Klinis</div>', unsafe_allow_html=True)
                st.markdown("""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:0.75rem 0.85rem; font-size:0.76rem; color:#334155; line-height:1.55;">
                    <div style="margin-bottom:0.45rem;">
                        <strong>1. Ambang Bimodal Otsu:</strong><br>
                        Memaksimalkan variansi antar-kelas piksel untuk memisahkan lesi secara otomatis tanpa tuning ambang batas manual.
                    </div>
                    <div style="margin-bottom:0.45rem;">
                        <strong>2. Margin Adaptif 15%:</strong><br>
                        Mencegah terpotongnya pola pigmentasi tepi (krusial bagi kriteria <em>Border</em> & <em>Asymmetry</em> pada ABCDE).
                    </div>
                    <div>
                        <strong>3. Minkowski Norm (p=6):</strong><br>
                        Eksponen p=6 memberikan bobot seimbang pada piksel lesi tanpa terdistorsi oleh specular highlights pantulan cairan dermatoskop.
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ─── FOOTER SEDERHANA ─────────────────────────────────────────────────────────
st.markdown("""
<div class="ds-footer">
    DERMASPARK &bull; Sistem Skrining & Triase Citra Lesi Kulit berbasis Deep Learning
</div>
""", unsafe_allow_html=True)
