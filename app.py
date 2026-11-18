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
    page_title="DERMASPARK - Sistem Skrining Citra Lesi Kulit",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── STYLING PROFESIONAL DISIPLIN & BERSIH (ANTI-AISLOP) ─────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #0F172A;
    }

    .stApp {
        background-color: #F8FAFC;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1140px !important;
    }

    /* Header Bersih & Disiplin */
    .app-header {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .app-header-left {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .app-header-icon {
        font-size: 1.6rem;
    }
    .app-header-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .app-header-desc {
        font-size: 0.82rem;
        color: #64748B;
    }
    .app-header-badge {
        font-size: 0.75rem;
        font-weight: 600;
        color: #475569;
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
    }

    /* Container Card */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        padding: 1.25rem !important;
    }

    .box-title {
        font-size: 0.90rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 1.0rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #F1F5F9;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    /* Status Banner Klinis */
    .status-card {
        border-radius: 8px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1.2rem;
    }
    .status-card.malignant {
        background-color: #FEF2F2;
        border-left: 4px solid #DC2626;
        color: #991B1B;
    }
    .status-card.benign {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        color: #166534;
    }
    .status-badge {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.25rem;
    }
    .status-title {
        font-size: 1.35rem;
        font-weight: 700;
        line-height: 1.25;
        margin-bottom: 0.3rem;
    }
    .status-urgency {
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .status-desc {
        font-size: 0.85rem;
        line-height: 1.5;
        color: #334155;
    }

    /* Tabel Hasil Uji Klinis (Disiplin & Terstruktur) */
    .lab-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.84rem;
        margin-bottom: 1.0rem;
    }
    .lab-table th {
        text-align: left;
        padding: 0.55rem 0.75rem;
        font-weight: 600;
        color: #64748B;
        border-bottom: 1px solid #E2E8F0;
        background: #F8FAFC;
    }
    .lab-table td {
        padding: 0.60rem 0.75rem;
        border-bottom: 1px solid #F1F5F9;
        color: #1E293B;
        vertical-align: middle;
    }
    .lab-table tr.top-row {
        background-color: #F8FAFC;
        font-weight: 600;
    }
    .tag-jinak {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        background: #DCFCE7;
        color: #15803D;
        font-size: 0.70rem;
        font-weight: 600;
    }
    .tag-ganas {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        background: #FEE2E2;
        color: #B91C1C;
        font-size: 0.70rem;
        font-weight: 600;
    }
    .bar-bg {
        width: 100%;
        height: 6px;
        background: #E2E8F0;
        border-radius: 3px;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        border-radius: 3px;
    }

    /* Ringkasan Parameter */
    .meta-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.65rem 0.95rem;
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 0.75rem;
    }

    /* Detail Informasi Klinis Lengkap */
    .clinical-detail-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.95rem 1.1rem;
        margin-top: 1.0rem;
    }
    .detail-sec-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.55rem;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 0.35rem;
    }
    .detail-item {
        font-size: 0.82rem;
        color: #334155;
        line-height: 1.55;
        margin-bottom: 0.45rem;
    }
    .detail-label {
        font-weight: 700;
        color: #0F172A;
    }

    /* Footer Sederhana */
    .app-footer {
        text-align: center;
        padding-top: 1.5rem;
        margin-top: 1.5rem;
        font-size: 0.80rem;
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
        "ciri_klinis": "Asimetris, batas tepi tidak beraturan/bergerigi, warna bervariasi (hitam, cokelat, kebiruan), diameter sering >6 mm, dan mengalami perubahan bentuk/ukuran (kriteria ABCDE).",
        "panduan_tindakan": "Segera lakukan rujukan ke dokter spesialis dermatologi untuk pemeriksaan dermoskopi digital lanjutan, biopsi eksisi diagnostik, dan penentuan ketebalan Breslow."
    },
    "BCC": {
        "nama_lengkap": "Basal Cell Carcinoma (Kanker Sel Basal)",
        "definisi": "Jenis kanker kulit non-melanoma paling sering ditemukan di dunia. Tumbuh lambat dari lapisan basal epidermis akibat akumulasi radiasi ultraviolet (UV) jangka panjang.",
        "ciri_klinis": "Benjolan mengkilap (pearly papule), pelebaran pembuluh darah halus (telangiektasia), tepi menggulung, dan lesi luka/borok kecil yang mudah berdarah atau sulit sembuh.",
        "panduan_tindakan": "Jarang bermetastasis ke organ jauh namun bersifat destruktif lokal terhadap jaringan sekitarnya. Disarankan konsultasi untuk tindakan eksisi bedah, bedah Mohs, atau kuretase."
    },
    "SCC": {
        "nama_lengkap": "Squamous Cell Carcinoma (Kanker Sel Skuamosa)",
        "definisi": "Kanker kulit ganas kedua paling umum, berasal dari keratinosit lapisan epidermis. Berkembang akibat paparan sinar matahari kronis atau lesi pra-kanker sebelumnya.",
        "ciri_klinis": "Plak kemerahan menebal bersisik, nodul keras yang mudah berdarah, atau luka berkeropeng tebal yang menetap di area terpapar matahari.",
        "panduan_tindakan": "Memiliki risiko penyebaran sekitar 2-5% ke kelenjar getah bening. Disarankan pemeriksaan histopatologi melalui biopsi serta tindakan eksisi bedah definitif."
    },
    "AKIEC": {
        "nama_lengkap": "Actinic Keratosis / Bowen's Disease (Lesi Pra-Kanker)",
        "definisi": "Lesi pra-kanker (pre-malignant) pada lapisan epidermis akibat kerusakan DNA oleh sinar UV kronis. Sekitar 10-20% lesi ini berpotensi bertransformasi menjadi Kanker Sel Skuamosa (SCC) invasif bila tidak ditangani.",
        "ciri_klinis": "Bercak kasar berpasir, kering, bersisik, berwarna kemerahan atau kecokelatan, sering muncul di wajah, daun telinga, bibir, atau punggung tangan.",
        "panduan_tindakan": "Dianjurkan konsultasi dermatologi untuk terapi pencegahan sebelum berkembang menjadi kanker invasif (krioterapi nitrogen cair, krim topikal, atau fotodinamik)."
    },
    "NV": {
        "nama_lengkap": "Melanocytic Nevus (Tahi Lalat Jinak)",
        "definisi": "Proliferasi jinak dari sel melanosit. Merupakan kondisi sangat umum dan non-kanker yang ditemukan pada hampir setiap orang.",
        "ciri_klinis": "Bentuk simetris bundar/oval, batas tepi tegas dan teratur, warna cokelat atau hitam merata dan homogen, permukaan rata atau sedikit menonjol.",
        "panduan_tindakan": "Tidak memerlukan tindakan medis khusus. Cukup lakukan observasi mandiri berkala (aturan ABCDE). Periksakan jika tahi lalat membesar cepat, gatal, atau berdarah."
    },
    "BKL": {
        "nama_lengkap": "Benign Keratosis / Seborrheic Keratosis (Lesi Jinak)",
        "definisi": "Pertumbuhan kulit jinak non-kanker yang sangat umum terjadi seiring bertambahnya usia, berasal dari keratinosit epidermis.",
        "ciri_klinis": "Tampak seperti 'menempel' pada kulit (stuck-on), permukaan berlilin, kasar atau berkeropeng, dengan warna bervariasi dari cokelat muda hingga hitam pekat.",
        "panduan_tindakan": "Kondisi sama sekali tidak berbahaya dan tidak berubah menjadi kanker. Pengangkatan (krioterapi/kuretase) hanya diperlukan jika mengalami iritasi atau alasan estetika."
    },
    "DF": {
        "nama_lengkap": "Dermatofibroma (Lesi Jaringan Ikat Jinak)",
        "definisi": "Proliferasi jinak fibroblas kutan pada lapisan dermis, sering kali dipicu oleh reaksi cedera minor sebelumnya (seperti gigitan serangga atau trauma tumpul kecil).",
        "ciri_klinis": "Nodul keras kecil (0.5 - 1 cm) berwarna merah kecokelatan yang memperlihatkan tanda cekungan ke dalam saat dijepit dari samping (pinch sign positif).",
        "panduan_tindakan": "Bersifat jinak dan stabil. Tidak memerlukan tindakan bedah kecuali jika menimbulkan keluhan nyeri atau gesekan."
    },
    "VASC": {
        "nama_lengkap": "Vascular Lesion / Cherry Angioma (Lesi Pembuluh Darah)",
        "definisi": "Proliferasi jinak pembuluh darah kapiler pada kulit, sering ditemukan pada orang dewasa dan bertambah banyak seiring pertambahan usia.",
        "ciri_klinis": "Bintik atau papula kecil berwarna merah terang, ungu, atau merah marun, permukaannya halus, dan memudar sesaat jika ditekan (blanching).",
        "panduan_tindakan": "Sangat jinak dan tidak memiliki potensi keganasan. Tindakan laser atau elektrokauter hanya dipertimbangkan untuk alasan kosmetik."
    }
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

def format_lab_table(probs, top_idx):
    order = np.argsort(probs)[::-1]
    rows = []
    for idx in order:
        k = KELAS[idx]
        nama = KELAS_NAMA[k]
        kat = KELAS_KATEGORI[k]
        p = float(probs[idx])
        pct = p * 100
        is_top = (idx == top_idx)
        is_mal = ("Ganas" in kat)
        
        row_cls = " class='top-row'" if is_top else ""
        tag_cls = "tag-ganas" if is_mal else "tag-jinak"
        bar_color = "#DC2626" if is_mal else "#2563EB"
        
        row_str = (
            f"<tr{row_cls}>"
            f"<td>{nama}</td>"
            f"<td><span class='{tag_cls}'>{kat}</span></td>"
            f"<td style='text-align:right;'>{pct:.2f}%</td>"
            f"<td style='width:28%;'><div class='bar-bg'><div class='bar-fill' style='width:{pct:.2f}%;background:{bar_color};'></div></div></td>"
            f"</tr>"
        )
        rows.append(row_str)
    
    table_html = (
        "<table class='lab-table'>"
        "<thead><tr>"
        "<th>Jenis Lesi Kulit</th>"
        "<th>Kategori</th>"
        "<th style='text-align:right;'>Probabilitas</th>"
        "<th>Distribusi</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )
    return table_html

# ─── HEADER BERSIH ────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-header-left">
        <span class="app-header-icon">🩺</span>
        <div>
            <div class="app-header-title">DERMASPARK</div>
            <div class="app-header-desc">Sistem Skrining & Triase Citra Lesi Kulit</div>
        </div>
    </div>
    <div class="app-header-badge">ConvNeXtTiny-MHSA &bull; 8 Kelas ISIC</div>
</div>
""", unsafe_allow_html=True)

# ─── WORKSPACE DUA KOLOM TERSTRUKTUR ──────────────────────────────────────────
col_input, col_report = st.columns([1, 1.3], gap="large")

with col_input:
    with st.container(border=True):
        st.markdown('<div class="box-title">1. Masukan Citra Dermoskopi</div>', unsafe_allow_html=True)

        mode = st.radio(
            "Pilih Sumber Citra:",
            ["Pilih dari Sampel", "Unggah Berkas Foto"],
            horizontal=True
        )

        citra_bgr = None
        label_sumber = ""

        if mode == "Pilih dari Sampel":
            pilihan_sampel = {
                "Melanoma (Kanker Ganas Agresif)": "Melanoma_ISIC_0024310.jpg",
                "Basal Cell Carcinoma (Kanker Kulit)": "BCC_ISIC_0024436.jpg",
                "Actinic Keratosis (Pra-Kanker)": "AKIEC_ISIC_0024329.jpg",
                "Melanocytic Nevus (Tahi Lalat Jinak)": "Nevus_ISIC_0000009.jpg",
                "Benign Keratosis (Lesi Jinak)": "BKL_ISIC_0012173.jpg"
            }
            terpilih = st.selectbox("Pilih Sampel Lesi:", list(pilihan_sampel.keys()))
            berkas = SAMPLES_DIR / pilihan_sampel[terpilih]
            if berkas.exists():
                citra_bgr = cv2.imread(str(berkas))
                label_sumber = terpilih
        else:
            file_up = st.file_uploader("Unggah foto lesi kulit (JPG / PNG):", type=["jpg", "jpeg", "png"])
            if file_up is not None:
                buf = np.asarray(bytearray(file_up.read()), dtype=np.uint8)
                citra_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                label_sumber = f"File: {file_up.name}"

        # Pratinjau Citra
        if citra_bgr is not None:
            tahap = proses_citra(citra_bgr)
            st.image(tahap["asli_rgb"], caption=f"Pratinjau: {label_sumber}", use_container_width=True)

        # Pengaturan Ambang Batas Klinis
        with st.expander("Pengaturan Ambang Batas (Opsional)", expanded=False):
            ambang = st.slider("Batas Ambang Keganasan:", 0.0, 1.0, 0.42, 0.01)
            st.caption("Batas ambang standar terkalibrasi: **0,42**. Nilai peluang keganasan $\ge 0,42$ dikategorikan Ganas.")

with col_report:
    with st.container(border=True):
        st.markdown('<div class="box-title">2. Hasil Skrining & Evaluasi</div>', unsafe_allow_html=True)

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

                # 1. STATUS BANNER KLINIS
                if is_ganas:
                    st.markdown(
                        f'<div class="status-card malignant">'
                        f'<div class="status-badge">⚠️ Terindikasi Ganas (Malignant)</div>'
                        f'<div class="status-title">{top_nama}</div>'
                        f'<div class="status-urgency">Tingkat Urgensi: Tinggi &bull; Rujukan Spesialis & Biopsi Jaringan</div>'
                        f'<div class="status-desc">Akumulasi estimasi potensi keganasan tercatat sebesar <strong>{prob_ganas*100:.1f}%</strong> (melebihi ambang batas {ambang:.2f}). Disarankan untuk melakukan konsultasi klinis langsung dengan dokter spesialis dermatologi.</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="status-card benign">'
                        f'<div class="status-badge">✓ Terindikasi Jinak (Benign)</div>'
                        f'<div class="status-title">{top_nama}</div>'
                        f'<div class="status-urgency">Tingkat Urgensi: Rendah &bull; Pemantauan Mandiri Rutin</div>'
                        f'<div class="status-desc">Akumulasi estimasi potensi keganasan berada pada tingkat rendah, yaitu <strong>{prob_ganas*100:.1f}%</strong> (di bawah ambang batas {ambang:.2f}). Lesi diprediksi non-kanker. Lakukan evaluasi berkala mandiri terhadap aturan ABCDE.</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                # 2. TABEL HASIL UJI LABORATORIUM (8 KELAS)
                st.markdown(format_lab_table(probs, top_idx), unsafe_allow_html=True)

                # 3. INFORMASI KLINIS MENDALAM
                info_med = PENJELASAN_MEDIS.get(KELAS[top_idx], {})
                if info_med:
                    st.markdown(
                        f'<div class="clinical-detail-card">'
                        f'<div class="detail-sec-title">📖 Informasi Medis: {info_med["nama_lengkap"]}</div>'
                        f'<div class="detail-item"><span class="detail-label">Definisi:</span> {info_med["definisi"]}</div>'
                        f'<div class="detail-item"><span class="detail-label">Ciri Dermoskopi:</span> {info_med["ciri_klinis"]}</div>'
                        f'<div class="detail-item"><span class="detail-label">Rekomendasi Tindakan:</span> {info_med["panduan_tindakan"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                # 4. METADATA FOOTER
                total_latency = tahap["prep_ms"] + durasi_inf_ms
                st.markdown(
                    f'<div class="meta-footer">'
                    f'<span>Waktu Komputasi: <strong>{total_latency:.0f} ms</strong> (Praproses {tahap["prep_ms"]:.0f}ms + Inferensi {durasi_inf_ms:.0f}ms)</span>'
                    f'<span>Resolusi: <strong>256×256 px</strong></span>'
                    f'<span>Model: <strong>ConvNeXtTiny-MHSA</strong></span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("Pilih sampel lesi kulit di sebelah kiri atau unggah berkas foto untuk melihat hasil skrining.")

# ─── ACCORDION TAHAPAN PRA-PEMROSESAN (TIDAK MEMBEBANI HALAMAN) ───────────────
if citra_bgr is not None:
    with st.expander("🔬 Lihat Alur Pra-pemrosesan Citra (4 Tahap)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.image(tahap["asli_rgb"], caption="1. Citra Asli (256×256)", use_container_width=True)
        with c2:
            st.image(tahap["masker_rgb"], caption="2. Masker Lingkaran & Otsu", use_container_width=True)
        with c3:
            st.image(tahap["roi_rgb"], caption="3. Segmentasi ROI (+15%)", use_container_width=True)
        with c4:
            st.image(tahap["sog_rgb"], caption="4. Normalisasi Warna (p=6)", use_container_width=True)

# ─── FOOTER SEDERHANA ─────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    DERMASPARK &bull; Sistem Skrining Citra Lesi Kulit
</div>
""", unsafe_allow_html=True)
