import os
import sys
import time
import json
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ─── KONFIGURASI HALAMAN ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="DERMASPARK - Deteksi Kanker Kulit",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── GAYA SEDERHANA & MINIMALIS (CLEAN MEDICAL UI) ────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 1.25rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .result-ganas {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        color: #991B1B;
    }
    .result-jinak {
        background-color: #F0FDF4;
        border: 2px solid #22C55E;
        color: #166534;
    }
    .result-title {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        margin-bottom: 0.25rem;
    }
    .result-prob {
        font-size: 1.05rem;
        font-weight: 600;
    }
    .info-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.9rem;
        font-size: 0.85rem;
        color: #475569;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── PATH DIREKTORI & KONSTANTA ───────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
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

# ─── PRA-PEMROSESAN CITRA ─────────────────────────────────────────────────────
def potong_roi(im_bgr):
    h, w = im_bgr.shape[:2]
    det = cv2.resize(im_bgr, (512, 512), interpolation=cv2.INTER_AREA)
    abu = cv2.GaussianBlur(cv2.cvtColor(det, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    _, th = cv2.threshold(abu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    lingkar = np.zeros((512, 512), np.uint8)
    cv2.circle(lingkar, (256, 256), 245, 255, -1)
    cnts, _ = cv2.findContours(cv2.bitwise_and(th, lingkar), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return cv2.resize(im_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    c = max(cnts, key=cv2.contourArea)
    if not (0.03 <= cv2.contourArea(c) / (512 * 512) <= 0.90):
        return cv2.resize(im_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    x, y, bw, bh = cv2.boundingRect(c)
    sisi = int(max(bw, bh) * 1.30)
    cx, cy = x + bw // 2, y + bh // 2
    x0, y0 = max(0, cx - sisi // 2), max(0, cy - sisi // 2)
    x1, y1 = min(512, x0 + sisi), min(512, y0 + sisi)
    sx, sy = w / 512.0, h / 512.0
    crop = im_bgr[int(y0 * sy):int(y1 * sy), int(x0 * sx):int(x1 * sx)]
    if crop.size == 0:
        return cv2.resize(im_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

def shades_of_gray(x_tensor, p=6):
    e = x_tensor.clamp_min(1e-6).pow(p).mean(dim=(2, 3), keepdim=True).pow(1.0 / p)
    e = e / e.norm(dim=1, keepdim=True)
    return (x_tensor / (e * math.sqrt(3))).clamp(0, 1)

@st.cache_resource
def muat_model():
    if not MODEL_PATH.exists():
        st.error(f"Berkas model tidak ditemukan: {MODEL_PATH}")
        return None
    model = ConvNeXtTinyMHSA(n_kelas=8)
    sd_half = torch.load(MODEL_PATH, map_location="cpu")
    sd_float = {k: v.float() if v.is_floating_point() else v for k, v in sd_half.items()}
    model.load_state_dict(sd_float)
    model.eval()
    return model

# ─── HEADER UTAMA ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">⚡ DERMASPARK</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Sistem Deteksi Lesi Kanker Kulit &bull; Model ConvNeXtTiny-MHSA (BIMA 2026 UTM)</div>', unsafe_allow_html=True)

# ─── LAYOUT SEDERHANA 2 KOLOM ─────────────────────────────────────────────────
col_input, col_output = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("1. Pilih atau Unggah Citra")

    metode = st.radio(
        "Pilih Sumber Gambar:",
        ["Gunakan Sampel Uji Riset", "Unggah Foto Lesi Sendiri"],
        horizontal=True,
        label_visibility="collapsed"
    )

    citra_bgr = None
    nama_sampel = ""

    if metode == "Gunakan Sampel Uji Riset":
        daftar_sampel = {
            "🔴 Melanoma (Kanker Ganas Agresif)": "Melanoma_ISIC_0000054.jpg",
            "🔴 Basal Cell Carcinoma (Kanker Kulit)": "BCC_ISIC_0024436.jpg",
            "🔴 Actinic Keratoses (Pra-Kanker)": "AKIEC_ISIC_0024329.jpg",
            "🟢 Melanocytic Nevi (Tahi Lalat Jinak)": "Nevus_ISIC_0000009.jpg",
            "🟢 Benign Keratosis (Lesi Jinak)": "BKL_ISIC_0012173.jpg"
        }
        pilihan = st.selectbox("Pilih Sampel:", list(daftar_sampel.keys()))
        path_sampel = SAMPLES_DIR / daftar_sampel[pilihan]
        if path_sampel.exists():
            citra_bgr = cv2.imread(str(path_sampel))
            nama_sampel = pilihan
    else:
        file_up = st.file_uploader("Pilih file foto dermoskopi (JPG/PNG):", type=["jpg", "jpeg", "png"])
        if file_up is not None:
            raw_bytes = np.asarray(bytearray(file_up.read()), dtype=np.uint8)
            citra_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
            nama_sampel = file_up.name

    if citra_bgr is not None:
        im_rgb = cv2.cvtColor(citra_bgr, cv2.COLOR_BGR2RGB)
        im_roi = cv2.cvtColor(potong_roi(citra_bgr), cv2.COLOR_BGR2RGB)
        
        c_img1, c_img2 = st.columns(2)
        with c_img1:
            st.image(cv2.resize(im_rgb, (IMAGE_SIZE, IMAGE_SIZE)), caption="Citra Asli", use_container_width=True)
        with c_img2:
            st.image(im_roi, caption="Hasil Segmentasi ROI", use_container_width=True)

    with st.expander("⚙️ Pengaturan Ambang (Threshold)"):
        ambang = st.slider("Batas Ambang Keganasan:", 0.0, 1.0, 0.42, 0.01, help="Ambang optimal riset: 0,42")

with col_output:
    st.subheader("2. Hasil Diagnostik Model")

    if citra_bgr is not None:
        model = muat_model()
        if model is not None:
            # Praproses
            im_roi_bgr = potong_roi(citra_bgr)
            im_roi_rgb = cv2.cvtColor(im_roi_bgr, cv2.COLOR_BGR2RGB)
            x_raw = torch.from_numpy(im_roi_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            x_sog = shades_of_gray(x_raw, p=6)
            
            mean_t = torch.tensor(NORM_MEAN).view(1, 3, 1, 1)
            std_t = torch.tensor(NORM_STD).view(1, 3, 1, 1)
            x_norm = (x_sog - mean_t) / std_t

            # Inferensi
            t0 = time.time()
            with torch.no_grad():
                logits = model(x_norm)
                probs = F.softmax(logits, dim=1)[0].numpy()
            durasi_ms = (time.time() - t0) * 1000

            p_ganas = float(sum(probs[i] for i in MALIGNANT_IDX))
            status_ganas = p_ganas >= ambang

            # Kartu Hasil
            if status_ganas:
                st.markdown(f"""
                <div class="result-card result-ganas">
                    <div class="result-title">⚠️ TERDETEKSI GANAS (MALIGNANT)</div>
                    <div class="result-prob">Peluang Keganasan: {p_ganas * 100:.2f}% (Batas Ambang: {ambang:.2f})</div>
                </div>
                """, unsafe_allow_html=True)
                st.error("Rekomendasi: Lesi berpotensi ganas. Disarankan untuk segera melakukan pemeriksaan biopsi histopatologi.")
            else:
                st.markdown(f"""
                <div class="result-card result-jinak">
                    <div class="result-title">🛡️ TERDETEKSI JINAK (BENIGN)</div>
                    <div class="result-prob">Peluang Keganasan: {p_ganas * 100:.2f}% (Batas Ambang: {ambang:.2f})</div>
                </div>
                """, unsafe_allow_html=True)
                st.success("Rekomendasi: Lesi terindikasi jinak. Tetap lakukan pemeriksaan berkala apabila terjadi perubahan warna atau ukuran.")

            # Rincian 8 Sub-Kelas
            st.markdown("##### Probabilitas Tiap Sub-Kelas:")
            df_kelas = pd.DataFrame({
                "Kelas": [KELAS[i] for i in range(8)],
                "Nama": [KELAS_NAMA[KELAS[i]].split(" (")[0] for i in range(8)],
                "Sifat": ["Ganas" if i in MALIGNANT_IDX else "Jinak" for i in range(8)],
                "Peluang": probs
            }).sort_values("Peluang", ascending=False)

            fig, ax = plt.subplots(figsize=(6, 2.8), dpi=120)
            warna = ["#EF4444" if s == "Ganas" else "#3B82F6" for s in df_kelas["Sifat"]]
            bars = ax.barh(df_kelas["Kelas"], df_kelas["Peluang"] * 100, color=warna, height=0.55)
            ax.set_xlim(0, 100)
            ax.invert_yaxis()
            ax.set_xlabel("Probabilitas (%)", fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(axis="x", linestyle=":", alpha=0.3)
            for bar in bars:
                w = bar.get_width()
                if w > 1:
                    ax.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=7.5, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.caption(f"⏱️ Waktu inferensi: {durasi_ms:.1f} ms | Resolusi: 256×256 px")
    else:
        st.info("👈 Pilih sampel di sebelah kiri atau unggah foto lesi kulit untuk melihat hasil diagnosa.")

# ─── FOOTER MINIMALIS ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 0.8rem; color: #94A3B8;">
    <strong>DERMASPARK</strong> &bull; Program Asistensi Hibah Penelitian BIMA UTM 2026<br>
    <em>Catatan: Sistem ini merupakan alat bantu skrining pendukung keputusan (decision support), bukan pengganti diagnosa medis dokter spesialis.</em>
</div>
""", unsafe_allow_html=True)
