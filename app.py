import os
import json
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# ─── KONFIGURASI HALAMAN ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="DERMASPARK - Evaluasi Data Uji ConvNeXtTiny-MHSA",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STYLING KHUSUS (ANTISLOP: CLEAN ACADEMIC & CLINICAL) ─────────────────────
st.markdown("""
<style>
    /* Tipografi & Warna Netral Medis */
    .main-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 1.2rem;
        line-height: 1.4;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
        border-radius: 9999px;
        font-size: 0.78rem;
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
        padding: 1.1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .metric-val {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.1;
    }
    .metric-lbl {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 0.35rem;
        letter-spacing: 0.04em;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #059669;
        font-weight: 500;
        margin-top: 0.35rem;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1E293B;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0.45rem;
        margin-top: 1.8rem;
        margin-bottom: 1.0rem;
    }
    .info-callout {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.85rem 1.1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
        color: #334155;
        margin: 1.0rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── PATH DATA ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"

# ─── FUNGSI STATISTIK ─────────────────────────────────────────────────────────
def wilson_interval(k, n, z=1.96):
    """Menghitung interval kepercayaan Wilson 95% untuk proporsi."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return max(0.0, (c - h) * 100), min(100.0, (c + h) * 100)

@st.cache_data
def muat_data():
    df_pred = pd.read_csv(DATA_DIR / "data_uji_prediksi.csv")
    df_eval = pd.read_csv(DATA_DIR / "hasil_evaluasi_uji.csv")
    df_abl = pd.read_csv(DATA_DIR / "ablasi_preprocessing.csv")
    ringkasan = json.load(open(DATA_DIR / "ringkasan_model.json"))
    return df_pred, df_eval, df_abl, ringkasan

try:
    df_pred, df_eval, df_abl, ringkasan = muat_data()
except Exception as e:
    st.error(f"Gagal memuat berkas data: {e}")
    st.stop()

# ─── HEADER UTAMA ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">⚡ DERMASPARK: Platform Evaluasi & Hasil Uji Citra Medis</div>', unsafe_allow_html=True)
st.markdown("""
<div class="sub-title">
    <strong>Asistensi Hibah Penelitian BIMA 2026</strong> &bull; Universitas Trunojoyo Madura<br>
    <em>Peneliti:</em> Aqbil Baraka (NIM 230411100077) &nbsp;|&nbsp; <em>Pembimbing:</em> Fifin Ayu Mufarroha, S.Kom., M.Kom.<br>
    <em>Topik:</em> Klasifikasi Kanker Kulit Berbasis Deep Learning & Analisis Granularitas Partisi Memori Apache Spark
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div>
    <span class="badge-pill highlight">⚡ DERMASPARK v1.0</span>
    <span class="badge-pill highlight">Data Uji Terisolasi (n = 3.291)</span>
    <span class="badge-pill highlight">Zero Patient Leakage</span>
    <span class="badge-pill">Backbone: ConvNeXtTiny-MHSA</span>
    <span class="badge-pill">Resolusi: 256x256 px</span>
    <span class="badge-pill">Praproses: ROI Otsu + Shades of Gray</span>
    <span class="badge-pill">Ambang Optimal: 0,42</span>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR: FILTER & KONTROL ────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Parameter Pengujian")

dataset_pilihan = st.sidebar.selectbox(
    "Pilih Dataset Uji:",
    ["Semua Dataset (Agregat)", "HAM10000", "ISIC 2017", "ISIC 2019"],
    index=0,
    help="Filter data pengujian berdasarkan repositori asal lesi kulit."
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🎯 Simulasi Ambang Keputusan (*Threshold*)")
ambang_aktif = st.sidebar.slider(
    "Ambang Keganasan (Cutoff):",
    min_value=0.00,
    max_value=1.00,
    value=float(ringkasan.get("thr_global", 0.42)),
    step=0.01,
    help="Probabilitas >= ambang diklasifikasikan sebagai Ganas (Malignant)."
)

st.sidebar.caption(
    f"📌 **Ambang Optimal Riset**: **`{ringkasan.get('thr_global', 0.42)}`**\n\n"
    f"Ambang per dataset riset:\n"
    f"- HAM10000: `{ringkasan['thr_dataset']['HAM10000']}`\n"
    f"- ISIC 2017: `{ringkasan['thr_dataset']['ISIC 2017']}`\n"
    f"- ISIC 2019: `{ringkasan['thr_dataset']['ISIC 2019']}`"
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Integritas Metodologi**\n\n"
    "Data uji diisolasi ketat pada level pasien/lesi (*Zero Patient Leakage*). "
    "Tidak ada citra dari pasien uji yang pernah terlihat pada data latih maupun validasi."
)

# ─── PEMISAH DATA SESUAI FILTER ───────────────────────────────────────────────
if dataset_pilihan == "Semua Dataset (Agregat)":
    sub_df = df_pred.copy()
else:
    sub_df = df_pred[df_pred["dataset"] == dataset_pilihan].copy()

# Kalkulasi metrik dinamis berdasarkan ambang slider
y_true = sub_df["target_binary"].values
y_prob = sub_df["prob"].values
y_pred = (y_prob >= ambang_aktif).astype(int)

tp = int(((y_pred == 1) & (y_true == 1)).sum())
tn = int(((y_pred == 0) & (y_true == 0)).sum())
fp = int(((y_pred == 1) & (y_true == 0)).sum())
fn = int(((y_pred == 0) & (y_true == 1)).sum())

n_total = len(y_true)
akurasi_dinamis = (tp + tn) / n_total * 100 if n_total > 0 else 0.0
sensitivitas_dinamis = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
spesifisitas_dinamis = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
ci_lo, ci_hi = wilson_interval(tp + tn, n_total)

# ─── BAGIAN 1: METRIK RINGKASAN DINAMIS ───────────────────────────────────────
st.markdown(f'<div class="section-header">1. Ringkasan Kinerja Uji ({dataset_pilihan} | Ambang: {ambang_aktif:.2f})</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Akurasi Pengujian</div>
        <div class="metric-val">{akurasi_dinamis:.2f}%</div>
        <div class="metric-sub">CI 95%: {ci_lo:.2f}% - {ci_hi:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Sensitivitas (Recall Ganas)</div>
        <div class="metric-val">{sensitivitas_dinamis:.2f}%</div>
        <div class="metric-sub">Terdeteksi: {tp} dari {tp + fn} kasus</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Spesifisitas (True Negative)</div>
        <div class="metric-val">{spesifisitas_dinamis:.2f}%</div>
        <div class="metric-sub">Jinak Benar: {tn} dari {tn + fp} kasus</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Total Citra Diuji</div>
        <div class="metric-val">{n_total:,}</div>
        <div class="metric-sub">Positif: {tp + fn} | Negatif: {tn + fp}</div>
    </div>
    """, unsafe_allow_html=True)

# ─── BAGIAN 2: MATRIKS KONFUSI & DETAIL SIMULASI ──────────────────────────────
st.markdown('<div class="section-header">2. Matriks Konfusi & Keseimbangan Diagnosis</div>', unsafe_allow_html=True)

cm_col1, cm_col2 = st.columns([1, 1.4])

with cm_col1:
    st.markdown("##### Matriks Konfusi (Hasil Ambang Terpilih)")
    
    # Render tabel matriks konfusi bersih
    cm_data = {
        "Kondisi Aktual": ["Aktual GANAS (Positif)", "Aktual JINAK (Negatif)"],
        "Prediksi GANAS": [f"TP: {tp:,} ({tp/n_total*100:.1f}%)", f"FP: {fp:,} ({fp/n_total*100:.1f}%)"],
        "Prediksi JINAK": [f"FN: {fn:,} ({fn/n_total*100:.1f}%)", f"TN: {tn:,} ({tn/n_total*100:.1f}%)"]
    }
    df_cm_display = pd.DataFrame(cm_data).set_index("Kondisi Aktual")
    st.table(df_cm_display)

    if ambang_aktif == 0.42:
        st.success("✅ **Ambang Riset Optimal (0,42)**: Menghasilkan trade-off ideal antara deteksi kanker ganas (Sensitivitas >81%) dan minimalisasi false alarm (Spesifisitas >87%).")
    elif ambang_aktif < 0.42:
        st.warning(f"⚠️ **Ambang Rendah ({ambang_aktif:.2f})**: Meningkatkan Sensitivitas ({sensitivitas_dinamis:.1f}%), namun False Positive meningkat (Spesifisitas turun ke {spesifisitas_dinamis:.1f}%).")
    else:
        st.info(f"ℹ️ **Ambang Tinggi ({ambang_aktif:.2f})**: Spesifisitas naik ({spesifisitas_dinamis:.1f}%), namun risiko False Negative meningkat ({fn} kasus ganas terlewat).")

with cm_col2:
    st.markdown("##### Sebaran Probabilitas Prediksi Model")
    
    fig_hist, ax_hist = plt.subplots(figsize=(6.5, 3.2), dpi=120)
    prob_ganas = sub_df[sub_df["target_binary"] == 1]["prob"]
    prob_jinak = sub_df[sub_df["target_binary"] == 0]["prob"]

    ax_hist.hist(prob_jinak, bins=35, alpha=0.65, color="#2563EB", label=f"Jinak / Benign (n={len(prob_jinak)})", density=True)
    ax_hist.hist(prob_ganas, bins=35, alpha=0.65, color="#DC2626", label=f"Ganas / Malignant (n={len(prob_ganas)})", density=True)
    ax_hist.axvline(ambang_aktif, color="#0F172A", linestyle="--", linewidth=1.5, label=f"Ambang ({ambang_aktif:.2f})")
    
    ax_hist.set_xlabel("Peluang Prediksi Keganasan P(Ganas)", fontsize=9)
    ax_hist.set_ylabel("Densitas", fontsize=9)
    ax_hist.tick_params(labelsize=8)
    ax_hist.legend(fontsize=8, loc="upper right")
    ax_hist.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_hist)
    plt.close()

# ─── BAGIAN 3: TABEL TOLOK UKUR RESMI RISET ───────────────────────────────────
st.markdown('<div class="section-header">3. Tabel Evaluasi Resmi Naskah Riset (Baseline Terkunci)</div>', unsafe_allow_html=True)
st.caption("Data berikut adalah hasil evaluasi formal yang dilaporkan dalam Laporan Akhir LPPM UTM & naskah JuTISI (SINTA 2).")

# Format tampilan tabel evaluasi
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

# ─── BAGIAN 4: BUKTI GRAFIS RISET & ABLASI PREPROCESSING ──────────────────────
st.markdown('<div class="section-header">4. Bukti Validasi Grafis & Studi Ablasi</div>', unsafe_allow_html=True)

tab_g1, tab_g2, tab_g3, tab_g4 = st.tabs([
    "📈 Kurva ROC & Sebaran Uji",
    "🔲 Matriks Konfusi Resmi",
    "⚡ Latensi & Parameter",
    "🧪 Ablasi Preprocessing (ROI + SoG)"
])

with tab_g1:
    col_r1, col_r2 = st.columns([1.5, 1])
    with col_r1:
        if (ASSETS_DIR / "roc_dan_sebaran.png").exists():
            st.image(str(ASSETS_DIR / "roc_dan_sebaran.png"), caption="Kurva ROC-AUC Agregat (0,9174) & Distribusi Skor Prediksi", use_container_width=True)
    with col_r2:
        st.markdown("""
        **Analisis Kurva ROC-AUC**:
        * **AUC Agregat 0,9174**: Menunjukkan daya pembeda model sangat unggul dalam memisahkan lesi ganas (*malignant*) dan jinak (*benign*).
        * Performa per repositori:
          * **HAM10000**: AUC `0,9500` (sangat tinggi)
          * **ISIC 2019**: AUC `0,8984`
          * **ISIC 2017**: AUC `0,8608`
        * Kurva ROC membuktikan stabilitas model pada berbagai variasi ambang operasi tanpa terjadi keruntuhan deteksi.
        """)

with tab_g2:
    col_m1, col_m2 = st.columns([1.5, 1])
    with col_m1:
        if (ASSETS_DIR / "matriks_konfusi.png").exists():
            st.image(str(ASSETS_DIR / "matriks_konfusi.png"), caption="Matriks Konfusi Standar & Terkalibrasi pada Data Uji", use_container_width=True)
    with col_m2:
        st.markdown("""
        **Analisis Matriks Konfusi Terkalibrasi**:
        * Pada data uji agregat ($n=3.291$), model mengidentifikasi **974 kasus ganas** dengan benar (*True Positive*) dan **1.832 kasus jinak** dengan benar (*True Negative*).
        * Tingkat akurasi tercatat **85,26%**, melampaui target batas minimal asistensi riset (85,00%).
        """)

with tab_g3:
    col_l1, col_l2 = st.columns([1.5, 1])
    with col_l1:
        if (ASSETS_DIR / "kinerja_dan_latensi_convnext.png").exists():
            st.image(str(ASSETS_DIR / "kinerja_dan_latensi_convnext.png"), caption="Grafik Latensi Inferensi & Parameter Model", use_container_width=True)
    with col_l2:
        st.markdown(f"""
        **Spesifikasi Komputasi Model**:
        * **Total Parameter**: `{ringkasan.get('parameter_juta', 33.08):.2f}` Juta Parameter.
        * **Latensi Standar (Tanpa TTA)**: `{ringkasan.get('ms_tanpa_tta', 3.76):.2f} ms` per citra (sangat cepat untuk *real-time clinical screening*).
        * **Latensi dengan TTA 4 Arah**: `{ringkasan.get('ms_tta', 12.21):.2f} ms` per citra (memberikan peningkatan akurasi dari 84,69% ke 85,26%).
        """)

with tab_g4:
    col_a1, col_a2 = st.columns([1.2, 1])
    with col_a1:
        st.markdown("##### Hasil Ablasi Praproses Citra (MobileNetV3 Backbone, 5 Epoch)")
        df_abl_tampil = df_abl.copy()
        df_abl_tampil.columns = ["Strategi Praproses", "Akurasi Val Epoch 5 (%)", "Akurasi Val Maks (%)", "AUC Val Epoch 5", "Akurasi Latih (%)"]
        st.dataframe(df_abl_tampil, use_container_width=True, hide_index=True)
    with col_a2:
        st.markdown("""
        **Justifikasi Pemilihan ROI + Shades of Gray**:
        * **Full Image**: Citra mentah rentan terganggu artefak rambut, vignetting lensa dermoskopi, dan bias saturasi warna.
        * **ROI Otsu**: Mengisolasi lesi aktif namun masih memiliki deviasi warna antar perangkat dermoskopi.
        * **ROI + SoG ($p=6$)**: Menghasilkan akurasi validasi tertinggi (**81,68%**) dan konsistensi representasi warna terbaik sebelum diumpankan ke model utama.
        """)

# ─── BAGIAN 5: PENJELAJAH DATA UJI INTERAKTIF ─────────────────────────────────
st.markdown('<div class="section-header">5. Penjelajah Sampel Data Uji (Audit Transparan)</div>', unsafe_allow_html=True)
st.caption("Cari dan periksa status prediksi setiap citra uji individual secara transparan.")

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    diagnosa_pilihan = st.selectbox(
        "Filter Kategori Diagnosa:",
        ["Semua Kategori", "Hanya GANAS (Malignant)", "Hanya JINAK (Benign)"],
        index=0
    )

with filter_col2:
    status_prediksi = st.selectbox(
        "Filter Kesesuaian Prediksi:",
        ["Semua Status", "True Positive (Benar Ganas)", "True Negative (Benar Jinak)", "False Positive (Keliru Ganas)", "False Negative (Ganas Terlewat)"],
        index=0
    )

with filter_col3:
    cari_id = st.text_input("Cari ID Citra (misal: ISIC_0000009):", "").strip().upper()

# Evaluasi status prediksi per baris
sub_df["status_diagnostik"] = np.where(sub_df["target_binary"] == 1, "GANAS", "JINAK")
sub_df["prediksi_label"] = np.where(sub_df["prob"] >= ambang_aktif, "GANAS", "JINAK")

def tentukan_status_eval(r):
    if r["target_binary"] == 1 and r["prob"] >= ambang_aktif:
        return "True Positive"
    elif r["target_binary"] == 0 and r["prob"] < ambang_aktif:
        return "True Negative"
    elif r["target_binary"] == 0 and r["prob"] >= ambang_aktif:
        return "False Positive"
    else:
        return "False Negative"

sub_df["kategori_evaluasi"] = sub_df.apply(tentukan_status_eval, axis=1)

# Terapkan filter
filtered_df = sub_df.copy()

if diagnosa_pilihan == "Hanya GANAS (Malignant)":
    filtered_df = filtered_df[filtered_df["target_binary"] == 1]
elif diagnosa_pilihan == "Hanya JINAK (Benign)":
    filtered_df = filtered_df[filtered_df["target_binary"] == 0]

if status_prediksi != "Semua Status":
    keyword_status = status_prediksi.split(" (")[0]
    filtered_df = filtered_df[filtered_df["kategori_evaluasi"] == keyword_status]

if cari_id:
    filtered_df = filtered_df[filtered_df["image_id"].str.contains(cari_id, na=False)]

st.markdown(f"Menampilkan **{len(filtered_df):,}** citra dari total **{len(sub_df):,}** sampel data uji:")

tampil_explorer = filtered_df[["image_id", "dataset", "unified_class", "status_diagnostik", "prob", "prediksi_label", "kategori_evaluasi"]].copy()
tampil_explorer["prob"] = tampil_explorer["prob"].map(lambda p: f"{p:.4f} ({p*100:.1f}%)")
tampil_explorer.columns = ["ID Citra", "Dataset", "Sub-Kelas", "Kondisi Riil", "Probabilitas Keganasan", "Hasil Klasifikasi", "Status Evaluasi"]

st.dataframe(tampil_explorer, use_container_width=True, hide_index=True, height=350)

# Unduh Data CSV
csv_export = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Unduh Data Hasil Uji Terfilter (CSV)",
    data=csv_export,
    file_name=f"hasil_uji_{dataset_pilihan.replace(' ', '_').lower()}.csv",
    mime="text/csv",
)

# ─── FOOTER & INTEGRITAS RISET ────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="font-size: 0.82rem; color: #64748B; text-align: center; line-height: 1.6;">
    <strong>⚡ DERMASPARK (Dermatology × Apache Spark)</strong> &bull; Platform Evaluasi Hasil Pengujian Model ConvNeXtTiny-MHSA<br>
    Asistensi Hibah Penelitian BIMA Kemendikbudristek / LPPM Universitas Trunojoyo Madura 2026<br>
    <em>Integritas Data: Bebas Bias Pasien (Zero Patient Leakage) | Catatan Klinis: Sistem ini berfungsi murni sebagai Decision Support System (DSS) dan komputasi performa partisi citra medis, bukan instrumen diagnosis mandiri tanpa supervisi spesialis patologi/dermatologi.</em>
</div>
""", unsafe_allow_html=True)
