# pages/2_Prédictions.py

import json, joblib
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import load_train_recent, load_items, load_stores, data_signature, load_metadata

st.set_page_config(
    page_title="Prédictions - Favorita",
    page_icon="🔮",
    layout="wide",
)

# ============================================================
# DESIGN PREMIUM - CSS MODERNE ULTRA
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.block-container {
    padding: 2rem 3rem 3rem 3rem;
    max-width: 1500px;
}

/* ===== MEGA HERO HEADER ===== */
.prediction-hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 28px;
    padding: 3.5rem 3rem;
    margin-bottom: 2.5rem;
    box-shadow: 0 25px 70px rgba(102, 126, 234, 0.35);
    position: relative;
    overflow: hidden;
}

.prediction-hero::before {
    content: '';
    position: absolute;
    top: -100px;
    right: -100px;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 60%);
    border-radius: 50%;
}

.prediction-hero::after {
    content: '';
    position: absolute;
    bottom: -80px;
    left: -80px;
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-content {
    position: relative;
    z-index: 1;
}

.hero-title {
    color: white;
    font-size: 3.2rem;
    font-weight: 900;
    margin: 0 0 1rem 0;
    letter-spacing: -0.03em;
    text-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.hero-subtitle {
    color: rgba(255,255,255,0.95);
    font-size: 1.2rem;
    margin: 0;
    font-weight: 400;
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.25);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    color: white;
    font-weight: 600;
    margin-top: 1.5rem;
    font-size: 0.95rem;
}

/* ===== INFO CARD ===== */
.info-card {
    background: white;
    border-radius: 20px;
    padding: 1.8rem;
    box-shadow: 0 6px 25px rgba(0,0,0,0.08);
    border: 1px solid rgba(0,0,0,0.05);
}

.info-label {
    font-size: 0.8rem;
    color: #999;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}

.info-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #667eea;
    margin-bottom: 0.3rem;
}

.info-detail {
    font-size: 0.85rem;
    color: #666;
}

/* ===== MEGA PREDICTION CARD ===== */
.mega-prediction {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    border-radius: 28px;
    padding: 3rem;
    margin: 2rem 0;
    box-shadow: 0 25px 60px rgba(240, 147, 251, 0.4);
    position: relative;
    overflow: hidden;
}

.mega-prediction::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}

.prediction-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 3rem;
    position: relative;
    z-index: 1;
}

.prediction-main {
    color: white;
}

.prediction-label {
    font-size: 1rem;
    opacity: 0.9;
    font-weight: 500;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.prediction-value {
    font-size: 5.5rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 1rem;
    text-shadow: 0 6px 20px rgba(0,0,0,0.2);
}

.prediction-unit {
    font-size: 1.2rem;
    opacity: 0.95;
    font-weight: 500;
}

.prediction-details {
    display: grid;
    gap: 1rem;
}

.detail-row {
    background: rgba(255,255,255,0.2);
    backdrop-filter: blur(10px);
    border-radius: 14px;
    padding: 1rem 1.3rem;
    color: white;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.detail-label {
    font-size: 0.9rem;
    opacity: 0.9;
}

.detail-value {
    font-size: 1.1rem;
    font-weight: 700;
}

/* ===== METRICS ===== */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.5rem;
    margin: 2rem 0;
}

.metric-box {
    background: white;
    border-radius: 18px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    border-top: 4px solid;
    transition: all 0.3s ease;
}

.metric-box:nth-child(1) { border-top-color: #667eea; }
.metric-box:nth-child(2) { border-top-color: #f093fb; }
.metric-box:nth-child(3) { border-top-color: #11998e; }

.metric-box:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
}

.metric-icon {
    font-size: 2.5rem;
    margin-bottom: 0.8rem;
}

.metric-label {
    font-size: 0.85rem;
    color: #999;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #1a1a1a;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8f9ff 0%, #fff 100%);
    border-right: 1px solid rgba(0,0,0,0.08);
}

section[data-testid="stSidebar"] h2 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.6rem;
    font-weight: 800;
    margin-bottom: 2rem;
}

/* ===== EXPANDER ===== */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, #f8f9ff 0%, #fff 100%);
    border-radius: 14px;
    padding: 1rem 1.3rem;
    font-weight: 700;
    border: 2px solid rgba(102, 126, 234, 0.2);
    transition: all 0.3s ease;
}

.streamlit-expanderHeader:hover {
    border-color: rgba(102, 126, 234, 0.5);
    background: linear-gradient(135deg, #f0f4ff 0%, #fff 100%);
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.8rem;
    background: white;
    border-radius: 18px;
    padding: 0.6rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 14px;
    padding: 1rem 2rem;
    font-weight: 700;
    font-size: 1.05rem;
    color: #666;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    transform: translateY(-2px);
}

/* ===== BUTTONS ===== */
div.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.85rem 2.2rem;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    transition: all 0.3s ease;
}

div.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5);
}

/* ===== INPUTS ===== */
.stSelectbox > div > div,
.stDateInput > div > div,
.stTextInput > div > div,
.stMultiSelect > div > div {
    border-radius: 14px;
    border: 2px solid #e8e8e8;
    transition: all 0.3s ease;
}

.stSelectbox > div > div:focus-within,
.stDateInput > div > div:focus-within,
.stTextInput > div > div:focus-within,
.stMultiSelect > div > div:focus-within {
    border-color: #667eea;
    box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.12);
}

/* ===== CHART CONTAINER ===== */
.chart-wrapper {
    background: white;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 6px 25px rgba(0,0,0,0.08);
    margin: 1.5rem 0;
}

.chart-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1a1a1a;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 3px solid #f0f0f0;
}

/* ===== ANIMATIONS ===== */
@keyframes slideInUp {
    from {
        opacity: 0;
        transform: translateY(40px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.85;
    }
}

.mega-prediction {
    animation: slideInUp 0.7s ease-out;
}

.prediction-value {
    animation: pulse 2s ease-in-out infinite;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS / PARAMS
# ============================================================
DATA_DIR = st.session_state.get("data_dir", "data/favorita_data")
WEEKS    = st.session_state.get("weeks_window", 12)

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "best_model.pkl"
PIPE_PATH  = MODELS_DIR / "feature_pipeline.pkl"
FEAT_PATH  = MODELS_DIR / "features.json"

# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_local_artifacts():
    model = joblib.load(MODEL_PATH)
    pipe  = joblib.load(PIPE_PATH)
    feature_cols = json.load(open(FEAT_PATH, "r", encoding="utf-8"))
    return model, pipe, feature_cols

# ============================================================
# CHECK ARTIFACTS
# ============================================================
if (not MODEL_PATH.exists()) or (not PIPE_PATH.exists()) or (not FEAT_PATH.exists()):
    st.error("❌ Modèle/pipeline introuvables. Va dans Admin → Retrain.")
    st.stop()

current_sig = data_signature(DATA_DIR, weeks=WEEKS)
meta = load_metadata(MODELS_DIR)

if meta.get("data_signature") != current_sig:
    st.warning("⚠️ Les données ont changé depuis le dernier entraînement.")
    st.info("Va dans Admin → clique sur Retrain pour générer un modèle compatible.")
    st.stop()

model, pipe, feature_cols = load_local_artifacts()

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_recent_data(data_dir: str, weeks: int):
    df_ = load_train_recent(data_dir, weeks=weeks)
    df_["date"] = pd.to_datetime(df_["date"], errors="coerce").dt.normalize()
    return df_

df = load_recent_data(DATA_DIR, WEEKS)

store_list = np.sort(df["store_nbr"].dropna().unique()).tolist()
item_list  = np.sort(df["item_nbr"].dropna().unique()).tolist()

min_d = df["date"].min().date()
max_d = df["date"].max().date()

# ============================================================
# HEADER
# ============================================================
col_hero, col_info = st.columns([0.7, 0.3])

with col_hero:
    st.markdown("""
    <div class="prediction-hero">
        <div class="hero-content">
            <div class="hero-title">🔮 Prédictions IA</div>
            <div class="hero-subtitle">Moteur de prévision des ventes basé sur Machine Learning avancé</div>
            <div class="hero-badge">✨ Modèle sklearn optimisé</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_info:
    st.markdown(f"""
    <div class="info-card">
        <div class="info-label">Fenêtre de données</div>
        <div class="info-value">{WEEKS} semaines</div>
        <div class="info-detail">📅 {min_d} → {max_d}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🎯 Configuration")

    with st.expander("⚡ Prédiction Instantanée", expanded=True):
        date_in = st.date_input("📅 Date", value=max_d, min_value=min_d, max_value=max_d)
        store_nbr = st.selectbox("🏪 Store", options=store_list, index=0)

        q = st.text_input("🔍 Rechercher un item", value="", placeholder="ID de l'item...")
        if q.strip():
            item_opts = [x for x in item_list if q.strip() in str(x)][:5000]
        else:
            item_opts = item_list[:5000]

        item_nbr = st.selectbox("📦 Item", options=item_opts, index=0)
        onpromotion = st.checkbox("🏷️ En promotion", value=False)

    with st.expander("📊 Prédiction sur Période", expanded=False):
        date_range = st.date_input(
            "Période",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d,
            key="pred_date_range",
        )
        store_sel = st.multiselect("Stores", options=store_list, default=[])
        item_sel  = st.multiselect("Items", options=item_opts, default=[])
        run_period = st.button("🚀 Lancer Prédiction")

# ============================================================
# TABS
# ============================================================
tab1, tab2 = st.tabs(["⚡ Instantané", "📈 Période"])

# ============================================================
# TAB 1 — SINGLE PRED
# ============================================================
with tab1:
    new_df = pd.DataFrame({
        "date": [pd.to_datetime(date_in)],
        "store_nbr": [int(store_nbr)],
        "item_nbr": [int(item_nbr)],
        "onpromotion": [bool(onpromotion)],
    })

    X_enriched = pipe.transform(new_df)
    X = (X_enriched
         .reindex(columns=feature_cols, fill_value=0)
         .replace([np.inf, -np.inf], np.nan)
         .fillna(0))

    pred_log = float(model.predict(X)[0])
    pred_sales = float(np.expm1(pred_log))

    st.markdown(f"""
    <div class="mega-prediction">
        <div class="prediction-grid">
            <div class="prediction-main">
                <div class="prediction-label">Prévision estimée</div>
                <div class="prediction-value">{pred_sales:.2f}</div>
                <div class="prediction-unit">unités vendues</div>
            </div>
            <div class="prediction-details">
                <div class="detail-row">
                    <div class="detail-label">📅 Date</div>
                    <div class="detail-value">{pd.to_datetime(date_in).strftime('%d/%m/%Y')}</div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">🏪 Store</div>
                    <div class="detail-value">{store_nbr}</div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">📦 Item</div>
                    <div class="detail-value">{item_nbr}</div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">🏷️ Promotion</div>
                    <div class="detail-value">{'✅ Oui' if onpromotion else '❌ Non'}</div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">🧮 Log(pred)</div>
                    <div class="detail-value">{pred_log:.4f}</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-box">
            <div class="metric-icon">🎯</div>
            <div class="metric-label">Ventes prévues</div>
            <div class="metric-value">{pred_sales:.2f}</div>
        </div>
        <div class="metric-box">
            <div class="metric-icon">🧪</div>
            <div class="metric-label">Log Prédiction</div>
            <div class="metric-value">{pred_log:.4f}</div>
        </div>
        <div class="metric-box">
            <div class="metric-icon">🏷️</div>
            <div class="metric-label">Statut Promo</div>
            <div class="metric-value">{'OUI' if onpromotion else 'NON'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔍 Détails de l'observation"):
        st.dataframe(new_df, use_container_width=True)

# ============================================================
# TAB 2 — PERIOD PRED
# ============================================================
with tab2:
    st.markdown("### 📊 Prédictions sur période avec filtres")

    if run_period:
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start_d = pd.to_datetime(date_range[0])
            end_d   = pd.to_datetime(date_range[1])
        else:
            start_d = pd.to_datetime(date_range)
            end_d   = pd.to_datetime(date_range)

        if start_d > end_d:
            start_d, end_d = end_d, start_d

        f = df.loc[(df["date"] >= start_d) & (df["date"] <= end_d)].copy()

        if store_sel:
            f = f.loc[f["store_nbr"].isin(store_sel)]
        if item_sel:
            f = f.loc[f["item_nbr"].isin(item_sel)]

        if len(f) == 0:
            st.warning("⚠️ Aucune ligne après filtres.")
            st.stop()

        nmax = 300_000
        if len(f) > nmax:
            f = f.sample(nmax, random_state=42)
            st.info(f"📊 Dataset échantillonné : {nmax:,} lignes")

        with st.spinner("⚙️ Calcul en cours..."):
            Xf_enriched = pipe.transform(f)
            Xf = (Xf_enriched
                  .reindex(columns=feature_cols, fill_value=0)
                  .replace([np.inf, -np.inf], np.nan)
                  .fillna(0))

            pred_log_arr = model.predict(Xf)
            pred = np.expm1(pred_log_arr)

        out = f[["date", "store_nbr", "item_nbr"]].copy()
        out["pred_unit_sales"] = pred.astype("float32")

        total = float(out["pred_unit_sales"].sum())
        avg   = float(out["pred_unit_sales"].mean())
        nrows = int(len(out))

        st.markdown(f"""
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-icon">📦</div>
                <div class="metric-label">Total prédit</div>
                <div class="metric-value">{total:,.0f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-icon">📈</div>
                <div class="metric-label">Moyenne / ligne</div>
                <div class="metric-value">{avg:.2f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-icon">🧾</div>
                <div class="metric-label">Lignes traitées</div>
                <div class="metric-value">{nrows:,}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">📈 Évolution temporelle</div>', unsafe_allow_html=True)
        g1 = out.groupby("date", as_index=False)["pred_unit_sales"].sum()
        st.line_chart(g1.set_index("date"))
        st.markdown('</div>', unsafe_allow_html=True)

        cA, cB = st.columns(2)
        with cA:
            st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">🏪 Top Stores</div>', unsafe_allow_html=True)
            g2 = (out.groupby("store_nbr", as_index=False)["pred_unit_sales"]
                    .sum().sort_values("pred_unit_sales", ascending=False).head(15))
            st.bar_chart(g2.set_index("store_nbr"))
            st.markdown('</div>', unsafe_allow_html=True)

        with cB:
            st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">📦 Top Items</div>', unsafe_allow_html=True)
            g3 = (out.groupby("item_nbr", as_index=False)["pred_unit_sales"]
                    .sum().sort_values("pred_unit_sales", ascending=False).head(15))
            st.bar_chart(g3.set_index("item_nbr"))
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("📄 Table de prédictions"):
            st.dataframe(out.head(200), use_container_width=True)
            # À ajouter dans CHAQUE page (app.py, 0_Admin.py, 1_Exploration.py, 2_Prédictions.py)
# Remplace ou ajoute ce CSS dans la section st.markdown() existante

            st.markdown("""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
            
            /* ===== SIDEBAR ULTRA PREMIUM ===== */
            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
                border-right: 2px solid rgba(0,0,0,0.06);
            }
            
            section[data-testid="stSidebar"] .block-container {
                padding-top: 1.5rem;
            }
            
            /* ===== SIDEBAR TITLES ===== */
            section[data-testid="stSidebar"] h1,
            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3 {
                background: linear-gradient(135deg, #1e3c72 0%, #7e22ce 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-weight: 900;
                margin-bottom: 1.5rem;
                padding-bottom: 1rem;
                border-bottom: 3px solid #f3f4f6;
                letter-spacing: -0.02em;
            }
            
            section[data-testid="stSidebar"] h2 {
                font-size: 1.4rem;
            }
            
            /* ===== SIDEBAR SECTIONS ===== */
            section[data-testid="stSidebar"] > div > div > div > div {
                padding: 0.5rem 0;
            }
            
            /* ===== INPUTS PREMIUM ===== */
            section[data-testid="stSidebar"] .stTextInput > div > div,
            section[data-testid="stSidebar"] .stSelectbox > div > div,
            section[data-testid="stSidebar"] .stDateInput > div > div,
            section[data-testid="stSidebar"] .stMultiSelect > div > div,
            section[data-testid="stSidebar"] .stNumberInput > div > div {
                border-radius: 14px;
                border: 2px solid #e5e7eb;
                background: white;
                transition: all 0.3s ease;
                box-shadow: 0 2px 8px rgba(0,0,0,0.03);
            }
            
            section[data-testid="stSidebar"] .stTextInput > div > div:focus-within,
            section[data-testid="stSidebar"] .stSelectbox > div > div:focus-within,
            section[data-testid="stSidebar"] .stDateInput > div > div:focus-within,
            section[data-testid="stSidebar"] .stMultiSelect > div > div:focus-within,
            section[data-testid="stSidebar"] .stNumberInput > div > div:focus-within {
                border-color: #667eea;
                box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.12);
                transform: translateY(-1px);
            }
            
            /* ===== LABELS PREMIUM ===== */
            section[data-testid="stSidebar"] label {
                font-weight: 600 !important;
                color: #374151 !important;
                font-size: 0.9rem !important;
                margin-bottom: 0.5rem !important;
                display: flex !important;
                align-items: center !important;
                gap: 0.5rem !important;
            }
            
            /* ===== CHECKBOX PREMIUM ===== */
            section[data-testid="stSidebar"] .stCheckbox {
                background: white;
                border-radius: 12px;
                padding: 0.8rem 1rem;
                border: 2px solid #e5e7eb;
                transition: all 0.3s ease;
            }
            
            section[data-testid="stSidebar"] .stCheckbox:hover {
                border-color: #667eea;
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(255,255,255,1) 100%);
            }
            
            /* ===== RADIO BUTTONS PREMIUM ===== */
            section[data-testid="stSidebar"] .stRadio {
                background: white;
                border-radius: 16px;
                padding: 1rem;
                box-shadow: 0 2px 12px rgba(0,0,0,0.04);
                border: 2px solid #f3f4f6;
            }
            
            section[data-testid="stSidebar"] .stRadio > label {
                font-weight: 700 !important;
                color: #1f2937 !important;
                margin-bottom: 1rem !important;
            }
            
            section[data-testid="stSidebar"] .stRadio > div {
                gap: 0.5rem;
            }
            
            section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
                background: linear-gradient(135deg, #f8fafc 0%, #fff 100%);
                border-radius: 10px;
                padding: 0.7rem 1rem;
                border: 2px solid #e5e7eb;
                transition: all 0.3s ease;
                margin: 0.3rem 0;
            }
            
            section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover {
                border-color: #667eea;
                background: linear-gradient(135deg, #f0f4ff 0%, #fff 100%);
                transform: translateX(4px);
            }
            
            /* ===== SLIDER PREMIUM ===== */
            section[data-testid="stSidebar"] .stSlider {
                padding: 1rem;
                background: white;
                border-radius: 14px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.04);
            }
            
            section[data-testid="stSidebar"] .stSlider > div > div > div > div {
                background: linear-gradient(90deg, #1e3c72 0%, #7e22ce 100%) !important;
            }
            
            section[data-testid="stSidebar"] .stSlider > div > div > div > div > div {
                background: white !important;
                border: 3px solid #667eea !important;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
            }
            
            /* ===== MULTISELECT PREMIUM ===== */
            section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                border: none !important;
                border-radius: 8px !important;
                padding: 0.3rem 0.7rem !important;
                font-weight: 600 !important;
            }
            
            /* ===== EXPANDER PREMIUM ===== */
            section[data-testid="stSidebar"] .streamlit-expanderHeader {
                background: linear-gradient(135deg, #f8fafc 0%, #fff 100%);
                border-radius: 14px;
                padding: 1rem 1.2rem;
                font-weight: 700;
                border: 2px solid #e5e7eb;
                transition: all 0.3s ease;
                color: #1f2937;
                box-shadow: 0 2px 8px rgba(0,0,0,0.03);
            }
            
            section[data-testid="stSidebar"] .streamlit-expanderHeader:hover {
                border-color: #667eea;
                background: linear-gradient(135deg, #f0f4ff 0%, #fff 100%);
                transform: translateX(4px);
            }
            
            section[data-testid="stSidebar"] .streamlit-expanderContent {
                background: white;
                border: 2px solid #f3f4f6;
                border-top: none;
                border-radius: 0 0 14px 14px;
                padding: 1rem;
            }
            
            /* ===== CAPTION / INFO TEXT ===== */
            section[data-testid="stSidebar"] .stCaption,
            section[data-testid="stSidebar"] small {
                color: #6b7280 !important;
                font-size: 0.85rem !important;
                font-weight: 500 !important;
                padding: 0.5rem 0;
                display: block;
            }
            
            /* ===== DIVIDER PREMIUM ===== */
            section[data-testid="stSidebar"] hr {
                border: none;
                height: 2px;
                background: linear-gradient(90deg, transparent, #667eea, transparent);
                margin: 1.5rem 0;
                border-radius: 2px;
            }
            
            /* ===== FILE UPLOADER PREMIUM ===== */
            section[data-testid="stSidebar"] .stFileUploader {
                background: white;
                border-radius: 14px;
                border: 2px dashed #e5e7eb;
                padding: 1rem;
                transition: all 0.3s ease;
            }
            
            section[data-testid="stSidebar"] .stFileUploader:hover {
                border-color: #667eea;
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.03) 0%, white 100%);
            }
            
            /* ===== BUTTON IN SIDEBAR ===== */
            section[data-testid="stSidebar"] .stButton > button {
                width: 100%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 0.8rem 1.5rem;
                font-size: 0.95rem;
                font-weight: 700;
                box-shadow: 0 4px 14px rgba(102, 126, 234, 0.3);
                transition: all 0.3s ease;
                margin-top: 0.5rem;
            }
            
            section[data-testid="stSidebar"] .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
            }
            
            /* ===== ANIMATIONS ===== */
            @keyframes slideInLeft {
                from {
                    opacity: 0;
                    transform: translateX(-20px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }
            
            section[data-testid="stSidebar"] > div > div > div > div {
                animation: slideInLeft 0.4s ease-out backwards;
            }
            
            section[data-testid="stSidebar"] > div > div > div > div:nth-child(1) { animation-delay: 0.1s; }
            section[data-testid="stSidebar"] > div > div > div > div:nth-child(2) { animation-delay: 0.15s; }
            section[data-testid="stSidebar"] > div > div > div > div:nth-child(3) { animation-delay: 0.2s; }
            section[data-testid="stSidebar"] > div > div > div > div:nth-child(4) { animation-delay: 0.25s; }
            section[data-testid="stSidebar"] > div > div > div > div:nth-child(5) { animation-delay: 0.3s; }
            
            /* ===== SCROLLBAR PREMIUM ===== */
            section[data-testid="stSidebar"] ::-webkit-scrollbar {
                width: 8px;
            }
            
            section[data-testid="stSidebar"] ::-webkit-scrollbar-track {
                background: #f3f4f6;
                border-radius: 10px;
            }
            
            section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 10px;
            }
            
            section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(135deg, #5568d3 0%, #6a3f8f 100%);
            }
            </style>
            """, unsafe_allow_html=True)
                        
                        # À ajouter dans CHAQUE page (app.py, 0_Admin.py, 1_Exploration.py, 2_Prédictions.py)
# Remplace ou ajoute ce CSS dans la section st.markdown() existante
            
            st.markdown("""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
            
            /* ===== SIDEBAR ULTRA PREMIUM ===== */
            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
                border-right: 2px solid rgba(0,0,0,0.06);
            }
            
            section[data-testid="stSidebar"] .block-container {
                padding-top: 1.5rem;
            }
            
            /* ===== SIDEBAR TITLES ===== */
            section[data-testid="stSidebar"] h1,
            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3 {
                background: linear-gradient(135deg, #1e3c72 0%, #7e22ce 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-weight: 900;
                margin-bottom: 1.5rem;
                padding-bottom: 1rem;
                border-bottom: 3px solid #f3f4f6;
                letter-spacing: -0.02em;
            }
            
            section[data-testid="stSidebar"] h2 {
                font-size: 1.4rem;
            }
            
            /* ===== SIDEBAR SECTIONS ===== */
            section[data-testid="stSidebar"] > div > div > div > div {
                padding: 0.5rem 0;
            }
            
            /* ===== INPUTS PREMIUM ===== */
            section[data-testid="stSidebar"] .stTextInput > div > div,
            section[data-testid="stSidebar"] .stSelectbox > div > div,
            section[data-testid="stSidebar"] .stDateInput > div > div,
            section[data-testid="stSidebar"] .stMultiSelect > div > div,
            section[data-testid="stSidebar"] .stNumberInput > div > div {
                border-radius: 14px;
                border: 2px solid #e5e7eb;
                background: white;
                transition: all 0.3s ease;
                box-shadow: 0 2px 8px rgba(0,0,0,0.03);
            }
            
            section[data-testid="stSidebar"] .stTextInput > div > div:focus-within,
            section[data-testid="stSidebar"] .stSelectbox > div > div:focus-within,
            section[data-testid="stSidebar"] .stDateInput > div > div:focus-within,
            section[data-testid="stSidebar"] .stMultiSelect > div > div:focus-within,
            section[data-testid="stSidebar"] .stNumberInput > div > div:focus-within {
                border-color: #667eea;
                box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.12);
                transform: translateY(-1px);
            }
            
            /* ===== LABELS PREMIUM ===== */
            section[data-testid="stSidebar"] label {
                font-weight: 600 !important;
                color: #374151 !important;
                font-size: 0.9rem !important;
                margin-bottom: 0.5rem !important;
                display: flex !important;
                align-items: center !important;
                gap: 0.5rem !important;
            }
            
            /* ===== CHECKBOX PREMIUM ===== */
            section[data-testid="stSidebar"] .stCheckbox {
                background: white;
                border-radius: 12px;
                padding: 0.8rem 1rem;
                border: 2px solid #e5e7eb;
                transition: all 0.3s ease;
            }
            
            section[data-testid="stSidebar"] .stCheckbox:hover {
                border-color: #667eea;
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(255,255,255,1) 100%);
            }
            
            /* ===== RADIO BUTTONS PREMIUM ===== */
            section[data-testid="stSidebar"] .stRadio {
                background: white;
                border-radius: 16px;
                padding: 1rem;
                box-shadow: 0 2px 12px rgba(0,0,0,0.04);
                border: 2px solid #f3f4f6;
            }
            
            section[data-testid="stSidebar"] .stRadio > label {
                font-weight: 700 !important;
                color: #1f2937 !important;
                margin-bottom: 1rem !important;
            }
            
            section[data-testid="stSidebar"] .stRadio > div {
                gap: 0.5rem;
            }
            
            section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
                background: linear-gradient(135deg, #f8fafc 0%, #fff 100%);
                border-radius: 10px;
                padding: 0.7rem 1rem;
                border: 2px solid #e5e7eb;
                transition: all 0.3s ease;
                margin: 0.3rem 0;
            }
            
            section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover {
                border-color: #667eea;
                background: linear-gradient(135deg, #f0f4ff 0%, #fff 100%);
                transform: translateX(4px);
            }
            
            /* ===== SLIDER PREMIUM ===== */
            section[data-testid="stSidebar"] .stSlider {
                padding: 1rem;
                background: white;
                border-radius: 14px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.04);
            }
            
            section[data-testid="stSidebar"] .stSlider > div > div > div > div {
                background: linear-gradient(90deg, #1e3c72 0%, #7e22ce 100%) !important;
            }
            
            section[data-testid="stSidebar"] .stSlider > div > div > div > div > div {
                background: white !important;
                border: 3px solid #667eea !important;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
            }
            
            /* ===== MULTISELECT PREMIUM ===== */
            section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                border: none !important;
                border-radius: 8px !important;
                padding: 0.3rem 0.7rem !important;
                font-weight: 600 !important;
            }
            
            /* ===== EXPANDER PREMIUM ===== */
            section[data-testid="stSidebar"] .streamlit-expanderHeader {
                background: linear-gradient(135deg, #f8fafc 0%, #fff 100%);
                border-radius: 14px;
                padding: 1rem 1.2rem;
                font-weight: 700;
                border: 2px solid #e5e7eb;
                transition: all 0.3s ease;
                color: #1f2937;
                box-shadow: 0 2px 8px rgba(0,0,0,0.03);
            }
            
            section[data-testid="stSidebar"] .streamlit-expanderHeader:hover {
                border-color: #667eea;
                background: linear-gradient(135deg, #f0f4ff 0%, #fff 100%);
                transform: translateX(4px);
            }
            
            section[data-testid="stSidebar"] .streamlit-expanderContent {
                background: white;
                border: 2px solid #f3f4f6;
                border-top: none;
                border-radius: 0 0 14px 14px;
                padding: 1rem;
            }
            
            /* ===== CAPTION / INFO TEXT ===== */
            section[data-testid="stSidebar"] .stCaption,
            section[data-testid="stSidebar"] small {
                color: #6b7280 !important;
                font-size: 0.85rem !important;
                font-weight: 500 !important;
                padding: 0.5rem 0;
                display: block;
            }
            
            /* ===== DIVIDER PREMIUM ===== */
            section[data-testid="stSidebar"] hr {
                border: none;
                height: 2px;
                background: linear-gradient(90deg, transparent, #667eea, transparent);
                margin: 1.5rem 0;
                border-radius: 2px;
            }
            
            /* ===== FILE UPLOADER PREMIUM ===== */
            section[data-testid="stSidebar"] .stFileUploader {
                background: white;
                border-radius: 14px;
                border: 2px dashed #e5e7eb;
                padding: 1rem;
                transition: all 0.3s ease;
            }
            
            section[data-testid="stSidebar"] .stFileUploader:hover {
                border-color: #667eea;
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.03) 0%, white 100%);
            }
            
            /* ===== BUTTON IN SIDEBAR ===== */
            section[data-testid="stSidebar"] .stButton > button {
                width: 100%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 0.8rem 1.5rem;
                font-size: 0.95rem;
                font-weight: 700;
                box-shadow: 0 4px 14px rgba(102, 126, 234, 0.3);
                transition: all 0.3s ease;
                margin-top: 0.5rem;
            }
            
            section[data-testid="stSidebar"] .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
            }
            
            /* ===== ANIMATIONS ===== */
            @keyframes slideInLeft {
                from {
                    opacity: 0;
                    transform: translateX(-20px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }
            
            section[data-testid="stSidebar"] > div > div > div > div {
                animation: slideInLeft 0.4s ease-out backwards;
            }
            
            section[data-testid="stSidebar"] > div > div > div > div:nth-child(1) { animation-delay: 0.1s; }
            section[data-testid="stSidebar"] > div > div > div > div:nth-child(2) { animation-delay: 0.15s; }
            section[data-testid="stSidebar"] > div > div > div > div:nth-child(3) { animation-delay: 0.2s; }
            section[data-testid="stSidebar"] > div > div > div > div:nth-child(4) { animation-delay: 0.25s; }
            section[data-testid="stSidebar"] > div > div > div > div:nth-child(5) { animation-delay: 0.3s; }
            
            /* ===== SCROLLBAR PREMIUM ===== */
            section[data-testid="stSidebar"] ::-webkit-scrollbar {
                width: 8px;
            }
            
            section[data-testid="stSidebar"] ::-webkit-scrollbar-track {
                background: #f3f4f6;
                border-radius: 10px;
            }
            
            section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 10px;
            }
            
            section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(135deg, #5568d3 0%, #6a3f8f 100%);
            }
            </style>
            """, unsafe_allow_html=True)