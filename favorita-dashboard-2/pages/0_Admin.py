# pages/0_Admin.py
# -*- coding: utf-8 -*-

from pathlib import Path
import streamlit as st

from utils.data_loader import (
    build_recent_parquet,
    load_train_recent,
    data_signature,
    load_metadata,
)
from utils.training import train_reference_model

st.set_page_config(page_title="Admin – Upload & Retrain", layout="wide", page_icon="⚙️")

# ============================================================
# DESIGN PREMIUM - CSS MODERNE
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

.block-container { padding: 2rem 3rem 3rem 3rem; max-width: 1400px; }

/* ===== HEADER HERO ===== */
.admin-hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 24px;
    padding: 3rem 2.5rem;
    margin-bottom: 2.5rem;
    box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
    position: relative;
    overflow: hidden;
}
.admin-hero::before{
    content:'';
    position:absolute;
    top:0; right:0;
    width:300px; height:300px;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    border-radius:50%;
    transform: translate(30%, -30%);
}
.admin-hero h1{
    color:white;
    font-size:2.8rem;
    font-weight:800;
    margin:0 0 0.5rem 0;
    letter-spacing:-0.02em;
}
.admin-hero p{
    color:rgba(255,255,255,0.9);
    font-size:1.1rem;
    margin:0;
    font-weight:400;
}

/* ===== SECTIONS ===== */
.section-card{
    background:white;
    border-radius:20px;
    padding:2rem;
    margin-bottom:2rem;
    box-shadow:0 4px 20px rgba(0,0,0,0.06);
    border:1px solid rgba(0,0,0,0.04);
    transition:all 0.3s ease;
}
.section-card:hover{
    box-shadow:0 8px 30px rgba(0,0,0,0.1);
    transform: translateY(-2px);
}
.section-header{
    display:flex;
    align-items:center;
    gap:1rem;
    margin-bottom:1.5rem;
    padding-bottom:1rem;
    border-bottom:2px solid #f0f0f0;
}
.section-icon{
    font-size:2rem;
    display:flex;
    align-items:center;
    justify-content:center;
    width:60px; height:60px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius:16px;
    box-shadow:0 4px 14px rgba(102,126,234,0.3);
}
.section-title{
    font-size:1.5rem;
    font-weight:700;
    color:#1a1a1a;
    margin:0;
}
.section-subtitle{
    font-size:0.9rem;
    color:#666;
    margin:0.3rem 0 0 0;
}

/* ===== STATUS CARDS ===== */
.status-card{
    border-radius:16px;
    padding:1.5rem;
    margin:1rem 0;
    display:flex;
    align-items:center;
    gap:1.5rem;
    box-shadow:0 4px 16px rgba(0,0,0,0.08);
}
.status-success{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color:white; }
.status-warning{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color:white; }
.status-error{ background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color:white; }
.status-icon{ font-size:3rem; opacity:0.9; }
.status-content h3{ margin:0 0 0.5rem 0; font-size:1.3rem; font-weight:700; }
.status-content p{ margin:0; font-size:0.95rem; opacity:0.95; }

/* ===== METRICS CARDS ===== */
.metrics-grid{
    display:grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap:1.5rem;
    margin:2rem 0;
}
.metric-card{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius:18px;
    padding:1.5rem;
    color:white;
    box-shadow:0 6px 20px rgba(102,126,234,0.3);
    transition:all 0.3s ease;
}
.metric-card:hover{
    transform: translateY(-4px);
    box-shadow:0 12px 30px rgba(102,126,234,0.4);
}
.metric-label{
    font-size:0.85rem;
    opacity:0.9;
    font-weight:500;
    text-transform:uppercase;
    letter-spacing:0.05em;
}
.metric-value{ font-size:2.2rem; font-weight:800; margin:0.5rem 0; }
.metric-subtitle{ font-size:0.8rem; opacity:0.8; }

/* ===== BUTTONS ===== */
div.stButton > button{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color:white;
    border:none;
    border-radius:14px;
    padding:0.75rem 2rem;
    font-size:1rem;
    font-weight:600;
    letter-spacing:0.02em;
    box-shadow:0 4px 14px rgba(102,126,234,0.4);
    transition:all 0.3s ease;
}
div.stButton > button:hover{
    transform: translateY(-2px);
    box-shadow:0 6px 20px rgba(102,126,234,0.5);
}
div.stButton > button:disabled{
    background: linear-gradient(135deg, #ccc 0%, #999 100%);
    box-shadow:none;
    cursor:not-allowed;
}

/* ===== ANIM ===== */
@keyframes slideIn{ from{opacity:0; transform:translateY(20px);} to{opacity:1; transform:translateY(0);} }
.section-card{ animation: slideIn 0.5s ease-out; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="admin-hero">
    <h1>⚙️ Administration</h1>
    <p>Gestion des données et entraînement du modèle de prédiction</p>
</div>
""", unsafe_allow_html=True)

DATA_DIR = st.session_state.get("data_dir", "data/favorita_data")

# ✅ On force 10 semaines max (last_10w)
WEEKS = 10
PARQUET_NAME = "train_last10w.parquet"

MODELS_DIR = Path("models")
DATA_DIR_PATH = Path(DATA_DIR)
DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)

# ============================================================
# SECTION 1: UPLOAD
# ============================================================
st.markdown(f"""
<div class="section-card">
    <div class="section-header">
        <div class="section-icon">📁</div>
        <div>
            <div class="section-title">Upload des fichiers</div>
            <div class="section-subtitle">
                Importez vos données CSV. Si vous uploadez <b>train.csv</b>, l’app construit automatiquement <b>{PARQUET_NAME}</b>.
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    up_train = st.file_uploader("📊 train.csv (optionnel, lourd)", type=["csv"], key="up_train")
    up_items = st.file_uploader("🏷️ items.csv", type=["csv"], key="up_items")
    up_stores = st.file_uploader("🏪 stores.csv", type=["csv"], key="up_stores")

with c2:
    up_trans = st.file_uploader("💳 transactions.csv", type=["csv"], key="up_trans")
    up_oil = st.file_uploader("🛢️ oil.csv", type=["csv"], key="up_oil")
    up_hol = st.file_uploader("🎉 holidays_events.csv", type=["csv"], key="up_hol")


def _save_upload(uploaded_file, target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


uploaded_any = any([up_train, up_items, up_stores, up_trans, up_oil, up_hol])

if uploaded_any:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("💾 Sauvegarder les fichiers uploadés"):
        if up_train:
            _save_upload(up_train, DATA_DIR_PATH / "train.csv")
        if up_items:
            _save_upload(up_items, DATA_DIR_PATH / "items.csv")
        if up_stores:
            _save_upload(up_stores, DATA_DIR_PATH / "stores.csv")
        if up_trans:
            _save_upload(up_trans, DATA_DIR_PATH / "transactions.csv")
        if up_oil:
            _save_upload(up_oil, DATA_DIR_PATH / "oil.csv")
        if up_hol:
            _save_upload(up_hol, DATA_DIR_PATH / "holidays_events.csv")

        st.success("✅ Upload sauvegardé.")

        # ✅ Si train.csv a été uploadé, on build le parquet 10w
        if up_train:
            with st.spinner(f"Construction de {PARQUET_NAME}…"):
                build_recent_parquet(DATA_DIR, weeks=WEEKS, out_filename=PARQUET_NAME)
            st.success(f"✅ train réduit aux {WEEKS} dernières semaines → {PARQUET_NAME}")

        st.session_state["data_changed"] = True
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# SECTION 2: STATUS
# ============================================================
st.markdown("""
<div class="section-card">
    <div class="section-header">
        <div class="section-icon">🎯</div>
        <div>
            <div class="section-title">Statut du système</div>
            <div class="section-subtitle">Vérification de la compatibilité modèle/données</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ✅ signature basée sur 10 semaines
current_sig = data_signature(DATA_DIR, weeks=WEEKS)

meta = load_metadata(str(MODELS_DIR))
model_sig = meta.get("data_signature")

if not (MODELS_DIR / "best_model.pkl").exists():
    st.markdown("""
    <div class="status-card status-error">
        <div class="status-icon">❌</div>
        <div class="status-content">
            <h3>Aucun modèle détecté</h3>
            <p>Veuillez entraîner un modèle en cliquant sur le bouton "Retrain" ci-dessous.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    need_retrain = True

elif model_sig != current_sig:
    st.markdown("""
    <div class="status-card status-warning">
        <div class="status-icon">⚠️</div>
        <div class="status-content">
            <h3>Données modifiées</h3>
            <p>Les données ont changé depuis le dernier entraînement. Un nouveau training est recommandé.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    need_retrain = True

else:
    st.markdown("""
    <div class="status-card status-success">
        <div class="status-icon">✅</div>
        <div class="status-content">
            <h3>Système opérationnel</h3>
            <p>Le modèle est compatible avec les données actuelles et prêt à l'emploi.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    need_retrain = False

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# SECTION 3: RETRAIN
# ============================================================
st.markdown("""
<div class="section-card">
    <div class="section-header">
        <div class="section-icon">🚀</div>
        <div>
            <div class="section-title">Entraînement du modèle</div>
            <div class="section-subtitle">Lancez l'entraînement pour générer un nouveau modèle sklearn</div>
        </div>
    </div>
""", unsafe_allow_html=True)

if st.button("🚀 Lancer Retrain", disabled=not need_retrain):
    with st.spinner("Entraînement en cours…"):
        # ✅ Charge toujours le parquet 10w
        df10 = load_train_recent(DATA_DIR, weeks=WEEKS)
        meta2 = train_reference_model(
            df10,
            DATA_DIR,
            weeks_window=WEEKS,
            data_signature=data_signature(DATA_DIR, weeks=WEEKS),
        )


    st.success("✅ Modèle entraîné et sauvegardé dans models/")

    m = meta2["metrics"]

    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">RMSE</div>
            <div class="metric-value">{m['RMSE']:.3f}</div>
            <div class="metric-subtitle">Root Mean Square Error</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">MAE</div>
            <div class="metric-value">{m['MAE']:.3f}</div>
            <div class="metric-subtitle">Mean Absolute Error</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">R²</div>
            <div class="metric-value">{m['R2']:.3f}</div>
            <div class="metric-subtitle">Coefficient de détermination</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">SMAPE</div>
            <div class="metric-value">{m['SMAPE_%']:.2f}%</div>
            <div class="metric-subtitle">Symmetric MAPE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        f"📊 Train: {meta2['n_obs_train']:,} obs | Validation: {meta2['n_obs_val']:,} obs | "
        f"⏱️ Durée: {meta2['runtime_sec']:.1f}s"
    )

    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
