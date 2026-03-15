"""
dashboard.py
============
Visualisation interactive des prédictions de langue de bois.

Pour chaque document du corpus, affiche le texte avec les phrases
détectées comme "langue de bois" surlignées. Un sélecteur permet
de basculer entre le modèle LR et XGBoost.

Lancement :
    streamlit run dashboard.py
    streamlit run dashboard.py -- --data outputs/final_predicted.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Langue de bois — Dashboard",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "outputs/final_predicted.parquet"

# Récupère le chemin passé en argument CLI (streamlit run dashboard.py -- --data ...)
if "--data" in sys.argv:
    idx = sys.argv.index("--data")
    if idx + 1 < len(sys.argv):
        DATA_PATH = sys.argv[idx + 1]

POS_LABEL = "langue_de_bois"

# ── Palette de couleurs ───────────────────────────────────────────────────────
# Intensité de la couleur proportionnelle à la probabilité prédite

def _proba_to_color(p: float) -> str:
    """
    Retourne une couleur CSS en dégradé jaune → orange → rouge
    selon la probabilité p ∈ [0, 1].
    """
    r = int(255)
    g = int(max(50, 220 - int(p * 200)))
    b = int(max(0, 100 - int(p * 100)))
    alpha = 0.25 + p * 0.65
    return f"rgba({r},{g},{b},{alpha:.2f})"


def render_html(sentences: pd.DataFrame, pred_col: str, proba_col: str) -> str:
    """
    Construit un bloc HTML avec les phrases surlignées selon leur label prédit.
    L'intensité du surlignage reflète la probabilité du modèle.
    """
    parts = []
    for _, row in sentences.iterrows():
        text  = str(row["sentence"])
        pred  = row[pred_col]
        proba = float(row[proba_col])

        if pred == POS_LABEL:
            color   = _proba_to_color(proba)
            tooltip = f"p = {proba:.3f}"
            parts.append(
                f'<span title="{tooltip}" style="'
                f"background-color:{color};"
                f"border-radius:4px;"
                f"padding:1px 4px;"
                f'margin:1px;">{text}</span>'
            )
        else:
            parts.append(text)

    return "<p style='line-height:1.9;font-size:15px;'>" + " ".join(parts) + "</p>"


# ── Chargement des données ─────────────────────────────────────────────────────

@st.cache_data(show_spinner="Chargement du corpus…")
def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("⚙️ Paramètres")

if not Path(DATA_PATH).exists():
    st.error(f"Fichier introuvable : `{DATA_PATH}`\n\nLance d'abord `python main.py --steps modelise`.")
    st.stop()

df = load_data(DATA_PATH)

# Sélecteur de modèle
model_choice = st.sidebar.radio(
    "Modèle",
    options=["Régression logistique", "XGBoost"],
    index=1,
)
pred_col  = "pred_lr"  if model_choice == "Régression logistique" else "pred_xgb"
proba_col = "proba_lr" if model_choice == "Régression logistique" else "proba_xgb"

# Filtre année / type
years   = ["Toutes"] + sorted(df["date"].unique().tolist())
classes = ["Tous"]   + sorted(df["classe"].unique().tolist())
sel_year  = st.sidebar.selectbox("Année", years)
sel_class = st.sidebar.selectbox("Type d'élection", classes)

df_filtered = df.copy()
if sel_year  != "Toutes":
    df_filtered = df_filtered[df_filtered["date"]   == sel_year]
if sel_class != "Tous":
    df_filtered = df_filtered[df_filtered["classe"] == sel_class]

doc_ids = sorted(df_filtered["doc_id"].unique().tolist())

if not doc_ids:
    st.warning("Aucun document pour ce filtre.")
    st.stop()

# Seuil de surlignage (filtre visuel uniquement)
proba_threshold = st.sidebar.slider(
    "Seuil d'affichage (probabilité min.)",
    min_value=0.0, max_value=1.0, value=0.5, step=0.01,
    help="Les phrases sous ce seuil ne sont pas surlignées, même si le modèle les prédit comme langue de bois.",
)

st.sidebar.markdown("---")
st.sidebar.caption(f"{len(doc_ids)} document(s) disponibles")

# ── Navigation ────────────────────────────────────────────────────────────────

st.title("🗣️ Langue de bois — Visualisation")

if "doc_index" not in st.session_state:
    st.session_state.doc_index = 0
st.session_state.doc_index = min(st.session_state.doc_index, len(doc_ids) - 1)

nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 4, 1])

with nav_col1:
    if st.button("⏮ Premier"):
        st.session_state.doc_index = 0
with nav_col2:
    if st.button("◀ Précédent") and st.session_state.doc_index > 0:
        st.session_state.doc_index -= 1
with nav_col4:
    if st.button("Suivant ▶") and st.session_state.doc_index < len(doc_ids) - 1:
        st.session_state.doc_index += 1

with nav_col3:
    selected_doc = st.selectbox(
        "Document",
        options=doc_ids,
        index=st.session_state.doc_index,
        label_visibility="collapsed",
    )
    st.session_state.doc_index = doc_ids.index(selected_doc)

st.caption(f"Document {st.session_state.doc_index + 1} / {len(doc_ids)}")

# ── Document ──────────────────────────────────────────────────────────────────

doc_df = (
    df_filtered[df_filtered["doc_id"] == selected_doc]
    .copy()
    .sort_values("PRIMARY_KEY")
    .reset_index(drop=True)
)

# Applique le seuil visuel
doc_df_display = doc_df.copy()
mask_below = doc_df_display[proba_col] < proba_threshold
doc_df_display.loc[mask_below, pred_col] = "non_langue_de_bois"

# Métadonnées
meta = doc_df.iloc[0]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Document", selected_doc)
m2.metric("Année", meta["date"])
m3.metric("Type", meta["classe"])
m4.metric("Phrases", len(doc_df))

# Barre de progression langue de bois
n_ldb = (doc_df_display[pred_col] == POS_LABEL).sum()
pct   = n_ldb / len(doc_df) if len(doc_df) > 0 else 0
st.progress(
    pct,
    text=f"**{n_ldb} / {len(doc_df)}** phrases langue de bois  ({pct:.1%})  — modèle : {model_choice}",
)

st.markdown("---")

# Légende
st.markdown(
    "<small>"
    "<span style='background:rgba(255,200,50,0.5);padding:2px 6px;border-radius:3px;'>faible</span>"
    " &nbsp;→&nbsp; "
    "<span style='background:rgba(255,120,10,0.8);padding:2px 6px;border-radius:3px;'>forte</span>"
    " &nbsp; intensité = probabilité prédite"
    "</small>",
    unsafe_allow_html=True,
)
st.markdown(" ")

# Texte du document
html = render_html(doc_df_display, pred_col, proba_col)
st.markdown(html, unsafe_allow_html=True)

# ── Tableau détaillé (expansible) ─────────────────────────────────────────────

with st.expander("📋 Tableau détaillé des phrases"):
    show_df = doc_df[["PRIMARY_KEY", "sentence", proba_col, pred_col]].copy()
    show_df.columns = ["PRIMARY_KEY", "Phrase", "Probabilité", "Label prédit"]
    show_df = show_df.sort_values("Probabilité", ascending=False)
    st.dataframe(
        show_df.style.background_gradient(subset=["Probabilité"], cmap="YlOrRd"),
        use_container_width=True,
        hide_index=True,
    )

# ── Statistiques globales (sidebar bas) ───────────────────────────────────────

with st.sidebar.expander("📊 Stats corpus filtré"):
    total_phrases = len(df_filtered)
    total_ldb     = (df_filtered[pred_col] == POS_LABEL).sum()
    st.metric("Phrases totales", f"{total_phrases:,}")
    st.metric("Langue de bois (prédit)", f"{total_ldb:,}  ({total_ldb/total_phrases:.1%})")
    st.metric("Documents", len(doc_ids))
