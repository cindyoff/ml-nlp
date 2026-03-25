"""
Dashboard Streamlit de résumé statistique des modèles de détection de langue de bois : LR + XGBoost
- Résumé des performance (train, val, test, AUC, F1, precision, recall, accuracy)
- Résumé hyperparamètres et features + seuils
- Test résidus
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.model_selection import train_test_split

# configuration
st.set_page_config(
    page_title="Résumé statistique — Modèles",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODELS_DIR   = Path("outputs/models")
LABELED_PATH = Path("outputs/final_labeled.parquet")
POS_LABEL    = "langue_de_bois"
NEG_LABEL    = "non_langue_de_bois"

LINGUISTIC_FEATURES = [
    "n_digits", "n_dates", "n_money", "n_percent", "numeric_ratio",
    "n_vague_words", "vague_ratio",
    "n_modal_verbs", "modal_ratio",
    "n_ent_org", "n_ent_loc", "n_ent_law", "n_ent_total",
    "sentiment_positive", "sentiment_negative", "sentiment_intensity",
    "filter_ratio",
]

# chargement
@st.cache_data(show_spinner=False)
def load_evaluation():
    with open(MODELS_DIR / "evaluation.json", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_params():
    with open(MODELS_DIR / "lr_params.json", encoding="utf-8") as f:
        lr = json.load(f)
    with open(MODELS_DIR / "xgb_params.json", encoding="utf-8") as f:
        xgb = json.load(f)
    return lr, xgb

@st.cache_resource(show_spinner=False)
def load_models():
    lr  = joblib.load(MODELS_DIR / "lr_estimator.joblib")
    xgb = joblib.load(MODELS_DIR / "xgb_estimator.joblib")
    return lr, xgb

@st.cache_data(show_spinner="Chargement des données annotées…")
def load_annotated(seed: int = 42, test_size: float = 0.15, val_size: float = 0.15):
    df     = pd.read_parquet(LABELED_PATH)
    df_ann = df[df["label"].isin([POS_LABEL, NEG_LABEL])].copy()
    df_ann["y"] = (df_ann["label"] == POS_LABEL).astype(int)
    train_val, test = train_test_split(
        df_ann, test_size=test_size, stratify=df_ann["y"], random_state=seed
    )
    relative_val = val_size / (1 - test_size)
    train, val   = train_test_split(
        train_val, test_size=relative_val, stratify=train_val["y"], random_state=seed
    )
    return train, val, test

def build_metrics_table(model_metrics: dict) -> pd.DataFrame:
    rows = []
    for split_name, split_data in model_metrics.items():
        rep = split_data.get("report", {})
        pos = rep.get(POS_LABEL, rep.get("1", {}))
        rows.append({
            "Split"    : split_name.capitalize(),
            "AUC-ROC"  : round(split_data.get("auc_roc", float("nan")), 4),
            "Seuil"    : round(split_data.get("threshold", float("nan")), 3),
            "F1 (LDB)" : round(pos.get("f1-score",  float("nan")), 4),
            "Précision": round(pos.get("precision",  float("nan")), 4),
            "Rappel"   : round(pos.get("recall",     float("nan")), 4),
            "Accuracy" : round(rep.get("accuracy",   float("nan")), 4),
        })
    return pd.DataFrame(rows)


def get_probas(lr_pipe, lr_params, xgb_model, train, val, test):
    lr_feats = lr_params["features_selected"]
    results  = {}
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        X_lr  = split_df[lr_feats].values.astype(np.float32)
        X_xgb = np.hstack([
            split_df[LINGUISTIC_FEATURES].values.astype(np.float32),
            np.stack(split_df["embedding"].values).astype(np.float32),
        ])
        results[name] = {
            "y"        : split_df["y"].values,
            "proba_lr" : lr_pipe.predict_proba(X_lr)[:, 1],
            "proba_xgb": xgb_model.predict_proba(X_xgb)[:, 1],
        }
    return results


def pearson_residuals(y_true, proba):
    p = np.clip(proba, 1e-8, 1 - 1e-8)
    return (y_true - p) / np.sqrt(p * (1 - p))


def deviance_residuals(y_true, proba):
    p   = np.clip(proba, 1e-8, 1 - 1e-8)
    dev = -2 * (y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    return np.sign(y_true - p) * np.sqrt(np.maximum(dev, 0))


def hosmer_lemeshow(y_true, proba, g=10):
    df = pd.DataFrame({"y": y_true, "p": proba})
    df["decile"] = pd.qcut(df["p"], q=g, duplicates="drop", labels=False)
    grp = df.groupby("decile").agg(
        obs_pos=("y", "sum"), total=("y", "count"), mean_p=("p", "mean")
    ).reset_index()
    grp["obs_neg"] = grp["total"] - grp["obs_pos"]
    grp["exp_pos"] = grp["total"] * grp["mean_p"]
    grp["exp_neg"] = grp["total"] * (1 - grp["mean_p"])
    chi2 = (
        ((grp["obs_pos"] - grp["exp_pos"]) ** 2 / grp["exp_pos"].clip(lower=1e-8)).sum()
      + ((grp["obs_neg"] - grp["exp_neg"]) ** 2 / grp["exp_neg"].clip(lower=1e-8)).sum()
    )
    pval = 1 - stats.chi2.cdf(chi2, df=len(grp) - 2)
    return float(chi2), float(pval), grp


def calibration_data(y_true, proba, n_bins=10):
    df = pd.DataFrame({"y": y_true, "p": proba})
    df["bin"] = pd.cut(df["p"], bins=n_bins, include_lowest=True)
    grp = df.groupby("bin", observed=True).agg(
        mean_pred=("p", "mean"), frac_pos=("y", "mean"), count=("y", "count")
    ).reset_index()
    return grp.dropna()

# app
st.title("Résumé statistique — Modèles de détection")
st.caption("Langue de bois dans les discours de campagne électorale française (1981–1993)")

for path in [MODELS_DIR / "evaluation.json", MODELS_DIR / "lr_params.json",
             MODELS_DIR / "xgb_params.json", LABELED_PATH]:
    if not path.exists():
        st.error(f"Fichier manquant : `{path}`. Lancer d'abord `python main.py --steps modelise`")
        st.stop()

with st.spinner("Chargement des modèles et prédictions"):
    evaluation            = load_evaluation()
    lr_params, xgb_params = load_params()
    lr_pipe, xgb_model    = load_models()
    train, val, test      = load_annotated()
    split_probas          = get_probas(lr_pipe, lr_params, xgb_model, train, val, test)

lr_eval  = evaluation["logistic_regression"]
xgb_eval = evaluation["xgboost"]


# ── Section 1 : Performances | Propriétés ────────────────────────────────────

col_left, col_right = st.columns(2, gap="large")

# ── Gauche : Performances ────────────────────────────────────────────────────
with col_left:
    st.subheader("Performances")
    tab_lr_m, tab_xgb_m = st.tabs(["Régression Logistique", "XGBoost"])

    for tab, eval_data, label in [
        (tab_lr_m,  lr_eval,  "LR"),
        (tab_xgb_m, xgb_eval, "XGBoost"),
    ]:
        with tab:
            df_m = build_metrics_table(eval_data)
            st.dataframe(
                df_m.style.background_gradient(
                    subset=["AUC-ROC", "F1 (LDB)", "Accuracy"], cmap="YlGn"
                ),
                use_container_width=True, hide_index=True,
            )

            fig = go.Figure()
            for metric, color in [
                ("AUC-ROC",  "#636EFA"),
                ("F1 (LDB)", "#EF553B"),
                ("Accuracy", "#00CC96"),
            ]:
                fig.add_trace(go.Bar(
                    name=metric, x=df_m["Split"], y=df_m[metric], marker_color=color,
                ))
            fig.update_layout(
                barmode="group", height=270,
                margin=dict(t=10, b=10),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig, use_container_width=True)


# ── Droite : Propriétés ───────────────────────────────────────────────────────
with col_right:
    st.subheader("Propriétés des modèles")
    tab_lr_p, tab_xgb_p = st.tabs(["Régression Logistique", "XGBoost"])

    with tab_lr_p:
        c1, c2, c3 = st.columns(3)
        c1.metric("Seuil de Bayes", f"{lr_params['threshold']:.3f}")
        c2.metric("IV seuil", lr_params["iv_threshold"])
        c3.metric("α Wald", lr_params["significance_alpha"])

        st.markdown(
            f"**Features retenues** : {len(lr_params['features_selected'])} / "
            f"{len(lr_params['features_after_iv'])}"
        )
        iv_rows = [
            {
                "Feature" : f,
                "IV"      : round(lr_params["iv_scores"].get(f, float("nan")), 4),
                "p-value" : round(lr_params["p_values"].get(f, float("nan")), 4),
                "Retenue" : "✓" if f in lr_params["features_selected"] else "✗",
            }
            for f in sorted(
                lr_params["features_after_iv"],
                key=lambda x: lr_params["iv_scores"].get(x, 0),
                reverse=True,
            )
        ]
        df_feat = pd.DataFrame(iv_rows)
        st.dataframe(
            df_feat.style.background_gradient(subset=["IV"], cmap="Blues"),
            use_container_width=True, hide_index=True, height=300,
        )
        with st.expander("Hyperparamètres LR"):
            st.json(lr_params.get("lr_params", {}))

    with tab_xgb_p:
        c1, c2, c3 = st.columns(3)
        c1.metric("Seuil de Bayes", f"{xgb_params['threshold']:.3f}")
        c2.metric("AUC CV (best)", f"{xgb_params['best_cv_auc']:.4f}")
        c3.metric("Nb features", xgb_params["n_features"])
        st.caption("Features = 17 linguistiques + 768 dims CamemBERT")
        with st.expander("Meilleurs hyperparamètres XGBoost", expanded=True):
            st.json(xgb_params["best_params"])


# ── Section 2 : Résidus ───────────────────────────────────────────────────────

st.divider()
st.subheader("Analyse des résidus")

split_choice = st.radio(
    "Jeu de données :", ["test", "val", "train"],
    horizontal=True, index=0,
)
data_split = split_probas[split_choice]
y_true     = data_split["y"]

tabs_res = st.tabs(["Régression Logistique", "XGBoost"])

for tab_res, proba_key, model_label in [
    (tabs_res[0], "proba_lr",  "LR"),
    (tabs_res[1], "proba_xgb", "XGBoost"),
]:
    with tab_res:
        proba = data_split[proba_key]

        # ── Hosmer-Lemeshow ──────────────────────────────────────────────────
        chi2_hl, pval_hl, hl_table = hosmer_lemeshow(y_true, proba, g=10)

        col_hl1, col_hl2 = st.columns([1, 2])

        with col_hl1:
            st.markdown("#### Test de Hosmer-Lemeshow")
            verdict = "Mauvaise calibration (p < 0.05)" if pval_hl < 0.05 else "Calibration acceptable (p >= 0.05)"
            st.metric("χ²", f"{chi2_hl:.3f}")
            st.metric("p-value", f"{pval_hl:.4f}")
            st.info(verdict)
            # st.caption("H₀ : le modèle est bien calibré.\np > 0.05 → on ne rejette pas H₀.")

        with col_hl2:
            fig_hl = go.Figure()
            x_deciles = hl_table["decile"].astype(str)
            fig_hl.add_trace(go.Bar(
                x=x_deciles, y=hl_table["obs_pos"],
                name="Observés +", marker_color="#EF553B",
            ))
            fig_hl.add_trace(go.Bar(
                x=x_deciles, y=hl_table["exp_pos"],
                name="Attendus +", marker_color="#636EFA", opacity=0.7,
            ))
            fig_hl.update_layout(
                barmode="group", height=220,
                margin=dict(t=10, b=10),
                xaxis_title="Décile de probabilité", yaxis_title="Nb phrases",
                legend=dict(orientation="h", y=-0.3),
            )
            st.plotly_chart(fig_hl, use_container_width=True)

        st.divider()

        col_r1, col_r2, col_r3 = st.columns(3)

        # calibration 
        with col_r1:
            st.markdown("#### Courbe de calibration")
            cal = calibration_data(y_true, proba, n_bins=10)
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Calibration parfaite",
                line=dict(dash="dash", color="gray"),
            ))
            fig_cal.add_trace(go.Scatter(
                x=cal["mean_pred"], y=cal["frac_pos"],
                mode="lines+markers", name=model_label,
                marker=dict(
                    size=(cal["count"] / cal["count"].max() * 18 + 5).tolist(),
                    color="#636EFA",
                ),
                line=dict(color="#636EFA"),
            ))
            fig_cal.update_layout(
                xaxis_title="Probabilité prédite",
                yaxis_title="Fraction positive réelle",
                height=300, margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_cal, use_container_width=True)

        # résidus Pearson
        with col_r2:
            st.markdown("#### Résidus de Pearson")
            p_res = pearson_residuals(y_true, proba)
            rng   = np.random.default_rng(42)
            sample = p_res if len(p_res) <= 5000 else p_res[rng.choice(len(p_res), 5000, replace=False)]
            sw_stat, sw_pval = stats.shapiro(sample)

            fig_pr = go.Figure()
            fig_pr.add_trace(go.Histogram(
                x=p_res, nbinsx=50, name="Pearson",
                marker_color="#AB63FA", opacity=0.8,
            ))
            fig_pr.update_layout(
                xaxis_title="Résidu de Pearson", yaxis_title="Count",
                height=270, margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_pr, use_container_width=True)
            st.caption(
                f"Shapiro-Wilk : W = {sw_stat:.4f},  p = {sw_pval:.4f}\n"
                f"Moyenne = {p_res.mean():.4f}  |  Std = {p_res.std():.4f}"
            )

        # résidus déviance
        with col_r3:
            st.markdown("#### Résidus de déviance")
            d_res = deviance_residuals(y_true, proba)

            fig_dr = go.Figure()
            fig_dr.add_trace(go.Scatter(
                x=proba, y=d_res, mode="markers",
                marker=dict(
                    size=4, opacity=0.55,
                    color=y_true.astype(int),
                    colorscale="RdBu",
                    colorbar=dict(title="Classe", thickness=10),
                ),
                name="Résidus",
            ))
            fig_dr.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_dr.update_layout(
                xaxis_title="Probabilité prédite",
                yaxis_title="Résidu de déviance",
                height=270, margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_dr, use_container_width=True)
            st.caption(
                f"Moyenne = {d_res.mean():.4f}  |  Std = {d_res.std():.4f}"
            )

        # distribution proba par classe
        st.markdown("#### Distribution des probabilités par classe réelle")
        fig_dist = go.Figure()
        for lval, lname, color in [
            (1, POS_LABEL, "#EF553B"),
            (0, NEG_LABEL, "#636EFA"),
        ]:
            mask = y_true == lval
            fig_dist.add_trace(go.Histogram(
                x=proba[mask], nbinsx=40, name=lname,
                marker_color=color, opacity=0.65,
            ))
        fig_dist.update_layout(
            barmode="overlay", height=280,
            xaxis_title="Probabilité prédite (langue_de_bois)",
            yaxis_title="Count",
            margin=dict(t=10, b=10),
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_dist, use_container_width=True)
