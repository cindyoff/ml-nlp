"""
Régression logistique et XGBoost

Étapes :
  1. split stratifié 70 (train), 15 (validation), 15 (test)
  2. Régression logistique :
       - sélection par IV (IV < 0.02)
       - test de Wald sur sélection par IV
       - calibration sur seuil de Bayes optimal sur la validation
  3. XGBoost :
       - linguistic features + embeddings
       - recharche hyperparamètres par RandomizedSearchCV
       - calibration du seuil de Bayes optimal sur la validation
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from pipeline.config import validate_schema
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# linguistic features
LINGUISTIC_FEATURES = [
    "n_digits", "n_dates", "n_money", "n_percent", "numeric_ratio",
    "n_vague_words", "vague_ratio",
    "n_modal_verbs", "modal_ratio",
    "n_ent_org", "n_ent_loc", "n_ent_law", "n_ent_total",
    "sentiment_positive", "sentiment_negative", "sentiment_intensity",
    "filter_ratio",
]

LABEL_COL = "label"
POS_LABEL = "langue_de_bois"
NEG_LABEL = "non_langue_de_bois"

# chargement + split
def load_and_split(
    input_path: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split train, test, validation
    """
    df = pd.read_parquet(input_path)
    validate_schema(df, set(LINGUISTIC_FEATURES) | {LABEL_COL, "embedding"}, "final_labeled.parquet")
    df_ann = df[df[LABEL_COL].isin([POS_LABEL, NEG_LABEL])].copy()
    df_ann["y"] = (df_ann[LABEL_COL] == POS_LABEL).astype(int)

    print(f"\nDonnées annotées : {len(df_ann)} phrases")
    print(f"{POS_LABEL}     : {(df_ann['y'] == 1).sum()}")
    print(f"{NEG_LABEL} : {(df_ann['y'] == 0).sum()}\n")

    train_val, test = train_test_split(
        df_ann, test_size=test_size, stratify=df_ann["y"], random_state=seed
    )

    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val["y"], random_state=seed
    )

    print(f"   Train : {len(train)}  |  Val : {len(val)}  |  Test : {len(test)}\n")
    return train, val, test

# iv
def compute_iv(series: pd.Series, target: pd.Series, bins: int = 10) -> float:
    """calcul IV pour chaque variable"""
    df = pd.DataFrame({"x": series.values, "y": target.values})
    try:
        df["bin"] = pd.qcut(df["x"], q=bins, duplicates="drop")
    except Exception:
        return 0.0

    stats = df.groupby("bin", observed=True)["y"].agg(["sum", "count"])
    stats.columns = ["events", "total"]
    stats["non_events"] = stats["total"] - stats["events"]

    total_ev  = stats["events"].sum()
    total_nev = stats["non_events"].sum()
    if total_ev == 0 or total_nev == 0:
        return 0.0

    stats["d_ev"]  = stats["events"]     / total_ev
    stats["d_nev"] = stats["non_events"] / total_nev
    mask = (stats["d_ev"] > 0) & (stats["d_nev"] > 0)
    stats = stats[mask]
    stats["woe"] = np.log(stats["d_ev"] / stats["d_nev"])
    stats["iv"]  = (stats["d_ev"] - stats["d_nev"]) * stats["woe"]
    return float(stats["iv"].sum())

def select_by_iv(
    df_train: pd.DataFrame,
    features: list[str],
    target_col: str = "y",
    threshold: float = 0.02,
) -> tuple[list[str], pd.DataFrame]:
    """tableau IV complet pour chaque feature"""
    iv_scores = {f: compute_iv(df_train[f], df_train[target_col]) for f in features}
    iv_df = (
        pd.DataFrame.from_dict(iv_scores, orient="index", columns=["IV"])
        .sort_values("IV", ascending=False)
    )
    selected = iv_df[iv_df["IV"] >= threshold].index.tolist()

    print(f"Information Value (seuil {threshold}) :")
    for feat, row in iv_df.iterrows():
        flag = "✓" if row["IV"] >= threshold else "✗"
        print(f"   {flag}  {feat:<32s}  IV = {row['IV']:.4f}")
    print(f"   → {len(selected)}/{len(features)} features retenues\n")
    return selected, iv_df

# wald test
def select_by_significance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    alpha: float = 0.05,
) -> tuple[list[str], pd.DataFrame]:
    """
    Test de Wald pour features pré-sélectionnées par IV
    """
    X_const = sm.add_constant(X_train, has_constant="add")
    model   = sm.Logit(y_train, X_const).fit(disp=False, maxiter=200)

    pvals    = pd.Series(model.pvalues[1:], index=feature_names, name="p_value")
    selected = pvals[pvals <= alpha].index.tolist()

    print(f"Test de Wald (alpha = {alpha}) :")
    for feat, pv in pvals.sort_values().items():
        flag = "✓" if pv <= alpha else "✗"
        print(f"   {flag}  {feat:<32s}  p = {pv:.4f}")
    print(f"   → {len(selected)}/{len(feature_names)} features significatives\n")
    return selected, pvals.to_frame()

# calibration seuil de bayes
def calibrate_threshold(
    proba: np.ndarray,
    y_true: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
    n_steps: int = 200,
) -> tuple[float, float]:
    """
    Minimisation du risque pour trouver le seuil optimal de Bayes
    """
    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_t, best_risk = 0.5, np.inf

    for t in thresholds:
        preds = (proba >= t).astype(int)
        fp    = ((preds == 1) & (y_true == 0)).sum()
        fn    = ((preds == 0) & (y_true == 1)).sum()
        risk  = cost_fp * fp + cost_fn * fn
        if risk < best_risk:
            best_risk = risk
            best_t    = t

    best_f1 = f1_score(y_true, (proba >= best_t).astype(int), zero_division=0)
    print(f"Seuil de Bayes optimal : {best_t:.3f}  (F1 val = {best_f1:.4f})\n")
    return float(best_t), float(best_f1)

# évaluation
def evaluate_model(
    name: str,
    proba: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
) -> dict:
    preds  = (proba >= threshold).astype(int)
    report = classification_report(y_true, preds, output_dict=True, zero_division=0)
    auc    = roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan")
    print(f"\n📊 {name}  (seuil = {threshold:.3f})")
    print(classification_report(y_true, preds, zero_division=0))
    print(f"AUC-ROC : {auc:.4f}\n")
    return {"threshold": threshold, "auc_roc": auc, "report": report}

# régression logistic
def train_logistic_regression(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    iv_threshold: float = 0.02,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[Pipeline, dict, dict]:
    """
    Régression logistique après IV et Wald
    """
    print("=" * 60)
    print("RÉGRESSION LOGISTIQUE")
    print("=" * 60 + "\n")

    # sélection par IV
    features_iv, iv_df = select_by_iv(train, LINGUISTIC_FEATURES, threshold=iv_threshold)
    if not features_iv:
        print("Aucune feature enlevées par IV — conservation de toutes les features")
        features_iv = LINGUISTIC_FEATURES

    scaler_iv  = StandardScaler()
    X_train_iv = scaler_iv.fit_transform(train[features_iv])
    y_train    = train["y"].values
    y_val      = val["y"].values
    y_test     = test["y"].values

    # seconde sélection par wald test
    selected_feats, pval_df = select_by_significance(
        X_train_iv, y_train, features_iv, alpha=alpha
    )
    if not selected_feats:
        print("Aucune feature significative — conservation des features IV")
        selected_feats = features_iv

    # entraînement final sur features sélectionnées
    scaler_final = StandardScaler()
    X_train_sel  = scaler_final.fit_transform(train[selected_feats])
    X_val_sel    = scaler_final.transform(val[selected_feats])
    X_test_sel   = scaler_final.transform(test[selected_feats])

    lr = LogisticRegression(max_iter=500, random_state=seed, class_weight="balanced")
    lr.fit(X_train_sel, y_train)

    # pipeline
    pipe = Pipeline([("scaler", scaler_final), ("lr", lr)])

    # calibration du seuil de Bayes
    print("Calibration du seuil de Bayes (validation) :")
    proba_val = lr.predict_proba(X_val_sel)[:, 1]
    threshold, _ = calibrate_threshold(proba_val, y_val)

    # predict proba
    proba_train = lr.predict_proba(X_train_sel)[:, 1]
    proba_test  = lr.predict_proba(X_test_sel)[:, 1]
    metrics = {
        "train": evaluate_model("LR — Train", proba_train, y_train, threshold),
        "val":   evaluate_model("LR — Val",   proba_val,   y_val,   threshold),
        "test":  evaluate_model("LR — Test",  proba_test,  y_test,  threshold),
    }

    params = {
        "model"             : "LogisticRegression",
        "iv_threshold"      : iv_threshold,
        "significance_alpha": alpha,
        "features_after_iv" : features_iv,
        "features_selected" : selected_feats,
        "threshold"         : threshold,
        "iv_scores"         : iv_df["IV"].to_dict(),
        "p_values"          : pval_df["p_value"].to_dict(),
        "lr_params"         : lr.get_params(),
    }
    return pipe, params, metrics

# xgboost
def build_xgb_features(df: pd.DataFrame) -> np.ndarray:
    """xgboost features"""
    X_ling = df[LINGUISTIC_FEATURES].values.astype(np.float32)
    embeds = np.stack(df["embedding"].values).astype(np.float32)
    return np.hstack([X_ling, embeds])


def train_xgboost(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    n_iter: int = 30,
    cv_folds: int = 3,
    seed: int = 42,
) -> tuple[xgb.XGBClassifier, dict, dict]:
    print("=" * 60)
    print("xgboost")
    print("=" * 60 + "\n")

    X_train = build_xgb_features(train)
    X_val   = build_xgb_features(val)
    X_test  = build_xgb_features(test)
    y_train = train["y"].values
    y_val   = val["y"].values
    y_test  = test["y"].values

    # Poids pour déséquilibre de classes
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos = n_neg / max(n_pos, 1)

    param_dist = {
        "n_estimators"    : [100, 200, 300, 500],
        "max_depth"       : [3, 4, 5, 6, 7],
        "learning_rate"   : [0.01, 0.05, 0.1, 0.2, 0.3],
        "subsample"       : [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
        "gamma"           : [0, 0.1, 0.5, 1.0],
        "reg_alpha"       : [0, 0.1, 1.0],
        "reg_lambda"      : [1.0, 2.0, 5.0],
    }

    base_xgb = xgb.XGBClassifier(
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )

    print(f"RandomizedSearchCV ({n_iter} itérations, CV={cv_folds} folds)\n")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        base_xgb,
        param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    best_xgb = search.best_estimator_
    print(f"\nMeilleurs hyperparamètres :\n   {search.best_params_}\n")

    # calibration du seuil de bayes
    print("Calibration du seuil de Bayes (validation) :")
    proba_val = best_xgb.predict_proba(X_val)[:, 1]
    threshold, _ = calibrate_threshold(proba_val, y_val)

    # evaluation
    proba_train = best_xgb.predict_proba(X_train)[:, 1]
    proba_test  = best_xgb.predict_proba(X_test)[:, 1]
    metrics = {
        "train": evaluate_model("XGB — Train", proba_train, y_train, threshold),
        "val":   evaluate_model("XGB — Val",   proba_val,   y_val,   threshold),
        "test":  evaluate_model("XGB — Test",  proba_test,  y_test,  threshold),
    }

    params = {
        "model"       : "XGBClassifier",
        "n_features"  : X_train.shape[1],
        "best_params" : search.best_params_,
        "best_cv_auc" : float(search.best_score_),
        "threshold"   : threshold,
    }
    return best_xgb, params, metrics

# prédiction sur corpus complet
def predict_all(
    full_path: str,
    lr_pipe: Pipeline,
    lr_features: list[str],
    lr_threshold: float,
    xgb_model: xgb.XGBClassifier,
    xgb_threshold: float,
) -> pd.DataFrame:
    """
      - proba_lr
      - pred_lr
      - proba_xgb
      - pred_xgb
    """
    print("=" * 60)
    print("PRÉDICTIONS — CORPUS COMPLET")
    print("=" * 60 + "\n")

    print(f"Chargement : {full_path}")
    df = pd.read_parquet(full_path)
    print(f"   {len(df)} phrases chargées\n")

    # régression logistique
    print("Prédictions LR...")
    X_lr        = df[lr_features].values.astype(np.float32)
    proba_lr    = lr_pipe.predict_proba(X_lr)[:, 1]
    df["proba_lr"] = proba_lr
    df["pred_lr"]  = np.where(proba_lr >= lr_threshold, POS_LABEL, NEG_LABEL)

    # xgboost
    print("Prédictions XGBoost...")
    X_xgb        = build_xgb_features(df)
    proba_xgb    = xgb_model.predict_proba(X_xgb)[:, 1]
    df["proba_xgb"] = proba_xgb
    df["pred_xgb"]  = np.where(proba_xgb >= xgb_threshold, POS_LABEL, NEG_LABEL)

    # résumé
    for model, col in [("LR", "pred_lr"), ("XGBoost", "pred_xgb")]:
        counts = df[col].value_counts()
        print(f"\n   {model} — distribution prédite :")
        for lbl, cnt in counts.items():
            print(f"{lbl} : {cnt}  ({cnt / len(df):.1%})")
    print()

    return df

def run(
    input_path: str,
    output_dir: str,
    full_path: str | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    n_iter: int = 30,
    seed: int = 42,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train, val, test = load_and_split(input_path, test_size, val_size, seed)

    lr_pipe,   lr_params,  lr_metrics  = train_logistic_regression(train, val, test, seed=seed)
    xgb_model, xgb_params, xgb_metrics = train_xgboost(train, val, test, n_iter=n_iter, seed=seed)

    # prédictions sur corpus complet
    corpus_path = full_path or str(Path(input_path).parent / "final.parquet")
    df_predicted = predict_all(
        full_path      = corpus_path,
        lr_pipe        = lr_pipe,
        lr_features    = lr_params["features_selected"],
        lr_threshold   = lr_params["threshold"],
        xgb_model      = xgb_model,
        xgb_threshold  = xgb_params["threshold"],
    )
    predicted_out = Path(output_dir).parent / "final_predicted.parquet"
    df_predicted.to_parquet(predicted_out, index=False)

    # sauvegarde
    print("=" * 60)
    print("  SAUVEGARDE")
    print("=" * 60 + "\n")

    joblib.dump(lr_pipe,   out / "lr_estimator.joblib")
    joblib.dump(xgb_model, out / "xgb_estimator.joblib")

    with open(out / "lr_params.json", "w", encoding="utf-8") as f:
        json.dump(lr_params, f, ensure_ascii=False, indent=2, default=str)

    with open(out / "xgb_params.json", "w", encoding="utf-8") as f:
        json.dump(xgb_params, f, ensure_ascii=False, indent=2, default=str)

    evaluation = {"logistic_regression": lr_metrics, "xgboost": xgb_metrics}
    with open(out / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2, default=str)

    print(f"final_predicted.parquet : {predicted_out}")
    print(f"lr_estimator.joblib : {out / 'lr_estimator.joblib'}")
    print(f"xgb_estimator.joblib : {out / 'xgb_estimator.joblib'}")
    print(f"lr_params.json : {out / 'lr_params.json'}")
    print(f"xgb_params.json : {out / 'xgb_params.json'}")
    print(f"evaluation.json : {out / 'evaluation.json'}\n")

def main():
    parser = argparse.ArgumentParser(description="Entraînement des classifieurs langue de bois")
    parser.add_argument("--input",     default="outputs/final_labeled.parquet",
                        help="Parquet annoté d'entrée (phrases avec label)")
    parser.add_argument("--full",      default="outputs/final.parquet",
                        help="Parquet complet des ~110 000 phrases pour la prédiction")
    parser.add_argument("--output",    default="outputs/models/",
                        help="Dossier de sortie pour les modèles et paramètres")
    parser.add_argument("--test_size", default=0.15, type=float,
                        help="Proportion du jeu de test (défaut : 0.15)")
    parser.add_argument("--val_size",  default=0.15, type=float,
                        help="Proportion du jeu de validation (défaut : 0.15)")
    parser.add_argument("--n_iter",    default=30, type=int,
                        help="Nombre d'itérations RandomizedSearchCV XGBoost (défaut : 30)")
    parser.add_argument("--seed",      default=42, type=int,
                        help="Graine aléatoire (défaut : 42)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    run(args.input, args.output, args.full, args.test_size, args.val_size, args.n_iter, args.seed)

if __name__ == "__main__":
    main()