"""
Annotation automatique locale avec zero-shot classification (HuggingFace)
Modèle par défaut : MoritzLaurer/mDeBERTa-v3-base-mnli-xnli  (multilingue, ~900 MB)

Labels :
  - langue_de_bois     : rhétorique vague, slogans creux, promesses floues
  - non_langue_de_bois : affirmations concrètes, faits précis, chiffres, actions spécifiques
  - autre              : texte administratif, métadonnées, entêtes, texte non-français, bruit OCR
"""

import argparse
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from transformers import pipeline

# Descriptions utilisées par le modèle zero-shot (une par label)
CANDIDATE_LABELS = [
    "langue de bois, rhétorique vague, slogans creux, promesses floues, généralités sans contenu concret",
    "faits concrets, chiffres précis, noms propres, actions spécifiques et vérifiables",
    "texte administratif, métadonnées, bruit OCR, texte non politique, langue étrangère",
]

# Correspondance → noms de labels finaux dans le CSV
LABEL_MAP = {
    CANDIDATE_LABELS[0]: "langue_de_bois",
    CANDIDATE_LABELS[1]: "non_langue_de_bois",
    CANDIDATE_LABELS[2]: "autre",
}

HYPOTHESIS_TEMPLATE = "Ce texte contient {}."

LABEL_COLORS = {
    "langue_de_bois":     "#e07b54",
    "non_langue_de_bois": "#5b9bd5",
    "autre":              "#a8a8a8",
}


def load_classifier(model_name: str):
    """Charge le pipeline zero-shot en utilisant MPS (Apple Silicon) si disponible."""
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = 0
    else:
        device = -1  # CPU

    print(f"Chargement du modèle '{model_name}' (device: {device}) …")
    clf = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )
    print("Modèle chargé.\n")
    return clf


def annotate_batch(classifier, batch: pd.DataFrame) -> dict[str, str]:
    sentences = [str(s) for s in batch["sentence"].tolist()]
    pks = batch["PRIMARY_KEY"].tolist()

    results = classifier(
        sentences,
        candidate_labels=CANDIDATE_LABELS,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )

    # pipeline renvoie une liste si plusieurs phrases, un dict si une seule
    if isinstance(results, dict):
        results = [results]

    labels = {}
    for pk, result in zip(pks, results):
        top_label = result["labels"][0]
        labels[pk] = LABEL_MAP[top_label]

    return labels


def save_annotation_stats(df: pd.DataFrame, output_dir: str = "outputs") -> str:
    """Génère un tableau de stats descriptives + un bar chart et sauvegarde en PNG."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "annotation_stats.png")

    annotated = df[df["label"].str.strip() != ""].copy()
    counts = annotated["label"].value_counts().sort_index()
    pct = (counts / counts.sum() * 100).round(1)

    stats = pd.DataFrame({
        "Label": counts.index,
        "Nb phrases": counts.values,
        "% du total": pct.values,
    })

    fig = plt.figure(figsize=(11, 5))
    fig.patch.set_facecolor("#f9f9f9")
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.1, 1], wspace=0.35)

    # --- Tableau ---
    ax_table = fig.add_subplot(gs[0])
    ax_table.axis("off")
    ax_table.set_title("Statistiques descriptives — annotation automatique",
                        fontsize=12, fontweight="bold", pad=14, loc="left")

    col_labels = stats.columns.tolist()
    cell_text  = stats.astype(str).values.tolist()
    cell_text.append(["TOTAL", str(counts.sum()), "100.0"])

    row_colors = []
    for lbl in stats["Label"]:
        row_colors.append([LABEL_COLORS.get(lbl, "#dddddd")] * 3)
    row_colors.append(["#333333"] * 3)

    table = ax_table.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0.05, 1, 0.85],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    for i, row_c in enumerate(row_colors, start=1):
        for j, fc in enumerate(row_c):
            cell = table[i, j]
            cell.set_facecolor(fc)
            if i == len(row_colors):
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_text_props(color="white")

    n_total = len(df)
    n_unannotated = (df["label"].str.strip() == "").sum()
    note = f"Phrases totales dans le fichier : {n_total}   |   Non annotées : {n_unannotated}"
    fig.text(0.01, 0.02, note, fontsize=8, color="#555555", style="italic")

    # --- Bar chart ---
    ax_bar = fig.add_subplot(gs[1])
    ax_bar.set_facecolor("#f9f9f9")

    bar_colors = [LABEL_COLORS.get(lbl, "#aaaaaa") for lbl in counts.index]
    bars = ax_bar.bar(counts.index, counts.values, color=bar_colors, edgecolor="white", width=0.5)

    for bar, p in zip(bars, pct.values):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + counts.max() * 0.02,
                    f"{p}%", ha="center", va="bottom", fontsize=9, color="#333333")

    ax_bar.set_title("Distribution des labels", fontsize=11, fontweight="bold", pad=10)
    ax_bar.set_ylabel("Nombre de phrases", fontsize=9)
    ax_bar.set_xticks(range(len(counts.index)))
    ax_bar.set_xticklabels(counts.index, rotation=15, ha="right", fontsize=9)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax_bar.set_axisbelow(True)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/labels/annotation_sample.csv")
    parser.add_argument("--output", default="data/labels/annotation_sample.csv")
    parser.add_argument("--batch",  default=32, type=int, help="Phrases par batch")
    parser.add_argument("--model",  default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                        help="Modèle HuggingFace zero-shot-classification")
    parser.add_argument("--resume", action="store_true",
                        help="Ignorer les phrases déjà annotées")
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype=str)
    df["label"] = df["label"].fillna("")

    if args.resume:
        todo = df[df["label"].str.strip() == ""].copy()
        print(f"Reprise : {len(todo)} phrases à annoter ({len(df) - len(todo)} déjà faites)")
    else:
        todo = df.copy()
        df["label"] = ""
        print(f"Annotation complète : {len(todo)} phrases")

    classifier = load_classifier(args.model)

    batches = [todo.iloc[i:i+args.batch] for i in range(0, len(todo), args.batch)]
    total = len(batches)
    t0 = time.time()

    for idx, batch in enumerate(batches, 1):
        print(f"Batch {idx}/{total} ({len(batch)} phrases)...", end=" ", flush=True)
        try:
            labels = annotate_batch(classifier, batch)
            for pk, label in labels.items():
                df.loc[df["PRIMARY_KEY"] == pk, "label"] = label

            elapsed = time.time() - t0
            remaining = elapsed / idx * (total - idx)
            print(f"OK — {len(labels)} annotées  (restant ~{remaining:.0f}s)")
        except Exception as e:
            print(f"ERREUR : {e}")
            df.to_csv(args.output, index=False, encoding="utf-8")
            raise

        if idx % 5 == 0:
            df.to_csv(args.output, index=False, encoding="utf-8")
            print(f"Sauvegarde intermédiaire : {args.output}")

    df.to_csv(args.output, index=False, encoding="utf-8")

    annotated = df[df["label"].str.strip() != ""]
    print(f"\nTerminé — {len(annotated)}/{len(df)} phrases annotées")
    print(f"Sauvegardé → {args.output}\n")
    print("Distribution des labels :")
    for lbl, cnt in sorted(df["label"].value_counts().items()):
        print(f"  {lbl:25s} : {cnt}")

    png_path = save_annotation_stats(df)
    print(f"\nStatistiques descriptives sauvegardées → {png_path}")


if __name__ == "__main__":
    main()