"""
Annotation automatique avec API Claude (annotation_sample.csv)

Labels :
  - langue_de_bois     : rhétorique vague, slogans creux, promesses floues
  - non_langue_de_bois : affirmations concrètes, faits précis, chiffres, actions spécifiques
  - autre              : texte administratif, métadonnées, entêtes, texte non-français, bruit OCR
"""

import argparse
import time
import pandas as pd
import anthropic

SYSTEM_PROMPT = """Tu es un expert en analyse du discours politique français.
Ta tâche : annoter des phrases extraites de tracts électoraux français.

Labels disponibles (un seul par phrase) :
- langue_de_bois     : rhétorique vague, slogans creux, promesses floues, généralités sans contenu concret, appels émotionnels vides
- non_langue_de_bois : affirmations concrètes avec faits précis, chiffres, noms propres, actions spécifiques et vérifiables, propositions législatives détaillées
- autre              : texte non-politique (en-têtes administratifs, métadonnées, texte en langue étrangère, bruit OCR, mentions d'imprimerie, cases à cocher ☐☒, etc.)

Réponds UNIQUEMENT avec un JSON array dans le format suivant, sans aucun texte supplémentaire :
[{"id": "PRIMARY_KEY", "label": "label_choisi"}, ...]"""

USER_TEMPLATE = """Annote les phrases suivantes :

{sentences_block}"""


def build_sentences_block(batch: pd.DataFrame) -> str:
    lines = []
    for _, row in batch.iterrows():
        sentence = str(row["sentence"]).replace('"', "'")
        lines.append(f'{row["PRIMARY_KEY"]}: "{sentence}"')
    return "\n".join(lines)


def annotate_batch(client: anthropic.Anthropic, batch: pd.DataFrame, model: str) -> dict[str, str]:
    block = build_sentences_block(batch)
    user_msg = USER_TEMPLATE.format(sentences_block=block)

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    import json
    text = response.content[0].text.strip()
    # extraction json
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    results = json.loads(text)
    return {item["id"]: item["label"] for item in results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/labels/annotation_sample.csv")
    parser.add_argument("--output", default="data/labels/annotation_sample.csv")
    parser.add_argument("--batch",  default=40, type=int, help="Phrases par requête API")
    parser.add_argument("--model",  default="claude-sonnet-4-6")
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

    client = anthropic.Anthropic()

    batches = [todo.iloc[i:i+args.batch] for i in range(0, len(todo), args.batch)]
    total = len(batches)

    for idx, batch in enumerate(batches, 1):
        print(f"Batch {idx}/{total} ({len(batch)} phrases)...", end=" ", flush=True)
        try:
            labels = annotate_batch(client, batch, args.model)
            for pk, label in labels.items():
                df.loc[df["PRIMARY_KEY"] == pk, "label"] = label
            print(f"OK — {len(labels)} annotées")
        except Exception as e:
            print(f"ERREUR : {e}")
            # sauvegarde partielle
            df.to_csv(args.output, index=False, encoding="utf-8")
            raise

        # sauvegarde progressive
        if idx % 5 == 0:
            df.to_csv(args.output, index=False, encoding="utf-8")
            print(f"Sauvegarde intermédiaire : {args.output}")

        if idx < total:
            time.sleep(0.5)

    df.to_csv(args.output, index=False, encoding="utf-8")

    annotated = df[df["label"].str.strip() != ""]
    print(f"\nTerminé — {len(annotated)}/{len(df)} phrases annotées")
    print(f"Sauvegardé → {args.output}\n")
    print("Distribution des labels :")
    for lbl, cnt in sorted(df["label"].value_counts().items()):
        print(f"  {lbl:25s} : {cnt}")

if __name__ == "__main__":
    main()