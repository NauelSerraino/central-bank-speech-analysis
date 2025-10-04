import os
from pathlib import Path
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from develop.utils.paths import DATA


def load_documents(folder_path, paragraphs_per_doc=1):
    """
    Load speeches and group paragraphs into docs.
    Assumes filenames like: YYYY_REGION_something.txt
    Example: 2015_EU_speech1.txt
    """
    docs, years, regions = [], [], []

    for file in Path(folder_path).glob("*.txt"):
        # Extract metadata from filename
        parts = file.stem.split("_")
        year = int(parts[0])
        region = parts[1].upper()  # EU or US

        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            
            for i in range(0, len(paragraphs), paragraphs_per_doc):
                grouped = " ".join(paragraphs[i:i+paragraphs_per_doc])
                if grouped.strip():
                    docs.append(grouped)
                    years.append(year)
                    regions.append(region)

    df = pd.DataFrame({"doc": docs, "year": years, "region": regions})
    return df


def train_and_save(df, embedding_model: SentenceTransformer) -> None:
    """Train a BERTopic model with given config and save outputs."""
    print(f"=== Training configuration ===")
    saving_folder = os.path.join(DATA, "01_bertopic")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        language="english",
        calculate_probabilities=True,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(df["doc"])
    df["topic"] = topics

    # Save model
    topic_model.save(f"bertopic_model")

    # Save topic words
    topic_info = topic_model.get_topic_info()
    topic_words = {
        topic: topic_model.get_topic(topic) 
        for topic in topic_info.Topic if topic != -1
    }
    topic_words_df = []
    for topic_id, words_probs in topic_words.items():
        for word, prob in words_probs:
            topic_words_df.append({"topic": topic_id, "word": word, "prob": prob})
    pd.DataFrame(topic_words_df).to_parquet(
        os.path.join(saving_folder, "bertopic_topic_words.parquet"),
        index=False
    )

    # Save topic prevalence
    topic_counts = (
        df.groupby(["year", "region", f"topic"])
          .size()
          .reset_index(name="count")
    )
    topic_counts.to_parquet(
        os.path.join(saving_folder, "bertopic_topic_counts_year_region.parquet"),
        index=False
    )

    print(f"Done: model, topic words, and topic counts saved.\n")


# === Main ===
if __name__ == "__main__":
    folder = os.path.join(DATA, "00_preprocessed_corpus")
    paragraphs_per_doc = 5

    df = load_documents(folder, paragraphs_per_doc)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    train_and_save(df.copy(), embedding_model)

    print("=== All models finished and saved ===")
