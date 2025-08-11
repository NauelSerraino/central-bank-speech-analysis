import os
import re
import logging
import pandas as pd
import spacy
import numpy as np
from nltk.corpus import stopwords
from nltk import download
from gensim.models.phrases import Phrases, Phraser

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

download('stopwords')
STOP_WORDS = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # only tag + lemma

country_map = {"United States": "USA", "Euro area": "EU"}

# -----------------------------
# Fast paragraph tokenizer
# -----------------------------
def tokenize_paragraphs_batch(paragraphs):
    """Tokenize and clean paragraphs in batch using spaCy.pipe."""
    results = []
    for doc in nlp.pipe(paragraphs, batch_size=64, n_process=4):
        tokens = [t.lemma_ for t in doc if t.is_alpha and t.text.lower() not in STOP_WORDS]
        results.append(tokens)
    return results

def tokenize_texts(df):
    """Split texts into paragraphs, then tokenize all in one batch."""
    logging.info("Splitting into paragraphs...")
    all_paragraphs = []
    doc_paragraph_indexes = []  # map doc -> list of paragraph indices
    for text in df["mistral_ocr"]:
        paragraphs = [re.sub(r"---\[PAGE_BREAK\]---", "", p.strip().lower())
                      for p in text.split("\n") if p.strip()]
        all_paragraphs.extend(paragraphs)
        doc_paragraph_indexes.append(list(range(len(all_paragraphs) - len(paragraphs), len(all_paragraphs))))

    logging.info(f"Tokenizing {len(all_paragraphs)} paragraphs...")
    tokenized_paragraphs = tokenize_paragraphs_batch(all_paragraphs)

    # Map back to documents
    docs_paragraphs_tokens = []
    for idxs in doc_paragraph_indexes:
        docs_paragraphs_tokens.append([tokenized_paragraphs[i] for i in idxs])
    return docs_paragraphs_tokens

# -----------------------------
# Bigram model
# -----------------------------
def train_bigram_model(docs_paragraphs_tokens, min_count=50, threshold=50):
    logging.info("Training bigram model...")
    all_paragraphs = [p for doc in docs_paragraphs_tokens for p in doc]
    phrases = Phrases(all_paragraphs, min_count=min_count, threshold=threshold)
    return Phraser(phrases)

def apply_bigram_model_to_docs(docs_paragraphs_tokens, bigram_model):
    logging.info("Applying bigram model to all docs...")
    return [
        "\n\n".join(" ".join(bigram_model[para]) for para in doc)
        for doc in docs_paragraphs_tokens
    ]

# -----------------------------
# Saving
# -----------------------------
def save_texts_by_year_and_region(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for country, code in country_map.items():
        sub_df = df[df['country'] == country]
        for year, group in sub_df.groupby(sub_df['date'].dt.year):
            file_path = os.path.join(output_dir, f"{year}_{code}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                for doc in group['mistral_ocr_preprocessed']:
                    if doc.strip():
                        f.write(doc.strip() + "\n\n")

# -----------------------------
# Main
# -----------------------------
logging.info("Loading data...")
df = pd.read_parquet("hf://datasets/istat-ai/ECB-FED-speeches/data/train-00000-of-00001.parquet")
df['date'] = pd.to_datetime(df['date'])

logging.info("Tokenizing corpus...")
docs_paragraphs_tokens = tokenize_texts(df)

logging.info("Training bigram model...")
bigram_model = train_bigram_model(docs_paragraphs_tokens)

logging.info("Applying bigram model...")
df['mistral_ocr_preprocessed'] = apply_bigram_model_to_docs(docs_paragraphs_tokens, bigram_model)

logging.info("Saving results...")
output_folder = os.path.join("your_data_alt_path", "02_preprocessed_corpus")
save_texts_by_year_and_region(df, output_folder)

logging.info("Done.")
