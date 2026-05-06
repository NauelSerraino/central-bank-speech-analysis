import os
import re
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk import download
from gensim.models.phrases import Phrases, Phraser

from develop.utils.paths import DATA
from develop.utils.logger import LoggerManager

log_mgr = LoggerManager(name="load_speeches", log_file="00_load_speeches.log", clear_log=True)
logger  = log_mgr.get_logger()

download('stopwords')
STOP_WORDS = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

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
    logger.info("Splitting into paragraphs...")
    all_paragraphs = []
    doc_paragraph_indexes = []
    for text in df["mistral_ocr"]:
        paragraphs = [re.sub(r"---\[PAGE_BREAK\]---", "", p.strip().lower())
                      for p in text.split("\n") if p.strip()]
        all_paragraphs.extend(paragraphs)
        doc_paragraph_indexes.append(list(range(len(all_paragraphs) - len(paragraphs), len(all_paragraphs))))

    logger.info(f"Tokenizing {len(all_paragraphs)} paragraphs from {len(df)} documents...")
    tokenized_paragraphs = tokenize_paragraphs_batch(all_paragraphs)

    non_empty = sum(1 for t in tokenized_paragraphs if t)
    logger.info(f"Tokenization done: {non_empty}/{len(all_paragraphs)} non-empty paragraphs")

    docs_paragraphs_tokens = []
    for idxs in doc_paragraph_indexes:
        docs_paragraphs_tokens.append([tokenized_paragraphs[i] for i in idxs])
    return docs_paragraphs_tokens

# -----------------------------
# Bigram model
# -----------------------------
def train_bigram_model(docs_paragraphs_tokens, min_count=50, threshold=50):
    logger.info(f"Training bigram model (min_count={min_count}, threshold={threshold})...")
    all_paragraphs = [p for doc in docs_paragraphs_tokens for p in doc]
    phrases = Phrases(all_paragraphs, min_count=min_count, threshold=threshold)
    bigram = Phraser(phrases)
    n_bigrams = len(bigram.phrasegrams)
    logger.info(f"Bigram model trained: {n_bigrams} bigram phrases detected")
    return bigram

def apply_bigram_model_to_docs(docs_paragraphs_tokens, bigram_model):
    logger.info("Applying bigram model to all docs...")
    return [
        "\n\n".join(" ".join(bigram_model[para]) for para in doc)
        for doc in docs_paragraphs_tokens
    ]

# -----------------------------
# Saving
# -----------------------------
def save_texts_by_year_and_region(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for country, code in country_map.items():
        sub_df = df[df['country'] == country]
        for year, group in sub_df.groupby(sub_df['date'].dt.year):
            file_path = os.path.join(output_dir, f"{year}_{code}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                for doc in group['mistral_ocr_preprocessed']:
                    if doc.strip():
                        f.write(doc.strip() + "\n\n")
            saved.append(f"{year}_{code} ({len(group)} docs)")
    logger.info(f"Saved {len(saved)} corpus slices to {output_dir}:")
    for s in saved:
        logger.info(f"  {s}")

# -----------------------------
# Main
# -----------------------------
logger.info("Loading dataset from HuggingFace...")
df = pd.read_parquet("hf://datasets/istat-ai/ECB-FED-speeches/data/train-00000-of-00001.parquet")
df['date'] = pd.to_datetime(df['date'])

logger.info(f"Dataset loaded: {len(df)} speeches | "
            f"years {df['date'].dt.year.min()}–{df['date'].dt.year.max()}")

n_before = len(df)
df = df[df['date'].dt.year >= 2000]
logger.info(f"Filtered to 2000+: {len(df)} speeches kept, {n_before - len(df)} dropped")
logger.info("Corpus breakdown by country:\n" +
            df['country'].value_counts().to_string())

logger.info("Tokenizing corpus...")
docs_paragraphs_tokens = tokenize_texts(df)

logger.info("Training bigram model...")
bigram_model = train_bigram_model(docs_paragraphs_tokens)

logger.info("Applying bigram model...")
df['mistral_ocr_preprocessed'] = apply_bigram_model_to_docs(docs_paragraphs_tokens, bigram_model)

output_folder = os.path.join(DATA, "00_preprocessed_corpus")
logger.info("Saving corpus slices by year and region...")
save_texts_by_year_and_region(df, output_folder)

logger.info("Preprocessing pipeline complete.")
