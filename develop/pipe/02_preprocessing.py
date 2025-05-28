import os
import re
import gc
import time
import multiprocessing as mp
from typing import Tuple, Dict, Optional, List

from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from ftlangdetect import detect

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, "../.."))))
from develop.utils.logger import LoggerManager

log_mgr = LoggerManager(
    name = "preprocessing", 
    log_file = "preprocessing.log",
    clear_log = True
    )
logger = log_mgr.get_logger()

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')
nltk.download('wordnet')
nltk.download('names')

def process_item(item: Tuple[int, str]) -> Tuple[int, Optional[str]]:
    """Cleans a corpus entry by removing newlines, extra spaces, and non-alphabetic characters.

    Args:
        item: A tuple of index and text string.

    Returns:
        A tuple containing the index and the cleaned text, or None if empty or all uppercase.
    """
    k, v = item
    re_newline = re.compile(r"[\r\n]+")
    re_multiple_spaces = re.compile(r"\s+")
    re_everything_except_alphabet = re.compile(r"[^\w\s]|_")
    re_non_4_digits = re.compile(r"\b(?!\d{4}\b)\d+\b")

    if not v.strip() or v.isupper():
        return k, None

    v = re_newline.sub(" ", v)
    v = re_multiple_spaces.sub(" ", v)
    v = re_everything_except_alphabet.sub("", v)
    v = re_non_4_digits.sub("", v)
    v = v.strip().lower()

    return k, v


class Preprocessor:
    """Handles full-cycle text preprocessing of a corpus using tokenization, stopword removal,
    lemmatization, and validation."""

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()


    def run(self, file_path: str, output_file_path: str) -> None:
        """Executes the entire preprocessing pipeline on a file.

        Args:
            file_path: Path to the input file.
            output_file_path: Path to write the processed file.
        """
        corpus = self._load_iterable_corpus(file_path)
        corpus = self._trim_corpus_with_regex(corpus)
        preprocessed_corpus = self._perform_text_preprocessing(corpus)
        self._save_preprocessed_corpus(preprocessed_corpus, output_file_path)

    def _load_iterable_corpus(self, file_path: str) -> Dict[int, str]:
        """Reads and splits a corpus file into paragraphs.

        Args:
            file_path: Path to the input file.

        Returns:
            A dictionary mapping indices to non-empty paragraphs.
        """
        logger.info(f"Loading corpus from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as input_file:
            corpus = input_file.read()
        corpus_iterable = corpus.split("\n")
        corpus_dict = {i: text for i, text in enumerate(corpus_iterable) if text.strip()}
        return corpus_dict

    def _trim_corpus_with_regex(self, corpus: Dict[int, str]) -> Dict[int, str]:
        """Cleans corpus using regex-based preprocessing.

        Args:
            corpus: Dictionary of raw text.

        Returns:
            Dictionary with cleaned text entries.
        """
        with mp.Pool(mp.cpu_count()) as pool:
            processed_items = list(
                tqdm(
                    pool.imap(process_item, corpus.items()),
                    total=len(corpus), 
                    desc="Trimming corpus with regex"
                    )
            )

        return {k: v for k, v in processed_items if v is not None}

    def _perform_text_preprocessing(self, corpus: Dict[int, str]) -> str:
        """Performs full tokenization and preprocessing pipeline.

        Args:
            corpus: Cleaned corpus dictionary.

        Returns:
            A single preprocessed text string.
        """
        start_time = time.time()
        with mp.Pool(mp.cpu_count()) as pool:
            tokens = list(tqdm(pool.imap(self._tokenize, corpus.items()), total=len(corpus), desc="Tokenizing"))
        log_mgr.log_resource_usage("_tokenize", start_time)

        start_time = time.time()
        with mp.Pool(mp.cpu_count()) as pool:
            tokens = list(tqdm(pool.imap(self._remove_stopwords, tokens), total=len(tokens), desc="Stopwords"))
        log_mgr.log_resource_usage("_remove_stopwords", start_time)

        start_time = time.time()
        with mp.Pool(mp.cpu_count()) as pool:
            tokens = list(tqdm(pool.imap(self._lemmatize, tokens), total=len(tokens), desc="Lemmatizing"))
        log_mgr.log_resource_usage("_lemmatize", start_time)

        start_time = time.time()
        with mp.Pool(mp.cpu_count()) as pool:
            tokens = list(tqdm(pool.imap(self._validate_words, tokens), total=len(tokens), desc="Validating words"))
        log_mgr.log_resource_usage("_validate_words", start_time)

        return " ".join(word for _, words in tokens for word in words)

    def _tokenize(self, pair: Tuple[int, str]) -> Tuple[int, List[str]]:
        k, v = pair
        return k, word_tokenize(v)

    def _remove_stopwords(self, pair: Tuple[int, List[str]]) -> Tuple[int, List[str]]:
        k, tokens = pair
        return k, [word for word in tokens if word not in self.stop_words]

    def _lemmatize(self, pair: Tuple[int, List[str]]) -> Tuple[int, List[str]]:
        k, tokens = pair
        return k, [self.lemmatizer.lemmatize(word) for word in tokens]

    def _validate_words(self, pair: Tuple[int, List[str]]) -> Tuple[int, List[str]]:
        k, tokens = pair
        return k, [word for word in tokens if 2 < len(word) < 20]

    def _save_preprocessed_corpus(self, preprocessed_corpus: str, output_file_path: str) -> None:
        """Writes processed text to disk.

        Args:
            preprocessed_corpus: The cleaned text.
            output_file_path: Destination path.
        """
        logger.info(f"Saving preprocessed corpus to {output_file_path}...")
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(preprocessed_corpus)

        log_mgr.log_resource_usage(f"_save_preprocessed_corpus_{output_file_path}", time.time())
        del preprocessed_corpus


def preprocess_sliced_corpus(input_dir: str, output_dir: str) -> None:
    """Preprocesses all corpus files from input_dir and saves to output_dir.

    Args:
        input_dir: Directory containing sliced corpus files.
        output_dir: Destination directory for processed files.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = os.listdir(input_dir)

    for file in tqdm(files, desc="Processing years"):
        input_file = os.path.join(input_dir, file)
        output_file = os.path.join(output_dir, file)

        prep = Preprocessor()
        prep.run(input_file, output_file)

        del prep
        gc.collect()


if __name__ == "__main__":
    from develop.utils.paths import DATA_ALT

    input_dir = os.path.join(DATA_ALT, "01_sliced_corpus")
    output_dir = os.path.join(DATA_ALT, "02_preprocessed_corpus")
    preprocess_sliced_corpus(input_dir, output_dir)
