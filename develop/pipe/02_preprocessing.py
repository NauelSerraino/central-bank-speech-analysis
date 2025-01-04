import os
import re
import string
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, "../.."))))

from develop.utils.paths import DATA

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_and_save(file_path, output_file_path):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
    additional_punctuations = "‘’“”"

    with open(file_path, 'r', encoding='utf-8') as input_file, \
         open(output_file_path, 'w', encoding='utf-8') as output_file:

        for line in tqdm(input_file, total=total_lines, desc="Processing lines"):
            if not line.strip() or line.isupper():
                continue

            if re.match(r"^\s*CHAPTER \w+\.", line) or re.match(r"^\s*[0-9]+", line):
                continue

            line = line.lower()
            line = re.sub(f"[{re.escape(string.punctuation + additional_punctuations)}]", " ", line)
            tokens = word_tokenize(line)
            tokens = [word for word in tokens if word not in stop_words]
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
            cleaned_line = " ".join(tokens)

            if cleaned_line.strip():
                output_file.write(cleaned_line + " ")

def preprocess_sliced_corpus(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for file in tqdm(os.listdir(input_dir), desc="Processing years"):
        input_file = os.path.join(input_dir, file)
        output_file = os.path.join(output_dir, file)
        preprocess_and_save(input_file, output_file)

if __name__ == "__main__":
    input_dir = os.path.join(DATA, "01_sliced_corpus")
    output_dir = os.path.join(DATA, "02_preprocessed_corpus")
    preprocess_sliced_corpus(input_dir, output_dir)
