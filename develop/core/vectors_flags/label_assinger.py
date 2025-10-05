import re
import numpy as np
from sentence_transformers import SentenceTransformer, util

class LabelAssigner:
    _instance = None
    _labels = [
        "monetary policy and interest rates",
        "inflation and consumer prices",
        "labor market and employment",
        "financial markets and banking",
        "fiscal policy and government spending",
        "international trade and global economy",
        "economic development and inequality",
        "climate change and sustainable finance",
        "industrial and sectoral economics",
        "behavioral economics and expectations",
        "policy communication and forward guidance"
    ]

    _threshold = 0.30
    _model_name = 'all-MiniLM-L6-v2'

    def __init__(self):
        self.model = SentenceTransformer(self._model_name)
        self.labels = self._labels
        self.threshold = self._threshold
        self.label_embs = self.model.encode(self.labels, convert_to_numpy=True)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _clean_word(self, word):
        word = word.replace('_', ' ')
        return re.sub(r"[\[\]'\"]", '', word)

    def assign_labels(self, df):
        df['word_clean'] = df['word'].astype(str).apply(self._clean_word)
        unique_words = df['word_clean'].dropna().unique()
        word_embs = self.model.encode(unique_words, convert_to_numpy=True)
        
        word_to_label = {}
        for word, vec in zip(unique_words, word_embs):
            sims = util.cos_sim(self.label_embs, vec).numpy().flatten()
            assigned = [self.labels[i] for i, s in enumerate(sims) if s > self.threshold]
            word_to_label[word] = assigned[0] if assigned else None
        
        df['label'] = df['word_clean'].map(word_to_label)
        df.drop(columns=['word_clean'], inplace=True)
        
        return df

