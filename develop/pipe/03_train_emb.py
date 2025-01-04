"""
Train embeddings using the TWEC model.
"""
import os
from twec.twec import TWEC
from gensim.models.word2vec import Word2Vec

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, "../.."))))

from develop.utils.paths import DATA

aligner = TWEC(size=100, siter=10, diter=10, workers=5)


aligner.train_compass(
    os.path.join(DATA, "02_preprocessed_corpus"),
    overwrite=False,
    ) 