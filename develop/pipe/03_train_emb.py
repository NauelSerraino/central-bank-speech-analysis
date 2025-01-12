"""
Train embeddings using the TWEC model.
"""
import os
from twec.twec import TWEC

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, "../.."))))
from develop.utils.paths import DATA, MODEL

input_dir = os.path.join(DATA, "02_preprocessed_corpus")
aligner = TWEC(size=100, siter=10, diter=10, workers=os.cpu_count())

if __name__ == "__main__": 
    aligner.train_compass(input_dir, overwrite=False) # COMPASS
    for file in os.listdir(input_dir):
        input_file = os.path.join(input_dir, file)
        aligner.train_slice(input_file, save=True) # SLICES