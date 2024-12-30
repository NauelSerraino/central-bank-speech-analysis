from twec.twec import TWEC
from gensim.models.word2vec import Word2Vec
import pdb

aligner = TWEC(size=30, siter=10, diter=10, workers=4)

# train the compass: the text should be the concatenation of the text from the slices

aligner.train_compass(
    "data/compass.txt", 
    overwrite=False
    ) # keep an eye on the overwrite behaviour objects
slice_one = aligner.train_slice(
    "data/txt-files.tar/txt-files/cache/epub/1/pg1.txt", 
    save=True
    )
slice_two = aligner.train_slice(
    "data/txt-files.tar/txt-files/cache/epub/2/pg2.txt", 
    save=True
    )

