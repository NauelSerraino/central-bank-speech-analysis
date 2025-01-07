from twec.twec import TWEC
from gensim.models.word2vec import Word2Vec

m = Word2Vec.load("model/compass.model")
breakpoint()
m2006 = Word2Vec.load("model/combined_2006.model")
m2007 = Word2Vec.load("model/combined_2007.model")
print(dir(m))