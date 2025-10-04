import os
from twec.twec import TWEC

# sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, "../.."))))
from develop.utils.paths import DATA

input_dir = os.path.join(DATA, "00_preprocessed_corpus")
aligner = TWEC(
        size=100, #50
        sg=0, #0
        siter=1, #20
        diter=1, #20
        window=5,
        workers=os.cpu_count(),
        opath=os.path.join(DATA, "03_twec")
    )

if __name__ == "__main__": 
    aligner.train_compass(input_dir, overwrite=True) # COMPASS
    aligner.diter = 100   
    aligner.siter = 100
    for file in os.listdir(input_dir):
        input_file = os.path.join(input_dir, file)
        aligner.train_slice(input_file, save=True) # SLICES