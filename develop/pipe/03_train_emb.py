import os
from twec.twec import TWEC

from develop.utils.paths import DATA
from develop.utils.logger import LoggerManager

log_mgr = LoggerManager(name="train_emb", log_file="03_train_emb.log", clear_log=True)
logger  = log_mgr.get_logger()

input_dir = os.path.join(DATA, "00_preprocessed_corpus")
output_dir = os.path.join(DATA, "03_twec")

COMPASS_SITER = 1
COMPASS_DITER = 1
SLICE_SITER   = 100
SLICE_DITER   = 100

aligner = TWEC(
    size=100,
    sg=0,
    siter=COMPASS_SITER,
    diter=COMPASS_DITER,
    window=5,
    seed=1,
    workers=os.cpu_count(),
    opath=output_dir,
)

if __name__ == "__main__":
    slices = sorted(os.listdir(input_dir))
    logger.info(f"TWEC config: size=100, sg=0, window=5, seed=1, workers={os.cpu_count()}")
    logger.info(f"Compass iterations: siter={COMPASS_SITER}, diter={COMPASS_DITER}")
    logger.info(f"Slice iterations:   siter={SLICE_SITER}, diter={SLICE_DITER}")
    logger.info(f"Corpus: {input_dir} | {len(slices)} slices")

    logger.info("Training compass...")
    aligner.train_compass(input_dir, overwrite=True)
    logger.info("Compass trained.")

    aligner.diter = SLICE_DITER
    aligner.siter = SLICE_SITER
    logger.info(f"Training {len(slices)} slices...")

    for file in slices:
        input_file = os.path.join(input_dir, file)
        aligner.train_slice(input_file, save=True)
        logger.info(f"Slice trained: {file}")

    logger.info(f"All slices saved to {output_dir}")
