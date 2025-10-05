import os

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = os.path.join(base_dir, 'data')
NOTEBOOKS_ANALYSIS = os.path.join(DATA, 'notebooks_analysis')
DEVELOP = os.path.join(base_dir, 'develop')
MODEL = os.path.join(base_dir, 'model')
DEV_LOGS = os.path.join(base_dir, 'develop', 'logs')

os.makedirs(NOTEBOOKS_ANALYSIS, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
os.makedirs(DEVELOP, exist_ok=True)
os.makedirs(DEV_LOGS, exist_ok=True)

data_folders = [
    "00_preprocessed_corpus",
    "01_bertopic",
    "03_twec",
    "04_create_metrics",
    "05_graphs"
    ]

for folder in data_folders:
    os.makedirs(os.path.join(DATA, folder),  exist_ok=True)