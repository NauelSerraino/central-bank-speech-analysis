import os

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = os.path.join(base_dir, 'data')
DATA_ALT = os.path.join(base_dir, 'data_alt')
NOTEBOOKS_ANALYSIS = os.path.join(DATA_ALT, 'notebooks_analysis')
DEVELOP = os.path.join(base_dir, 'develop')
MODEL = os.path.join(base_dir, 'model')
MODEL_USA = os.path.join(base_dir, 'model_USA')
MODEL_EU = os.path.join(base_dir, 'model_EU')
DEV_LOGS = os.path.join(base_dir, 'develop', 'logs')

os.makedirs(DATA, exist_ok=True)
os.makedirs(NOTEBOOKS_ANALYSIS, exist_ok=True)
os.makedirs(DATA_ALT, exist_ok=True)
os.makedirs(DEVELOP, exist_ok=True)
os.makedirs(MODEL_USA, exist_ok=True)
os.makedirs(MODEL_EU, exist_ok=True)
os.makedirs(DEV_LOGS, exist_ok=True)