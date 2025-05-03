import os

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = os.path.join(base_dir, 'data')
DATA_ALT = os.path.join(base_dir, 'data_alt')
DEVELOP = os.path.join(base_dir, 'develop')
MODEL = os.path.join(base_dir, 'model')
DEV_LOGS = os.path.join(base_dir, 'develop', 'logs')

os.makedirs(DATA, exist_ok=True)
os.makedirs(DATA_ALT, exist_ok=True)
os.makedirs(DEVELOP, exist_ok=True)
os.makedirs(MODEL, exist_ok=True)
os.makedirs(DEV_LOGS, exist_ok=True)