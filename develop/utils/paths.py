import os

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = os.path.join(base_dir, 'data')
DEVELOP = os.path.join(base_dir, 'develop')
MODEL = os.path.join(base_dir, 'model')

os.makedirs(DATA, exist_ok=True)
os.makedirs(DEVELOP, exist_ok=True)
os.makedirs(MODEL, exist_ok=True)

print(f"Data path: {DATA}")
print(f"Develop path: {DEVELOP}")
print(f"Model path: {MODEL}")