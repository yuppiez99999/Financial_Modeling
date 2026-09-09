from pathlib import Path
import joblib
from src.utils.dummy_models import DummyModel

p = Path(__file__).resolve().parents[1] / 'models'
p.mkdir(exist_ok=True)
filenames = [
    'timesfm_short_term_5d.pkl',
    'timesfm_mid_term_10d.pkl',
    'timesfm_long_term_20d.pkl',
]
for fn in filenames:
    joblib.dump(DummyModel(), p / fn)
print('wrote', [str(p / fn) for fn in filenames])
