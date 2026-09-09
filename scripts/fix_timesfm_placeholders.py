from pathlib import Path
import joblib
from src.utils.dummy_models import DummyScaler

p = Path(__file__).resolve().parents[1] / 'models'
filenames = [
    'timesfm_short_term_5d.pkl',
    'timesfm_mid_term_10d.pkl',
    'timesfm_long_term_20d.pkl',
]
for fn in filenames:
    path = p / fn
    if not path.exists():
        print('missing', path)
        continue
    obj = joblib.load(path)
    # 如果已是 dict 且包含 model，则跳过
    if isinstance(obj, dict) and 'model' in obj:
        print('already wrapped', path)
        continue
    wrapped = {'model': obj, 'scaler': DummyScaler()}
    joblib.dump(wrapped, path)
    print('rewrote', path)
