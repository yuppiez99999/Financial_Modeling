# research/trial_run.py
"""小范围试跑最小入口。"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def make_case():
    return {
        "prices": [100.0, 101.0, 102.0, 101.0, 103.0, 105.0],
        "y_true": [0, 1, 1, 0, 1, 1],
    }


def run():
    case = make_case()
    return {"case": case, "status": "ok"}


def main():
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
