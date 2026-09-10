# finrl_experiments/run_minimal.py
"""FinRL 最小实验入口。"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def make_toy_env():
    return {
        "prices": [100.0, 101.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0],
        "actions": [0, 1, 1, 0, 1, 1, 0, 1],
    }


def run():
    env = make_toy_env()
    return {"env": env, "status": "ok"}


def main():
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
