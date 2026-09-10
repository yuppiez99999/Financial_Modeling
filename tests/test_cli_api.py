"""测试 CLI 入口与 API 应用创建（可选依赖降级）。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main as main_cli


def test_cli_parser_has_commands():
    parser = main_cli.build_parser()
    ns = parser.parse_args(["train"])
    assert ns.command == "train"


def test_cli_load_config():
    cfg = main_cli.load_config()
    assert "model" in cfg and "data" in cfg


def test_api_create_app_maybe_skip():
    from src.api import server
    if not server._HAS_FASTAPI:
        return  # 未安装 FastAPI 时跳过
    app = server.create_app(str(PROJECT_ROOT / "configs" / "config.yaml"))
    assert app is not None
