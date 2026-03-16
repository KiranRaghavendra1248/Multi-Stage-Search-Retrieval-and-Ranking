import os
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def detect_environment() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "remote"
    except ImportError:
        pass
    return "local"


def load_config(override_env: str | None = None) -> DictConfig:
    base = OmegaConf.load(_CONFIG_DIR / "settings.yaml")

    env = override_env or (
        base.environment if base.environment != "auto" else detect_environment()
    )
    override_path = _CONFIG_DIR / f"{env}_settings.yaml"
    override = OmegaConf.load(override_path)

    cfg = OmegaConf.merge(base, override)
    OmegaConf.set_readonly(cfg, False)
    cfg.environment = env
    return cfg
