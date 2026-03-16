import os
import subprocess
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _vast_host() -> str:
    user = os.environ.get("VAST_AI_USER", "root")
    ip = os.environ.get("VAST_AI_IP")
    if not ip:
        raise EnvironmentError("VAST_AI_IP not set. Copy .env.example to .env and fill in values.")
    return f"{user}@{ip}"


def push_to_remote(local_path: str, remote_path: str) -> None:
    host = _vast_host()
    cmd = ["rsync", "-avz", "--progress", local_path, f"{host}:{remote_path}"]
    logger.info("Pushing %s → %s:%s", local_path, host, remote_path)
    subprocess.run(cmd, check=True)


def pull_from_remote(remote_path: str, local_path: str) -> None:
    host = _vast_host()
    cmd = ["rsync", "-avz", "--progress", f"{host}:{remote_path}", local_path]
    logger.info("Pulling %s:%s → %s", host, remote_path, local_path)
    subprocess.run(cmd, check=True)
