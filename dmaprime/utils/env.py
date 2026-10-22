import os
from pathlib import Path


def load_env_file(env_path=".env"):
    env_file = Path(env_path)
    if not env_file.exists():
        return {}

    loaded = {}
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value
        loaded[key] = os.environ.get(key, value)

    return loaded
