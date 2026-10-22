"""Download hub data files into a local directory, rewriting only files that changed.

Hub modules (dmaprime/hubs/*.py) define which files to fetch; run a hub's sync with
`python -m dmaprime.data.sync [hub]` (default: flusight).
"""
import hashlib
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def _sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def _read_bytes_if_exists(path):
    if not path.exists():
        return None
    return path.read_bytes()


def _download_bytes(url, timeout=30):
    with urlopen(url, timeout=timeout) as response:
        return response.read()


def ensure_current(base_url, remote_files, data_dir, verbose=True):
    """Fetch base_url/<remote_file> into data_dir/<file name> for every remote file."""
    dataset_path = Path(data_dir)
    dataset_path.mkdir(parents=True, exist_ok=True)
    status = {}

    for remote_file in remote_files:
        filename = Path(remote_file).name
        local_path = dataset_path / filename
        remote_url = f"{base_url}/{remote_file}"
        local_bytes = _read_bytes_if_exists(local_path)

        try:
            remote_bytes = _download_bytes(remote_url)
        except (URLError, HTTPError) as exc:
            raise RuntimeError(f"Failed to fetch {remote_url}: {exc}") from exc

        local_hash = _sha256_bytes(local_bytes) if local_bytes is not None else None
        remote_hash = _sha256_bytes(remote_bytes)

        if local_hash != remote_hash:
            local_path.write_bytes(remote_bytes)
            action = "downloaded" if local_bytes is None else "updated"
        else:
            action = "current"

        status[str(local_path)] = {
            "action": action,
            "remote_url": remote_url,
            "sha256": remote_hash,
        }

        if verbose:
            print(f"{filename}: {action}")

    return status


if __name__ == "__main__":
    from dmaprime.hubs import HUBS
    HUBS[sys.argv[1] if len(sys.argv) > 1 else 'flusight'].sync()
