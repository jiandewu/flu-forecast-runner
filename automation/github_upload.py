"""Upload a file to the hub fork and keep the fork in sync, through the GitHub REST API.

Used by automation/auto_submit.py. Reads GITHUB_TOKEN, GITHUB_REPO and GITHUB_BRANCH (default
main) from the environment or .env; the token is a fine-grained token for the fork only.
"""
import base64
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from dmaprime.utils.env import load_env_file


def _settings(branch=None):
    """(token, repo, branch) from the environment or .env."""
    load_env_file(".env")
    return (os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPO"),
            branch or os.environ.get("GITHUB_BRANCH", "main"))


def _github_request(url, method="GET", token=None, payload=None):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "FluSight-GitHub-Uploader",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_existing_file_sha(repo, remote_path, branch, token):
    encoded_path = quote(remote_path.strip("/"), safe="/")
    url = f"https://api.github.com/repos/{repo}/contents/{encoded_path}?ref={quote(branch)}"
    try:
        response = _github_request(url, token=token)
        return response.get("sha")
    except HTTPError as exc:
        if exc.code == 404:
            return None
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub lookup failed ({exc.code}): {error_body}") from exc


def file_exists(remote_path, branch=None):
    """Whether remote_path exists on the fork's branch."""
    token, repo, branch = _settings(branch)
    if not repo:
        raise ValueError("Missing GITHUB_REPO. Set it in .env as owner/repo.")
    return _get_existing_file_sha(repo, remote_path, branch, token) is not None


def sync_fork(branch=None):
    """Fast-forward a fork's branch to its upstream (GitHub merge-upstream API).

    Returns the API response, or None with a warning if the sync fails. A stale fork does
    not block a pull request, so callers should still upload when this returns None.
    Upstream changes under .github/workflows need the token's Workflows permission.
    """
    token, repo, branch = _settings(branch)
    if not token or not repo:
        print("WARNING fork sync skipped: GITHUB_TOKEN or GITHUB_REPO is not set")
        return None
    url = f"https://api.github.com/repos/{repo}/merge-upstream"
    try:
        response = _github_request(url, method="POST", token=token, payload={"branch": branch})
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"WARNING fork sync of {repo}:{branch} failed ({exc.code}), uploading anyway: {error_body}")
        return None
    print(f"Fork sync {repo}:{branch}: {response.get('message')}")
    return response


def push_file_to_github(local_file_path, remote_path, commit_message, branch=None):
    """Create or update remote_path on the fork's branch with one commit.

    Returns {'repo', 'branch', 'html_url', 'action'} ('created' or 'updated').
    """
    token, repo, branch = _settings(branch)
    if not token:
        raise ValueError("Missing GITHUB_TOKEN. Add it to .env or your environment.")
    if not repo:
        raise ValueError("Missing GITHUB_REPO. Set it in .env as owner/repo.")

    local_path = Path(os.path.expanduser(local_file_path)).resolve()
    if not local_path.is_file():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    encoded_path = quote(remote_path.strip("/"), safe="/")
    upload_url = f"https://api.github.com/repos/{repo}/contents/{encoded_path}"
    existing_sha = _get_existing_file_sha(repo, remote_path, branch, token)

    payload = {
        "message": commit_message,
        "content": base64.b64encode(local_path.read_bytes()).decode("utf-8"),
        "branch": branch,
    }
    if existing_sha:
        payload["sha"] = existing_sha

    try:
        response = _github_request(upload_url, method="PUT", token=token, payload=payload)
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub upload failed ({exc.code}): {error_body}") from exc

    return {
        "repo": repo,
        "branch": branch,
        "html_url": response.get("content", {}).get("html_url"),
        "action": "updated" if existing_sha else "created",
    }
