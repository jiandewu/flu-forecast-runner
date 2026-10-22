"""Open a pull request from the fork to the upstream hub with the GitHub CLI (gh).

The fine-grained token in .env is scoped to the fork and cannot open pull requests on the
upstream repository, so this uses the gh login instead (`gh auth status` must show an
account with the `repo` scope). On GitHub Actions, where gh has no login, a classic token
with the `public_repo` scope is given as PR_TOKEN and used instead. Last season's PRs were opened the same way: head
DMA-PRIME:main, title "Forecast for week <reference_date>", empty body.
"""
import json
import os
import shutil
import subprocess
import tempfile


def _gh(args):
    # Run outside any git checkout so gh never tries to push local branches. Drop the fork
    # token that auto_submit loads from .env: gh prefers GITHUB_TOKEN/GH_TOKEN over its own
    # login, and the fork token cannot open pull requests on the upstream hub.
    env = {k: v for k, v in os.environ.items() if k not in ('GITHUB_TOKEN', 'GH_TOKEN')}
    if os.environ.get('PR_TOKEN'):
        env['GH_TOKEN'] = os.environ['PR_TOKEN']
    with tempfile.TemporaryDirectory() as cwd:
        return subprocess.run(['gh'] + args, cwd=cwd, env=env, capture_output=True, text=True, encoding='utf-8')


def upstream_of(repo):
    result = _gh(['api', f'repos/{repo}', '--jq', '.parent.full_name'])
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f'Could not find the upstream of {repo}: {result.stderr.strip()}')
    return result.stdout.strip()


def open_pull_request(repo, branch, title, dry_run=False):
    """Open a PR from repo:branch to its upstream's main; reuse an open PR from the same head.

    Returns the PR URL, or None (with a message) if gh is missing or the PR could not be opened.
    The upload has already succeeded at this point, so failures here never raise.
    """
    if shutil.which('gh') is None:
        print('WARNING gh is not installed; open the pull request manually')
        return None
    try:
        upstream = upstream_of(repo)
    except RuntimeError as exc:
        print(f'WARNING {exc}; open the pull request manually')
        return None
    owner = repo.split('/')[0]

    listed = _gh(['pr', 'list', '-R', upstream, '--state', 'open', '--author', '@me',
                  '--json', 'number,url,title,headRefName,headRepositoryOwner'])
    if listed.returncode != 0:
        print(f'WARNING could not list open PRs on {upstream}: {listed.stderr.strip()}')
        return None
    for pr in json.loads(listed.stdout or '[]'):
        if pr['headRepositoryOwner']['login'] == owner and pr['headRefName'] == branch:
            # A PR's head is a branch, so the new commit is already part of this PR
            print(f'Open PR #{pr["number"]} ({pr["title"]}) already uses {owner}:{branch}; '
                  f'the upload was added to it: {pr["url"]}')
            return pr['url']

    args = ['pr', 'create', '-R', upstream, '--base', 'main', '--head', f'{owner}:{branch}',
            '--title', title, '--body', '']
    if dry_run:
        args.append('--dry-run')
    created = _gh(args)
    if created.returncode != 0:
        print(f'WARNING could not open the pull request on {upstream}: {created.stderr.strip()}')
        return None
    url = created.stdout.strip()
    print(f'{"Dry run, PR not created" if dry_run else "Opened PR"}: {url}')
    return url
