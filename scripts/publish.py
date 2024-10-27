"""Fetches the latest build artifacts from a GitHub Actions workflow and uploads them to PyPI using twine."""
import os
import subprocess
import requests
import tempfile
import zipfile


GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')


def fetch_workflow_actions(sha):
    print(f"Fetching workflow actions for {sha}")
    url = "https://api.github.com/repos/softwaredoug/searcharray/actions/runs"
    params = {
        "branch": "main",
        "status": "completed",
        "head_sha": sha,
    }
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
    }
    response = requests.get(url, headers=headers,
                            params=params)
    response.raise_for_status()
    return response.json()


def fetch_artifacts(artifact_url, dest="build/"):
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
    }
    response = requests.get(artifact_url, headers=headers)
    response.raise_for_status()
    artifacts = response.json()
    download_dir = tempfile.mkdtemp()
    for artifact in artifacts['artifacts']:
        download_url = artifact['archive_download_url']
        response = requests.get(download_url, headers=headers)
        response.raise_for_status()
        print(f"Downloading {artifact['name']} to {download_dir}")
        with open(download_dir + artifact['name'] + '.zip', 'wb') as f:
            f.write(response.content)
            with zipfile.ZipFile(download_dir + artifact['name'] + '.zip', 'r') as zip_ref:
                print(f"Extracting {artifact['name']} to {dest}")
                zip_ref.extractall(dest)


def wheels(git_sha):
    actions = fetch_workflow_actions(git_sha)
    name = "Build Wheels"
    artifact_urls = []
    for run in actions['workflow_runs']:
        if run['name'] == name:
            artifact_urls.append(run['artifacts_url'])
    return artifact_urls


def twine_upload(from_dir):
    subprocess.run(["twine", "upload", "--skip-existing", f"{from_dir}"], check=True)


def main():
    git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    git_sha_short = git_sha[:7]
    dest = f"build/{git_sha_short}/"
    os.makedirs(dest, exist_ok=True)
    wheels_to_fetch = wheels(git_sha)
    if not wheels_to_fetch:
        raise ValueError("No artifacts found")
    for artifact_url in wheels_to_fetch:
        fetch_artifacts(artifact_url, dest)
    twine_upload(dest + "*.whl")


if __name__ == "__main__":
    main()
