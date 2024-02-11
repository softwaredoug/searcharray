"""A function for memray profiling to access all CLI params"""
import pandas as pd
import pathlib
import requests

# Set python path to parent
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from searcharray import SearchArray  # noqa


def download_file(url):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        print(f"Downloading {url}")
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded to {local_filename}")
    return local_filename


def msmarco_path():
    return "data/msmarco-docs.tsv.gz"


def msmarco_exists():
    path = pathlib.Path(msmarco_path())
    return path.exists()


def download_msmarco():
    # Download to fixtures
    print("Downloading MSMARCO")

    url = "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz"
    download_file(url)
    # Ensure data directory
    pathlib.Path("data").mkdir(exist_ok=True)
    # Move to data directory
    path = "msmarco-docs.tsv.gz"
    pathlib.Path(path).rename(f"data/{path}")


def msmarco_download():
    if not msmarco_exists():
        download_msmarco()
    return msmarco_path()


def msmarco100k_raw(msmarco_download):
    msmarco_raw_path = 'data/msmarco100k_raw.pkl'
    msmarco100k_raw_path = pathlib.Path(msmarco_raw_path)

    if not msmarco100k_raw_path.exists():
        print("Loading docs...")
        msmarco = pd.read_csv(msmarco_download, sep="\t",
                              nrows=100000,
                              header=None, names=["id", "url", "title", "body"])

        msmarco.to_pickle(msmarco_raw_path)
        return msmarco
    else:
        return pd.read_pickle(msmarco_raw_path)


def run_msmarco10k_indexing(msmarco100k_raw):
    SearchArray.index(msmarco100k_raw["body"].sample(10000))


if __name__ == "__main__":
    path = msmarco_download()
    msmarco100k_raw_df = msmarco100k_raw(path)
    run_msmarco10k_indexing(msmarco100k_raw_df)
