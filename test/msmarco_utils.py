import csv
import requests
import pathlib
import pandas as pd
import sys

csv.field_size_limit(sys.maxsize)


def msmarco_path():
    return "data/msmarco-docs.tsv.gz"


def msmarco_exists():
    path = pathlib.Path(msmarco_path())
    return path.exists()


# Use csv iterator for memory efficiency
def csv_col_iter(msmarco_unzipped_path, col_no, num_docs=None):
    with open(msmarco_unzipped_path, "rt") as f:
        csv_reader = csv.reader(f, delimiter="\t")
        for idx, row in enumerate(csv_reader):
            col = row[col_no]
            yield col
            if num_docs is not None and idx >= num_docs:
                break


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


def msmarco1m_raw_path():
    msmarco_raw_path = 'data/msmarco1m_raw.pkl'
    msmarco1m_raw_path = pathlib.Path(msmarco_raw_path)

    if not msmarco1m_raw_path.exists():
        print("Loading docs...")
        msmarco = pd.read_csv(msmarco_path(), sep="\t",
                              nrows=1000000,
                              header=None, names=["id", "url", "title", "body"])

        msmarco.to_pickle(msmarco_raw_path)
        return msmarco
    return msmarco1m_raw_path


def msmarco100k_raw_path():
    msmarco_raw_path = 'data/msmarco100k_raw.pkl'
    msmarco100k_raw_path = pathlib.Path(msmarco_raw_path)

    if not msmarco100k_raw_path.exists():
        print("Loading docs...")
        msmarco = pd.read_csv(msmarco_path(), sep="\t",
                              nrows=100000,
                              header=None, names=["id", "url", "title", "body"])

        msmarco.to_pickle(msmarco_raw_path)
    return msmarco100k_raw_path


def msmarco_all_raw_path():
    print("Loading docs...")
    msmarco_raw_path = 'data/msmarco_all_raw.pkl'
    msmarco_all_raw_path = pathlib.Path(msmarco_raw_path)

    if not msmarco_all_raw_path.exists():
        print("Loading docs...")
        msmarco = pd.read_csv(msmarco_path(), sep="\t",
                              header=None, names=["id", "url", "title", "body"])

        msmarco.to_pickle(msmarco_raw_path)

    return msmarco_raw_path


if not msmarco_exists():
    download_msmarco()
