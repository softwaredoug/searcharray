import pytest
import pandas as pd
import pathlib
import requests
import string
from time import perf_counter
from searcharray.postings import PostingsArray


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


def msmarco_exists():
    path = pathlib.Path("data/msmarco-docs.tsv.gz")
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


@pytest.fixture(scope="session")
def msmarco100k():
    msmarco100k_path = pathlib.Path("data/msmarco100k.pkl")

    if not msmarco100k_path.exists():
        start = perf_counter()
        if not msmarco_exists():
            download_msmarco()
        print("Loading docs...")
        msmarco = pd.read_csv("data/msmarco-docs.tsv.gz", sep="\t",
                              nrows=100000,
                              header=None, names=["id", "url", "title", "body"])
        print(f"Loaded {len(msmarco)} docs in {perf_counter() - start:.4f}s")

        def ws_punc_tokenizer(text):
            split = text.lower().split()
            return [token.translate(str.maketrans('', '', string.punctuation))
                    for token in split]

        print("Indexing...")
        msmarco["title_ws"] = PostingsArray.index(msmarco["title"])
        print(f"Indexed in {perf_counter() - start:.4f}s")
        msmarco["body_ws"] = PostingsArray.index(msmarco["body"])
        print(f"Indexed in {perf_counter() - start:.4f}s")

        # Save as pickle
        msmarco.to_pickle("data/msmarco100k.pkl")
        return msmarco
    else:
        return pd.read_pickle("data/msmarco100k.pkl")


@pytest.fixture(scope="session")
def msmarco():
    start = perf_counter()
    if not msmarco_exists():
        download_msmarco()
    print("Loading docs...")
    msmarco = pd.read_csv("data/msmarco-docs.tsv.gz", sep="\t", header=None, names=["id", "url", "title", "body"])
    print(f"Loaded {len(msmarco)} docs in {perf_counter() - start:.4f}s")

    def ws_punc_tokenizer(text):
        split = text.lower().split()
        return [token.translate(str.maketrans('', '', string.punctuation))
                for token in split]

    print("Indexing...")
    # msmarco["title_ws"] = PostingsArray.index(msmarco["title"])
    msmarco["body_ws"] = PostingsArray.index(msmarco["body"])
    print(f"Indexed in {perf_counter() - start:.4f}s")

    return msmarco


@pytest.mark.skip
def test_msmarco(msmarco100k):
    phrase_search = ["what", "is"]
    start = perf_counter()
    print("Phrase search...")
    results = msmarco100k['body_ws'].array.bm25(phrase_search)
    print(f"msmarco phraes search: {perf_counter() - start:.4f}s")

