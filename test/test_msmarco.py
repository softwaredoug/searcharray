import pytest
import pandas as pd
import pathlib
import requests
import string
from searcharray import SearchArray
from test_utils import Profiler, profile_enabled


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


@pytest.fixture(scope="session")
def msmarco_download():
    if not msmarco_exists():
        download_msmarco()
    return msmarco_path()


@pytest.fixture(scope="session")
def msmarco_raw(msmarco_download):
    msmarco_raw_path = 'data/msmarco_raw.pkl'
    msmarco100k_raw_path = pathlib.Path(msmarco_raw_path)

    if not msmarco100k_raw_path.exists():
        print("Loading docs...")
        msmarco = pd.read_csv(msmarco_download, sep="\t",
                              header=None, names=["id", "url", "title", "body"])

        msmarco.to_pickle(msmarco_raw_path)
        return msmarco
    else:
        return pd.read_pickle(msmarco_raw_path)


@pytest.fixture(scope="session")
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


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
@pytest.fixture(scope="session")
def msmarco100k(msmarco100k_raw):
    msmarco_path = 'data/msmarco100k.pkl'
    msmarco100k_path = pathlib.Path(msmarco_path)

    if not msmarco100k_path.exists():
        def ws_punc_tokenizer(text):
            split = text.lower().split()
            return [token.translate(str.maketrans('', '', string.punctuation))
                    for token in split]

        msmarco = msmarco100k_raw
        msmarco["title_ws"] = SearchArray.index(msmarco["title"])
        msmarco["body_ws"] = SearchArray.index(msmarco["body"])

        msmarco.to_pickle(msmarco_path)
        return msmarco
    else:
        return pd.read_pickle(msmarco_path)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
@pytest.fixture(scope="session")
def msmarco(msmarco_raw):
    msmarco_path_str = 'data/msmarco100k.pkl'
    msmarco_path = pathlib.Path(msmarco_path_str)

    if not msmarco_path.exists():
        def ws_punc_tokenizer(text):
            split = text.lower().split()
            return [token.translate(str.maketrans('', '', string.punctuation))
                    for token in split]

        msmarco = msmarco_raw
        msmarco["title_ws"] = SearchArray.index(msmarco["title"])
        msmarco["body_ws"] = SearchArray.index(msmarco["body"])
        msmarco.to_pickle(msmarco_path_str)
        return msmarco
    else:
        return pd.read_pickle(msmarco_path_str)


# Memory usage
#
# Indexed in 14.7362s
# [postings.py:303 - _build_index_from_dict() ] Padded Posn memory usage: 4274.036334991455 MB
# [postings.py:304 - _build_index_from_dict() ] Bitwis Posn memory usage: 800.7734680175781 MB

# (venv)  $ git co 60ad46d1a2edc1504942b2c80b71b38673ff6426                                              search-array$
# Previous HEAD position was 55c3594 Add mask for diff, but one test still fails
# HEAD is now at 60ad46d Save different phrase implementations
# (venv)  $ python -m pytest -s "test/test_msmarco.py"                                                   search-array$
# ================================================ test session starts ================================================
# platform darwin -- Python 3.11.6, pytest-7.4.3, pluggy-1.3.0
# rootdir: /Users/douglas.turnbull/src/search-array
# plugins: cov-4.1.0
# collected 1 item
#
# test/test_msmarco.py Phrase search...
# msmarco phraes search: 1.9268s
#
# After looping different widths
# e6980396976231a8a124a1d8d58ee939d8f27482
# test/test_msmarco.py Phrase search...
# msmarco phraes search: 1.5184s
#
# Before col cache
# test/test_msmarco.py msmarco phrase search ['what', 'is']: 2.0513s
# .msmarco phrase search ['what', 'is', 'the']: 2.6227s
# .msmarco phrase search ['what', 'is', 'the', 'purpose']: 1.0535s
# .msmarco phrase search ['what', 'is', 'the', 'purpose', 'of']: 1.2327s
# .msmarco phrase search ['what', 'is', 'the', 'purpose', 'of', 'cats']: 1.1104s
# .msmarco phrase search ['star', 'trek']: 0.4251s
# .msmarco phrase search ['star', 'trek', 'the', 'next', 'generation']: 0.9067s
#
# After col cache
# test/test_msmarco.py msmarco phrase search ['what', 'is']: 1.7201s
# .msmarco phrase search ['what', 'is', 'the']: 2.2504s
# .msmarco phrase search ['what', 'is', 'the', 'purpose']: 0.4560s
# .msmarco phrase search ['what', 'is', 'the', 'purpose', 'of']: 0.4879s
# .msmarco phrase search ['what', 'is', 'the', 'purpose', 'of', 'cats']: 0.1907s
# .msmarco phrase search ['star', 'trek']: 0.2590s
# .msmarco phrase search ['star', 'trek', 'the', 'next', 'generation']: 0.2521s
#
# After new binary representation
# test/test_msmarco.py msmarco phrase search ['what', 'is']. Found 5913. 0.9032s
# .msmarco phrase search ['what', 'is', 'the']. Found 978. 2.9973s
# .msmarco phrase search ['what', 'is', 'the', 'purpose']. Found 12. 0.7181s
# .msmarco phrase search ['what', 'is', 'the', 'purpose', 'of']. Found 9. 0.9779s
# .msmarco phrase search ['what', 'is', 'the', 'purpose', 'of', 'cats']. Found 0. 0.2539s
# .msmarco phrase search ['star', 'trek']. Found 4. 0.2690s
# .msmarco phrase search ['star', 'trek', 'the', 'next', 'generation']. Found 0. 0.2918s
# .msmarco phrase search ['what', 'what', 'what']. Found 0. 0.4040s
#
# Before removing scipy
# Memory Usage (BODY): 1167.23 MB
#
# Removing scipy
# Memory Usage (BODY): 985.34 MB
#
@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
@pytest.mark.parametrize("phrase_search", ["what is", "what is the", "what is the purpose", "what is the purpose of", "what is the purpose of cats", "star trek", "star trek the next generation", "what what what"])
def test_msmarco(phrase_search, msmarco100k, benchmark):
    profiler = Profiler(benchmark)
    phrase_search = phrase_search.split()
    print(f"STARTING {phrase_search}")
    print(f"Memory Usage (BODY): {msmarco100k['body_ws'].array.memory_usage() / 1024 ** 2:.2f} MB")
    profiler.run(msmarco100k['body_ws'].array.score, phrase_search)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
def test_msmarco10k_indexing(msmarco100k_raw, benchmark):
    profiler = Profiler(benchmark)
    # Random 10k
    tenk = msmarco100k_raw['body'].sample(10000)
    results = profiler.run(SearchArray.index, tenk)
    assert len(results) == 10000
