import pytest
import pandas as pd
import numpy as np
import csv
import gzip
import pathlib
import requests
import string
import logging
import sys
from searcharray import SearchArray
from test_utils import Profiler, profile_enabled


csv.field_size_limit(sys.maxsize)


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
def msmarco_unzipped(msmarco_download):
    print("Unzipping...")
    msmarco_unzipped_path = 'data/msmarco-docs.tsv'
    msmarco_unzipped_path = pathlib.Path(msmarco_unzipped_path)

    if not msmarco_unzipped_path.exists():
        with gzip.open(msmarco_download, 'rb') as f_in:
            with open(msmarco_unzipped_path, 'wb') as f_out:
                f_out.write(f_in.read())
    return msmarco_unzipped_path


@pytest.fixture(scope="session")
def msmarco_all_raw(msmarco_download):
    print("Loading docs...")
    msmarco_raw_path = 'data/msmarco_all_raw.pkl'
    msmarco_all_raw_path = pathlib.Path(msmarco_raw_path)

    if not msmarco_all_raw_path.exists():
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


@pytest.fixture(scope="session")
def msmarco1m_raw(msmarco_download):
    msmarco_raw_path = 'data/msmarco1m_raw.pkl'
    msmarco1m_raw_path = pathlib.Path(msmarco_raw_path)

    if not msmarco1m_raw_path.exists():
        print("Loading docs...")
        msmarco = pd.read_csv(msmarco_download, sep="\t",
                              nrows=1000000,
                              header=None, names=["id", "url", "title", "body"])

        msmarco.to_pickle(msmarco_raw_path)
        return msmarco
    else:
        print("Loading pkl docs...")
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
def msmarco1m(msmarco1m_raw):
    msmarco_path = 'data/msmarco1m.pkl'
    msmarco1m_path = pathlib.Path(msmarco_path)

    if not msmarco1m_path.exists():
        def ws_punc_tokenizer(text):
            split = text.lower().split()
            return [token.translate(str.maketrans('', '', string.punctuation))
                    for token in split]

        msmarco = msmarco1m_raw
        msmarco["title_ws"] = SearchArray.index(msmarco["title"])
        msmarco["body_ws"] = SearchArray.index(msmarco["body"])

        msmarco.to_pickle(msmarco_path)
        return msmarco
    else:
        return pd.read_pickle(msmarco_path)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
@pytest.fixture(scope="session")
def msmarco_all(msmarco_all_raw):
    msmarco_path_str = 'data/msmarco_all.pkl'
    msmarco_path = pathlib.Path(msmarco_path_str)

    if not msmarco_path.exists():
        def ws_punc_tokenizer(text):
            split = text.lower().split()
            return [token.translate(str.maketrans('', '', string.punctuation))
                    for token in split]

        msmarco = msmarco_all_raw
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
def test_msmarco100k_phrase(phrase_search, msmarco100k, benchmark):
    profiler = Profiler(benchmark)
    phrase_search = phrase_search.split()
    print(f"STARTING {phrase_search}")
    print(f"Memory Usage (BODY): {msmarco100k['body_ws'].array.memory_usage() / 1024 ** 2:.2f} MB")
    profiler.run(msmarco100k['body_ws'].array.score, phrase_search)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
@pytest.mark.parametrize("phrase_search", ["what is", "what is the", "what is the purpose", "what is the purpose of", "what is the purpose of cats", "star trek", "star trek the next generation", "what what what"])
def test_msmarco1m_phrase(phrase_search, msmarco1m, benchmark):
    profiler = Profiler(benchmark)
    phrase_search = phrase_search.split()
    print(f"STARTING {phrase_search}")
    print(f"Memory Usage (BODY): {msmarco1m['body_ws'].array.memory_usage() / 1024 ** 2:.2f} MB")
    profiler.run(msmarco1m['body_ws'].array.score, phrase_search)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
def test_msmarco10k_indexing(msmarco100k_raw, benchmark):
    profiler = Profiler(benchmark)
    # Random 10k
    tenk = msmarco100k_raw['body'].sample(10000)
    results = profiler.run(SearchArray.index, tenk, autowarm=False)
    assert len(results) == 10000


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
def test_msmarco10k_indexing_warm(msmarco100k_raw, benchmark):
    profiler = Profiler(benchmark)
    # Random 10k
    tenk = msmarco100k_raw['body'].sample(10000)
    results = profiler.run(SearchArray.index, tenk, autowarm=True)
    assert len(results) == 10000


@pytest.mark.skip(reason="Not used on every run")
def test_msmarco1m_indexall(msmarco1m_raw, benchmark, caplog):
    caplog.set_level(logging.DEBUG)

    body = msmarco1m_raw['body']
    idxed = SearchArray.index(body)
    assert len(idxed) == len(body)


@pytest.mark.skip(reason="Not used on every run")
def test_msmarco_indexall(msmarco_unzipped, benchmark, caplog):
    caplog.set_level(logging.DEBUG)
    # Get an iterator through the msmarco dataset

    body_iter = csv_col_iter(msmarco_unzipped, 3)
    title_iter = csv_col_iter(msmarco_unzipped, 2)
    df = pd.DataFrame()
    print("Indexing body")
    df['body_tokens'] = SearchArray.index(body_iter, truncate=True)
    print("Indexing title")
    df['title_tokens'] = SearchArray.index(title_iter, truncate=True)
    print("Saving ids")
    df['msmarco_id'] = pd.read_csv(msmarco_unzipped, delimiter="\t", usecols=[0], header=None)
    print("Getting URL")
    df['msmarco_id'] = pd.read_csv(msmarco_unzipped, delimiter="\t", usecols=[1], header=None)
    # Save to pickle
    df.to_pickle("data/msmarco_indexed.pkl")


# FAILED test/test_msmarco.py::test_msmarco1m_or_search_unwarmed[what is] - assert False
@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
@pytest.mark.parametrize("query", ["what is", "what is the", "what is the purpose", "what is the purpose of", "what is the purpose of cats", "star trek", "star trek the next generation", "what what what"])
def test_msmarco1m_or_search_unwarmed(query, msmarco1m, benchmark, caplog):
    profiler = Profiler(benchmark)

    caplog.set_level(logging.DEBUG)

    def sum_scores(query):
        return np.sum([msmarco1m['body_ws'].array.score(query_term) for query_term in query.split()], axis=0)
    scores = profiler.run(sum_scores, query)
    assert len(scores) == len(msmarco1m['body_ws'].array), f"Expected {len(msmarco1m['body_ws'].array)}, got {len(scores)}"
    assert np.any(scores > 0), "No scores > 0"


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
@pytest.mark.parametrize("query", ["what is", "what is the", "what is the purpose", "what is the purpose of", "what is the purpose of cats", "star trek", "star trek the next generation", "what what what"])
def test_msmarco1m_or_search_no_cache(query, msmarco1m, benchmark, caplog):
    profiler = Profiler(benchmark)

    caplog.set_level(logging.DEBUG)

    df_cache = msmarco1m['body_ws'].array.posns.docfreq_cache
    tf_cache = msmarco1m['body_ws'].array.posns.termfreq_cache
    msmarco1m['body_ws'].array.posns.clear_cache()

    def sum_scores(query):
        msmarco1m['body_ws'].array.posns.clear_cache()
        return np.sum([msmarco1m['body_ws'].array.score(query_term) for query_term in query.split()], axis=0)
    scores = profiler.run(sum_scores, query)
    # Restore cache
    msmarco1m['body_ws'].array.posns.docfreq_cache = df_cache
    msmarco1m['body_ws'].array.posns.termfreq_cache = tf_cache
    assert len(scores) == len(msmarco1m['body_ws'].array)
    assert np.any(scores > 0)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
@pytest.mark.parametrize("query", ["what is", "what is the", "what is the purpose", "what is the purpose of", "what is the purpose of cats", "star trek", "star trek the next generation", "what what what"])
def test_msmarco1m_or_search_warmed(query, msmarco1m, benchmark, caplog):
    profiler = Profiler(benchmark)

    caplog.set_level(logging.DEBUG)

    def sum_scores(query):
        return np.sum([msmarco1m['body_ws'].array.score(query_term) for query_term in query.split()], axis=0)
    score_first = sum_scores(query)  # Warmup
    scores = profiler.run(sum_scores, query)
    assert len(scores) == len(msmarco1m['body_ws'].array)
    assert np.all(score_first == scores)
    assert np.any(scores > 0)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
@pytest.mark.parametrize("query", ["what is", "what is the", "what is the purpose", "what is the purpose of", "what is the purpose of cats", "star trek", "star trek the next generation", "what what what"])
def test_msmarco1m_or_search_warming(query, msmarco1m, benchmark, caplog):
    profiler = Profiler(benchmark)

    caplog.set_level(logging.DEBUG)

    def sum_scores(query):
        return np.sum([msmarco1m['body_ws'].array.score(query_term) for query_term in query.split()], axis=0)
    sum_scores(query)  # Warmup
    scores = profiler.run(sum_scores, query)
    assert len(scores) == len(msmarco1m['body_ws'].array)
    assert np.any(scores > 0)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
@pytest.mark.parametrize("query", ["what is", "what is the", "what is the purpose", "what is the purpose of", "what is the purpose of cats", "star trek", "star trek the next generation", "what what what"])
def test_msmarco1m_or_search_max_posn(query, msmarco1m, benchmark, caplog):
    profiler = Profiler(benchmark)

    caplog.set_level(logging.DEBUG)

    def sum_scores(query):
        return np.sum([msmarco1m['body_ws'].array.score(query_term, max_posn=17)
                       for query_term in query.split()], axis=0)
    sum_scores(query)  # Warmup
    scores = profiler.run(sum_scores, query)
    assert len(scores) == len(msmarco1m['body_ws'].array)
    assert np.any(scores > 0)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
@pytest.mark.parametrize("query", ["what is", "what is the", "what is the purpose", "what is the purpose of", "what is the purpose of cats", "star trek", "star trek the next generation", "what what what"])
def test_msmarco100k_or_search_unwarmed(query, msmarco100k, benchmark, caplog):
    profiler = Profiler(benchmark)

    caplog.set_level(logging.DEBUG)

    def sum_scores(query):
        return np.sum([msmarco100k['body_ws'].array.score(query_term) for query_term in query.split()], axis=0)
    scores = profiler.run(sum_scores, query)
    assert len(scores) == len(msmarco100k['body_ws'].array)
    assert np.any(scores > 0)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
@pytest.mark.parametrize("query", ["what is", "what is the", "what is the purpose", "what is the purpose of", "what is the purpose of cats", "star trek", "star trek the next generation", "what what what"])
def test_msmarco100k_or_search_no_cache(query, msmarco100k, benchmark, caplog):
    profiler = Profiler(benchmark)

    caplog.set_level(logging.DEBUG)

    df_cache = msmarco100k['body_ws'].array.posns.docfreq_cache
    tf_cache = msmarco100k['body_ws'].array.posns.termfreq_cache
    msmarco100k['body_ws'].array.posns.clear_cache()

    def sum_scores(query):
        msmarco100k['body_ws'].array.posns.clear_cache()
        return np.sum([msmarco100k['body_ws'].array.score(query_term) for query_term in query.split()], axis=0)
    scores = profiler.run(sum_scores, query)
    # Restore cache
    msmarco100k['body_ws'].array.posns.docfreq_cache = df_cache
    msmarco100k['body_ws'].array.posns.termfreq_cache = tf_cache
    assert len(scores) == len(msmarco100k['body_ws'].array)
    assert np.any(scores > 0)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
@pytest.mark.parametrize("query", ["what is", "what is the", "what is the purpose", "what is the purpose of", "what is the purpose of cats", "star trek", "star trek the next generation", "what what what"])
def test_msmarco100k_or_search_warmed(query, msmarco100k, benchmark, caplog):
    profiler = Profiler(benchmark)

    caplog.set_level(logging.DEBUG)

    def sum_scores(query):
        return np.sum([msmarco100k['body_ws'].array.score(query_term) for query_term in query.split()], axis=0)
    score_first = sum_scores(query)  # Warmup
    scores = profiler.run(sum_scores, query)
    assert len(scores) == len(msmarco100k['body_ws'].array)
    assert np.all(score_first == scores)
    assert np.any(scores > 0)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
@pytest.mark.parametrize("query", ["what is", "what is the", "what is the purpose", "what is the purpose of", "what is the purpose of cats", "star trek", "star trek the next generation", "what what what"])
def test_msmarco100k_or_search_max_posn(query, msmarco100k, benchmark, caplog):
    profiler = Profiler(benchmark)

    caplog.set_level(logging.DEBUG)

    def sum_scores(query):
        return np.sum([msmarco100k['body_ws'].array.score(query_term, max_posn=17)
                       for query_term in query.split()], axis=0)
    sum_scores(query)  # Warmup
    scores = profiler.run(sum_scores, query)
    assert len(scores) == len(msmarco100k['body_ws'].array)
    assert np.any(scores > 0)
