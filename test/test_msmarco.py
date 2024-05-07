import pytest
import pandas as pd
import numpy as np
import pathlib
import string
import logging
from typing import Dict, List, Any, Optional
from searcharray import SearchArray
from searcharray.solr import edismax
from searcharray.utils.sort import SetOfResults
from test_utils import Profiler, profile_enabled
from msmarco_utils import msmarco1m_raw_path, msmarco100k_raw_path, msmarco_all_raw_path, csv_col_iter


def ws_punc_tokenizer(text):
    split = text.lower().split()
    return [token.translate(str.maketrans('', '', string.punctuation))
            for token in split]


@pytest.fixture(scope="session")
def msmarco_all_raw():
    return pd.read_pickle(msmarco_all_raw_path())


@pytest.fixture(scope="session")
def msmarco100k_raw(msmarco_download):
    return pd.read_pickle(msmarco100k_raw_path())


@pytest.fixture(scope="session")
def msmarco1m_raw(msmarco_download):
    return pd.read_pickle(msmarco1m_raw_path())


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
@pytest.fixture(scope="session")
def msmarco100k():
    msmarco100k_raw = pd.read_pickle(msmarco100k_raw_path())
    msmarco_path = 'data/msmarco100k.pkl'
    msmarco100k_path = pathlib.Path(msmarco_path)

    if not msmarco100k_path.exists():
        msmarco = msmarco100k_raw
        print("Indexing 100k docs...")
        msmarco['title'].fillna('', inplace=True)
        msmarco['body'].fillna('', inplace=True)
        print(" Index Title")
        msmarco["title_ws"] = SearchArray.index(msmarco["title"], tokenizer=ws_punc_tokenizer)
        print(" Index Body")
        msmarco["body_ws"] = SearchArray.index(msmarco["body"], tokenizer=ws_punc_tokenizer)
        print(" Done!... Saving")

        msmarco.to_pickle(msmarco_path)
        return msmarco
    else:
        return pd.read_pickle(msmarco_path)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
@pytest.fixture(scope="session")
def msmarco1m():
    msmarco1m_raw = pd.read_pickle(msmarco1m_raw_path())
    msmarco_path = 'data/msmarco1m.pkl'
    msmarco1m_path = pathlib.Path(msmarco_path)

    if not msmarco1m_path.exists():
        print("Indexing 1m docs...")
        msmarco = msmarco1m_raw
        msmarco['title'].fillna('', inplace=True)
        msmarco['body'].fillna('', inplace=True)
        print(" Index Title")
        msmarco["title_ws"] = SearchArray.index(msmarco["title"], tokenizer=ws_punc_tokenizer)
        print(" Index Body")
        msmarco["body_ws"] = SearchArray.index(msmarco["body"], tokenizer=ws_punc_tokenizer)

        print(" DONE!... Saving")
        msmarco.to_pickle(msmarco_path)
        return msmarco
    else:
        print("Loading idxed pkl docs...")
        msmarco = pd.read_pickle(msmarco_path)
        return msmarco


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
@pytest.fixture(scope="session")
def msmarco_all(msmarco_download):
    msmarco_all_raw = pd.read_pickle(msmarco_all_raw_path())
    msmarco_path_str = 'data/msmarco_all.pkl'
    msmarco_path = pathlib.Path(msmarco_path_str)

    if not msmarco_path.exists():
        print("Indexing all docs...")

        msmarco = msmarco_all_raw(msmarco_download)
        msmarco['title'].fillna('', inplace=True)
        msmarco['body'].fillna('', inplace=True)
        msmarco["title_ws"] = SearchArray.index(msmarco["title"], tokenizer=ws_punc_tokenizer)
        msmarco["body_ws"] = SearchArray.index(msmarco["body"], tokenizer=ws_punc_tokenizer)
        msmarco.to_pickle(msmarco_path_str)
        return msmarco
    else:
        print("Loading idxed pkl docs...")
        msmarco = pd.read_pickle(msmarco_path_str)
        print(f"Loaded msmarco -- {len(msmarco)} -- {msmarco['body_ws'].array.memory_usage() / 1024 ** 2:.2f} MB | {msmarco['title_ws'].array.memory_usage() / 1024 ** 2:.2f} MB")
        return msmarco


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
@pytest.mark.parametrize("phrase_search", ["what is", "what is the", "what is the purpose", "what is the purpose of", "what is the purpose of cats", "star trek", "star trek the next generation", "what what what", "the purpose"])
def test_msmarco1m_phrase(phrase_search, msmarco1m, benchmark):
    profiler = Profiler(benchmark)
    phrase_search = phrase_search.split()
    print(f"STARTING {phrase_search}")
    print(f"Memory Usage (BODY): {msmarco1m['body_ws'].array.memory_usage() / 1024 ** 2:.2f} MB")
    profiler.run(msmarco1m['body_ws'].array.score, phrase_search)


@pytest.mark.parametrize("phrase_search", ["what is", "what is the", "what is the purpose", "what is the purpose of", "what is the purpose of cats", "star trek", "star trek the next generation", "what what what", "the purpose"])
def test_msmarco1m_phrase_memray(phrase_search, msmarco1m, benchmark):
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
@pytest.mark.parametrize("query", ["what is", "what is the", "what is the purpose", "what is the purpose of", "what is the purpose of cats", "star trek", "star trek the next generation", "what what what", "the purpose"])
def test_msmarco1m_edismax(query, msmarco1m, benchmark, caplog):
    profiler = Profiler(benchmark)

    caplog.set_level(logging.DEBUG)

    def run_edismax(query):
        return edismax(msmarco1m, q=query,
                       mm=2,
                       qf=['title_ws^1.0', 'body_ws^0.5'],
                       pf=['title_ws^1.0', 'body_ws^0.5'],
                       pf2=['title_ws^1.0', 'body_ws^0.5'],
                       pf3=['title_ws^1.0', 'body_ws^0.5'],
                       tie=0.3)

    scores, explain = profiler.run(run_edismax, query)
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
def test_msmarco100k_edismax(query, msmarco100k, benchmark, caplog):
    profiler = Profiler(benchmark)

    caplog.set_level(logging.DEBUG)

    def run_edismax(query):
        return edismax(msmarco100k, q=query,
                       mm=2,
                       qf=['title_ws^1.0', 'body_ws^0.5'],
                       pf=['title_ws^1.0', 'body_ws^0.5'],
                       pf2=['title_ws^1.0', 'body_ws^0.5'],
                       # pf3=['title_ws^1.0', 'body_ws^0.5'],
                       tie=0.3)

    scores, explain = profiler.run(run_edismax, query)
    assert len(scores) == len(msmarco100k['body_ws'].array)
    assert np.any(scores > 0)


class SetOfResultsNaive:
    """Gather multiple sets of search results, ie for testing."""

    def __init__(self, df: pd.DataFrame):
        self.all_results: List[pd.DataFrame] = []
        self.df = df

    def ins_top_n(self, scores, N=10,
                  query: str = '', metadata: Optional[Dict[str, List[Any]]] = None):
        """Sort a dataframe by a score column.

        Args:
            df (pd.DataFrame): The dataframe to sort.
            score_col (str): The column to sort by.
            N (int): The number of rows to return.

        Returns:
            pd.DataFrame: The sorted dataframe.
        """
        top_n = np.argpartition(scores, -N)[-N:]
        results = self.df.iloc[top_n, :]
        results['score'] = scores[top_n]
        results['query'] = query
        results_sorted = results.sort_values('score', ascending=False)
        results_sorted['rank'] = np.arange(N) + 1
        self.all_results.append(results_sorted)

    def get_all(self) -> pd.DataFrame:
        return pd.concat(self.all_results).sort_values(['query', 'rank']).reset_index(drop=True)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
def test_msmarco100k_gather(msmarco100k, benchmark, caplog):
    """Run a query for a set of queries and compile a DF of results."""
    profiler = Profiler(benchmark)

    caplog.set_level(logging.DEBUG)

    queries = ['star trek',
               'star trek the next generation',
               'what is the purpose of cats',
               'what is the purpose of',
               'what is the purpose',
               'what is the',
               'what is',
               'what what what',
               'best buy',
               'beauty and the beast',
               'beauty and the beast the musical',
               'bears',
               'hat',
               'what is a hat',
               'what is a hat made of',
               'who are the beatles']

    def search_many(df, queries):
        results = SetOfResults(df)
        for query in queries:
            query_tokenized = msmarco100k['body_ws'].array.tokenizer(query)
            scores = np.sum([msmarco100k['body_ws'].array.score(query_term)
                             for query_term in query_tokenized], axis=0)
            results.ins_top_n(scores, query=query)

        return results.get_all()

    df = profiler.run(search_many, msmarco100k, queries)
    assert len(df) == len(queries) * 10


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
