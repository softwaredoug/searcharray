import pytest
import gzip
from time import perf_counter
import json
import pandas as pd
import numpy as np
import sys
from searcharray.postings import PostingsArray
from test_utils import Profiler, profile_enabled


should_profile = '--benchmark-disable' in sys.argv


@pytest.fixture(scope="session")
def tmdb_raw_data():
    path = 'fixtures/tmdb.json.gz'
    with gzip.open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def tmdb_data(tmdb_raw_data):
    ids = tmdb_raw_data.keys()
    titles = []
    overviews = []
    for id in ids:
        try:
            titles.append(tmdb_raw_data[id]['title'])
        except KeyError:
            titles.append('')

        try:
            overviews.append(tmdb_raw_data[id]['overview'])
        except KeyError:
            overviews.append('')

    assert len(ids) == len(titles) == len(overviews)

    df = pd.DataFrame({'title': titles, 'overview': overviews, 'doc_id': ids}, index=ids)
    indexed = PostingsArray.index(df['title'])
    df['title_tokens'] = indexed

    indexed = PostingsArray.index(df['overview'])
    df['overview_tokens'] = indexed
    return df


def test_tokenize_tmdb(tmdb_raw_data):
    ids = tmdb_raw_data.keys()
    titles = []
    overviews = []
    for id in ids:
        try:
            titles.append(tmdb_raw_data[id]['title'])
        except KeyError:
            titles.append('')

        try:
            overviews.append(tmdb_raw_data[id]['overview'])
        except KeyError:
            overviews.append('')

    assert len(ids) == len(titles) == len(overviews)

    df = pd.DataFrame({'title': titles, 'overview': overviews}, index=ids)
    # Create tokenized versions of each
    start = perf_counter()
    print("Indexing title...")
    indexed = PostingsArray.index(df['title'])
    stop = perf_counter()
    df['title_tokens'] = indexed
    print(f"Memory usage: {indexed.memory_usage()}")
    print(f"Time: {stop - start}")

    start = perf_counter()
    print("Indexing overview...")
    indexed = PostingsArray.index(df['overview'])
    stop = perf_counter()
    df['overview_tokens'] = indexed
    print(f"Memory usage: {indexed.memory_usage()}")
    print(f"Time: {stop - start}")

    assert len(df) == len(ids)


def test_slice_then_search(tmdb_data):
    star_wars_in_title = tmdb_data['title_tokens'].array.match(["Star", "Wars"])
    star_wars_in_title = tmdb_data[star_wars_in_title]
    skywalker_bm25 = star_wars_in_title['overview_tokens'].array.score(["Skywalker"])
    assert skywalker_bm25.shape[0] == 3


tmdb_term_matches = [
    ("Star", ['11', '330459', '76180']),
    ("Black", ['374430']),
]


@pytest.mark.parametrize("term,expected_matches", tmdb_term_matches)
def test_term_freqs(tmdb_data, term, expected_matches):
    sliced = tmdb_data[tmdb_data['doc_id'].isin(expected_matches)]
    term_freqs = sliced['title_tokens'].array.term_freq(term)
    assert np.all(term_freqs == 1)


tmdb_phrase_matches = [
    (["Star", "Wars"], ['11', '330459', '76180']),
    (["Black", "Mirror:"], ['374430']),
    (["this", "doesnt", "match", "anything"], []),
]


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
@pytest.mark.parametrize("phrase,expected_matches", tmdb_phrase_matches)
def test_phrase_match_tmdb(phrase, expected_matches, tmdb_data, benchmark):
    prof = Profiler(benchmark)
    mask = prof.run(tmdb_data['title_tokens'].array.match, phrase)
    matches = tmdb_data[mask].index.sort_values()
    assert (matches == expected_matches).all()


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
def test_index_benchmark(benchmark, tmdb_data):
    prof = Profiler(benchmark)
    results = prof.run(PostingsArray.index, tmdb_data['overview'])
    assert len(results) == len(tmdb_data)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
def test_copy_benchmark(benchmark, tmdb_data):
    prof = Profiler(benchmark)
    results = prof.run(tmdb_data['overview_tokens'].array.copy)
    assert len(results) == len(tmdb_data)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
def test_slice_benchmark(benchmark, tmdb_data):
    # Slice the first 1000 elements
    prof = Profiler(benchmark)
    results = prof.run(tmdb_data['overview_tokens'].array[:1000].copy)
    assert len(results) == 1000


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
def test_repr_html_benchmark(benchmark, tmdb_data):
    prof = Profiler(benchmark)
    results = prof.run(tmdb_data._repr_html_)
    assert len(results) > 0


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
@pytest.mark.parametrize("term", ['the', 'cat', 'star', 'skywalker'])
def test_term_freq(benchmark, tmdb_data, term):
    prof = Profiler(benchmark)
    results = prof.run(tmdb_data['overview_tokens'].array.term_freq, term)
    assert len(results) > 0


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
def test_gather_results(benchmark, tmdb_data):
    """Gathering results typical of a search operation."""
    def gather_multiple_results():
        N = 10
        all_results = []
        for keywords in [['Star', 'Wars'], ['Black', 'Mirror:'], ['rambo']]:
            score = tmdb_data['title_tokens'].array.score(keywords)
            score += tmdb_data['overview_tokens'].array.score(keywords)
            tmdb_data['score'] = score
            top_n = tmdb_data.sort_values('score', ascending=False)[:N].copy()
            top_n.loc[:, 'doc_id'] = top_n['doc_id'].astype(int)
            top_n.loc[:, 'rank'] = np.arange(N) + 1
            top_n.loc[:, 'keywords'] = " ".join(keywords)
            all_results.append(top_n)
        return pd.concat(all_results)
    prof = Profiler(benchmark)
    results = prof.run(gather_multiple_results)
    assert len(results) > 0


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
def test_eq_benchmark(benchmark, tmdb_data):
    prof = Profiler(benchmark)
    idx_again = PostingsArray.index(tmdb_data['overview'])
    compare_amount = 10000
    results = prof.run(tmdb_data['overview_tokens'][:compare_amount].array.__eq__, idx_again[:compare_amount])
    assert np.sum(results) == compare_amount

    # eq = benchmark(tmdb_data['overview_tokens'].array.__eq__, idx_again)
