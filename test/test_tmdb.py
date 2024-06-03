import pytest
import gzip
from time import perf_counter
import json
import pandas as pd
import numpy as np
import sys
from searcharray.postings import SearchArray
from searcharray.solr import edismax
from test_utils import Profiler, profile_enabled, naive_find_term


should_profile = '--benchmark-disable' in sys.argv


@pytest.fixture(scope="session")
def tmdb_raw_data():
    path = 'fixtures/tmdb.json.gz'
    with gzip.open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def tmdb_pd_data(tmdb_raw_data):
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
    return df


@pytest.fixture(scope="session")
def tmdb_data(tmdb_pd_data):
    df = tmdb_pd_data
    indexed = SearchArray.index(df['title'])
    df['title_tokens'] = indexed

    indexed = SearchArray.index(df['overview'])
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
    indexed = SearchArray.index(df['title'])
    stop = perf_counter()
    df['title_tokens'] = indexed
    print(f"Memory usage: {indexed.memory_usage()}")
    print(f"Time: {stop - start}")

    start = perf_counter()
    print("Indexing overview...")
    indexed = SearchArray.index(df['overview'])
    stop = perf_counter()
    df['overview_tokens'] = indexed
    print(f"Memory usage: {indexed.memory_usage()}")
    print(f"Time: {stop - start}")

    assert len(df) == len(ids)


def test_slice_then_search(tmdb_data):
    star_wars_in_title = tmdb_data['title_tokens'].array.termfreqs(["Star", "Wars"]) > 0
    star_wars_in_title = tmdb_data[star_wars_in_title]
    skywalkec_docfreq = star_wars_in_title['overview_tokens'].array.docfreq("Skywalker")
    assert skywalkec_docfreq <= star_wars_in_title['overview_tokens'].array.corpus_size
    skywalker_bm25 = star_wars_in_title['overview_tokens'].array.score(["Skywalker"])
    assert skywalker_bm25.shape[0] == len(star_wars_in_title)
    assert np.all(skywalker_bm25 >= 0)


def test_batch_sizes_give_same(tmdb_data):
    with_batch_10k = SearchArray.index(tmdb_data['overview'], batch_size=10000)
    with_batch_5k = SearchArray.index(tmdb_data['overview'], batch_size=5000)
    # We don't expect the full array to be compatible given term dict assigned
    # different term ids given threading, but individual docs should be the same
    assert np.all(with_batch_10k[-1] == with_batch_5k[-1])
    assert np.all(with_batch_10k[100] == with_batch_5k[100])
    assert np.all(with_batch_10k[5000] == with_batch_5k[5000])
    assert np.all(with_batch_10k[5001] == with_batch_5k[5001])


tmdb_term_matches = [
    ("Star", ['11', '330459', '76180']),
    ("Black", ['374430']),
]


@pytest.mark.parametrize("term,expected_matches", tmdb_term_matches)
def test_term_freqs(tmdb_data, term, expected_matches):
    sliced = tmdb_data[tmdb_data['doc_id'].isin(expected_matches)]
    term_freqs = sliced['title_tokens'].array.termfreqs(term)
    assert np.all(term_freqs == 1)


queries = [
    "Star Wars",
    "the next generation",
    "bartender fights a cow and",
    "to be or not to be",
    "the quick brown fox jumps over the lazy dog",
    "bill and ted's excellent adventure",
    "thirty years after defeating the galactic empire",
    "a film about a daughter of a refugee family",
    "have one thing in mind: to find a way to kill each other without risk. After listening to a radio show, Paul decided",
    "executive who can't stop his career downspiral is invited into his daughter's imaginary world, where solutions to his"
]


@pytest.mark.parametrize("query", queries)
def test_tmdb_expected_edismax(query, tmdb_data):

    title_tokenizer = tmdb_data['title_tokens'].array.tokenizer
    overview_tokenizer = tmdb_data['overview_tokens'].array.tokenizer
    title_has_term = np.sum([naive_find_term(tmdb_data['title'],
                             query_term,
                             title_tokenizer) for query_term in title_tokenizer(query)], axis=0) > 0
    overview_has_term = np.sum([naive_find_term(tmdb_data['overview'],
                                query_term,
                                overview_tokenizer) for query_term in overview_tokenizer(query)], axis=0) > 0

    matches, _ = edismax(tmdb_data, q=query,
                         qf=["title_tokens^2", "overview_tokens"],
                         pf=["title_tokens^2", "overview_tokens"],
                         pf2=["title_tokens^2", "overview_tokens"],
                         tie=0.1,
                         mm=1)
    matches = tmdb_data[matches > 0]
    expected_matches = tmdb_data[title_has_term | overview_has_term].index
    print(f"Query - {query} | Expected: {len(expected_matches)}")
    assert np.all(matches.index == expected_matches)


@pytest.mark.parametrize("query", queries)
def test_tmdb_expected_edismax_and_query(query, tmdb_data):

    title_tokenizer = tmdb_data['title_tokens'].array.tokenizer
    overview_tokenizer = tmdb_data['overview_tokens'].array.tokenizer
    num_terms = len(query.split())
    title_matches = np.asarray([naive_find_term(tmdb_data['title'],
                                query_term,
                                title_tokenizer) for query_term in title_tokenizer(query)]) > 0
    overview_matches = np.asarray([naive_find_term(tmdb_data['overview'],
                                   query_term,
                                   overview_tokenizer) for query_term in overview_tokenizer(query)]) > 0

    either_has_term = (title_matches + overview_matches)
    all_terms_have_match = np.sum(either_has_term, axis=0) == num_terms

    matches, _ = edismax(tmdb_data, q=query,
                         qf=["title_tokens^2", "overview_tokens"],
                         pf=["title_tokens^2", "overview_tokens"],
                         pf2=["title_tokens^2", "overview_tokens"],
                         tie=0.1,
                         mm="100%")
    matches = tmdb_data[matches > 0]

    expected_matches = tmdb_data[all_terms_have_match].index
    print(f"Query - {query} | Expected: {len(expected_matches)}")
    assert np.all(matches.index == expected_matches)


tmdb_phrase_matches = [
    (["Star", "Wars"], ['11', '330459', '76180']),
    (["Black", "Mirror:"], ['374430']),
    (["this", "doesnt", "match", "anything"], []),
]


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
@pytest.mark.parametrize("phrase,expected_matches", tmdb_phrase_matches)
def test_phrase_match_tmdb(phrase, expected_matches, tmdb_data, benchmark):
    prof = Profiler(benchmark)
    mask = prof.run(tmdb_data['title_tokens'].array.score, phrase)
    matches = tmdb_data[mask].index.sort_values()
    assert (matches == expected_matches).all()


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
def test_repr_html(tmdb_data, benchmark):
    prof = Profiler(benchmark)
    html = prof.run(tmdb_data._repr_html_)
    assert "<th>title</th>" in html


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
def test_unique(tmdb_data, benchmark):
    prof = Profiler(benchmark)
    prof.run(tmdb_data['overview'].array.unique)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
def test_index_benchmark(benchmark, tmdb_pd_data):
    prof = Profiler(benchmark)
    results = prof.run(SearchArray.index, tmdb_pd_data['overview'], autowarm=False)
    assert len(results) == len(tmdb_pd_data)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
def test_index_benchmark_warmed(benchmark, tmdb_pd_data):
    prof = Profiler(benchmark)
    results = prof.run(SearchArray.index, tmdb_pd_data['overview'], autowarm=True)
    assert len(results) == len(tmdb_pd_data)


@pytest.mark.skipif(not profile_enabled, reason="Profiling disabled")
def test_index_benchmark_1k_random(benchmark, tmdb_pd_data):
    prof = Profiler(benchmark)
    thousand_random = np.random.choice(tmdb_pd_data['overview'], size=1000)
    results = prof.run(SearchArray.index, thousand_random, autowarm=False)
    assert len(results) == 1000


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
    results = prof.run(tmdb_data['overview_tokens'].array.termfreqs, term)
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
    idx_again = SearchArray.index(tmdb_data['overview'])
    compare_amount = 10000
    results = prof.run(tmdb_data['overview_tokens'][:compare_amount].array.__eq__, idx_again[:compare_amount])
    assert np.sum(results) == compare_amount

    # eq = benchmark(tmdb_data['overview_tokens'].array.__eq__, idx_again)
