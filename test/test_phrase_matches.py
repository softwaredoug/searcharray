from searcharray.postings import PostingsArray


def test_phrase_match():
    data = PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25)
    matches = data.phrase_match(["foo", "bar"])
    assert (matches == [True, False, False, False] * 25).all()


def test_phrase_match_three_terms():
    data = PostingsArray.index(["foo bar bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25)
    matches = data.phrase_match(["bunny", "funny", "wunny"])
    assert (matches == [False, False, False, True] * 25).all()


def test_phrase_match_three_terms_spread_out_doesnt_match():
    spread_out = PostingsArray.index(["foo bar EEK foo URG bar baz", "data2", "data3 bar", "bunny funny wunny"] * 25)
    matches = spread_out.phrase_match(["foo", "bar", "baz"])
    assert (matches == [False, False, False, False] * 25).all()


def test_phrase_match_same_term_matches():
    spread_out = PostingsArray.index(["foo foo foo", "data2", "data3 bar", "bunny funny wunny"] * 25)
    matches = spread_out.phrase_match(["foo", "foo", "foo"])
    assert (matches == [True, False, False, False] * 25).all()


def test_phrase_match_duplicate_phrases():
    multiple = PostingsArray.index(["foo bar foo bar", "data2", "data3 bar", "bunny funny wunny"] * 25)
    matches = multiple.phrase_match(["foo", "bar"])
    assert (matches == [True, False, False, False] * 25).all()
