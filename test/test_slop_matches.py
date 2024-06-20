from searcharray.postings import SearchArray
from test_utils import w_scenarios


scenarios = {
    "direct_phrase": {
        "phrase": "intergalactic bounty hunters",
        "doc": """A massive ball of furry creatures from another world eat their way through a small mid-western town followed by intergalactic bounty hunters opposed only by militant townspeople.""",
        "slop": 0
    },
    "slop 1": {
        "phrase": "massive ball furry",
        "doc": """A massive ball of furry creatures from another world eat their way through a small mid-western town followed by intergalactic bounty hunters opposed only by militant townspeople.""",
        "slop": 1
    },
    "two_after_the": {
        "phrase": "the to be",
        "doc": """
            Broke and alone on New Year's Eve, Wilson just wants to spend the rest of a very bad year in bed.
            But, when his best friend convinces him to post a personal ad,
            he meets a woman bent on finding the right guy to be with at midnight.""",
        "slop": 2
    },
    "slop_3_order": {
        "phrase": "the to be",
        "doc": """
            A series of hideous murders is taking place, and Inspector Capell and
            cop-turned-novelist Lonergan are investigating.
            The murders are found to be the work of an out-of-control experiment in genetic engineering.
            The two men must descend into the city's sewer systems to destroy the horrific miscreation.
            It won't be hard to find, as it's already looking for its next victims...""",
        "slop": 3
    },
    "slop_5": {
        "phrase": "spice found substance",
        "doc": """
In the year 10,191, the world is at war for control of the desert planet Dune—the only place where the time-travel substance spice can be found But when one leader gives up control, it's only so he can stage a coup with some unsavory characters.""",
        "slop": 5
    },
}


@w_scenarios(scenarios)
def test_phrase_slop(phrase, doc, slop):
    sa = SearchArray.index([doc, " empty ", doc + " " + doc, " empty"])
    phrase_toks = sa.tokenizer(phrase)
    # assert sa.score(phrase_toks, slop=slop)
    for match_slop in range(slop, max(slop, 10)):
        assert sa.score(phrase_toks, slop=match_slop)[0] > 0
        assert sa.score(phrase_toks, slop=match_slop)[1] == 0
        assert sa.score(phrase_toks, slop=match_slop)[2] > 0
        assert sa.score(phrase_toks, slop=match_slop)[3] == 0
