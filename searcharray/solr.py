"""Utility functions for Solr users of searcharray."""
import re
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from searcharray.postings import SearchArray
from searcharray.similarity import Similarity, default_bm25


def parse_min_should_match(num_clauses: int, spec: str) -> int:
    """Parse Solr's min should match (ie mm) spec.

    See this ChatGPT translation of mm code from Solr's Java code for parsing this
    https://chat.openai.com/share/76642aec-7e05-420f-a53a-83b8e2eea8fb

    Parameters
    ----------
    num_clauses : int
    spec : str

    Returns
    -------
    int : the number of clauses that must match
    """
    def checked_parse_int(value, error_message):
        try:
            return int(value)
        except ValueError:
            raise ValueError(error_message)

    result = num_clauses
    spec = spec.strip()

    if '<' in spec:
        # we have conditional spec(s)
        space_around_less_than_pattern = re.compile(r'\s*<\s*')
        spec = space_around_less_than_pattern.sub('<', spec)
        for s in spec.split():
            parts = s.split('<', 1)
            if len(parts) < 2:
                raise ValueError("Invalid 'mm' spec: '" + s + "'. Expecting values before and after '<'")
            upper_bound = checked_parse_int(parts[0], "Invalid 'mm' spec. Expecting an integer.")
            if num_clauses <= upper_bound:
                return result
            else:
                result = parse_min_should_match(num_clauses, parts[1])
        return result

    # otherwise, simple expression
    if '%' in spec:
        # percentage - assume the % was the last char. If not, let int() fail.
        spec = spec[:-1]
        percent = checked_parse_int(spec, "Invalid 'mm' spec. Expecting an integer.")
        calc = (result * percent) * (1 / 100)
        result = result + int(calc) if calc < 0 else int(calc)
    else:
        calc = checked_parse_int(spec, "Invalid 'mm' spec. Expecting an integer.")
        result = result + calc if calc < 0 else calc

    return min(num_clauses, max(result, 0))


def parse_field_boosts(field_lists: List[str]) -> dict:
    """Parse Solr's qf, pf, pf2, pf3 field boosts."""
    if not field_lists:
        return {}

    out = {}
    carat_pattern = re.compile(r'\^')

    for field in field_lists:
        parts = carat_pattern.split(field)
        out[parts[0]] = None if len(parts) == 1 else float(parts[1])

    return out


def get_field(frame, field) -> SearchArray:
    if field not in frame.columns:
        raise ValueError(f"Field {field} not in dataframe")
    if not isinstance(frame[field].array, SearchArray):
        raise ValueError(f"Field {field} is not a searcharray field")
    return frame[field].array


def parse_query_terms(frame: pd.DataFrame,
                      query: str,
                      query_fields: List[str]):

    search_terms: Dict[str, List[str]] = {}
    num_search_terms = 0
    term_centric = True

    for field in query_fields:
        arr = get_field(frame, field)

        tokenizer = arr.tokenizer
        search_terms[field] = []
        field_num_search_terms = 0
        for posn, term in enumerate(tokenizer(query)):
            search_terms[field].append(term)
            field_num_search_terms += 1
        if num_search_terms == 0:
            num_search_terms = field_num_search_terms
        elif field_num_search_terms != num_search_terms:
            term_centric = False

    return num_search_terms, search_terms, term_centric


def _edismax_term_centric(frame: pd.DataFrame,
                          query_fields: Dict[str, float],
                          num_search_terms: int,
                          search_terms: Dict[str, List[str]],
                          mm: str,
                          similarity: Similarity) -> Tuple[np.ndarray, str]:
    explain = []
    term_scores = []
    for term_posn in range(num_search_terms):
        max_scores = np.zeros(len(frame))
        term_explain = []
        for field, boost in query_fields.items():
            term = search_terms[field][term_posn]
            post_arr = get_field(frame, field)
            field_term_score = post_arr.score(term, similarity=similarity) * (1 if boost is None else boost)
            boost_exp = f"{boost}" if boost is not None else "1"
            term_explain.append(f"{field}:{term}^{boost_exp}")
            max_scores = np.maximum(max_scores, field_term_score)
        term_scores.append(max_scores)
        explain.append("(" + " | ".join(term_explain) + ")")

    min_should_match = parse_min_should_match(num_search_terms, spec=mm)
    qf_scores = np.asarray(term_scores)
    matches_gt_mm = np.sum(qf_scores > 0, axis=0) >= min_should_match
    qf_scores = np.sum(term_scores, axis=0)
    qf_scores[~matches_gt_mm] = 0
    return qf_scores, "(" + " ".join(explain) + f")~{min_should_match}"


def _edismax_field_centric(frame: pd.DataFrame,
                           query_fields: Dict[str, float],
                           num_search_terms: int,
                           search_terms: Dict[str, List[str]],
                           mm: str,
                           similarity: Similarity = default_bm25) -> Tuple[np.ndarray, str]:
    field_scores = []
    explain = []
    for field, boost in query_fields.items():
        post_arr = get_field(frame, field)
        term_scores = np.array([post_arr.score(term, similarity=similarity)
                                for term in search_terms[field]])
        min_should_match = parse_min_should_match(len(search_terms[field]), spec=mm)
        exp = " ".join([f"{field}:{term}" for term in search_terms[field]])
        boost_exp = f"{boost}" if boost is not None else "1"
        exp = "(" + exp + f")~{min(min_should_match, len(search_terms[field]))}"
        exp = "(" + exp + f")^{boost_exp}"

        matches_gt_mm = np.sum(term_scores > 0, axis=0) >= min(min_should_match, len(search_terms[field]))
        sum_terms_bm25 = np.sum(term_scores, axis=0)
        sum_terms_bm25[~matches_gt_mm] = 0
        field_scores.append(sum_terms_bm25 * (1 if boost is None else boost))
        explain.append(exp)
    # Take maximum field scores as qf
    qf_scores = np.asarray(field_scores)
    qf_scores = np.max(qf_scores, axis=0)
    return qf_scores, " | ".join(explain)


def edismax(frame: pd.DataFrame,
            q: str,
            qf: List[str],
            mm: Optional[str] = None,
            pf: Optional[List[str]] = None,
            pf2: Optional[List[str]] = None,
            pf3: Optional[List[str]] = None,
            q_op: str = "OR",
            similarity: Similarity = default_bm25) -> Tuple[np.ndarray, str]:
    """Run edismax search over dataframe with searcharray fields.

    Parameters
    ----------
    q : str
        The query string
    mm : str
        The minimum should match spec
    qf : list
        The fields to search
    pf : list
        The fields to search for phrase matches
    pf2 : list
        The fields to search for bigram matches
    pf3 : list
        The fields to search for trigram matches
    q_op : str, optional
        The default operator, by default "OR"

    Returns
    -------
    np.ndarray
        The search results
    """
    def listify(x):
        return x if isinstance(x, list) else [x]

    query_fields = parse_field_boosts(listify(qf))
    phrase_fields = parse_field_boosts(listify(pf)) if pf else {}
    if mm is None:
        mm = "1"
    if q_op == "AND":
        mm = "100%"

    # bigram_fields = parse_field_boosts(pf2) if pf2 else {}
    # trigram_fields = parse_field_boosts(pf3) if pf3 else {}

    num_search_terms, search_terms, term_centric = parse_query_terms(frame, q, list(query_fields.keys()))
    if term_centric:
        qf_scores, explain = _edismax_term_centric(frame, query_fields,
                                                   num_search_terms, search_terms, mm,
                                                   similarity=similarity)
    else:
        qf_scores, explain = _edismax_field_centric(frame, query_fields,
                                                    num_search_terms, search_terms, mm,
                                                    similarity=similarity)

    phrase_scores = []
    for field, boost in phrase_fields.items():
        arr = get_field(frame, field)
        terms = search_terms[field]
        field_phrase_score = arr.score(terms, similarity=similarity) * (1 if boost is None else boost)
        boost_exp = f"{boost}" if boost is not None else "1"
        explain += f" ({field}:\"{' '.join(terms)}\")^{boost_exp}"
        phrase_scores.append(field_phrase_score)

    if len(phrase_scores) > 0:
        phrase_scores = np.sum(phrase_scores, axis=0)
        # Add where term_scores > 0
        term_match_idx = np.where(qf_scores)[0]

        qf_scores[term_match_idx] += phrase_scores[term_match_idx]
    return qf_scores, explain
