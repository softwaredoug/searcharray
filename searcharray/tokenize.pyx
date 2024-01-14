import numpy as np

from searcharray.term_dict import TermDict
from searcharray.utils.mat_set import SparseMatSetBuilder

def tokenize(array, tokenizer):
    term_dict = TermDict()
    term_doc = SparseMatSetBuilder()

    all_terms = []
    all_docs = []
    all_posns = []

    for doc_id, doc in enumerate(array):
        terms = [term_dict.add_term(token)
                 for token in tokenizer(doc)]
        doc_ids = [doc_id] * len(terms)
        all_terms.extend(terms)
        all_docs.extend(doc_ids)
        all_posns.extend(list(range(len(terms))))

        term_doc.append(set(terms))

    terms_w_posns = np.vstack([all_terms, all_docs, all_posns])
    return terms_w_posns, term_dict, term_doc
