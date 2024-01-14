import sys


class TermMissingError(KeyError):

    def __init__(self, msg):
        super().__init__(msg)


class TermDict:

    def __init__(self):
        self.term_to_ids = {}
        self.id_to_terms = {}

    def add_term(self, term):
        if term in self.term_to_ids:
            return self.term_to_ids[term]
        term_id = len(self.term_to_ids)
        self.term_to_ids[term] = term_id
        self.id_to_terms[term_id] = term
        return term_id

    def copy(self):
        new_dict = TermDict()
        new_dict.term_to_ids = dict(self.term_to_ids)
        new_dict.id_to_terms = dict(self.id_to_terms.copy())
        return new_dict

    def get_term_id(self, term):
        try:
            return self.term_to_ids[term]
        except KeyError:
            raise TermMissingError(f"Term {term} not present in dictionary. Reindex to add.")

    def get_term(self, term_id):
        try:
            return self.id_to_terms[term_id]
        except KeyError:
            raise TermMissingError(f"Term at {term_id} not present in dictionary. Reindex to add.")

    def compatible(self, other) -> bool:
        # Intersect the terms in both dictionaries
        terms_self = list(self.term_to_ids.keys())
        terms_other = list(other.term_to_ids.keys())
        shortest = min(len(terms_self), len(terms_other))
        return terms_self[:shortest] == terms_other[:shortest]
        # If the intersection is empty, the dictionaries are not compatible

    def __len__(self):
        return len(self.term_to_ids)

    def __repr__(self):
        return repr(self.term_to_ids)

    @property
    def nbytes(self):
        bytes_used = sys.getsizeof(self.term_to_ids) + sys.getsizeof(self.id_to_terms)
        return bytes_used
