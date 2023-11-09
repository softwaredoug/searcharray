

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

    def __len__(self):
        return len(self.term_to_ids)

    def __repr__(self):
        return repr(self.term_to_ids)
