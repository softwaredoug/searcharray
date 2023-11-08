

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
        return self.term_to_ids[term]

    def get_term(self, term_id):
        return self.id_to_terms[term_id]

    def __len__(self):
        return len(self.term_to_ids)
