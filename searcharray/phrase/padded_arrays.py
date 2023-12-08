"""Query-doc term positions implemented as a padded array."""
from collections import defaultdict
from searcharray.utils.row_viewable_matrix import RowViewableMatrix
from scipy.sparse import dok_matrix
from time import perf_counter
import logging
import numpy as np

logger = logging.getLogger(__name__)

# When running in pytest
import sys  # noqa
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _pad_to_len(arr, fill, every=10):
    """Pad an array to a multiple of every."""
    if len(arr) % every != 0:
        arr = np.pad(arr, (0, every - len(arr) % every),
                     'constant', constant_values=fill)
    return arr


def mat_index_from_width(width, mats):
    """Return the index of the matrix to use for a given width."""
    if width < len(mats) - 1:
        return width // 10
    else:
        return len(mats) - 1


class PaddedPosnArraysBuilder():
    def __init__(self, pad=-100):
        # Matrix per 10 positions
        self.posns_mats = [defaultdict(int), defaultdict(int),
                           defaultdict(int), defaultdict(int),
                           defaultdict(int)]

        self.posns_lookup = [np.array([]), ]  # 0th is invalid
        self.pad = pad

    def add_posns(self, doc_id, term_id, positions):
        idx = len(self.posns_lookup)
        positions = np.array(positions)

        # PAD TO LENGTH % 10
        positions = _pad_to_len(positions, self.pad)

        self.posns_lookup.append(positions)
        mat_to_use = mat_index_from_width(len(positions), self.posns_mats)
        self.posns_mats[mat_to_use][doc_id, term_id] += idx

    def _build_mat(self, mat_dict, num_postings, num_terms):
        posns_dok = dok_matrix((num_postings, num_terms), dtype=np.uint32)
        dict.update(posns_dok, mat_dict)
        posns_csr = posns_dok.tocsr()
        posns_mat = RowViewableMatrix(posns_csr)
        return posns_mat

    def build(self, num_postings, num_terms):
        mats = [self._build_mat(mat_dict, num_postings, num_terms)
                for mat_dict in self.posns_mats]

        padded_posn_arrays = PaddedPosnArrays()
        padded_posn_arrays.posns_mats = mats
        padded_posn_arrays.posns_lookup = self.posns_lookup
        return padded_posn_arrays


class PaddedPosnArrays():

    def __init__(self, pad=-100):
        self.posns_mats = None
        self.posns_lookup = [np.array([])]  # 0th is invalid
        self.pad = pad

    def ensure_capacity(self, num_postings, num_terms):
        for mat in self.posns_mats:
            mat.resize((num_postings, num_terms))
        max_lookup = max(mat.mat.max() for mat in self.posns_mats)
        if max_lookup > len(self.posns_lookup):
            self.posns_lookup = np.resize(self.posns_lookup, max_lookup + 1)

    @property
    def nbytes(self):
        posns_lookup_bytes = sum(x.nbytes for x in self.posns_lookup)
        posns_mats_bytes = sum(x.nbytes for x in self.posns_mats)
        return posns_mats_bytes + posns_lookup_bytes

    def insert(self, key, term_ids_to_posns):
        for update_doc_idx, new_posns_row in enumerate(term_ids_to_posns):
            for term_id, positions in new_posns_row:
                mat_idx = mat_index_from_width(len(positions), self.posns_mats)
                update_docs = self.posns_mats[mat_idx][key]
                lookup_location = update_docs[update_doc_idx][0, term_id]
                self.posns_lookup[lookup_location] = positions

    def positions(self, term_id, key=None, padded=False, width_hint=None):

        if width_hint is not None:
            mat_to_use = mat_index_from_width(width_hint, self.posns_mats)
            posns_to_lookup = self.posns_mats[mat_to_use].copy_col_at(term_id)
        else:
            for mat in self.posns_mats:
                if key is not None:
                    posns_to_lookup = mat.copy_col_at(term_id)[key]
                else:
                    posns_to_lookup = self.posns_mat.copy_col_at(term_id)
                if posns_to_lookup.nnz > 0:
                    break

        # This could be faster if posns_lookup was more row slicable
        posns_to_lookup = posns_to_lookup.toarray().flatten()
        if padded:
            posns = [self.posns_lookup[lookup] for lookup in posns_to_lookup]
        else:
            posns = [self.posns_lookup[lookup][self.posns_lookup[lookup] != self.pad] for lookup in posns_to_lookup]
        return posns

    def copy(self):
        new = PaddedPosnArrays()
        new.posns_mats = [mat.copy() for mat in self.posns_mats]
        new.posns_lookup = self.posns_lookup.copy()
        return new

    def slice(self, key):
        new = PaddedPosnArrays()
        new.posns_mats = [mat.slice(key) for mat in self.posns_mats]
        new.posns_lookup = self.posns_lookup
        return new

    def __getitem__(self, key):
        return self.posns_mat[key]

    def _compute_phrase_freqs(term_posns, slop=1):
        start = perf_counter()

        prior_term = term_posns[0]
        logger.info(f"Preamble took {perf_counter() - start:.4f} seconds")
        for term in term_posns[1:]:
            is_same_term = (term.shape == prior_term.shape) and np.all(term[0] == prior_term[0])

            # Compute positional differences
            #
            # Each row of posn_diffs is a term posn diff matrix
            # Where columns are prior_term posns, rows are term posns
            # This shows every possible term diff
            #
            # Example:
            #   prior_term = array([[0, 4],[0, 4])
            #         term = array([[1, 2, 3],[1, 2, 3]])
            #
            #
            #   posn_diffs =
            #
            #     array([[ term[0] - prior_term[0], term[0] - prior_term[1] ],
            #            [ term[1] - prior_term[0], ...
            #            [ term[2] - prior_term[0], ...
            #
            #    or in our example
            #
            #     array([[ 1, -3],
            #            [ 2, -2],
            #            [ 3, -1]])
            #
            #  We care about all locations where posn == slop (or perhaps <= slop)
            #  that is term is slop away from prior_term. Usually slop == 1 (ie 1 posn away)
            #  for normal phrase matching
            #
            posn_diffs = term[:, :, np.newaxis] - prior_term[:, np.newaxis, :]
            logger.info(f"posn_diffs took {perf_counter() - start:.4f} seconds")

            # For > 2 terms, we need to connect a third term by making prior_term = term
            # and repeating
            #
            # BUT
            # we only want those parts of term that are adjacent to prior_term
            # before continuing, so we don't accidentally get a partial phrase
            # so we need to make sure to
            # Pad out any rows in 'term' where posn diff != slop
            # so they're not considered on subsequent iterations
            term_mask = np.any(posn_diffs == 1, axis=2)
            term[~term_mask] = -100
            logger.info(f"term mask took {perf_counter() - start:.4f} seconds")

            # Count how many times the row term is 1 away from the col term
            per_doc_diffs = np.sum(posn_diffs == slop, axis=1, dtype=np.int8)
            logger.info(f"count slops took {perf_counter() - start:.4f} seconds")

            # Doc-wise sum to get a 'term freq' for the prior_term - term bigram
            bigram_freqs = np.sum(per_doc_diffs == slop, axis=1)
            if is_same_term:
                satisfies_slop = per_doc_diffs == slop
                consecutive_ones = satisfies_slop[:, 1:] & satisfies_slop[:, :-1]
                consecutive_ones = np.sum(consecutive_ones, axis=1)
                # ceiling divide?
                # Really these show up as
                # 1 1 1 0 1
                # we need to treat the 2nd consecutive 1 as 'not a match'
                # and also update 'term' to not include it
                bigram_freqs -= -np.floor_divide(consecutive_ones, -2)
                logger.info(f"same terms took {perf_counter() - start:.4f} seconds")

            # Last loop, bigram_freqs is the full phrase term freq

            # Update mask to eliminate any non-matches
            phrase_freqs[to_compute] = bigram_freqs

            # Should only keep positions of 'prior term' that are adjacent to the
            # one prior to it...
            prior_term = term
            logger.info(f"Loop took {perf_counter() - start:.4f} seconds")

        return phrase_freqs


