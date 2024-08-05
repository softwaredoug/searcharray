import numpy as np
from numpy.typing import NDArray
import math
import os
import sys
from typing import Iterable, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from searcharray.phrase.middle_out import MAX_POSN, PosnBitArrayFromFlatBuilder, PosnBitArrayBuilder, PosnBitArrayAlreadyEncBuilder
from searcharray.term_dict import TermDict
from searcharray.utils.mat_set import SparseMatSetBuilder
from searcharray.utils.row_viewable_matrix import RowViewableMatrix

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)


# Set to stdout for debugging
# logging.basicConfig(level=logging.DEBUG)


def home_directory():
    return os.path.expanduser("~")


def searcharray_home():
    searcharray_dir = os.path.join(home_directory(), ".searcharray")
    # Ensure exists
    os.makedirs(searcharray_dir, exist_ok=True)
    return searcharray_dir


def _compute_doc_lens(posns: np.ndarray, doc_ids: np.ndarray, num_docs: int) -> NDArray[np.float32]:
    """Given an array of positions, compute the length of each document."""
    doc_lens = np.zeros(num_docs, dtype=np.float32)

    # Find were we ave posns for each doc
    non_empty_doc_lens = -np.diff(posns).astype(np.float32) + 1

    non_empty_idxs = np.argwhere(non_empty_doc_lens > 0).flatten()
    non_empty_doc_ids = doc_ids[non_empty_idxs]
    non_empty_doc_lens = non_empty_doc_lens[non_empty_idxs]
    doc_lens[non_empty_doc_ids] = non_empty_doc_lens
    if len(doc_ids) > 0 and doc_ids[-1] not in non_empty_doc_ids:
        doc_lens[doc_ids[-1]] = np.float32(posns[-1] + 1)
    return doc_lens


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def _gather_tokens(array, tokenizer,
                   term_dict, term_doc, start_doc_id=0, trunc_posn=None):
    all_terms = []
    all_docs = []
    all_posns = []

    logger.info(f"Tokenizing {len(array)} documents")

    for idx, doc in enumerate(array):
        doc_id = start_doc_id + idx

        # Speedup - use predefined vocabulary
        terms = np.asarray([term_dict.add_term(token)
                            for token in tokenizer(doc)], dtype=np.uint32)[:trunc_posn]
        doc_ids = np.full(len(terms), doc_id, dtype=np.uint32)[:trunc_posn]
        all_terms.append(terms)
        all_docs.append(doc_ids)

        all_posns.append(np.arange(len(terms)))

        term_doc.append(np.unique(terms))

        if idx % 10000 == 0 and idx > 0:
            logger.info(f"Tokenized {doc_id} ({100.0 * (idx / len(array))}%)")

    # Flatten each
    all_terms = np.concatenate(all_terms)
    all_docs = np.concatenate(all_docs)
    all_posns = np.concatenate(all_posns)

    logger.info("Tokenization -- vstacking")
    terms_w_posns = np.vstack([all_terms, all_docs, all_posns])
    # del all_terms, all_docs, all_posns
    # gc.collect()
    logger.info("Tokenization -- DONE")
    return terms_w_posns, term_doc


def _lex_sort(terms_w_posns):
    """Sort terms, then doc_id, then posn."""
    # Because docs / posns already sorted, we can just sort on terms
    # Equivelant to np.lexsort(terms_w_posns[[::-1], :])
    return np.argsort(terms_w_posns[0, :], kind='stable')


def _invert_docs_terms(terms_w_posns):
    """Sort terms, then doc_id, then posn."""
    # An in place sort could be faster if we could figure this out,
    # an np.sort(arr) is about 1/3 the time of np.argsort(arr)
    # possibly more with just arr.sort()
    lexsort = _lex_sort(terms_w_posns)
    return terms_w_posns[:, lexsort]


def _tokenize_batch(array, tokenizer, term_dict, batch_size, batch_beg, truncate=False):
    logger.info(f"{batch_beg} Batch Start tokenization")
    trunc_posn = None
    if truncate:
        trunc_posn = MAX_POSN

    term_doc = SparseMatSetBuilder()
    terms_w_posns, term_doc = _gather_tokens(array, tokenizer,
                                             term_dict, term_doc, start_doc_id=batch_beg,
                                             trunc_posn=trunc_posn)

    # Use posns to compute doc lens
    doc_lens = _compute_doc_lens(posns=terms_w_posns[2, :],
                                 doc_ids=(terms_w_posns[1, :] - batch_beg),
                                 num_docs=len(array))
    logger.info("Inverting docs->terms")
    terms_w_posns = _invert_docs_terms(terms_w_posns)

    # Encode posns to bit array
    logger.info("Encoding positions to bit array")
    posns = PosnBitArrayFromFlatBuilder(terms_w_posns, max_doc_id=(batch_beg + len(array) - 1))
    bit_posns = posns.build()

    if np.any(doc_lens > MAX_POSN):
        raise ValueError(f"Document length exceeds maximum of {MAX_POSN}")

    logger.info("Batch tokenization complete")
    return batch_beg, term_doc, bit_posns, doc_lens


def batch_iterator(iterator, batch_size):
    """

    :param iterator: The iterator to slice into batches.
    :param batch_size: The size of each batch.
    """
    batch_beg = 0
    it = iter(iterator)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch_beg, batch
        batch_beg += batch_size


def _process_batches(term_doc, batch_size,
                     futures,
                     last_batch_beg_processed,
                     bit_posns=None,
                     doc_lens=None,
                     truncate=False):
    batch_results = [None] * len(futures)
    batch_beg = 0
    for future in as_completed(futures):
        try:
            batch_beg, batch_term_doc, batch_bit_posns, batch_doc_lens = future.result()
            idx = (batch_beg - last_batch_beg_processed) // batch_size
            assert batch_results[idx] is None
            batch_results[idx] \
                = (batch_beg, batch_term_doc, batch_bit_posns, batch_doc_lens)
        except ValueError as e:
            logger.error(e)
            logger.error(f"Batch {batch_beg} failed to tokenize")
            raise e

    logger.info(f"(main thread) Processing {len(batch_results)} batch results")
    for result in batch_results:
        assert result is not None
        batch_beg, batch_term_doc, batch_bit_posns, batch_doc_lens = result
        term_doc.concat(batch_term_doc)
        if bit_posns is None:
            bit_posns = batch_bit_posns
        else:
            logger.info("(main thread) Concatenating bit positions")
            bit_posns.concat(batch_bit_posns)
            logger.info("(main thread) Concatenated bit positions... Done")

        doc_lens.append(batch_doc_lens)
    return bit_posns


def build_index_no_workers(array: Iterable, tokenizer, batch_size=10000,
                           data_dir: Optional[str] = None,
                           cache_gt_than=25,
                           truncate=False):
    term_dict = TermDict()
    term_doc = SparseMatSetBuilder()
    doc_lens: List[np.ndarray] = []
    bit_posns = None

    logger.info("Indexing begins w/ NO workers")
    for batch_beg, batch in batch_iterator(array, batch_size):
        batch_beg, batch_term_doc, batch_bit_posns, batch_doc_lens = _tokenize_batch(batch, tokenizer, term_dict, batch_size, batch_beg, truncate=truncate)
        term_doc.concat(batch_term_doc)
        if bit_posns is None:
            bit_posns = batch_bit_posns
        else:
            bit_posns.concat(batch_bit_posns)
        doc_lens.append(batch_doc_lens)

    doc_lens = np.concatenate(doc_lens)

    avg_doc_length = np.mean(doc_lens)

    term_doc_built = RowViewableMatrix(term_doc.build())
    logger.info("Indexing from tokenization complete")
    assert bit_posns is not None
    # if data_dir is None:
    #     data_dir = searcharray_home()
    if data_dir is not None:
        logger.info(f"Memmapping bit positions to {data_dir}")
        bit_posns.memmap(data_dir)
    bit_posns.cache_gt_than = cache_gt_than
    return term_doc_built, bit_posns, term_dict, avg_doc_length, np.array(doc_lens)


def build_index_from_tokenizer(array: Iterable, tokenizer, batch_size=10000,
                               data_dir: Optional[str] = None,
                               truncate=False, workers=4,
                               cache_gt_than=25):
    """Build index directly from tokenizing docs (array of string)."""
    if workers == 1:
        return build_index_no_workers(array, tokenizer, batch_size=batch_size,
                                      cache_gt_than=cache_gt_than,
                                      data_dir=data_dir, truncate=truncate)
    term_dict = TermDict()
    term_doc = SparseMatSetBuilder()
    doc_lens: List[np.ndarray] = []
    bit_posns = None

    logger.info(f"Indexing begins w/ {workers} workers")
    futures = []
    last_batch_beg_processed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:

        for batch_beg, batch in batch_iterator(array, batch_size):
            futures.append(executor.submit(_tokenize_batch,
                                           batch,
                                           tokenizer,
                                           term_dict,
                                           batch_size,
                                           batch_beg,
                                           truncate=truncate))

            if len(futures) >= workers:
                logger.info(f"Collecting {len(futures)} futures")
                bit_posns = _process_batches(term_doc, batch_size,
                                             futures,
                                             last_batch_beg_processed,
                                             bit_posns, doc_lens, truncate=truncate)
                logger.info(f"{batch_beg} Batch Complete")
                logger.info(f"Roaringish NBytes -- {convert_size(bit_posns.nbytes)}")
                logger.info(f"Term Dict Size -- {len(term_dict)}")

                last_batch_beg_processed += len(futures) * batch_size
                futures = []
        if len(futures) > 0:
            bit_posns = _process_batches(term_doc, batch_size,
                                         futures,
                                         last_batch_beg_processed,
                                         bit_posns, doc_lens, truncate=truncate)

    doc_lens = np.concatenate(doc_lens)

    avg_doc_length = np.mean(doc_lens)

    term_doc_built = RowViewableMatrix(term_doc.build())
    logger.info("Indexing from tokenization complete")
    assert bit_posns is not None
    # if data_dir is None:
    #     data_dir = searcharray_home()
    if data_dir is not None:
        logger.info(f"Memmapping bit positions to {data_dir}")
        bit_posns.memmap(data_dir)
    bit_posns.cache_gt_than = cache_gt_than
    return term_doc_built, bit_posns, term_dict, avg_doc_length, np.array(doc_lens, dtype=np.float32)


def build_index_from_terms_list(postings, Terms):
    """Bulid an index from postings that are already tokenized and point at their term frequencies/posns."""
    term_dict = TermDict()
    term_doc = SparseMatSetBuilder()
    doc_lens = []
    avg_doc_length = 0
    num_postings = 0
    posns = PosnBitArrayBuilder()
    posns_enc = PosnBitArrayAlreadyEncBuilder()

    # COPY 1
    # Consume generator (tokenized postings) into list
    # its faster this way?
    postings = list(postings)

    # COPY 2
    for doc_id, tokenized in enumerate(postings):
        if isinstance(tokenized, dict):
            tokenized = Terms(tokenized, doc_len=len(tokenized))
        elif not isinstance(tokenized, Terms):
            raise TypeError("Expected a Terms or a dict")

        if tokenized.encoded:
            posns = posns_enc

        doc_lens.append(tokenized.doc_len)
        avg_doc_length += doc_lens[-1]
        terms = []
        for token, term_freq in tokenized.terms():
            term_id = term_dict.add_term(token)
            terms.append(term_id)
            positions = tokenized.positions(token)
            if positions is not None:
                posns.add_posns(doc_id, term_id, positions)

        term_doc.append(terms)

        posns.ensure_capacity(doc_id)
        num_postings += 1

    if num_postings > 0:
        avg_doc_length /= num_postings

    bit_posns = posns.build()
    return RowViewableMatrix(term_doc.build()), bit_posns, term_dict, avg_doc_length, np.array(doc_lens)
