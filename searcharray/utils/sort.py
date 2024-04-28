"""Utility to sort a dataframe by a score."""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
from searcharray.postings import SearchArray


class SetOfResults:
    """Gather multiple sets of search results, ie for testing."""

    def __init__(self, df: pd.DataFrame, searchable=False):
        self.df = df
        self.metadata: Dict[str, List[Any]] = defaultdict(list)
        self.indices: List[int] = []
        self.searchable_cols = []
        for col in df.columns:
            if not searchable and isinstance(df[col].array, SearchArray):
                self.searchable_cols.append(col)

    def ins_top_n(self, scores, N=10, query: str = '',
                  metadata: Optional[Dict[str, List[Any]]] = None):
        """Insert the top N rows into the set of results."""
        top_n = np.argpartition(scores, -N)[-N:]
        self.indices.extend(top_n)
        self.metadata['score'].extend(scores[top_n])
        self.metadata['query'].extend([query] * len(top_n))
        if metadata is None:
            return
        for key, values in metadata.items():
            if not isinstance(values, list):
                values = [values] * len(top_n)
            self.metadata[key].extend(values)
            if len(self.metadata[key]) != len(self.indices):
                raise ValueError("Metadata must have same length as scores.")

    def get_all(self) -> pd.DataFrame:
        subset = self.df.iloc[self.indices, ~self.df.columns.isin(self.searchable_cols)]
        for key, values in self.metadata.items():
            subset[key] = values
        # Sort by query, then by score
        sorted_subset = subset.sort_values(['query', 'score'], ascending=[True, False])
        # Assign rank within each query
        sorted_subset['rank'] = sorted_subset.groupby('query').cumcount() + 1
        return sorted_subset.reset_index(drop=True)
