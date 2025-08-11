from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import pandas as pd
from scipy.sparse import coo_matrix
import numpy as np

from develop.core.metrics.metric_type import MetricType


class BaseMetric(ABC):
    @property
    @abstractmethod
    def metric_type(self) -> MetricType:
        pass
    
    @abstractmethod
    def pairwise(self, batch_emb: np.ndarray, full_emb: np.ndarray) -> np.ndarray:
        pass
    
    @staticmethod
    def compute_sparse_matrix(embeddings: np.ndarray, 
                              metric: 'BaseMetric',
                              threshold: Optional[float] = None,
                              top_k: Optional[int] = None,
                              batch_size: int = 1000) -> coo_matrix:
        n = embeddings.shape[0]
        rows, cols, data = [], [], []

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = embeddings[start:end]
            dist_mat = metric.pairwise(batch, embeddings)

            for i, row in enumerate(dist_mat):
                row_idx = start + i

                if threshold is not None:
                    mask = row < threshold
                    inds = np.where(mask)[0]
                elif top_k is not None:
                    if top_k >= len(row):
                        inds = np.arange(len(row))
                    else:
                        inds = np.argpartition(row, top_k)[:top_k]
                else:
                    inds = np.arange(len(row))

                vals = row[inds]

                rows.extend([row_idx] * len(inds))
                cols.extend(inds)
                data.extend(vals)

        return coo_matrix((data, (rows, cols)), shape=(n, n))

    def extract_topn_from_sparse(self, 
                                 sparse_matrix: coo_matrix, 
                                 words: List[str], 
                                 top_n: int = 5) -> List[List[Tuple[str, float]]]:
        """
        Extract top-n highest scoring words per row from a sparse similarity/distance matrix.

        Args:
            sparse_matrix (coo_matrix): Sparse matrix representing pairwise scores (e.g., similarities or distances).
            words (List[str]): List of words corresponding to matrix rows/columns.
            top_n (int, optional): Number of top elements to extract per row. Defaults to 5.

        Returns:
            List[List[Tuple[str, float]]]: For each row, a list of tuples (word, score) sorted descending by score.
            If a row has no entries, returns an empty list for that row.
        """
        csr = sparse_matrix.tocsr()
        results = []
        for i in range(csr.shape[0]):
            row_start = csr.indptr[i]
            row_end = csr.indptr[i+1]
            indices = csr.indices[row_start:row_end]
            data = csr.data[row_start:row_end]
            
            mask = indices != i # remove self word
            indices = indices[mask]
            data = data[mask]

            if len(data) == 0:
                results.append([])
                continue

            top_n_idx = np.argsort(data)[:top_n]
            top_words_scores = [(words[indices[idx]], round(data[idx], 4)) for idx in top_n_idx]
            results.append(top_words_scores)
        return results

    def create_df_words_embeddings(
        self,
        model,
        year: int,
        region: str,
        compass_vocabulary: List[str],
        top_k: int = 100,
        top_n: int = 10,
        batch_size: int = 1000,
    ) -> pd.DataFrame:
        emb = model.wv.vectors
        w = model.wv.index_to_key

        word_to_idx = {word: i for i, word in enumerate(w)}
        filtered_words = [word for word in compass_vocabulary if word in word_to_idx]
        indices = [word_to_idx[word] for word in filtered_words]

        filtered_emb = emb[indices]
        counts = [model.wv.expandos['count'][i] for i in indices]

        sparse_sim = self.compute_sparse_matrix(
            filtered_emb, self, top_k=top_k, batch_size=batch_size
        )

        top_similar = self.extract_topn_from_sparse(sparse_sim, filtered_words, top_n=top_n)

        df = pd.DataFrame(
            {
                "count": counts,
                "year": year,
                f"most_similar_{self.metric_type.value}": top_similar,
                "embedding": list(filtered_emb),
                'region': region
            },
            index=filtered_words,
        )
        return df.reindex(compass_vocabulary)
    
    def compute_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return self.pairwise(a[None, :], b[None, :])[0, 0]