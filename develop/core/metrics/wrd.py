import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import ot

from develop.core.metrics.base import BaseMetric


class WRDMetric(BaseMetric):
    def __init__(self, top_k_cosine: int = 100, top_k_euclidean: int = 100):
        self.top_k_cosine = top_k_cosine
        self.top_k_euclidean = top_k_euclidean


    def pairwise(self, batch_emb: np.ndarray, full_emb: np.ndarray) -> np.ndarray:
        top_index_sets = self._filter_candidates(batch_emb, full_emb)

        scores = np.zeros((batch_emb.shape[0],), dtype=np.float32)

        for i, top_ids in enumerate(top_index_sets):
            query = batch_emb[i].reshape(1, -1)
            candidates = full_emb[list(top_ids)]
            scores[i] = self._compute_wrd(query, candidates)

        return scores
    
    def _filter_candidates(self, batch_emb: np.ndarray, full_emb: np.ndarray) -> list[set]:
        cosine_dist = 1 - cosine_similarity(batch_emb, full_emb)
        euclidean_dist = euclidean_distances(batch_emb, full_emb)

        top_indices = []
        for i in range(batch_emb.shape[0]):
            top_cosine = np.argpartition(cosine_dist[i], self.top_k_cosine)[:self.top_k_cosine]
            top_euclidean = np.argpartition(euclidean_dist[i], self.top_k_euclidean)[:self.top_k_euclidean]
            merged = set(top_cosine).union(set(top_euclidean))
            top_indices.append(merged)

        return top_indices

    def _compute_wrd(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        # Normalize each token embedding
        query_norms = np.linalg.norm(query, axis=1, keepdims=True)
        candidate_norms = np.linalg.norm(candidates, axis=1, keepdims=True)

        query_normed = query / np.clip(query_norms, 1e-8, None)
        candidate_normed = candidates / np.clip(candidate_norms, 1e-8, None)

        # Weights: proportional to L2 norm
        query_weights = (query_norms.squeeze() / np.sum(query_norms)).astype(np.float64)
        candidate_weights = (candidate_norms.squeeze() / np.sum(candidate_norms)).astype(np.float64)

        # Distance matrix: cosine distance
        dist_matrix = 1 - np.dot(query_normed, candidate_normed.T)

        # Compute EMD
        wrd_score = ot.emd2(query_weights, candidate_weights, dist_matrix.astype(np.float64))
        return wrd_score
