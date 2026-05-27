import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_distances

class LabelAssigner:
    _instance = None
    _labels = [
        "monetary policy and interest rates",
        "inflation and consumer prices",
        "labor market and employment",
        "financial markets and banking",
        "fiscal policy and government spending",
        "international trade and global economy",
        "economic development and inequality",
        "climate change and sustainable finance",
        "industrial and sectoral economics",
        "behavioral economics and expectations",
        "policy communication and forward guidance"
    ]

    _model_name = 'all-MiniLM-L6-v2'

    def __init__(self):
        self.model = SentenceTransformer(self._model_name)
        self.labels = self._labels
        self.label_embs = self.model.encode(self.labels, convert_to_numpy=True)
        self.threshold = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _clean_word(self, word):
        word = word.replace('_', ' ')
        return re.sub(r"[\[\]'\"]", '', word)
    
    def optimal_threshold(self, embs, thresholds=np.arange(0.2, 0.60, 0.05)):
        sims = util.cos_sim(self.label_embs, embs).numpy()
        best_t, best_score, best_db_norm, best_coverage = None, -1, None, None

        for t in thresholds:
            labels = []
            for i in range(sims.shape[1]):
                label_idx = np.argmax(sims[:, i]) if np.max(sims[:, i]) > t else -1
                labels.append(label_idx)

            labels = np.array(labels)
            valid_idx = labels != -1

            n_valid    = int(valid_idx.sum())
            n_clusters = len(np.unique(labels[valid_idx]))
            if n_clusters < 2 or n_clusters >= n_valid:
                continue

            e = embs[valid_idx]
            lv = labels[valid_idx]
            unique_labels = np.unique(lv)
            centroids = np.stack([e[lv == k].mean(axis=0) for k in unique_labels])
            dists_to_c = cosine_distances(e, centroids)
            s = np.array([dists_to_c[lv == k, i].mean() for i, k in enumerate(unique_labels)])
            c_dists = cosine_distances(centroids)
            np.fill_diagonal(c_dists, np.inf)
            R = ((s[:, None] + s[None, :]) / c_dists).max(axis=1)
            db = R.mean()
            db_norm = 1 / (1 + db)
            assigned_labels = set(labels[valid_idx])
            coverage = len(assigned_labels) / len(self.labels)

            combined_score = (db_norm + coverage) / 2

            if combined_score > best_score:
                best_t, best_score = t, combined_score
                best_db_norm, best_coverage = db_norm, coverage
        return best_t, best_score, best_db_norm, best_coverage


    def _auto_threshold(self, word_embs):
        sims = util.cos_sim(self.label_embs, word_embs).numpy()
        avg_per_word = sims.mean(axis=0)  # average similarity to all topics
        best_per_word = sims.max(axis=0)  # similarity to best topic

        # Compute distances between each word’s average and best similarity
        dist = np.abs(best_per_word - avg_per_word)

        # Optimal cutoff: halfway between global mean and 1 std above mean
        mean_d, std_d = dist.mean(), dist.std()
        threshold = mean_d + 0.5 * std_d

        # clip to [0.1, 0.7] to avoid degenerate thresholds
        threshold = float(np.clip(threshold, 0.1, 0.7))
        return threshold

    def assign_labels(self, df, auto_threshold=True):
        df['word_clean'] = df['word'].astype(str).apply(self._clean_word)
        unlabelled_topics = df['word_clean'].dropna().unique()
        unlabelled_topics_emb = self.model.encode(unlabelled_topics, convert_to_numpy=True)

        self.threshold, _, self.best_db_norm, self.best_coverage = self.optimal_threshold(unlabelled_topics_emb)

        sims = util.cos_sim(self.label_embs, unlabelled_topics_emb).numpy()
        word_to_label = {}
        for i, topic in enumerate(unlabelled_topics):
            sim_vec = sims[:, i]
            assigned = [self.labels[j] for j, s in enumerate(sim_vec) if s > self.threshold]
            word_to_label[topic] = assigned[0] if assigned else None

        df['label'] = df['word_clean'].map(word_to_label)
        df.drop(columns=['word_clean'], inplace=True)
        return df
