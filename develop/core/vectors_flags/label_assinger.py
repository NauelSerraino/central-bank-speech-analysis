import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import silhouette_score

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
        best_t, best_score = None, -1

        print(f"{'Threshold':>10} | {'Silhouette':>10} | {'Coverage':>8} | {'Combined':>10} | {'#Assigned labels':>15}")
        print("-" * 65)

        for t in thresholds:
            labels = []
            for i in range(sims.shape[1]):
                label_idx = np.argmax(sims[:, i]) if np.max(sims[:, i]) > t else -1
                labels.append(label_idx)

            labels = np.array(labels)
            valid_idx = labels != -1

            if len(np.unique(labels[valid_idx])) < 2:
                print(f"{t:10.2f} | {'-':>10} | {'-':>8} | {'-':>10} | {0:15}")
                continue  # silhouette needs at least 2 clusters

            sil_score = silhouette_score(embs[valid_idx], labels[valid_idx], metric="cosine")
            assigned_labels = set(labels[valid_idx])
            coverage = len(assigned_labels) / len(self.labels)

            alpha = 0.40
            combined_score = alpha * sil_score + (1 - alpha) * coverage

            print(f"{t:10.2f} | {sil_score:10.4f} | {coverage:8.4f} | {combined_score:10.4f} | {len(assigned_labels):15}")

            if combined_score > best_score:
                best_t, best_score = t, combined_score

        print(f"\nBest threshold = {best_t:.2f}, best combined score = {best_score:.4f}")
        return best_t, best_score


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

        self.threshold, sil_score = self.optimal_threshold(unlabelled_topics_emb)
        print(f"Auto threshold = {self.threshold:.2f} (silhouette={sil_score:.3f})")

        sims = util.cos_sim(self.label_embs, unlabelled_topics_emb).numpy()
        word_to_label = {}
        for i, topic in enumerate(unlabelled_topics):
            sim_vec = sims[:, i]
            assigned = [self.labels[j] for j, s in enumerate(sim_vec) if s > self.threshold]
            word_to_label[topic] = assigned[0] if assigned else None

        df['label'] = df['word_clean'].map(word_to_label)
        df.drop(columns=['word_clean'], inplace=True)
        return df
