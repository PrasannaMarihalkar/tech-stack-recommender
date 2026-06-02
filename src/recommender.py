import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


class TechStackRecommender:
    """
    Content-based filtering recommender using TF-IDF vectors and cosine similarity.
    Implements the 4-step pipeline: Ingestion → Scoring → Sorting → Filtering.
    """

    def __init__(self, top_n: int = 3):
        self.top_n = top_n
        self.df = None
        self.tfidf_matrix = None
        self.is_fitted = False

        # token_pattern extended to capture C++, C#, .NET style tokens
        # ngram_range=(1,2) lets it recognize "machine learning" as one concept
        # sublinear_tf applies log(1+tf) — dampens the effect of repeated terms
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"[a-zA-Z0-9][a-zA-Z0-9\+\#\.]*",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            norm="l2",
        )

    def fit(self, df: pd.DataFrame) -> "TechStackRecommender":
        """
        Step 1 (Ingestion) + Build phase:
        Fit the TF-IDF vectorizer on the full corpus of job role skill strings.
        This learns the vocabulary and computes IDF weights across all documents.
        """
        self.df = df.reset_index(drop=True)
        corpus = self.df["skills_cleaned"].tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.is_fitted = True
        return self

    def recommend(self, user_skills_str: str) -> List[Tuple[str, float, str]]:
        """
        Steps 2, 3, 4 — Scoring, Sorting, Filtering.

        user_skills_str: cleaned, space-joined string of user skills
        Returns: list of (job_role, similarity_score, description) tuples
        """
        if not self.is_fitted:
            raise RuntimeError("Recommender must be fitted before calling recommend().")

        if not user_skills_str.strip():
            raise ValueError("User skill profile is empty — cannot compute similarity.")

        # Transform user input using the SAME vectorizer (same vocabulary, same IDF weights)
        # This is critical — the vocabulary must be shared (see slide 9 of the PDF)
        user_vector = self.vectorizer.transform([user_skills_str])

        # Compute cosine similarity: dot(user, item) / (|user| * |item|)
        # Returns a (1, n_items) array; flatten to 1D
        raw_scores = cosine_similarity(user_vector, self.tfidf_matrix).flatten()

        # Sort descending — highest similarity first
        sorted_indices = np.argsort(raw_scores)[::-1]

        # Filter to top_n (prevents choice overload — slide 19)
        top_indices = sorted_indices[: self.top_n]

        results = []
        for idx in top_indices:
            role = self.df.iloc[idx]["job_role"]
            score = float(raw_scores[idx])
            description = self.df.iloc[idx].get("description", "No description available.")
            results.append((role, round(score, 4), description))

        return results

    def explain_match(self, user_skills_list: List[str], job_role: str) -> List[str]:
        """
        Bonus feature: shows which specific skills drove the recommendation.
        Makes the system transparent — production systems always need explainability.
        """
        if not self.is_fitted:
            return []

        role_row = self.df[self.df["job_role"].str.lower() == job_role.lower()]
        if role_row.empty:
            return []

        role_skill_tokens = set(role_row.iloc[0]["skills_cleaned"].lower().split())
        user_tokens = set(" ".join(user_skills_list).lower().split())
        matched = sorted(role_skill_tokens & user_tokens)
        return matched

    def get_stats(self) -> dict:
        """Returns metadata about the fitted model — useful for the README/presentation."""
        if not self.is_fitted:
            return {}
        return {
            "total_job_roles": len(self.df),
            "vocabulary_size": len(self.vectorizer.vocabulary_),
            "tfidf_matrix_shape": self.tfidf_matrix.shape,
            "top_n": self.top_n,
        }