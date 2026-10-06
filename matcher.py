"""Core matching logic: feature engineering + (optionally trained) scoring model."""
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from skills import extract_skills, extract_years

MODEL_PATH = Path(__file__).parent / "model.joblib"
FEATURES = ["tfidf_sim", "skill_recall", "skill_jaccard", "exp_ratio", "keyword_cov"]


def clean(text: str) -> str:
    text = re.sub(r"http\S+|\S+@\S+", " ", text.lower())
    return re.sub(r"[^a-z0-9+#./\s-]", " ", text)


@dataclass
class MatchResult:
    score: float                 # 0-100
    features: dict
    matched_skills: list
    missing_skills: list
    extra_skills: list


class ResumeMatcher:
    def __init__(self, corpus=None):
        """corpus: optional list of texts to fit TF-IDF on (better IDF weights)."""
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english",
                                   sublinear_tf=True, min_df=1, preprocessor=clean)
        self.corpus = corpus or []
        self.model = None
        self._fitted = False
        if self.corpus:
            self.vec.fit(self.corpus)
            self._fitted = True
        if MODEL_PATH.exists():
            self.model = joblib.load(MODEL_PATH)

    # ---------- features ----------
    def features(self, resume: str, jd: str) -> dict:
        if self._fitted:
            X = self.vec.transform([resume, jd])
        else:  # fall back to fitting on the pair itself
            X = TfidfVectorizer(ngram_range=(1, 2), stop_words="english",
                                sublinear_tf=True, preprocessor=clean).fit_transform([resume, jd])
        tfidf_sim = float(cosine_similarity(X[0], X[1])[0, 0])

        rs, js = extract_skills(resume), extract_skills(jd)
        inter = rs & js
        skill_recall = len(inter) / len(js) if js else 0.0           # JD skills the resume covers
        skill_jaccard = len(inter) / len(rs | js) if (rs | js) else 0.0

        need, have = extract_years(jd), extract_years(resume)
        exp_ratio = min(have / need, 1.5) if need else 1.0

        jd_terms = set(self._top_terms(jd, 25))
        r_tokens = set(clean(resume).split())
        keyword_cov = len(jd_terms & r_tokens) / len(jd_terms) if jd_terms else 0.0

        return dict(tfidf_sim=tfidf_sim, skill_recall=skill_recall, skill_jaccard=skill_jaccard,
                    exp_ratio=exp_ratio, keyword_cov=keyword_cov)

    def _top_terms(self, text, k):
        v = TfidfVectorizer(stop_words="english", preprocessor=clean)
        try:
            m = v.fit_transform([text])
        except ValueError:
            return []
        idx = np.argsort(m.toarray()[0])[::-1][:k]
        names = v.get_feature_names_out()
        return [names[i] for i in idx]

    # ---------- scoring ----------
    def score(self, resume: str, jd: str) -> MatchResult:
        f = self.features(resume, jd)
        x = np.array([[f[n] for n in FEATURES]])
        if self.model is not None:
            s = float(self.model.predict_proba(x)[0, 1])
        else:  # sensible hand-tuned blend when no trained model exists
            s = (0.30 * f["tfidf_sim"] + 0.40 * f["skill_recall"] +
                 0.10 * f["skill_jaccard"] + 0.10 * min(f["exp_ratio"], 1) + 0.10 * f["keyword_cov"])
        rs, js = extract_skills(resume), extract_skills(jd)
        return MatchResult(round(100 * s, 1), f, sorted(rs & js), sorted(js - rs), sorted(rs - js))
