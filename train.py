import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matcher
from matcher import FEATURES, ResumeMatcher, MODEL_PATH


def main():
    df = pd.read_csv("data/pairs.csv")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=0, stratify=df.match)

    # fit TF-IDF only on training text to avoid leakage
    corpus = list(train_df.resume) + list(train_df.jd)
    matcher.MODEL_PATH = MODEL_PATH.with_name("_none.joblib")  # ensure no old model is loaded
    m = ResumeMatcher(corpus)

    def featurize(d):
        return np.array([[m.features(r, j)[k] for k in FEATURES] for r, j in zip(d.resume, d.jd)])

    Xtr, Xte = featurize(train_df), featurize(test_df)
    ytr, yte = train_df.match.values, test_df.match.values

    candidates = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "gboost": GradientBoostingClassifier(random_state=0),
    }
    best, best_auc = None, 0
    for name, clf in candidates.items():
        auc = cross_val_score(clf, Xtr, ytr, cv=5, scoring="roc_auc").mean()
        print(f"{name}: CV AUC = {auc:.3f}")
        if auc > best_auc:
            best, best_auc, best_name = clf, auc, name

    best.fit(Xtr, ytr)
    proba = best.predict_proba(Xte)[:, 1]
    print(f"\nBest: {best_name} | Test AUC = {roc_auc_score(yte, proba):.3f}")
    print(classification_report(yte, proba > 0.5))
    joblib.dump(best, MODEL_PATH)
    joblib.dump(corpus, MODEL_PATH.with_name("corpus.joblib"))
    print("Saved", MODEL_PATH)


if __name__ == "__main__":
    main()
