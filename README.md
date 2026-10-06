# Resume vs Job Description Matcher

Scores how well a resume fits a job description and explains why.

## How it works
1. **Text features**: bigram TF-IDF cosine similarity, JD keyword coverage.
2. **Skill features**: curated skill vocabulary (+ aliases like `sklearn`→`scikit-learn`); recall of JD skills, Jaccard overlap.
3. **Experience feature**: years required in JD vs years claimed in resume.
4. **Model**: logistic regression / gradient boosting (best by 5-fold CV AUC) learns how to weigh the features → probability = match score.
   Without a trained model, a hand-tuned weighted blend is used.

## Run
```bash
pip install -r requirements.txt
python make_data.py     # synthetic labelled pairs -> data/pairs.csv
python train.py         # trains + saves model.joblib
python demo.py          # CLI example
streamlit run app.py    # web UI
```

## Make it real
- Replace `data/pairs.csv` (columns: resume, jd, match) with real labelled pairs. The synthetic data is only a demo, so its metrics are optimistic.
- Add sentence-transformer embeddings (`all-MiniLM-L6-v2`) as another feature for semantic matching.
- Grow `skills.py` for your domain; watch for bias (names, gender, college) — keep such fields out of the features.
