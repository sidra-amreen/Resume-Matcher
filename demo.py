from matcher import ResumeMatcher
resume = """Data Scientist with 4 years of experience. Skills: Python, pandas, sklearn, SQL,
XGBoost, statistics, A/B testing. Built churn models and deployed with Docker."""
jd = """Looking for a Data Scientist with 3+ years of experience. Must know Python, SQL,
machine learning, scikit-learn, and AWS. Experience with feature engineering is a plus."""
r = ResumeMatcher().score(resume, jd)
print(f"Score: {r.score}%\nMatched: {r.matched_skills}\nMissing: {r.missing_skills}\nFeatures: {r.features}")
