import random
import pandas as pd

ROLES = {
    "data scientist": ["python", "pandas", "scikit-learn", "machine learning", "statistics", "sql", "feature engineering", "a/b testing", "xgboost"],
    "ml engineer": ["python", "pytorch", "tensorflow", "docker", "kubernetes", "mlops", "aws", "deep learning", "ci/cd"],
    "data engineer": ["python", "sql", "spark", "airflow", "kafka", "aws", "etl", "postgresql", "docker"],
    "backend developer": ["java", "spring", "sql", "rest api", "docker", "postgresql", "redis", "git", "kubernetes"],
    "frontend developer": ["javascript", "typescript", "react", "git", "rest api", "graphql", "node.js", "agile"],
    "data analyst": ["sql", "excel", "tableau", "power bi", "python", "data analysis", "data visualization", "statistics"],
    "nlp engineer": ["python", "nlp", "transformers", "pytorch", "llm", "deep learning", "fastapi", "docker"],
}
FILLER = ["Collaborated with cross-functional teams", "Delivered projects on schedule",
          "Strong communication and ownership", "Mentored junior colleagues", "Worked in an agile environment"]
OTHER = [s for v in ROLES.values() for s in v]


def make_jd(role, rng):
    sk = rng.sample(ROLES[role], k=rng.randint(4, 7))
    yrs = rng.choice([1, 2, 3, 5, 7])
    return (f"We are hiring a {role}. Requirements: {yrs}+ years of experience. "
            f"Must know {', '.join(sk)}. Responsibilities include building and shipping production solutions.")


def make_resume(role, rng, years):
    sk = rng.sample(ROLES[role], k=rng.randint(3, 8))
    sk += rng.sample(OTHER, k=rng.randint(0, 3))
    return (f"{role.title()} with {years} years of experience. Skills: {', '.join(sorted(set(sk)))}. "
            + " ".join(rng.sample(FILLER, 3)))


def build(n=1500, seed=42):
    rng = random.Random(seed)
    roles, rows = list(ROLES), []
    for _ in range(n):
        jd_role = rng.choice(roles)
        same = rng.random() < 0.5
        r_role = jd_role if same else rng.choice([r for r in roles if r != jd_role])
        jd = make_jd(jd_role, rng)
        resume = make_resume(r_role, rng, rng.randint(0, 9))
        label = int(same)
        if rng.random() < 0.08:         
            label = 1 - label
        rows.append((resume, jd, label))
    return pd.DataFrame(rows, columns=["resume", "jd", "match"])


if __name__ == "__main__":
    df = build()
    df.to_csv("data/pairs.csv", index=False)
    print(df.shape, df.match.mean())
