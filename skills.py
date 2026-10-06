import re

SKILLS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "sql", "r", "scala",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "xgboost", "lightgbm",
    "nlp", "computer vision", "deep learning", "machine learning", "statistics", "data analysis",
    "data visualization", "tableau", "power bi", "excel", "spark", "hadoop", "kafka", "airflow",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd", "git", "linux",
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi", "spring", "rest api",
    "graphql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "etl", "mlops",
    "a/b testing", "feature engineering", "llm", "transformers", "agile", "scrum",
    "communication", "leadership", "problem solving",
}

ALIASES = {
    "sklearn": "scikit-learn", "scikit learn": "scikit-learn", "js": "javascript",
    "postgres": "postgresql", "k8s": "kubernetes", "nodejs": "node.js", "node": "node.js",
    "ml": "machine learning", "dl": "deep learning", "natural language processing": "nlp",
    "powerbi": "power bi", "large language models": "llm", "golang": "go",
}


def _pattern(term: str) -> re.Pattern:
    return re.compile(r"(?<![\w+#.])" + re.escape(term) + r"(?![\w+#])", re.I)


_PATTERNS = {s: _pattern(s) for s in SKILLS}
_ALIAS_PATTERNS = {a: (_pattern(a), c) for a, c in ALIASES.items()}


def extract_skills(text: str) -> set:
    found = {s for s, p in _PATTERNS.items() if p.search(text)}
    found |= {canon for _, (p, canon) in _ALIAS_PATTERNS.items() if p.search(text)}
    return found


def extract_years(text: str) -> float:
    """Largest 'N years' / 'N+ years' figure mentioned (0 if none)."""
    nums = re.findall(r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)", text, re.I)
    return float(max(map(int, nums))) if nums else 0.0
