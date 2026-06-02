# tests/test_recommender.py
import pandas as pd
import pytest
from src.preprocessor import load_dataset, preprocess_dataframe, clean_skills_text, parse_user_skills
from src.recommender import TechStackRecommender


@pytest.fixture
def sample_df():
    data = {
        "job_role": ["Data Scientist", "DevOps Engineer", "Frontend Developer"],
        "skills_required": [
            "Python SQL Machine Learning TensorFlow Pandas",
            "AWS Docker Kubernetes CI/CD Linux",
            "HTML CSS JavaScript React TypeScript",
        ],
        "description": ["Desc A", "Desc B", "Desc C"],
    }
    df = pd.DataFrame(data)
    return preprocess_dataframe(df)



def test_clean_skills_text():
    result = clean_skills_text("Python, SQL, AWS!")
    assert "python" in result
    assert "sql" in result
    assert "aws" in result
    assert clean_skills_text(None) == ""
    assert clean_skills_text("") == ""


def test_recommender_fits_without_error(sample_df):
    rec = TechStackRecommender(top_n=2)
    rec.fit(sample_df)
    assert rec.is_fitted is True


def test_recommender_returns_correct_count(sample_df):
    rec = TechStackRecommender(top_n=2)
    rec.fit(sample_df)
    results = rec.recommend(parse_user_skills(["Python", "Machine Learning", "SQL"]))
    assert len(results) == 2


def test_data_scientist_tops_for_ml_skills(sample_df):
    rec = TechStackRecommender(top_n=3)
    rec.fit(sample_df)
    results = rec.recommend(parse_user_skills(["Python", "Machine Learning", "TensorFlow"]))
    top_role = results[0][0]
    assert top_role == "Data Scientist"


def test_scores_between_zero_and_one(sample_df):
    rec = TechStackRecommender(top_n=3)
    rec.fit(sample_df)
    results = rec.recommend(parse_user_skills(["Python", "Docker", "React"]))
    for _, score, _ in results:
        assert 0.0 <= score <= 1.0


def test_explain_match_returns_overlapping_terms(sample_df):
    rec = TechStackRecommender(top_n=3)
    rec.fit(sample_df)
    matched = rec.explain_match(["Python", "SQL", "TensorFlow"], "Data Scientist")
    assert "python" in matched or "sql" in matched


def test_recommend_raises_if_not_fitted():
    rec = TechStackRecommender()
    with pytest.raises(RuntimeError):
        rec.recommend("python sql aws")