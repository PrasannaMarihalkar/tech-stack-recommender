import argparse
import sys

from src.preprocessor import load_dataset, preprocess_dataframe, parse_user_skills
from src.recommender import TechStackRecommender
from src.utils import (
    get_user_skills_interactive,
    validate_skills,
    display_results,
)

DEFAULT_DATA = "data/raw_skills.csv"
DEFAULT_TOP_N = 3


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tech-stack-recommender",
        description="DecodeLabs Project 3 — AI-powered career path recommender",
    )
    parser.add_argument(
        "--skills",
        nargs="+",
        metavar="SKILL",
        help="Your skills (e.g. --skills Python Docker AWS)",
        default=None,
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_N,
        metavar="N",
        help="Number of recommendations to return (default: 3)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA,
        metavar="PATH",
        help="Path to job roles CSV dataset",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        df_raw = load_dataset(args.data)
        df = preprocess_dataframe(df_raw)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] {e}\n")
        sys.exit(1)

    recommender = TechStackRecommender(top_n=args.top)
    recommender.fit(df)

    stats = recommender.get_stats()
    print(f"\n[✓] Model ready — {stats['total_job_roles']} job roles, "
          f"{stats['vocabulary_size']} vocabulary terms")

    if args.skills:
        user_skills = args.skills
        if not validate_skills(user_skills):
            print("\n[ERROR] Provide at least 3 skills (--skills Python SQL AWS ...)\n")
            sys.exit(1)
    else:
        user_skills = get_user_skills_interactive()

    user_profile_str = parse_user_skills(user_skills)

    try:
        results = recommender.recommend(user_profile_str)
    except ValueError as e:
        print(f"\n[ERROR] {e}\n")
        sys.exit(1)

    matched_skills = {
        role: recommender.explain_match(user_skills, role)
        for role, _, _ in results
    }

    display_results(user_skills, results, matched_skills, stats)


if __name__ == "__main__":
    main()