import sys
from typing import List, Tuple


BANNER = """
╔══════════════════════════════════════════════════════════╗
║         TECH STACK RECOMMENDER  |  DecodeLabs           ║
║              AI Recommendation Logic — Project 3         ║
╚══════════════════════════════════════════════════════════╝
"""


def get_user_skills_interactive() -> List[str]:
    print(BANNER)
    print("  Tell me your skills and I'll recommend your ideal career path.")
    print("  (Enter at least 3 skills. Press Enter with no input when done.)\n")

    skills = []
    MIN_SKILLS = 3

    while True:
        remaining = max(0, MIN_SKILLS - len(skills))

        if remaining > 0:
            prompt = f"  Skill #{len(skills) + 1} (need {remaining} more): "
        else:
            prompt = f"  Skill #{len(skills) + 1} (or press Enter to finish): "

        user_input = input(prompt).strip()

        if user_input == "":
            if len(skills) >= MIN_SKILLS:
                break
            else:
                print(f"\n  ⚠  You need at least {MIN_SKILLS} skills for accurate matching.\n")
                continue

        if user_input.lower() in [s.lower() for s in skills]:
            print(f"  ('{user_input}' already added — skipping duplicate)")
            continue

        skills.append(user_input)

    return skills


def validate_skills(skills: List[str]) -> bool:
    cleaned = [s.strip() for s in skills if s.strip()]
    return len(cleaned) >= 3


def display_results(
    user_skills: List[str],
    results: List[Tuple[str, float, str]],
    matched_skills: dict,
    model_stats: dict,
) -> None:
    MEDALS = ["  🥇 RANK 1", "  🥈 RANK 2", "  🥉 RANK 3"]
    SEPARATOR = "─" * 60

    print(f"\n{SEPARATOR}")
    print(f"  YOUR SKILL PROFILE")
    print(f"  Skills: {', '.join(user_skills)}")
    print(f"{SEPARATOR}")

    print(f"\n  TOP {len(results)} RECOMMENDED CAREER PATHS\n")

    for i, (role, score, desc) in enumerate(results):
        rank_label = MEDALS[i] if i < len(MEDALS) else f"  #{i+1}"
        match_pct = f"{score * 100:.1f}%"
        matched = matched_skills.get(role, [])

        print(rank_label)
        print(f"  Role        : {role}")
        print(f"  Match Score : {match_pct}")
        print(f"  Description : {desc}")
        if matched:
            print(f"  Key Matches : {', '.join(matched)}")
        else:
            print(f"  Key Matches : (no direct keyword overlap — matched via context)")
        print()

    print(SEPARATOR)
    print(f"  Engine Stats: {model_stats.get('total_job_roles', '?')} roles | "
          f"{model_stats.get('vocabulary_size', '?')} vocab terms")
    print(SEPARATOR + "\n")