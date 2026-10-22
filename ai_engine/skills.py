import re

COMMON_SKILLS = [
    "python", "java", "c++", "c", "sql", "mysql", "postgresql",
    "machine learning", "deep learning", "data science",
    "flask", "django", "fastapi",
    "html", "css", "javascript", "react", "node",
    "aws", "docker", "kubernetes",
    "linux", "git", "tensorflow", "pytorch"
]

def skill_pattern(skill):
    r"""Match a skill as a standalone token.

    \b does not work for skills ending in a non-word character: r"\bc\+\+\b"
    never matches "c++" because there is no word boundary after the '+'.
    Explicit non-word lookarounds handle c++, c#, .net and node.js correctly.
    """
    return r"(?<![\w+#.])" + re.escape(skill) + r"(?![\w+#])"


def extract_skills(text):
    """Kept for compatibility. app.py now uses extract_skills_comprehensive(),
    which draws on the full SKILL_DATABASE instead of these 25 entries."""
    text = text.lower()
    found_skills = set()

    for skill in COMMON_SKILLS:
        if re.search(skill_pattern(skill), text):
            found_skills.add(skill.title())

    return sorted(list(found_skills))
