import re

SKILL_DICT = {
    "python": ["python"],
    "java": ["java"],
    "c++": ["c++"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl"],
    "nlp": ["nlp", "natural language processing"],
    "sql": ["sql"],
    "excel": ["excel"],
    "aws": ["aws", "amazon web services"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch"],
    "data analysis": ["data analysis", "data analytics"],
    "flask": ["flask"],
    "django": ["django"],
    "react": ["react", "reactjs", "react.js"],
    "node": ["node", "nodejs", "node.js"],
    "mongodb": ["mongodb", "mongo"]
}


def extract_skills(text):
    found_skills = []
    text = text.lower()

    for skill, aliases in SKILL_DICT.items():
        for alias in aliases:
            if re.search(r'\b' + re.escape(alias) + r'\b', text):
                found_skills.append(skill)
                break

    return list(set(found_skills))


def skill_gap(resume_skills, jd_skills):
    return list(set(jd_skills) - set(resume_skills))
