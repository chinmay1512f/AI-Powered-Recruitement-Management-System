import re

def extract_experience(text):
    matches = re.findall(r"(\d+)\s*\+?\s*years?", text.lower())
    if matches:
        return max(map(int, matches))
    return 0
