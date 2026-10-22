import re

def extract_education(text):
    lines = text.split("\n")
    education_lines = []
    capture = False

    for line in lines:
        clean = line.strip()

        if re.search(r"\b(education|academic|qualification)\b", clean.lower()):
            capture = True
            continue

        if capture and re.search(
            r"\b(skills|experience|projects|internship|certification)\b",
            clean.lower()
        ):
            break

        if capture and clean:
            education_lines.append(clean)

    return education_lines
