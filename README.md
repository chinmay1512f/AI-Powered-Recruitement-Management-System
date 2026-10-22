# AI Powered Recruitment Management System

> Flask app that ranks job applicants against a posted opening using Sentence-BERT
> semantic matching, per-skill evidence extraction and optional Gemini LLM assessment.
> Runs fully offline; includes an evaluation harness measuring Spearman rank
> correlation against hand-labelled ground truth.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)
![Sentence-BERT](https://img.shields.io/badge/Sentence--BERT-all--MiniLM--L6--v2-FFB300)
![Gemini](https://img.shields.io/badge/Gemini-optional-4285F4?logo=googlegemini&logoColor=white)
![Offline capable](https://img.shields.io/badge/Offline-capable-2E7D32)

---

## Overview

Shortlisting resumes by hand does not scale, and keyword filters fail in both
directions: they reject strong candidates who phrase their experience differently,
and they promote weak ones who simply list the right words. This project replaces
that with a scoring pipeline that reads a resume the way a reviewer would — looking
for *evidence* behind each claimed skill rather than the presence of a string.

HR posts a job with its required skills, minimum experience and a deadline.
Candidates register and apply with a PDF resume. The system extracts the text
(with OCR fallback for scanned documents), scores every applicant across five
weighted dimensions, and ranks them. Every score is explained: HR can open any
candidate and see which required skills were found, the exact line of the resume
that evidences each one, a confidence band, and how that candidate compares with
the rest of the applicant pool. From there HR can accept or reject, schedule
interviews, email candidates and export the shortlist as a PDF.

The ranking runs entirely on the local machine using Sentence-BERT, so the system
works with no internet connection and no API costs. Supplying a Gemini API key is
optional and turns on a second pass in which the LLM assesses each required skill
individually and returns a quoted justification. Resume text is PII-redacted — name,
email, phone and profile links stripped — before any text leaves the machine.

Because "it feels better" is not a result, ranking quality is measured rather than
asserted. `eval/rank_eval.py` scores the system's ordering against hand-labelled
ground truth using Spearman rank correlation, NDCG and top-K overlap; the numbers
are in the *Accuracy* section below.

### What makes it different from a keyword filter

- **Evidence, not occurrence.** A skill counts when the resume shows it being used,
  and the matching line is surfaced to the reviewer.
- **Resistant to keyword stuffing.** In testing, a resume padded with unsupported
  skill names ranked #2 under local scoring and #8 once LLM assessment was enabled.
- **No truncation.** Long resumes are chunked with overlap, so Sentence-BERT's
  256-token limit does not silently discard the second half of a CV.
- **Explainable end to end.** Every ranking decision can be traced to a component
  score and a quoted line, which matters for a hiring decision.
- **Private by default.** Fully functional with no network access at all.

---

## How scoring works

Each resume gets a weighted score out of 100:

| Component | Weight | Method |
|---|---|---|
| Skills | 40% | Per-skill evidence matching (regex with word-boundary lookarounds, so `c++` and `java` vs `javascript` behave correctly) |
| Semantic fit | 30% | Sentence-BERT (`all-MiniLM-L6-v2`) over overlapping chunks, so long resumes are not truncated at 256 tokens |
| Experience | 20% | Years extracted from the resume vs the job minimum |
| TF-IDF overlap | 5% | scikit-learn cosine similarity |
| Education | 5% | Degree level detection |

With `GEMINI_API_KEY` set, each required skill is additionally assessed by the LLM,
which returns a confidence and a quoted line of evidence from the resume. Resume text
is PII-redacted (name, email, phone, profile links) before it leaves the machine.

## Accuracy

Measured on 10 CSE resumes against a Software Engineer posting, using
`eval/rank_eval.py` and a hand-labelled `eval/ground_truth.json`:

| Metric | Local only | With Gemini |
|---|---|---|
| Spearman rank correlation | +0.613 (p = 0.140) | **+0.879 (p = 0.001)** |
| Top-3 overlap with ground truth | 2 / 3 | **3 / 3** |
| NDCG | 0.921 | 0.874 |

The clearest difference: a keyword-stuffing resume that listed skills it had no evidence
for ranked #2 locally and #8 with the LLM.

## Features

- HR: create and close jobs, view applications, AI sort with a live progress page,
  accept/reject, schedule interviews, email candidates, export an accepted-candidate PDF
- Candidate: register, browse open jobs, apply with a resume, track status
- Per-candidate skill-gap analysis with evidence, confidence bands and pool benchmarking
- Pool analytics across all applicants for a job
- OCR fallback (Tesseract) for scanned resumes

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate on Linux/macOS
pip install -r requirements.txt

copy .env.example .env         # cp on Linux/macOS
# fill in FLASK_SECRET_KEY and HR_ADMIN_PASSWORD at minimum
python check_setup.py          # verifies config without printing any secret
python app.py
```

Open http://127.0.0.1:5000. The HR login is whatever you set as
`HR_ADMIN_USERNAME` / `HR_ADMIN_PASSWORD` in `.env`.

### Optional: LLM extraction

Get a free key at https://aistudio.google.com/apikey and set `GEMINI_API_KEY` in `.env`.
Without it the app stays fully offline — no network calls at all.

## Layout

```
app.py                   routes, scoring orchestration, PDF export
ai_engine/
  semantic_matcher.py    chunked Sentence-BERT similarity
  skills.py              skill extraction and evidence matching
  experience.py          years-of-experience parsing
  education.py           degree detection
  llm_extractor.py       Gemini / Anthropic extraction, PII redaction
resume_parser/           PDF, DOCX and OCR text extraction
database/models.py       SQLAlchemy models
templates/               Jinja2 views
eval/rank_eval.py        Spearman / NDCG / top-K evaluation harness
test_data/               generates 10 sample CSE resumes for testing
tests/smoke_test.py      end-to-end checks (auth, CSRF, uploads, PDF)
```

## Testing

```bash
python tests/smoke_test.py     # safe against real data; cleans up after itself
python eval/rank_eval.py       # ranking quality vs ground truth
```

## Security notes

- All secrets live in `.env`, which is gitignored. Nothing is hardcoded.
- CSRF tokens on every form; all mutations are POST-only.
- Uploaded filenames are namespaced per candidate and job, so two people uploading
  `resume.pdf` cannot overwrite each other.
- Uploaded resumes (`uploads/`) and the database are gitignored — they contain real PII.
