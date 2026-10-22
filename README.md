# AI Powered Recruitment Management System

A Flask application that ranks job applicants against a posted opening. HR creates a
job with required skills, candidates apply with a PDF resume, and the system scores
every resume on five dimensions, explains each score, and surfaces the skill gaps.

It runs **fully offline** using local models. Adding a Gemini API key is optional and
measurably improves ranking quality (see *Accuracy* below).

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
