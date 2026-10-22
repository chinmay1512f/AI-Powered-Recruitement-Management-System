# Changes: bug fixes, evaluation harness, and LLM extraction

Backups of the originals are at `app.py.backup`, `resume_system.db.backup`, and
`templates.backup/`.

---

## Do these two things first

**1. Revoke the exposed Gmail app password.** It was committed in `app.py` in
plain text (`REDACTED - see git history of the original file`). Revoke it at
<https://myaccount.google.com/apppasswords>, generate a new one, and put it in `.env`.

**2. Create your `.env`.** Copy `.env.example` to `.env` and fill it in:

```bash
cp .env.example .env
```

At minimum set `FLASK_SECRET_KEY`, or every restart invalidates all logged-in
sessions (the key is now randomly generated when unset, instead of being a
hardcoded string anyone could read):

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## Phase 1 - bugs fixed

| # | Problem | Fix |
|---|---------|-----|
| 1 | Uploads saved under the raw browser filename, so two candidates uploading `resume.pdf` overwrote each other | `build_resume_filename()` - `secure_filename` + `c{candidate}_j{job}_{timestamp}_` prefix |
| 2 | TF-IDF returned exactly 0.0 for every candidate (`max_df=0.95` prunes every term shared by both documents when there are only two) | `max_df=1.0` |
| 3 | Score breakdown shown to HR was the total re-scaled by the weights - it could not distinguish candidates | Real components persisted in `candidate_result.breakdown_json`; rows scored before this show "breakdown unavailable" rather than fake numbers |
| 4 | `\bc\+\+\b` never matches "c++"; `Java.` from a job posting matched nothing | `skill_pattern()` uses non-word lookarounds; `normalize_skill()` strips trailing punctuation |
| 5 | Skills column came from a 24-entry list while a 100+ entry `SKILL_DATABASE` sat unused | `extract_skills_comprehensive()` |
| 6 | Accept, reject, delete, close, reopen, cancel and complete were GET links - any prefetch or `<img>` tag could fire them | All POST-only, with CSRF tokens on every form |
| 7 | No CSRF protection anywhere | Session token, `@app.before_request` check, `{{ csrf_token() }}` in all 9 forms |
| 8 | Secrets hardcoded in source | `.env` via a dependency-free `load_env()` |
| 9 | `db.create_all()` only ran under `__main__`, so `gunicorn app:app` started with no schema | `init_db()` runs on import |
| 10 | `debug=True` hardcoded (remote code execution if exposed) | `FLASK_DEBUG` env var, defaults off |
| 11 | Candidates could apply twice by hitting `/apply/<id>` directly | Server-side duplicate check |
| 12 | Bulk reject silently swallowed email failures | Failures counted and flashed |
| 13 | No upload size or type limit | 10 MB cap, extension allowlist |

Run `python migrate_db.py` once to add the new columns (already done; it is
idempotent and safe to re-run).

### Data damage that could not be repaired

`Chinmay Jandrapalli` and `Nishant Mishra` both point at
`Resume_7_Rahul_Singh.pdf` - one overwrote the other before the fix. The
original file is gone. **Both need to re-apply** before their scores mean
anything. `migrate_db.py` reports collisions like this on every run.

---

## Phase 1b - local scoring quality

Four further fixes, all offline, found by testing the components directly.

**Semantic truncation.** `all-MiniLM-L6-v2` accepts 256 word-piece tokens and
silently truncates the rest. A two-page resume is 400-1000 tokens, so the old
whole-document encode read roughly its first quarter. Measured: the same resume
with its qualifications moved from top to bottom scored **36.03 vs 14.30** - a
21-point swing from layout alone. `semantic_matcher.py` now splits the resume
into overlapping windows under the limit, embeds each, and blends the best
window with the mean of the top 3. Same test after the fix: **32.19 vs 27.91**,
a 4.3-point spread.

**TF-IDF re-weighted 0.15 -> 0.05.** Measured against a four-skill job posting:

| Resume phrasing | TF-IDF |
|---|---|
| exact wording | 60.8 |
| "Postgres, JS, ML, K8s" | **0.0** |
| skills described, not named | **0.0** |
| the job's keywords pasted twice | **71.0** |

It returns zero for correct answers and rewards keyword stuffing above an honest
resume. Most of its weight moved to skills. All weights now live in one
`SCORE_WEIGHTS` dict, and both scorers route through `combine_scores()`.

**"java" was matching "JavaScript".** `extract_skills_precise` fell back to a raw
substring test when the regex missed, so "java" matched inside "javascript",
"c" matched inside almost any word, and "react" matched "reactive". Removed;
matching is regex-only now.

**Singular/plural and concept aliases.** Job postings say "databases" and
"algorithms"; resumes say "database" and "SQL". Added inflection handling plus
alias lists for `database`, `algorithm`, `data structure`, `rest api` and
`frontend`.

### What this did to the metrics: nothing, and that is worth knowing

Before and after these four fixes the numbers are identical - Spearman +0.618,
NDCG 0.834, top-3 3/3. The *reasons* changed completely: the backend candidate
rose from 4th to 3rd because "databases" now matches SQL, and the frontend
candidate correctly fell to last because the java/JavaScript false positive is
gone. Two errors were cancelling out.

These changes are justified by correctness, not by the score. Do not read a
6-candidate sample as evidence either way, and do not tune the weights against
it - that is fitting noise.

---

## Phase 1c - skill-gap page corrections

The skill-gap page kept its own copy of the matching logic, so it contradicted
the ranking page on the same candidate:

```
Frontend candidate whose resume says "JavaScript", job requires "Java":
  ranking page   -> 0.0    skill-gap page -> 100% matched (via substring)

Backend candidate with SQL + MongoDB, job requires "databases":
  ranking page   -> 51.0   skill-gap page -> 0%
```

`analyze_skill_gap` now uses the same `match_skill_variant` helper as the
ranking path. After the fix both pages agree (0 and 0; 51.0 and 50%).
Skills found only through an equivalent - SQL for "databases" - are now reported
as **partial** matches and count half in the match percentage, which is what
that previously-always-empty section of the UI was for.

**Priority is derived from the job.** Every skill on that page was typed into
the job posting by the HR user, so labelling it "nice to have" was wrong - a
Software Engineer posting requiring data structures, algorithms and databases
had all three marked nice-to-have. They are now `important`, or `critical` if
they appear in the curated high-demand list.

**ROI score was inverted and unreachable.** It returned **0 for a candidate with
no gaps** - ranking a perfect match as the worst possible investment - and
produced values around 1.0 on a scale the template rendered as a percentage with
a `> 70` threshold that could never fire. Now 0-100 and behaves sensibly:

| Case | Before | After |
|---|---|---|
| Perfect candidate, no gaps | 0 | 100 |
| One critical, quick-to-learn gap | ~0.3 | 90 |
| Five low-value gaps | 1.0 | 45.6 |

The scale constant is a heuristic and is labelled as one in the UI.

**"Days to Upskill" says when it is guessing.** Only 8 skills have researched
hour counts; the rest fall back to a flat 100 hours. The card now shows
"4 of 5 use a default estimate" instead of implying precision.

**Proficiency scoring counted substrings.** `resume_lower.count("java")` treated
every "JavaScript" as evidence of Java. Now token-matched.

**Proficiency inference credited other people's seniority.** The level badge
(beginner/intermediate/advanced/expert) came from scanning a fixed 200-character
window around each skill mention, and the years pattern `(\d+)\s*years?.*?skill`
matched across any distance. Two rules fixed both:

1. context is now the sentence or line the skill appears in, not a character window;
2. "N years" only counts within 40 characters of the skill.

| Resume text | Before | After |
|---|---|---|
| "Skills: Python. Junior Developer, reporting to the Senior Architect." | expert | beginner |
| "8 years in retail sales. Recently learned Python in a bootcamp." | expert | beginner |
| "Python developer with 6 years building production services." | expert | expert |
| "Senior Python Engineer at Acme, led the platform team." | expert | expert |

This mattered beyond the badge: `extract_skills_precise` multiplies each skill's
confidence by proficiency (beginner 0.6 -> expert 1.0), so a false "expert" was
inflating the ranking score by about 40% for that skill. Also fixed "advanced"
being listed as an expert-level indicator, which promoted every "advanced"
mention two levels.

---

## Phase 5 - analytics and explainability

**The evidence was being computed and never shown.** For every required skill the
LLM records the resume text that justified its confidence - and none of it
reached a template. It is now on both pages:

* results page: a "Why?" button per candidate opens a drawer with the AI's
  written verdict plus each skill's supporting quote and confidence;
* skill-gap page: a "Why the AI scored it this way" panel, listing quotes for
  matched skills and an explicit "nothing in the resume demonstrates this" for
  the gaps.

**Real pool analytics on the results page.** Four counters said nothing about the
shape of an applicant pool. `build_pool_analytics()` now produces a score
histogram (with average, median and spread) and a "requirements the pool is
failing" panel counting how many candidates miss each required skill. When one
requirement is missed by 70%+ of applicants the page says so directly - that
usually means the requirement is unrealistic rather than the pool being weak.

**ROI removed entirely.** The score, its calibration constant, the metric card,
the investment panel, the recommendation line and the dead CSS are all gone. It
was arbitrary constants divided by arbitrary constants presented as a percentage.

**Charts made informative.** The proficiency bars are coloured by match status
instead of uniform blue, so a matched skill no longer looks identical to a
missing one. The doughnut - which duplicated the three stat boxes directly below
it - is replaced by a horizontal per-skill coverage chart (green direct match,
amber partial, red missing).

---

## Phase 2 - measuring accuracy

```bash
python eval/rank_eval.py --job 1
```

Compares the system's ranking against human relevance grades in
`eval/ground_truth.json` using Spearman rank correlation, NDCG, and top-3 overlap.

**Baseline on job 1 (local scorer): Spearman +0.618, NDCG 0.834, top-3 3/3.**

The grades in `ground_truth.json` are provisional - drafted by reading each
resume against the job requirements. **Edit them to match your judgement**; the
metric is only as good as the labels. Note that with 6 candidates the
correlation is not statistically significant (p = 0.19) - it is a directional
signal, not proof. More labelled candidates would fix that.

---

## Phase 3 - LLM extraction (ready, needs your key)

`ai_engine/llm_extractor.py`. The model is used as an **extractor, not a judge**:
it reads the resume and returns structured facts, then the existing weighted
scoring code turns those facts into a number. Rankings stay reproducible, and
re-tuning weights costs no API calls.

Two providers are supported. To turn it on, add **one** key to `.env`:

```
GEMINI_API_KEY=...          # Google AI Studio - has a free tier
GEMINI_MODEL=gemini-2.5-flash
```

or

```
ANTHROPIC_API_KEY=sk-ant-...
RESUME_EXTRACTION_MODEL=claude-opus-5
```

That is the only step. `get_extractor()` picks Gemini if its key is set, then
Claude, then returns an unavailable stub so `score_resume()` falls through to the
fully offline pipeline. Both providers share one `BaseExtractor.score()`, so the
scoring maths cannot drift between them.

Gemini specifics: the JSON schema is sanitised (`gemini_schema()` strips
`additionalProperties`, which Gemini's `Schema` type rejects), `temperature=0`
keeps extraction deterministic, free-tier 429s are retried with backoff, an
unknown model name falls back through a list of alternatives, and an invalid key
fails on the first call rather than retrying every model.

What it does:
- **Evidence-based skill assessment.** Judges "data structures" from an LRU-cache
  project rather than looking for the literal phrase. This is the fix for the
  four-of-five required skills currently scoring 0.
- **Real experience figures** computed from date ranges, with internships at 0.5x.
- **Written justification** per candidate, shown on the results page.
- **PII redacted before transmission** - emails, phones, profile links, and the
  candidate's name are stripped. Skill names like `node.js` and `asp.net` are
  deliberately preserved.
- **Prompt caching** on the rubric and job block, which are identical across
  every applicant to one job.
- **Falls back to local scoring** on any API error, refusal, or missing key.

Cost: roughly 3c per resume on Opus 5 before caching.

---

## Phase 4 - measuring the delta (run once you have the key)

```bash
python eval/rank_eval.py --job 1 --compare
```

Runs both scorers and prints the change in Spearman and NDCG. This is the number
that tells you whether the API was worth it. **It spends real money** - about 20c
for the six candidates on job 1.

---

## Still open

Things I did not change, in rough priority order:

1. **Job requirements changing does not invalidate cached scores.** Results are
   only recomputed when the resume file changes, so editing a job's required
   skills leaves stale scores in place.
2. **Scoring runs synchronously inside the request.** With a large applicant
   pool "AI Sort" will hang the page; this wants a background job.
3. **Ten templates load fonts, icons, and Chart.js from CDNs**, so the UI
   degrades offline and the skill-gap charts do not render at all. Vendoring
   them into `static/` is about half an hour.
4. **`database/models.py` is empty and `database/database.db` is unused** - the
   app uses `resume_system.db` at the project root. I left both alone rather
   than delete files with unknown contents.
5. `templates/upload.html` and `templates/history.html` pointed at routes that
   no longer exist; moved to `templates/_deprecated/`.
