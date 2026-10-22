"""LLM-backed resume extraction, with pluggable providers.

Design note: the model is used as an EXTRACTOR, not as a judge. It reads the
messy resume and returns structured facts; the existing deterministic code in
app.py turns those facts into a score. That keeps rankings reproducible and
auditable - the same facts always produce the same number, and a change to the
scoring weights is arithmetic rather than a re-run of every API call.

Two providers are supported:
  * Gemini    (GEMINI_API_KEY)    - Google AI Studio, has a free tier
  * Claude    (ANTHROPIC_API_KEY) - Anthropic

get_extractor() picks whichever key is configured. With neither, the app runs
fully offline on the local embedding + keyword pipeline.
"""
import json
import os
import re
import threading
import time

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"

# Tried in order if the configured Gemini model is rejected as unknown - free
# tier model availability changes, and a wrong name should degrade rather than
# break. `check_setup.py` lists what a given key can actually reach.
# Verified against a live free-tier key. Note that models.list() reports models
# that generateContent then rejects - gemini-2.5-flash is listed but returns
# "no longer available to new users" - so this list is what actually worked.
GEMINI_MODEL_FALLBACKS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
]

# A model that just returned 503 (overloaded) or 429 (quota exhausted) will do
# the same for the next resume too. Remember that for a few minutes instead of
# rediscovering it every call - that rediscovery, with its retry sleeps, was
# costing ~49s of every 52s call.
MODEL_COOLDOWN_SECONDS = 300
_model_cooldown = {}
_cooldown_lock = threading.Lock()


def _usable_models(preferred):
    now = time.time()
    ordered = [preferred] + [m for m in GEMINI_MODEL_FALLBACKS if m != preferred]
    with _cooldown_lock:
        usable = [m for m in ordered if _model_cooldown.get(m, 0) <= now]
    return usable or ordered  # everything is cooling down: try anyway


def _put_on_cooldown(model, seconds=MODEL_COOLDOWN_SECONDS):
    with _cooldown_lock:
        _model_cooldown[model] = time.time() + seconds


def _clear_cooldown(model):
    with _cooldown_lock:
        _model_cooldown.pop(model, None)


EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "required_skill_assessment": {
            "type": "array",
            "description": "One entry per required skill, in the order given.",
            "items": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string"},
                    "demonstrated": {
                        "type": "boolean",
                        "description": "True only if the resume shows real evidence.",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "0.0-1.0. How strongly the resume demonstrates this skill.",
                    },
                    "evidence": {
                        "type": "string",
                        "description": "Short quote or paraphrase from the resume, or '' if absent.",
                    },
                    "proficiency": {
                        "type": "string",
                        "enum": ["none", "beginner", "intermediate", "advanced", "expert"],
                    },
                },
                "required": ["skill", "demonstrated", "confidence", "evidence", "proficiency"],
                "additionalProperties": False,
            },
        },
        "total_years_experience": {
            "type": "number",
            "description": "Professional experience in years, computed from date ranges. Internships count as 0.5x. 0 for a fresher.",
        },
        "relevant_years_experience": {
            "type": "number",
            "description": "Of that total, the years relevant to this specific role.",
        },
        "highest_education_level": {
            "type": "string",
            "enum": ["none", "high_school", "diploma", "bachelors", "masters", "phd"],
        },
        "seniority": {
            "type": "string",
            "enum": ["intern", "junior", "mid", "senior", "lead"],
        },
        "all_skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Every concrete technical skill the resume demonstrates.",
        },
        "summary": {
            "type": "string",
            "description": "Two sentences on fit for this role, citing resume evidence.",
        },
    },
    "required": [
        "required_skill_assessment",
        "total_years_experience",
        "relevant_years_experience",
        "highest_education_level",
        "seniority",
        "all_skills",
        "summary",
    ],
    "additionalProperties": False,
}

RUBRIC = """You assess resumes against job requirements for a recruitment system.

Rules:
- Judge each required skill on EVIDENCE, not on keyword presence. A resume that
  lists "LRU cache implementation in C++" demonstrates data structures even
  though the phrase "data structures" never appears. A resume that merely lists
  a word in a skills blob with no supporting project or role is weak evidence -
  give it a confidence around 0.4, not 1.0.
- Concepts like "data structures", "algorithms" and "databases" are almost never
  stated literally. Infer them from projects, coursework, competitive
  programming, and job responsibilities.
- Compute years of experience from date ranges. Treat internships as 0.5x their
  duration. A recent graduate with only internships has 0 professional years.
- Never infer anything from a candidate's name, gender, nationality, or the
  prestige of their school. Judge only demonstrated capability.
- Set confidence to 0.0 and demonstrated to false when there is no evidence. Do
  not be generous; an inflated score is worse than a low one.
- Quote real text in the evidence field. Never invent evidence.
- Return only the JSON object described by the schema."""

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
# Profile hosts only. A general bare-domain rule would also eat skill names
# like node.js, next.js and asp.net, which the model needs to see.
PROFILE_RE = re.compile(
    r"\b(?:github|gitlab|linkedin|bitbucket|leetcode|kaggle)\.com/\S+", re.IGNORECASE
)

EDUCATION_LEVELS = {
    "none": 1,
    "high_school": 1,
    "diploma": 2,
    "bachelors": 3,
    "masters": 4,
    "phd": 5,
}

PROFICIENCY_WEIGHT = {
    "none": 0.0,
    "beginner": 0.6,
    "intermediate": 0.8,
    "advanced": 0.9,
    "expert": 1.0,
}


def redact_pii(text, candidate_name=None):
    """Remove contact details before the resume leaves the machine.

    The system already knows who the candidate is; the model does not need to.
    This cuts the most obvious bias surface and limits what is transmitted.
    """
    text = EMAIL_RE.sub("[email]", text)
    text = PHONE_RE.sub("[phone]", text)
    text = URL_RE.sub("[url]", text)
    text = PROFILE_RE.sub("[profile]", text)
    if candidate_name:
        for part in [p for p in candidate_name.split() if len(p) > 2]:
            text = re.sub(re.escape(part), "[name]", text, flags=re.IGNORECASE)
    return text


def build_job_block(job_title, job_skills, min_experience):
    return (
        "ROLE UNDER CONSIDERATION\n"
        "Title: {}\n"
        "Minimum experience: {} years\n"
        "Required skills (assess each one, in this order):\n{}".format(
            job_title,
            min_experience,
            "\n".join("- {}".format(s) for s in job_skills),
        )
    )


def gemini_schema(schema):
    """Strip keys Gemini's Schema type rejects (notably additionalProperties)."""
    if isinstance(schema, dict):
        return {
            key: gemini_schema(value)
            for key, value in schema.items()
            if key not in ("additionalProperties",)
        }
    if isinstance(schema, list):
        return [gemini_schema(item) for item in schema]
    return schema


class BaseExtractor(object):
    """Shared scoring logic. Subclasses implement extract()."""

    provider = "none"

    def __init__(self):
        self._last_usage = None

    def available(self):
        return False

    def extract(self, resume_text, job_title, job_skills, min_experience, candidate_name=None):
        raise NotImplementedError

    def last_usage(self):
        """Token usage from the most recent call."""
        return self._last_usage

    def score(self, resume_text, job_title, job_skills, min_experience, candidate_name=None):
        """Score a resume using LLM-extracted facts and the existing weights.

        Returns (final_score, breakdown) matching calculate_final_score_improved,
        so callers can swap scorers without changing anything downstream.
        """
        import app  # lazy: avoids a circular import at module load

        facts = self.extract(
            resume_text, job_title, job_skills, min_experience, candidate_name
        )
        if facts is None:
            job_text = app.build_job_text(job_title, job_skills, min_experience)
            score, breakdown = app.calculate_final_score_improved(
                resume_text, job_text, job_skills, min_experience, job_title
            )
            breakdown["scored_by"] = "local (llm unavailable)"
            return score, breakdown

        # --- skills: evidence-weighted, from the model's per-skill assessment ---
        assessments = facts.get("required_skill_assessment", [])
        if assessments:
            total = 0.0
            for item in assessments:
                confidence = max(0.0, min(1.0, float(item.get("confidence", 0) or 0)))
                weight = PROFICIENCY_WEIGHT.get(item.get("proficiency", "none"), 0.6)
                total += confidence * max(weight, 0.6 if confidence > 0 else 0.0)
            skills_score = (total / len(assessments)) * 100
        else:
            skills_score = 0.0

        # --- experience: same shape as the local scorer, better inputs ---
        years = float(facts.get("relevant_years_experience", 0) or 0)
        total_years = float(facts.get("total_years_experience", 0) or 0)
        if min_experience > 0:
            exp_score = min((years / min_experience) * 100, 150)
        else:
            exp_score = 100 if years > 0 else 50
        if facts.get("seniority") in ("senior", "lead"):
            exp_score = min(exp_score + 20, 150)

        # --- education ---
        edu_level = EDUCATION_LEVELS.get(facts.get("highest_education_level", "none"), 1)
        edu_score = app.calculate_education_score_improved(edu_level, min_experience)

        # --- semantic + tfidf still come from the local models (free, offline) ---
        job_text = app.build_job_text(job_title, job_skills, min_experience)
        from ai_engine.semantic_matcher import match_resume

        semantic_score = min(float(match_resume(job_text, resume_text)) * 1.1, 100)
        tfidf_score = app.calculate_tfidf_similarity(resume_text, job_text)

        final_score = app.combine_scores(
            semantic_score, tfidf_score, skills_score, exp_score, edu_score
        )

        breakdown = {
            "semantic": round(semantic_score, 1),
            "tfidf": round(tfidf_score, 1),
            "skills": round(skills_score, 1),
            "experience": round(min(exp_score, 100), 1),
            "education": round(edu_score, 1),
            "years": int(round(total_years)),
            "education_level": edu_level,
            "matched_skills": {
                item.get("skill", ""): round(float(item.get("confidence", 0) or 0), 2)
                for item in assessments
            },
            "evidence": {
                item.get("skill", ""): item.get("evidence", "") for item in assessments
            },
            "summary": facts.get("summary", ""),
            "all_skills": facts.get("all_skills", []),
            "scored_by": self.provider,
        }
        return round(final_score, 2), breakdown


class GeminiExtractor(BaseExtractor):
    """Google AI Studio / Gemini. Has a free tier, which is why it is the default."""

    provider = "gemini"

    def __init__(self, model=None, api_key=None):
        BaseExtractor.__init__(self)
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._local = threading.local()

    def available(self):
        if not self.api_key:
            return False
        try:
            from google import genai  # noqa: F401
        except ImportError:
            return False
        return True

    def client(self):
        """One client per thread.

        A single genai.Client shared across a thread pool raised "Cannot send a
        request, as the client has been closed" - its underlying HTTP transport
        is not safe to share. Clients are cheap, so give each thread its own.
        """
        from google import genai

        existing = getattr(self._local, "client", None)
        if existing is None:
            existing = genai.Client(api_key=self.api_key)
            self._local.client = existing
        return existing

    def list_models(self):
        """Model names this key can actually reach - used by check_setup.py."""
        names = []
        for model in self.client().models.list():
            name = getattr(model, "name", "") or ""
            actions = getattr(model, "supported_actions", None) or []
            if not actions or "generateContent" in actions:
                names.append(name.replace("models/", ""))
        return names

    def extract(self, resume_text, job_title, job_skills, min_experience, candidate_name=None):
        if not self.available():
            return None

        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=RUBRIC + "\n\n" + build_job_block(
                job_title, job_skills, min_experience
            ),
            response_mime_type="application/json",
            response_schema=gemini_schema(EXTRACTION_SCHEMA),
            max_output_tokens=8192,
            temperature=0,  # deterministic extraction: same resume, same facts
        )
        prompt = "RESUME\n\n{}".format(redact_pii(resume_text, candidate_name))

        last_error = None

        for model_name in _usable_models(self.model):
            try:
                response = self.client().models.generate_content(
                    model=model_name, contents=prompt, config=config
                )
            except Exception as exc:
                message = str(exc)
                last_error = exc

                # A bad key fails identically on every model - stop immediately.
                if any(token in message for token in
                       ("API_KEY_INVALID", "PERMISSION_DENIED", "UNAUTHENTICATED")):
                    print("[LLM] Gemini rejected the API key - falling back to local scoring")
                    return None

                # Overloaded or out of quota: park this model and move on rather
                # than sleeping. Sleeping here cost ~30s per resume and never
                # helped, because an exhausted daily quota does not clear in 15s.
                if any(token in message for token in
                       ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")):
                    _put_on_cooldown(model_name)
                    print("[LLM] {} unavailable ({}), skipping it for {}s".format(
                        model_name,
                        "quota" if "429" in message or "RESOURCE_EXHAUSTED" in message
                        else "overloaded",
                        MODEL_COOLDOWN_SECONDS))
                    continue

                print("[LLM] {} -> {}".format(model_name, message.split(chr(10))[0][:140]))
                _put_on_cooldown(model_name)
                continue

            _clear_cooldown(model_name)
            if model_name != self.model:
                print("[LLM] using Gemini model '{}'".format(model_name))
                self.model = model_name   # sticky: try this one first next time

            self._last_usage = getattr(response, "usage_metadata", None)
            text = getattr(response, "text", None)
            if not text:
                print("[LLM] empty response from Gemini")
                return None
            try:
                return json.loads(text)
            except ValueError as exc:
                print("[LLM] could not parse Gemini JSON: {}".format(exc))
                return None

        print("[LLM] every Gemini model is unavailable: {}".format(
            str(last_error)[:160] if last_error else "unknown"))
        return None


class AnthropicExtractor(BaseExtractor):
    """Anthropic / Claude."""

    provider = "claude"

    def __init__(self, model=None, api_key=None):
        BaseExtractor.__init__(self)
        self.model = model or os.environ.get("RESUME_EXTRACTION_MODEL") or DEFAULT_ANTHROPIC_MODEL
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def available(self):
        if not self.api_key:
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def extract(self, resume_text, job_title, job_skills, min_experience, candidate_name=None):
        if not self.available():
            return None

        job_block = build_job_block(job_title, job_skills, min_experience)
        try:
            response = self.client().messages.create(
                model=self.model,
                max_tokens=16000,
                # Two cached blocks: the rubric never changes, and the job block
                # is identical for every applicant to the same job. Only the
                # resume below varies, so most of each request is a cache read.
                system=[
                    {"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": job_block, "cache_control": {"type": "ephemeral"}},
                ],
                messages=[
                    {
                        "role": "user",
                        "content": "RESUME\n\n{}".format(
                            redact_pii(resume_text, candidate_name)
                        ),
                    }
                ],
                thinking={"type": "adaptive"},
                output_config={
                    "effort": "low",
                    "format": {"type": "json_schema", "schema": EXTRACTION_SCHEMA},
                },
            )
        except Exception as exc:
            print("[LLM] extraction failed: {}: {}".format(type(exc).__name__, exc))
            return None

        if getattr(response, "stop_reason", None) == "refusal":
            print("[LLM] request refused by safety classifier - falling back to local scoring")
            return None

        self._last_usage = getattr(response, "usage", None)
        try:
            text = next(b.text for b in response.content if b.type == "text")
            return json.loads(text)
        except (StopIteration, ValueError) as exc:
            print("[LLM] could not parse response: {}".format(exc))
            return None


# Back-compat: the original single-provider name.
LLMExtractor = AnthropicExtractor


def get_extractor():
    """The configured extractor: Gemini, then Claude, else an unavailable stub.

    An unavailable extractor makes score_resume() fall through to the fully
    offline pipeline, so the app never breaks for want of a key.
    """
    gemini = GeminiExtractor()
    if gemini.available():
        return gemini

    claude = AnthropicExtractor()
    if claude.available():
        return claude

    return gemini  # unavailable, but reports the provider we would prefer
