"""Measure how well the ranker orders candidates against human judgement.

Usage:
    python eval/rank_eval.py                 # job 1, local scorer
    python eval/rank_eval.py --job 1
    python eval/rank_eval.py --scorer llm    # needs ANTHROPIC_API_KEY (phase 3)
    python eval/rank_eval.py --compare       # run both and show the delta

Metrics:
    Spearman rho  - rank correlation with the human ordering (-1..1, higher is better)
    NDCG          - rewards putting genuinely strong candidates near the top (0..1)
    Top-K overlap - how many of the human's top K the system also puts in its top K
"""
import argparse
import json
import math
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DB = os.path.join(ROOT, "resume_system.db")
GROUND_TRUTH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth.json")


def load_ground_truth(job_id):
    with open(GROUND_TRUTH, encoding="utf-8") as fh:
        data = json.load(fh)
    entry = data.get(str(job_id))
    if not entry:
        sys.exit(
            "No ground truth for job {}. Add a '{}' entry to eval/ground_truth.json.".format(
                job_id, job_id
            )
        )
    return {name.strip(): grade for name, grade in entry["relevance"].items()}, entry


def load_applications(job_id):
    conn = sqlite3.connect(DB)
    job = conn.execute(
        "SELECT job_title, required_skills, min_experience FROM job_analysis WHERE id=?",
        (job_id,),
    ).fetchone()
    if not job:
        sys.exit("No job with id {}".format(job_id))
    rows = conn.execute(
        "SELECT candidate_name, resume_file FROM job_application "
        "WHERE job_id=? AND status='Active'",
        (job_id,),
    ).fetchall()
    conn.close()
    return job, [(name.strip(), rf) for name, rf in rows]


def score_all(job, applications, scorer):
    """Return [(candidate_name, score)] using the requested scorer."""
    from app import build_job_text, calculate_final_score_improved, parse_job_skills
    from resume_parser.extract_text import extract_text

    title, skills_raw, min_exp = job
    job_skills = parse_job_skills(skills_raw)
    # Both scorers must see byte-identical job text, or the semantic component
    # differs for reasons unrelated to extraction quality.
    job_text = build_job_text(title, job_skills, min_exp)

    extractor = None
    if scorer == "llm":
        from ai_engine.llm_extractor import get_extractor

        extractor = get_extractor()
        if not extractor.available():
            sys.exit(
                "LLM scorer unavailable: set GEMINI_API_KEY (or ANTHROPIC_API_KEY) "
                "in .env and install the matching SDK."
            )
        print("Using {} extractor: {}".format(extractor.provider, extractor.model))

    results = []
    fallbacks = 0
    for name, resume_file in applications:
        path = os.path.join(ROOT, "uploads", resume_file)
        if not os.path.exists(path):
            print("  skipping {} - missing file {}".format(name, resume_file))
            continue
        text = extract_text(path)

        if extractor is not None:
            score, breakdown = extractor.score(text, title, job_skills, min_exp)
            # score() falls back to local scoring when a call fails. Without this
            # check, a run reports local numbers under the "llm" label.
            if str(breakdown.get("scored_by", "")).startswith("local"):
                fallbacks += 1
        else:
            score, _ = calculate_final_score_improved(
                text, job_text, job_skills, min_exp, title
            )
        results.append((name, round(float(score), 2)))

    if extractor is not None and fallbacks:
        print("\n  !! {} of {} candidates fell back to LOCAL scoring - the LLM call "
              "failed for them.".format(fallbacks, len(results)))
        print("     These results are NOT a valid measure of the LLM scorer.\n")

    results.sort(key=lambda pair: pair[1], reverse=True)
    return results, fallbacks


def spearman(system_order, truth):
    """Rank correlation, with average ranks for ties on the human side."""
    from scipy.stats import spearmanr

    names = [name for name, _ in system_order if name in truth]
    system_ranks = [i + 1 for i, _ in enumerate(names)]
    human_grades = [truth[name] for name in names]
    # higher grade = better, so negate to align with "rank 1 is best"
    rho, p = spearmanr(system_ranks, [-g for g in human_grades])
    return rho, p


def ndcg(system_order, truth):
    gains = [truth.get(name, 0) for name, _ in system_order]
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(truth.values(), reverse=True)
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def top_k_overlap(system_order, truth, k=3):
    system_top = {name for name, _ in system_order[:k]}
    human_top = {name for name, _ in sorted(truth.items(), key=lambda kv: -kv[1])[:k]}
    return len(system_top & human_top), k


def report(label, system_order, truth):
    rho, p = spearman(system_order, truth)
    n = ndcg(system_order, truth)
    hit, k = top_k_overlap(system_order, truth)

    human_order = sorted(truth.items(), key=lambda kv: -kv[1])

    print("\n  {}".format(label))
    print("  " + "-" * 62)
    print("  {:<4} {:<24} {:>7}   {:<24}".format("#", "system ranking", "score", "human ranking"))
    for i in range(max(len(system_order), len(human_order))):
        sys_name, sys_score = system_order[i] if i < len(system_order) else ("", "")
        hum_name = "{} (grade {})".format(*human_order[i]) if i < len(human_order) else ""
        score_text = "{:>7}".format(sys_score) if sys_score != "" else " " * 7
        print("  {:<4} {:<24} {}   {:<24}".format(i + 1, sys_name[:24], score_text, hum_name))
    print("  " + "-" * 62)
    print("  Spearman rho : {:+.3f}  (p={:.3f})".format(rho, p))
    print("  NDCG         :  {:.3f}".format(n))
    print("  Top-{} overlap:  {}/{}".format(k, hit, k))
    return {"spearman": rho, "ndcg": n, "top_k": hit}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=int, default=1)
    parser.add_argument("--scorer", choices=["local", "llm"], default="local")
    parser.add_argument("--compare", action="store_true", help="run local and llm, show the delta")
    args = parser.parse_args()

    truth, entry = load_ground_truth(args.job)
    job, applications = load_applications(args.job)

    print("\nJob {}: {}".format(args.job, entry.get("job_title", job[0])))
    print("Requirements: {}".format(entry.get("requirements", job[1])))
    print("{} applications, {} labelled candidates".format(len(applications), len(truth)))

    scorers = ["local", "llm"] if args.compare else [args.scorer]
    metrics = {}
    fallback_counts = {}
    for scorer in scorers:
        order, fallbacks = score_all(job, applications, scorer)
        fallback_counts[scorer] = fallbacks
        metrics[scorer] = report("scorer = {}".format(scorer), order, truth)

    if args.compare and len(metrics) == 2:
        if fallback_counts.get("llm"):
            print("\n  DELTA NOT REPORTED - {} LLM call(s) failed and fell back to local "
                  "scoring,".format(fallback_counts["llm"]))
            print("  so the two tables above are not actually different scorers.")
        else:
            d_rho = metrics["llm"]["spearman"] - metrics["local"]["spearman"]
            d_ndcg = metrics["llm"]["ndcg"] - metrics["local"]["ndcg"]
            print("\n  DELTA (llm - local): Spearman {:+.3f}   NDCG {:+.3f}".format(d_rho, d_ndcg))

    print("")


if __name__ == "__main__":
    main()
