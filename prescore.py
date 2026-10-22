"""Score every pending candidate BEFORE a demo, so the UI is instant during it.

Scores are cached in the database, so this is a one-time cost per candidate.
Run it 10 minutes before you present and every results page opens immediately.

    python prescore.py              # score all active jobs
    python prescore.py --job 5      # score one job
    python prescore.py --offline    # ignore the API key, use local scoring only
                                    # (about 0.2s per resume - use if quota is gone)
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=int, help="only this job id")
    parser.add_argument("--offline", action="store_true",
                        help="force local scoring, no API calls")
    args = parser.parse_args()

    if args.offline:
        # Must happen before app is imported, so the extractor sees no key.
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ["SKIP_LLM"] = "1"

    import app as A

    if args.offline:
        A.os.environ.pop("GEMINI_API_KEY", None)
        A.os.environ.pop("ANTHROPIC_API_KEY", None)
        A._llm_extractor = None

    with A.app.app_context():
        jobs = ([A.JobAnalysis.query.get(args.job)] if args.job
                else A.JobAnalysis.query.all())
        jobs = [j for j in jobs if j is not None]

        if not jobs:
            print("No jobs found.")
            return 1

        extractor = A.get_llm_extractor()
        mode = "{} ({})".format(extractor.provider, extractor.model) \
            if extractor.available() else "local / offline"
        print("Scoring mode: {}\n".format(mode))

        grand_total = 0
        started = time.time()

        for job in jobs:
            outstanding = A.applications_needing_scoring(job.id)
            if not outstanding:
                print("  job {:<3} {:<34} already scored".format(job.id, job.job_title[:34]))
                continue

            print("  job {:<3} {:<34} scoring {} resume(s)...".format(
                job.id, job.job_title[:34], len(outstanding)))
            t = time.time()
            A.run_scoring_job(job.id)
            progress = A.get_progress(job.id)

            if progress.get("state") == "error":
                print("        FAILED: {}".format(progress.get("message")))
            else:
                done = progress.get("done", 0)
                grand_total += done
                elapsed = time.time() - t
                print("        done: {} scored in {:.1f}s ({:.1f}s each)".format(
                    done, elapsed, elapsed / max(done, 1)))

        print("\n{} candidate(s) scored in {:.1f}s. Every results page is now instant.".format(
            grand_total, time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
