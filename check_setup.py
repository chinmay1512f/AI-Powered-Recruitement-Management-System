"""Check that everything this project needs is configured and working.

Run it any time, especially after editing .env or before a demo:

    python check_setup.py

It never prints your passwords or API key - only whether they work.
"""
import os
import smtplib
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

OK, WARN, FAIL = "  [ OK ]", "  [WARN]", "  [FAIL]"
problems = []


def report(status, label, detail=""):
    print("{} {}{}".format(status, label, ("  - " + detail) if detail else ""))
    if status == FAIL:
        problems.append(label)


def check_core():
    print("\nCore application")
    try:
        import app  # noqa: F401
        report(OK, "app.py imports and database is reachable")
    except Exception as exc:
        report(FAIL, "app.py failed to import", str(exc))
        return None

    import app as A
    with A.app.app_context():
        hr = A.HRUser.query.first()
        report(OK if hr else FAIL, "HR account exists",
               "username: {}".format(hr.username) if hr else "run the app once to seed it")
        report(OK, "jobs in database", str(A.JobAnalysis.query.count()))
        report(OK, "candidates in database", str(A.CandidateUser.query.count()))

    if os.environ.get("FLASK_SECRET_KEY"):
        report(OK, "FLASK_SECRET_KEY set", "logins survive a restart")
    else:
        report(WARN, "FLASK_SECRET_KEY missing",
               "a new key is generated each start, so restarting logs everyone out")
    return A


def check_offline_model():
    print("\nOffline scoring")
    try:
        from ai_engine.semantic_matcher import model, MAX_TOKENS
        report(OK, "embedding model loaded", "{} token window".format(MAX_TOKENS))
    except Exception as exc:
        report(FAIL, "embedding model failed to load", str(exc))

    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        report(OK, "Tesseract OCR available", "scanned PDFs will be readable")
    except Exception:
        report(WARN, "Tesseract OCR not available",
               "text PDFs still work; scanned/image resumes will not")


def check_mail(A):
    print("\nEmail (accept / reject / interview notifications)")
    user = os.environ.get("MAIL_USERNAME")
    password = os.environ.get("MAIL_PASSWORD")

    if not user or not password:
        report(WARN, "email not configured",
               "decisions still work, but the UI will show 'Failed to send email'")
        print("         set MAIL_USERNAME and MAIL_PASSWORD in .env to enable it")
        return

    try:
        server = smtplib.SMTP(A.app.config["MAIL_SERVER"], A.app.config["MAIL_PORT"], timeout=15)
        server.starttls()
        server.login(user, password)
        server.quit()
        report(OK, "SMTP login succeeded", "emails will send from {}".format(user))
    except smtplib.SMTPAuthenticationError:
        report(FAIL, "SMTP login rejected",
               "wrong address or app password - use a 16-character Google App Password, not your normal one")
    except Exception as exc:
        report(FAIL, "could not reach the mail server", str(exc))


def check_llm():
    print("\nLLM extraction (optional)")
    try:
        from ai_engine.llm_extractor import get_extractor, GeminiExtractor
    except Exception as exc:
        report(FAIL, "llm_extractor failed to import", str(exc))
        return

    extractor = get_extractor()
    if not extractor.available():
        if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
            report(WARN, "no API key set",
                   "the app runs fully offline; skills scored by keyword matching only")
            print("         set GEMINI_API_KEY in .env (free tier) to enable it")
        else:
            report(FAIL, "API key set but the SDK is missing",
                   "run: pip install google-genai   (or: pip install anthropic)")
        return

    report(OK, "{} extractor configured".format(extractor.provider),
           "model: {}".format(extractor.model))

    if isinstance(extractor, GeminiExtractor):
        try:
            names = extractor.list_models()
        except Exception as exc:
            report(FAIL, "the Gemini key was rejected", str(exc)[:120])
            return
        flash = [n for n in names if "flash" in n and "vision" not in n][:6]
        report(OK, "Gemini key works", "{} models reachable".format(len(names)))
        if flash:
            print("         usable models: {}".format(", ".join(flash)))
        if extractor.model not in names:
            report(WARN, "configured model not in that list",
                   "'{}' may fall back at run time".format(extractor.model))
    else:
        print("         key is NOT verified here - a real call costs money.")

    print("         to test extraction:  python eval/rank_eval.py --job 1 --scorer llm")


def main():
    print("=" * 64)
    print("  Resume Intelligence System - setup check")
    print("=" * 64)

    A = check_core()
    check_offline_model()
    if A:
        check_mail(A)
    check_llm()

    print("\n" + "=" * 64)
    if problems:
        print("  {} problem(s) need attention: {}".format(len(problems), ", ".join(problems)))
    else:
        print("  No blocking problems. Anything marked WARN is optional.")
    print("=" * 64 + "\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
