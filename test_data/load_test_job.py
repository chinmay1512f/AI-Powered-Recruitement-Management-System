"""Load the generated test resumes into the app as a real job with applicants.

Additive and reversible - it creates one job, eight candidate accounts and
eight applications, and touches nothing that already exists.

    python test_data/load_test_job.py            # create
    python test_data/load_test_job.py --remove   # undo

Login for any test candidate: username as printed, password  test1234
"""
import argparse
import os
import shutil
import sys
from datetime import date, datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESUME_DIR = os.path.join(ROOT, "test_data", "resumes")
JOB_TITLE = "Software Engineer (Test Set)"
PASSWORD = "test1234"

# file stem -> (display name, email)
CANDIDATES = [
    ("01_meera_krishnan", "Meera Krishnan", "meera.k@example.com"),
    ("02_arjun_desai", "Arjun Desai", "arjun.d@example.com"),
    ("03_rohit_bhalla", "Rohit Bhalla", "rohit.b@example.com"),
    ("04_priya_raghavan", "Priya Raghavan", "priya.r@example.com"),
    ("05_sameer_qureshi", "Sameer Qureshi", "sameer.q@example.com"),
    ("06_ananya_bose", "Ananya Bose", "ananya.b@example.com"),
    ("07_nikhil_pawar", "Nikhil Pawar", "nikhil.p@example.com"),
    ("08_ritika_chandra", "Ritika Chandra", "ritika.c@example.com"),
    ("09_devendra_shetty", "Devendra Shetty", "devendra.s@example.com"),
    ("10_farah_siddiqui", "Farah Siddiqui", "farah.s@example.com"),
]


def username_for(stem):
    return "test_" + stem.split("_", 1)[1]


def remove(A):
    with A.app.app_context():
        job = A.JobAnalysis.query.filter_by(job_title=JOB_TITLE).first()
        if job:
            A.CandidateResult.query.filter_by(job_id=job.id).delete()
            A.JobApplication.query.filter_by(job_id=job.id).delete()
            A.InterviewSchedule.query.filter_by(job_id=job.id).delete()
            A.db.session.delete(job)
        for stem, _, _ in CANDIDATES:
            user = A.CandidateUser.query.filter_by(username=username_for(stem)).first()
            if user:
                A.JobApplication.query.filter_by(candidate_id=user.id).delete()
                A.CandidateResult.query.filter_by(candidate_id=user.id).delete()
                A.db.session.delete(user)
        A.db.session.commit()
    for name in os.listdir(os.path.join(ROOT, "uploads")):
        if name.startswith("test_"):
            os.remove(os.path.join(ROOT, "uploads", name))
    print("Removed the test job, its candidates, applications and uploaded files.")


def create(A):
    if not os.path.isdir(RESUME_DIR):
        sys.exit("No resumes found. Run: python test_data/generate_resumes.py")

    from werkzeug.security import generate_password_hash

    uploads = os.path.join(ROOT, "uploads")
    os.makedirs(uploads, exist_ok=True)

    with A.app.app_context():
        job = A.JobAnalysis.query.filter_by(job_title=JOB_TITLE).first()
        if job:
            sys.exit("The test job already exists. Run with --remove first.")

        job = A.JobAnalysis(
            job_title=JOB_TITLE,
            required_skills="data structures, algorithms, databases, Python, Java",
            min_experience=2,
            deadline=date.today() + timedelta(days=30),
            status="Active",
        )
        A.db.session.add(job)
        A.db.session.commit()
        print("Created job #{}: {}".format(job.id, JOB_TITLE))

        stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        for stem, display_name, email in CANDIDATES:
            source = os.path.join(RESUME_DIR, stem + ".pdf")
            if not os.path.exists(source):
                print("  ! missing {} - skipped".format(source))
                continue

            user = A.CandidateUser(
                name=display_name,
                phone="9000000000",
                email=email,
                username=username_for(stem),
                password_hash=generate_password_hash(PASSWORD),
            )
            A.db.session.add(user)
            A.db.session.commit()

            stored = "test_c{}_j{}_{}_{}.pdf".format(user.id, job.id, stamp, stem)
            shutil.copyfile(source, os.path.join(uploads, stored))

            A.db.session.add(A.JobApplication(
                job_id=job.id,
                candidate_id=user.id,
                candidate_name=display_name,
                resume_file=stored,
            ))
            A.db.session.commit()
            print("  + {:<18} username: {}".format(display_name, user.username))

        print("\nAll candidate logins use the password: {}".format(PASSWORD))
        print("Open /hr/sort/{} to rank them.".format(job.id))
        print("Evaluate with:  python eval/rank_eval.py --job {} --compare".format(job.id))
        return job.id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove", action="store_true", help="undo everything this script created")
    args = parser.parse_args()

    import app as A

    if args.remove:
        remove(A)
    else:
        create(A)


if __name__ == "__main__":
    main()
