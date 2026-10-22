"""End-to-end smoke test. Run it after any change:

    python tests/smoke_test.py

Covers auth, CSRF, POST-only mutations, upload naming, duplicate applications,
path traversal, page rendering and the shortlist PDF. Creates a temporary job
and candidate and removes them again, so it is safe to run against real data.
"""
import io
import os
import sys
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import app as A  # noqa: E402

failures = []


def check(label, condition, detail=""):
    print("  [{}] {}{}".format("PASS" if condition else "FAIL", label,
                               ("  <- " + detail) if detail and not condition else ""))
    if not condition:
        failures.append(label)


def token(client):
    with client.session_transaction() as sess:
        return sess.get("csrf_token")


A.app.config["TESTING"] = True

print("\n== 1. public pages ==")
with A.app.test_client() as c:
    check("GET /", c.get("/").status_code == 200)
    check("GET /admin-login", c.get("/admin-login").status_code == 200)
    check("GET /candidate-login", c.get("/candidate-login").status_code == 200)
    check("GET /candidate-register", c.get("/candidate-register").status_code == 200)
    check("HR dashboard redirects when logged out", c.get("/hr-dashboard").status_code == 302)

print("\n== 2. CSRF enforcement ==")
with A.app.test_client() as c:
    c.get("/admin-login")
    resp = c.post("/admin-login", data={"username": "hradmin", "password": "admin123"})
    check("tokenless login POST rejected", resp.status_code == 302)
    with c.session_transaction() as sess:
        check("no session from tokenless POST", not sess.get("hr_logged_in"))

print("\n== 3. HR login ==")
hr = A.app.test_client()
hr.get("/admin-login")
tok = token(hr)
check("csrf token issued", bool(tok))
hr.post("/admin-login", data={"username": "hradmin", "password": "admin123", "csrf_token": tok})
with hr.session_transaction() as sess:
    check("HR session established", bool(sess.get("hr_logged_in")))

body = hr.get("/hr-dashboard").get_data(as_text=True)
check("dashboard renders", "HR Dashboard" in body)
check("no GET delete links", 'href="/hr/delete-job/' not in body)

print("\n== 4. state-changing routes reject GET ==")
for url in ("/hr/close-job/999", "/hr/delete-job/999", "/hr/decision/999/accept",
            "/hr/cancel-interview/999", "/hr/complete-interview/999"):
    check("GET {} -> 405".format(url), hr.get(url).status_code == 405)

print("\n== 5. HR pages render ==")
with A.app.app_context():
    job = A.JobAnalysis.query.first()
    job_id = job.id if job else None
    result = A.CandidateResult.query.first()
    result_id = result.id if result else None

if job_id:
    for path in ("/hr/sort/{}".format(job_id), "/hr/applications/{}".format(job_id),
                 "/create-job", "/hr/interviews"):
        check("GET {}".format(path), hr.get(path).status_code == 200)
if result_id:
    check("GET /hr/skill-gap/{}".format(result_id),
          hr.get("/hr/skill-gap/{}".format(result_id)).status_code == 200)

print("\n== 6. shortlist PDF ==")
if job_id:
    with A.app.app_context():
        pending = A.CandidateResult.query.filter_by(job_id=job_id).first()
        restore = None
        if pending:
            restore = (pending.id, pending.status)
            pending.status = "Accepted"
            A.db.session.commit()

    resp = hr.get("/hr/shortlist/{}".format(job_id))
    if restore:
        check("returns a PDF", resp.status_code == 200 and resp.get_data()[:5] == b"%PDF-")
        check("sent as attachment", "attachment" in resp.headers.get("Content-Disposition", ""))
        with A.app.app_context():
            row = A.CandidateResult.query.get(restore[0])
            row.status = restore[1]
            A.db.session.commit()
    else:
        check("no candidates - redirects", resp.status_code == 302)

print("\n== 7. candidate registration and login ==")
cand = A.app.test_client()
cand.get("/candidate-register")
tok = token(cand)
uname = "smoketest_user"
with A.app.app_context():
    old = A.CandidateUser.query.filter_by(username=uname).first()
    if old:
        A.JobApplication.query.filter_by(candidate_id=old.id).delete()
        A.CandidateResult.query.filter_by(candidate_id=old.id).delete()
        A.db.session.delete(old)
        A.db.session.commit()

cand.post("/candidate-register", data={
    "name": "Smoke Test", "phone": "9999999999", "email": "smoke@example.com",
    "username": uname, "password": "pw12345", "confirm_password": "pw12345",
    "csrf_token": tok})
with A.app.app_context():
    check("account created", A.CandidateUser.query.filter_by(username=uname).first() is not None)

cand.get("/candidate-login")
tok = token(cand)
cand.post("/candidate-login", data={"username": uname, "password": "pw12345", "csrf_token": tok})
with cand.session_transaction() as sess:
    check("candidate session established", bool(sess.get("candidate_logged_in")))
    cand_id = sess.get("candidate_id")
check("candidate dashboard renders", cand.get("/candidate-dashboard").status_code == 200)

print("\n== 8. upload naming and duplicate guard ==")
with A.app.app_context():
    temp_job = A.JobAnalysis(job_title="SMOKE TEST JOB", required_skills="python, sql",
                             min_experience=0, deadline=date.today() + timedelta(days=7),
                             status="Active")
    A.db.session.add(temp_job)
    A.db.session.commit()
    temp_job_id = temp_job.id

before = set(os.listdir("uploads"))
cand.get("/apply/{}".format(temp_job_id))
tok = token(cand)
cand.post("/apply/{}".format(temp_job_id),
          data={"csrf_token": tok,
                "resume": (io.BytesIO(b"%PDF-1.4 smoke"), "resume.pdf")},
          content_type="multipart/form-data")
new_files = set(os.listdir("uploads")) - before
check("one file stored", len(new_files) == 1, str(new_files))
if new_files:
    stored = new_files.pop()
    check("filename namespaced", stored.startswith("c{}_j{}_".format(cand_id, temp_job_id)), stored)
    os.remove(os.path.join("uploads", stored))

cand.get("/apply/{}".format(temp_job_id))
tok = token(cand)
before = set(os.listdir("uploads"))
cand.post("/apply/{}".format(temp_job_id),
          data={"csrf_token": tok,
                "resume": (io.BytesIO(b"%PDF-1.4 again"), "resume.pdf")},
          content_type="multipart/form-data")
check("duplicate application blocked", set(os.listdir("uploads")) == before)

print("\n== 9. filename safety ==")
name = A.build_resume_filename("../../evil.pdf", 1, 1)
check("traversal stripped", name and "/" not in name and "\\" not in name, str(name))
check("bad extension rejected", A.build_resume_filename("payload.exe", 1, 1) is None)

print("\n== 10. cleanup ==")
with A.app.app_context():
    user = A.CandidateUser.query.filter_by(username=uname).first()
    if user:
        A.JobApplication.query.filter_by(candidate_id=user.id).delete()
        A.CandidateResult.query.filter_by(candidate_id=user.id).delete()
        A.db.session.delete(user)
    A.JobApplication.query.filter_by(job_id=temp_job_id).delete()
    A.CandidateResult.query.filter_by(job_id=temp_job_id).delete()
    temp = A.JobAnalysis.query.get(temp_job_id)
    if temp and temp.job_title == "SMOKE TEST JOB":
        A.db.session.delete(temp)
    A.db.session.commit()
    print("  temporary job and candidate removed")

print("\n" + ("ALL CHECKS PASSED" if not failures else "FAILURES: {}".format(failures)))
sys.exit(1 if failures else 0)
