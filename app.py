from resume_parser.extract_text import extract_text
from ai_engine.semantic_matcher import match_resume
from ai_engine.skills import extract_skills
from ai_engine.education import extract_education
from ai_engine.experience import extract_experience
from ai_engine.llm_extractor import get_extractor
from concurrent.futures import ThreadPoolExecutor
import threading

from datetime import datetime, date, timedelta
from flask import (Flask, render_template, request, redirect, session, flash,
                   send_from_directory, send_file, jsonify)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import secrets as pysecrets
from sqlalchemy import exists, and_
from sqlalchemy.orm import joinedload
from flask_mail import Mail, Message
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import fitz
import io
import os
import re
import json

# -------------------- APP CONFIG --------------------

def load_env(path=None):
    """Minimal .env reader so secrets stay out of source. No external dependency."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or pysecrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload ceiling

# -------------------- EMAIL CONFIGURATION --------------------
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = (
    os.environ.get('MAIL_SENDER_NAME', 'HR Team'),
    os.environ.get('MAIL_USERNAME') or 'no-reply@localhost'
)
MAIL_CONFIGURED = bool(app.config['MAIL_USERNAME'] and app.config['MAIL_PASSWORD'])

mail = Mail(app)

# -------------------- DATABASE CONFIG (SQLITE) --------------------

basedir = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(basedir, "resume_system.db")

# Absolute, so the app behaves the same whatever directory it is started from.
UPLOAD_DIR = os.path.join(basedir, "uploads")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# -------------------- DATABASE MODELS --------------------

class HRUser(db.Model):
    __tablename__ = "hr_user"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)


class CandidateUser(db.Model):
    __tablename__ = "candidate_user"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class JobAnalysis(db.Model):
    __tablename__ = "job_analysis"
    id = db.Column(db.Integer, primary_key=True)
    job_title = db.Column(db.String(200))
    required_skills = db.Column(db.Text)
    min_experience = db.Column(db.Integer)
    deadline = db.Column(db.Date)
    status = db.Column(db.String(20), default="Active")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class JobApplication(db.Model):
    __tablename__ = "job_application"
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey("job_analysis.id"), nullable=False)
    candidate_id = db.Column(db.Integer, db.ForeignKey("candidate_user.id"), nullable=False)
    candidate_name = db.Column(db.String(150), nullable=False)
    resume_file = db.Column(db.String(300), nullable=False)
    applied_at = db.Column(db.DateTime, default=datetime.utcnow)
    withdrawn_at = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(20), default="Active")


class CandidateResult(db.Model):
    __tablename__ = "candidate_result"
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey("job_analysis.id"))
    candidate_id = db.Column(db.Integer, db.ForeignKey("candidate_user.id"), nullable=False)
    candidate_name = db.Column(db.String(200))
    score = db.Column(db.Float)
    skills = db.Column(db.Text)
    education = db.Column(db.Text)
    experience = db.Column(db.Integer)
    resume_file = db.Column(db.String(300))
    status = db.Column(db.String(20), default="Pending")
    breakdown_json = db.Column(db.Text)
    scored_by = db.Column(db.String(20), default="local")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class InterviewSchedule(db.Model):
    __tablename__ = "interview_schedule"
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey("job_analysis.id"), nullable=False)
    candidate_id = db.Column(db.Integer, db.ForeignKey("candidate_user.id"), nullable=False)
    candidate_name = db.Column(db.String(200), nullable=False)
    candidate_email = db.Column(db.String(150), nullable=False)
    interview_date = db.Column(db.DateTime, nullable=False)
    interview_type = db.Column(db.String(50), default="Video Call")
    interview_link = db.Column(db.String(500), nullable=True)
    location = db.Column(db.String(300), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), default="Scheduled")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    sent_at = db.Column(db.DateTime, nullable=True)
    
    job = db.relationship('JobAnalysis', backref='interviews')

# -------------------- ENHANCED SKILL DATABASE --------------------

SKILL_DATABASE = {
    'technical': [
        'python', 'java', 'javascript', 'typescript', 'react', 'node', 'nodejs', 'sql', 'mysql', 'postgresql',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'github', 'gitlab', 'linux', 'bash',
        'html', 'css', 'sass', 'scss', 'bootstrap', 'tailwind', 'jquery', 'ajax', 'json', 'xml',
        'mongodb', 'redis', 'elasticsearch', 'kafka', 'rabbitmq', 'nginx', 'apache',
        'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn',
        'django', 'flask', 'fastapi', 'spring', 'springboot', 'hibernate', 'express', 'next.js', 'nuxt.js',
        'php', 'laravel', 'symfony', 'ruby', 'rails', 'go', 'golang', 'rust', 'c', 'c++', 'csharp', '.net',
        'swift', 'kotlin', 'flutter', 'react native', 'ionic', 'cordova',
        'ansible', 'terraform', 'jenkins', 'circleci', 'travis', 'github actions', 'gitlab ci',
        'prometheus', 'grafana', 'elk stack', 'splunk', 'new relic', 'datadog',
        'graphql', 'rest api', 'soap', 'grpc', 'websocket', 'oauth', 'jwt', 'sso', 'ldap',
        'cicd', 'devops', 'microservices', 'serverless', 'lambda', 'ecs', 'eks', 'gke'
    ],
    'soft': ['communication', 'leadership', 'teamwork', 'problem solving', 'critical thinking', 'time management', 'adaptability', 'creativity', 'emotional intelligence'],
    'domain': ['machine learning', 'deep learning', 'data science', 'web development', 'cloud computing', 'devops', 'cybersecurity', 'blockchain', 'iot', 'big data', 'nlp', 'computer vision'],
    'alternatives': {
        'react': ['reactjs', 'react.js', 'react native', 'next.js', 'gatsby', 'preact', 'frontend', 'frontend development', 'ui development'],
        'python': ['django', 'flask', 'fastapi', 'python3', 'py', 'data analysis', 'scripting', 'automation'],
        'aws': ['amazon web services', 'ec2', 's3', 'rds', 'lambda', 'cloudformation', 'cloudfront', 'route53', 'azure', 'gcp', 'cloud infrastructure', 'cloud platforms'],
        'docker': ['containerization', 'containers', 'kubernetes', 'k8s', 'container orchestration', 'podman', 'cri-o'],
        'sql': ['mysql', 'postgresql', 'sqlite', 'database', 'rdbms', 'relational database', 'nosql', 'mongodb', 'query optimization', 'database design'],
        'git': ['github', 'gitlab', 'bitbucket', 'version control', 'source control', 'svn', 'mercurial'],
        'javascript': ['js', 'es6', 'es7', 'typescript', 'ts', 'node', 'nodejs', 'frontend', 'backend'],
        'machine learning': ['ml', 'deep learning', 'ai', 'artificial intelligence', 'tensorflow', 'pytorch', 'scikit-learn', 'predictive modeling', 'neural networks'],
        'node': ['nodejs', 'node.js', 'express', 'npm', 'backend javascript'],
        'kubernetes': ['k8s', 'container orchestration', 'docker swarm', 'openshift', 'rancher'],
        'devops': ['cicd', 'continuous integration', 'continuous deployment', 'infrastructure as code', 'iac', 'site reliability', 'sre'],
        'database': ['sql', 'mysql', 'postgresql', 'postgres', 'mongodb', 'sqlite', 'oracle', 'rdbms', 'nosql', 'dbms', 'redis', 'dynamodb', 'schema design', 'query optimization', 'stored procedures'],
        'algorithm': ['leetcode', 'competitive programming', 'dynamic programming', 'time complexity', 'big o', 'sorting', 'searching', 'graph algorithms', 'codeforces'],
        'data structure': ['linked list', 'binary tree', 'hash map', 'hashmap', 'graph traversal', 'stack', 'queue', 'heap', 'trie', 'leetcode', 'bst'],
        'rest api': ['rest', 'restful', 'api development', 'endpoints', 'http api'],
        'frontend': ['html', 'css', 'javascript', 'react', 'vue', 'angular', 'ui development']
    },
    'learning_paths': {
        'python': {
            'beginner': ['Python for Everybody (Coursera)', 'Automate the Boring Stuff', 'Codecademy Python'],
            'intermediate': ['Django for Beginners', 'Flask Web Development', 'Python Testing'],
            'advanced': ['Advanced Python Patterns', 'System Design with Python', 'ML with Python']
        },
        'react': {
            'beginner': ['React Official Docs', 'Scrimba React Course', 'React for Beginners'],
            'intermediate': ['Advanced React Patterns', 'React Performance', 'Testing React Apps'],
            'advanced': ['React Architecture', 'React Native', 'Next.js Mastery']
        },
        'aws': {
            'beginner': ['AWS Cloud Practitioner', 'AWS Free Tier Labs', 'Cloud Computing Basics'],
            'intermediate': ['AWS Solutions Architect', 'AWS Developer Associate', 'Serverless Architecture'],
            'advanced': ['AWS DevOps Engineer', 'AWS Security Specialty', 'Multi-Cloud Strategies']
        },
        'machine learning': {
            'beginner': ['ML Crash Course (Google)', 'Andrew Ng ML Course', 'Kaggle Learn'],
            'intermediate': ['Deep Learning Specialization', 'ML Engineering', 'MLOps'],
            'advanced': ['Advanced ML (MIT)', 'Research Papers Implementation', 'Production ML Systems']
        }
    },
    'learning_time': {
        'python': {'beginner': 40, 'intermediate': 100, 'expert': 300},
        'react': {'beginner': 60, 'intermediate': 150, 'expert': 400},
        'aws': {'beginner': 80, 'intermediate': 200, 'expert': 500},
        'machine learning': {'beginner': 100, 'intermediate': 300, 'expert': 800},
        'docker': {'beginner': 20, 'intermediate': 60, 'expert': 150},
        'sql': {'beginner': 30, 'intermediate': 80, 'expert': 200},
        'kubernetes': {'beginner': 40, 'intermediate': 120, 'expert': 300},
        'javascript': {'beginner': 40, 'intermediate': 100, 'expert': 250}
    },
    'priority': {
        'critical': ['python', 'machine learning', 'aws', 'sql', 'docker', 'kubernetes', 'react', 'javascript', 'git'],
        'important': ['typescript', 'node', 'java', 'golang', 'ci/cd', 'microservices', 'cloud computing', 'devops'],
        'nice_to_have': ['angular', 'vue', 'php', 'ruby', 'c++', 'rust', 'blockchain', 'iot']
    }
}

# -------------------- UPLOAD SAFETY --------------------

ALLOWED_RESUME_EXTENSIONS = {".pdf", ".docx", ".doc", ".png", ".jpg", ".jpeg"}


def build_resume_filename(original_name, candidate_id, job_id):
    """Return a collision-proof, path-safe name for an uploaded resume.

    The previous code saved under the raw browser-supplied filename, so two
    candidates who both uploaded 'resume.pdf' silently overwrote each other and
    ended up scored on the same document.
    """
    safe = secure_filename(original_name or "")
    if not safe:
        return None
    ext = os.path.splitext(safe)[1].lower()
    if ext not in ALLOWED_RESUME_EXTENSIONS:
        return None
    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return "c{}_j{}_{}_{}".format(candidate_id, job_id, stamp, safe)


# -------------------- CSRF PROTECTION --------------------

def get_csrf_token():
    token = session.get("csrf_token")
    if not token:
        token = pysecrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


@app.before_request
def csrf_protect():
    if request.method != "POST":
        return None
    submitted = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
    if not submitted or not pysecrets.compare_digest(
        str(submitted), str(session.get("csrf_token", ""))
    ):
        flash("Security check failed. Please try again.", "warning")
        return redirect(request.referrer or "/")
    return None


app.jinja_env.globals["csrf_token"] = get_csrf_token


@app.errorhandler(500)
def internal_error(error):
    """Never show a raw traceback to an HR user."""
    db.session.rollback()
    flash("Something went wrong on that action. It was not completed.", "warning")
    return redirect(request.referrer or "/hr-dashboard"), 302


@app.errorhandler(413)
def file_too_large(error):
    """MAX_CONTENT_LENGTH rejects the upload before the view runs, so without
    this a candidate sees a bare "Request Entity Too Large" page."""
    flash("That file is too large. Please upload a resume under 10 MB.", "warning")
    return redirect(request.referrer or "/candidate-dashboard"), 302


# -------------------- SKILL NORMALISATION --------------------

def normalize_skill(raw):
    """Trim punctuation that leaks in from comma-separated job postings.

    'Java.' -> 'java', ' C++ ' -> 'c++'. Without this a trailing period makes a
    skill impossible to match against any resume.
    """
    return raw.strip().strip(".,;:|/").strip().lower()


def skill_pattern(skill):
    r"""Build a regex matching a skill as a standalone token.

    \b fails for skills ending in a non-word character: r"\bc\+\+\b" never
    matches "c++" because there is no word boundary after the '+'. Explicit
    non-word lookarounds handle c++, .net, c# and node.js correctly.
    """
    return r"(?<![\w+#.])" + re.escape(skill) + r"(?![\w+#])"


def skill_variants(skill):
    """A skill and its singular/plural form.

    Job postings say "databases" and "algorithms"; resumes say "database" and
    "algorithm". Without this, the plural form matches nothing.
    """
    variants = [skill]
    if skill.endswith("s") and len(skill) > 3:
        variants.append(skill[:-1])
    else:
        variants.append(skill + "s")
    return variants


def match_skill_variant(skill, text):
    """Return the variant that matched, or None.

    Deliberately regex-only. The previous code fell back to a raw substring
    test (`skill in resume_lower`), which matched "java" inside "JavaScript",
    "c" inside almost any word, and "react" inside "reactive" - manufacturing
    skills the candidate never claimed.
    """
    for variant in skill_variants(skill):
        if re.search(skill_pattern(variant), text):
            return variant
    return None


def parse_job_skills(raw_skills):
    """Split a job's required_skills field into clean, matchable tokens."""
    if not raw_skills:
        return []
    return [s for s in (normalize_skill(part) for part in raw_skills.split(",")) if s]


def extract_skills_comprehensive(text):
    """Extract the skills a resume demonstrates, using the full SKILL_DATABASE.

    Replaces the 24-entry COMMON_SKILLS list in ai_engine/skills.py, which also
    could not match 'c++' at all.
    """
    lowered = text.lower()
    found = set()
    for group in ("technical", "domain", "soft"):
        for skill in SKILL_DATABASE[group]:
            if re.search(skill_pattern(skill), lowered):
                found.add(skill.title())
    return sorted(found)


# -------------------- JOB EXPIRY FUNCTIONS --------------------

def check_and_close_expired_jobs():
    today = date.today()
    expired_jobs = JobAnalysis.query.filter(
        JobAnalysis.deadline < today,
        JobAnalysis.status == 'Active'
    ).all()
    
    for job in expired_jobs:
        job.status = 'Closed'
    
    if expired_jobs:
        db.session.commit()
        print(f"[AUTO-CLOSE] Closed {len(expired_jobs)} expired job(s)")
    
    return len(expired_jobs)


def get_job_status_badge(job):
    if job.status == 'Closed':
        return 'closed'
    elif job.deadline and job.deadline < date.today():
        return 'expired'
    elif job.deadline:
        days_left = (job.deadline - date.today()).days
        if days_left <= 3:
            return 'urgent'
        elif days_left <= 7:
            return 'warning'
    return 'active'


def get_days_remaining(job):
    if not job.deadline:
        return None
    if job.status == 'Closed':
        return -1
    days = (job.deadline - date.today()).days
    return days if days >= 0 else -1

# -------------------- EMAIL HELPER FUNCTIONS --------------------

def send_email(to_email, subject, body, html_body=None):
    try:
        msg = Message(
            subject=subject,
            recipients=[to_email],
            body=body,
            html=html_body
        )
        mail.send(msg)
        print(f"[EMAIL SENT] To: {to_email}, Subject: {subject}")
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send email to {to_email}: {str(e)}")
        return False


def send_application_status_email(candidate_email, candidate_name, job_title, status, interview_details=None):
    if status == "Accepted":
        subject = f"Congratulations! You've been selected for {job_title}"
        body = f"""
Dear {candidate_name},

Great news! We are pleased to inform you that your application for the position of "{job_title}" has been ACCEPTED.

We were impressed with your qualifications and experience, and we believe you would be a valuable addition to our team.

"""
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #22c55e, #16a34a); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
        .content {{ background: #f9fafb; padding: 30px; border-radius: 0 0 10px 10px; }}
        .success-badge {{ background: #dcfce7; color: #166534; padding: 10px 20px; border-radius: 20px; display: inline-block; font-weight: bold; margin: 10px 0; }}
        .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎉 Congratulations!</h1>
        </div>
        <div class="content">
            <p>Dear <strong>{candidate_name}</strong>,</p>
            
            <p>Great news! We are pleased to inform you that your application has been:</p>
            
            <div class="success-badge">✓ ACCEPTED</div>
            
            <p><strong>Position:</strong> {job_title}</p>
            
            <p>We were impressed with your qualifications and experience, and we believe you would be a valuable addition to our team.</p>
            
            <p><strong>Next Steps:</strong><br>
            Our HR team will contact you shortly regarding the next steps in the hiring process.</p>
            
            <div class="footer">
                <p>Best regards,<br>
                <strong>HR Team</strong><br>
                Resume Intelligence System</p>
            </div>
        </div>
    </div>
</body>
</html>
"""
        
        if interview_details:
            body += f"\nInterview Scheduled:\nDate: {interview_details['date']}\nType: {interview_details['type']}"
            if interview_details.get('link'):
                body += f"\nLink: {interview_details['link']}"
            if interview_details.get('location'):
                body += f"\nLocation: {interview_details['location']}"
    
    else:
        subject = f"Update on your application for {job_title}"
        body = f"""
Dear {candidate_name},

Thank you for your interest in the "{job_title}" position and for taking the time to apply.

After careful consideration of all applications, we regret to inform you that we have decided to move forward with other candidates whose qualifications more closely match our current needs.

This decision was not easy, as we received many strong applications. We encourage you to apply for future openings that match your skills and experience.

We wish you all the best in your job search and future career endeavors.

Best regards,
HR Team
Resume Intelligence System
"""
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #64748b, #475569); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
        .content {{ background: #f9fafb; padding: 30px; border-radius: 0 0 10px 10px; }}
        .status-badge {{ background: #fee2e2; color: #991b1b; padding: 10px 20px; border-radius: 20px; display: inline-block; font-weight: bold; margin: 10px 0; }}
        .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Application Update</h1>
        </div>
        <div class="content">
            <p>Dear <strong>{candidate_name}</strong>,</p>
            
            <p>Thank you for your interest in the position and for taking the time to apply.</p>
            
            <p>After careful consideration, we regret to inform you that your application has been:</p>
            
            <div class="status-badge">Not Selected</div>
            
            <p><strong>Position:</strong> {job_title}</p>
            
            <p>This decision was not easy, as we received many strong applications. We encourage you to apply for future openings that match your skills and experience.</p>
            
            <p>We wish you all the best in your job search and future career endeavors.</p>
            
            <div class="footer">
                <p>Best regards,<br>
                <strong>HR Team</strong><br>
                Resume Intelligence System</p>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    return send_email(candidate_email, subject, body, html_body)


def send_interview_email(candidate_email, candidate_name, job_title, interview_details, is_update=False):
    action = "Updated" if is_update else "Scheduled"
    subject = f"Interview {action}: {job_title} Position"
    
    date_str = interview_details['date'].strftime('%A, %B %d, %Y at %I:%M %p')
    
    body = f"""
Dear {candidate_name},

Your interview for the "{job_title}" position has been {action.lower()}.

Interview Details:
Date & Time: {date_str}
Type: {interview_details['type']}
"""
    
    if interview_details.get('location'):
        body += f"Location: {interview_details['location']}\n"
    if interview_details.get('link'):
        body += f"Meeting Link: {interview_details['link']}\n"
    
    if interview_details.get('notes'):
        body += f"\nAdditional Notes: {interview_details['notes']}\n"
    
    body += """
Please confirm your attendance by replying to this email.

If you need to reschedule, please contact us at least 24 hours in advance.

Best regards,
HR Team
"""
    
    html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
        .content {{ background: #f9fafb; padding: 30px; border-radius: 0 0 10px 10px; }}
        .detail-box {{ background: white; padding: 20px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #3b82f6; }}
        .detail-row {{ margin: 10px 0; }}
        .detail-label {{ font-weight: 600; color: #6b7280; display: inline-block; width: 120px; }}
        .detail-value {{ color: #1f293b; }}
        .btn {{ background: #3b82f6; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block; margin: 10px 0; }}
        .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📅 Interview {action}</h1>
        </div>
        <div class="content">
            <p>Dear <strong>{candidate_name}</strong>,</p>
            
            <p>Your interview for the <strong>{job_title}</strong> position has been <strong>{action.lower()}</strong>.</p>
            
            <div class="detail-box">
                <div class="detail-row">
                    <span class="detail-label">Date & Time:</span>
                    <span class="detail-value">{date_str}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Type:</span>
                    <span class="detail-value">{interview_details['type']}</span>
                </div>
                {'<div class="detail-row"><span class="detail-label">Location:</span><span class="detail-value">' + interview_details['location'] + '</span></div>' if interview_details.get('location') else ''}
                {'<div class="detail-row"><span class="detail-label">Meeting Link:</span><span class="detail-value"><a href="' + interview_details['link'] + '">' + interview_details['link'] + '</a></span></div>' if interview_details.get('link') else ''}
                {'<div class="detail-row"><span class="detail-label">Notes:</span><span class="detail-value">' + interview_details['notes'] + '</span></div>' if interview_details.get('notes') else ''}
            </div>
            
            <p>Please confirm your attendance by replying to this email.</p>
            
            <p><em>If you need to reschedule, please contact us at least 24 hours in advance.</em></p>
            
            <div class="footer">
                <p>Best regards,<br>
                <strong>HR Team</strong><br>
                Resume Intelligence System</p>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    return send_email(candidate_email, subject, body, html_body)

# -------------------- IMPROVED AI SCORING FUNCTIONS --------------------

def preprocess_text(text):
    """Clean and normalize text for better matching"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text


def extract_section(text, section_name):
    """Extract specific section from resume with improved accuracy"""
    lines = text.split('\n')
    section_lines = []
    capture = False
    
    section_patterns = {
        'experience': ['experience', 'work experience', 'employment', 'work history', 'professional experience', 'career history'],
        'education': ['education', 'academic background', 'qualifications', 'academic qualifications', 'degrees'],
        'skills': ['skills', 'technical skills', 'core competencies', 'expertise', 'proficiencies', 'technologies'],
        'projects': ['projects', 'personal projects', 'professional projects', 'key projects'],
        'certifications': ['certifications', 'certificates', 'professional certifications', 'accreditations']
    }
    
    patterns = section_patterns.get(section_name, [section_name])
    
    for line in lines:
        clean = line.strip().lower()
        
        if not capture:
            for pattern in patterns:
                if clean.startswith(pattern) or pattern in clean[:len(pattern) + 5]:
                    capture = True
                    break
            continue
        
        for other_section, other_patterns in section_patterns.items():
            if other_section != section_name:
                for other_pattern in other_patterns:
                    if clean.startswith(other_pattern) or other_pattern in clean[:len(other_pattern) + 5]:
                        return ' '.join(section_lines)
        
        if capture and line.strip():
            section_lines.append(line.strip())
    
    return ' '.join(section_lines)


def calculate_tfidf_similarity(resume_text, job_text):
    """Calculate TF-IDF based cosine similarity for keyword matching"""
    try:
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0  # with only 2 documents, anything lower prunes all shared terms
        )
        
        texts = [preprocess_text(resume_text), preprocess_text(job_text)]
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity * 100)
    except Exception as e:
        print(f"TF-IDF calculation error: {e}")
        return 0.0


def extract_skills_precise(resume_text, job_skills):
    """Extract skills with precise matching and context analysis"""
    resume_lower = resume_text.lower()
    job_skills_lower = [s for s in (normalize_skill(s) for s in job_skills) if s]
    
    synonyms = SKILL_DATABASE['alternatives']
    
    matched_skills = {}
    total_confidence = 0
    skill_details = []
    
    for skill in job_skills_lower:
        skill_info = {
            'skill': skill,
            'matched': False,
            'confidence': 0,
            'match_type': 'none',
            'proficiency': 'beginner',
            'context': ''
        }
        
        hit = match_skill_variant(skill, resume_lower)
        if hit:
            skill_info['matched'] = True
            skill_info['confidence'] = 1.0 if hit == skill else 0.95
            skill_info['match_type'] = 'exact' if hit == skill else 'inflection ({})'.format(hit)
        
        if not skill_info['matched']:
            candidate_synonyms = []
            for variant in skill_variants(skill):
                candidate_synonyms.extend(synonyms.get(variant, []))
            for syn in candidate_synonyms:
                syn_hit = match_skill_variant(syn, resume_lower)
                if syn_hit:
                    skill_info['matched'] = True
                    skill_info['confidence'] = 0.85
                    skill_info['match_type'] = 'synonym ({})'.format(syn_hit)
                    break
        
        if skill_info['matched']:
            skill_info['proficiency'] = determine_proficiency(resume_text, skill)
            proficiency_multiplier = {
                'expert': 1.0,
                'advanced': 0.9,
                'intermediate': 0.8,
                'beginner': 0.6
            }
            skill_info['confidence'] *= proficiency_multiplier.get(skill_info['proficiency'], 0.7)
        
        matched_skills[skill] = skill_info['confidence']
        total_confidence += skill_info['confidence']
        skill_details.append(skill_info)
    
    if job_skills_lower:
        skill_score = (total_confidence / len(job_skills_lower)) * 100
    else:
        # No parseable required skills. Awarding 100 would silently hand every
        # candidate full marks on 40% of the score; 0 is the honest neutral.
        skill_score = 0
    
    return round(skill_score, 2), matched_skills, skill_details


# Proficiency inference reads the words surrounding a skill. Two rules keep it
# from crediting the candidate with someone else's seniority:
#   1. context is the sentence/line the skill sits in, not a fixed character
#      window - "Junior Developer, reporting to the Senior Architect" must not
#      make the candidate an expert;
#   2. "N years" only counts when it sits next to the skill - "8 years in retail
#      sales. Recently learned Python" is not 8 years of Python.
SENTENCE_BOUNDARY_RE = re.compile(r"[.\n;\r|\u2022]")
YEARS_RE = re.compile(r"(\d+)\s*\+?\s*years?")
YEARS_ADJACENCY_CHARS = 40

EXPERT_WORDS = ['expert', 'senior', 'lead', 'architect', 'specialist', 'principal']
ADVANCED_WORDS = ['advanced', 'experienced', 'proficient', 'skilled', 'strong']
INTERMEDIATE_WORDS = ['intermediate', 'familiar', 'working knowledge', 'exposure']
BEGINNER_WORDS = ['beginner', 'basic', 'learning', 'bootcamp', 'introductory']


def sentence_around(text, start, end):
    """The sentence or line containing a span."""
    left = 0
    for boundary in SENTENCE_BOUNDARY_RE.finditer(text, 0, start):
        left = boundary.end()
    right_match = SENTENCE_BOUNDARY_RE.search(text, end)
    right = right_match.start() if right_match else len(text)
    return text[left:right]


def years_beside(text, start, end, window=YEARS_ADJACENCY_CHARS):
    """Largest "N years" appearing within `window` characters of a span."""
    low = max(0, start - window)
    high = min(len(text), end + window)
    best = 0
    for match in YEARS_RE.finditer(text, low, high):
        best = max(best, int(match.group(1)))
    return best


def years_to_level(years):
    if years >= 5:
        return 'expert'
    if years >= 3:
        return 'advanced'
    if years >= 1:
        return 'intermediate'
    return None


def determine_proficiency(resume_text, skill):
    """beginner / intermediate / advanced / expert for one skill."""
    lowered = resume_text.lower()
    skill_norm = normalize_skill(skill)

    matches = list(re.finditer(skill_pattern(skill_norm), lowered))
    if not matches:
        return 'beginner'

    best_years = 0
    contexts = []
    for match in matches:
        best_years = max(best_years, years_beside(lowered, match.start(), match.end()))
        contexts.append(sentence_around(lowered, match.start(), match.end()))

    level_from_years = years_to_level(best_years)
    if level_from_years:
        return level_from_years

    context = ' '.join(contexts)
    if any(word in context for word in BEGINNER_WORDS):
        return 'beginner'
    if any(word in context for word in EXPERT_WORDS):
        return 'expert'
    if any(word in context for word in ADVANCED_WORDS):
        return 'advanced'
    if any(word in context for word in INTERMEDIATE_WORDS):
        return 'intermediate'
    return 'beginner'


def calculate_experience_score_improved(resume_text, min_years, job_title):
    """Calculate experience score with job relevance weighting"""
    exp_section = extract_section(resume_text, 'experience')
    
    years_matches = re.findall(r"(\d+)\s*\+?\s*years?", exp_section.lower())
    total_years = max([int(y) for y in years_matches]) if years_matches else 0
    
    if min_years > 0:
        base_score = min((total_years / min_years) * 100, 150)
    else:
        base_score = 100 if total_years > 0 else 50
    
    job_keywords = set(job_title.lower().split())
    relevance_score = 0
    
    for keyword in job_keywords:
        if len(keyword) > 3:
            count = exp_section.lower().count(keyword)
            relevance_score += min(count * 15, 30)
    
    seniority_indicators = ['senior', 'lead', 'principal', 'staff', 'manager', 'director', 'head of', 'architect']
    seniority_score = 0
    for indicator in seniority_indicators:
        if indicator in exp_section.lower():
            seniority_score = 20
            break
    
    final_score = min(base_score + relevance_score + seniority_score, 150)
    return round(final_score, 2), total_years


def calculate_education_score_improved(edu_level, min_exp):
    """Calculate education score based on experience requirements"""
    expected_level = 2
    
    if min_exp >= 7:
        expected_level = 4
    elif min_exp >= 5:
        expected_level = 3
    elif min_exp >= 3:
        expected_level = 3
    
    if edu_level >= expected_level:
        return 100
    elif edu_level == expected_level - 1:
        return 85
    elif edu_level == expected_level - 2:
        return 70
    else:
        return max(edu_level * 20, 40)


def extract_education_level_improved(text):
    """Extract education level with improved accuracy"""
    education_section = extract_section(text, 'education')
    text_lower = education_section.lower() if education_section else text.lower()
    
    education_patterns = {
        5: ['phd', 'doctorate', 'ph.d', 'doctor of philosophy', 'd.phil', 'ed.d'],
        4: ['masters', 'master', 'mba', 'm.s', 'm.sc', 'm.tech', 'm.e', 'm.eng', 'ma', 'm.a', 'post graduate', 'postgraduate', 'm.b.a'],
        3: ['bachelors', 'bachelor', 'b.tech', 'b.e', 'b.eng', 'b.s', 'b.sc', 'b.a', 'undergraduate', 'graduate', 'ba', 'bs', 'be', 'btech'],
        2: ['diploma', 'associate', 'a.s', 'a.a', 'certification', 'certificate', 'vocational'],
        1: ['high school', 'secondary', 'hsc', 'ssc', '12th', '10+2']
    }
    
    max_level = 1
    
    for level, patterns in education_patterns.items():
        for pattern in patterns:
            if re.search(r"\b" + re.escape(pattern) + r"\b", text_lower):
                max_level = max(max_level, level)
                break
    
    gpa_pattern = r'(?:gpa|cgpa)[\s:]*(\d+\.?\d*)'
    gpa_match = re.search(gpa_pattern, text_lower)
    if gpa_match:
        gpa = float(gpa_match.group(1))
        if gpa > 10:
            gpa = gpa / 10
        if gpa >= 3.5 or (gpa >= 7 and gpa <= 10):
            max_level += 0.5
    
    return min(max_level, 5)


# Component weights. Must sum to 1.0 - assert_weights() checks that on import.
#
# TF-IDF was cut from 0.15 to 0.05: measured on this codebase it returns 0.0 for
# common abbreviations ("Postgres", "JS", "ML", "K8s") and 0.0 for skills
# described rather than named, while scoring a resume that just pastes the job
# keywords HIGHER than an honest one. It is weak, gameable evidence, so most of
# its weight moved to skills - the only component that checks the requirements.
SCORE_WEIGHTS = {
    "semantic": 0.30,
    "tfidf": 0.05,
    "skills": 0.40,
    "experience": 0.20,
    "education": 0.05,
}


def assert_weights():
    total = sum(SCORE_WEIGHTS.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError("SCORE_WEIGHTS must sum to 1.0, got {}".format(total))


assert_weights()


def combine_scores(semantic, tfidf, skills, experience, education):
    """Apply SCORE_WEIGHTS. Both scorers route through here so they cannot drift."""
    return max(0.0, min(
        (semantic * SCORE_WEIGHTS["semantic"])
        + (tfidf * SCORE_WEIGHTS["tfidf"])
        + (skills * SCORE_WEIGHTS["skills"])
        + (min(experience, 100) * SCORE_WEIGHTS["experience"])
        + (education * SCORE_WEIGHTS["education"]),
        100.0,
    ))


def calculate_final_score_improved(resume_text, job_text, job_skills, min_experience, job_title):
    """Calculate final score with improved precision using hybrid approach"""
    
    semantic_raw = match_resume(job_text, resume_text)
    semantic_score = min(semantic_raw * 1.1, 100)
    
    tfidf_score = calculate_tfidf_similarity(resume_text, job_text)
    
    skills_score, matched_skills, skill_details = extract_skills_precise(resume_text, job_skills)
    
    exp_score, total_years = calculate_experience_score_improved(resume_text, min_experience, job_title)
    
    edu_level = extract_education_level_improved(resume_text)
    edu_score = calculate_education_score_improved(edu_level, min_experience)
    
    final_score = combine_scores(
        semantic_score, tfidf_score, skills_score, exp_score, edu_score
    )
    
    breakdown = {
        'semantic': round(semantic_score, 1),
        'tfidf': round(tfidf_score, 1),
        'skills': round(skills_score, 1),
        'experience': round(min(exp_score, 100), 1),
        'education': round(edu_score, 1),
        'years': total_years,
        'education_level': edu_level,
        'matched_skills': matched_skills,
        'skill_details': skill_details
    }
    
    return round(final_score, 2), breakdown


# -------------------- SCORER SELECTION --------------------

_llm_extractor = None


def get_llm_extractor():
    """Cached extractor: Gemini if GEMINI_API_KEY is set, else Claude, else none."""
    global _llm_extractor
    if _llm_extractor is None:
        _llm_extractor = get_extractor()
    return _llm_extractor


def build_job_text(job_title, job_skills, min_experience):
    """The job description text fed to the embedding and TF-IDF models.

    Every scorer must build this identically. The local and LLM paths used to
    format it differently (whitespace, raw vs normalised skills), which changed
    the semantic score for reasons unrelated to extraction quality and
    contaminated any local-vs-LLM comparison.
    """
    return (
        "Job Title: {}\n"
        "Required Skills: {}\n"
        "Minimum Experience: {} years"
    ).format(job_title, ", ".join(job_skills), min_experience)


SCREENING_LEVELS = {
    "strong": ("Strong match", "This candidate evidences most of what the role asks for."),
    "promising": ("Promising, with gaps", "A solid base, but some requirements need checking."),
    "partial": ("Partial match", "Meets some requirements; several are unproven."),
    "weak": ("Weak match", "Little of what the role requires is evidenced in this resume."),
}


def build_screening_summary(analysis, pool, years, min_experience):
    """A description of what was found, plus what to ask about.

    Deliberately not a hire/no-hire verdict. The system sees a resume - not the
    interview, the references or the team - so it reports findings and leaves
    the decision to the recruiter.
    """
    match = analysis["match_percentage"]
    weak = analysis["weak_claims"]
    partial = [s for s in analysis["partial_matches"] if s["band"] == "partial"]
    missing = analysis["skill_gaps_detailed"]
    meets_experience = years >= min_experience

    if match >= 80 and meets_experience:
        level = "strong"
    elif match >= 60:
        level = "promising"
    elif match >= 35:
        level = "partial"
    else:
        level = "weak"

    facts = [{
        "icon": "chart-simple",
        "text": "{}% of the required skills are evidenced in the resume".format(match),
    }]

    if min_experience > 0:
        facts.append({
            "icon": "briefcase",
            "text": "{} year(s) of relevant experience against a {}-year minimum{}".format(
                years, min_experience,
                "" if meets_experience else " - below the bar"),
            "warn": not meets_experience,
        })
    else:
        facts.append({"icon": "briefcase",
                      "text": "{} year(s) of relevant experience (no minimum set)".format(years)})

    if pool and pool.get("rank"):
        facts.append({
            "icon": "users",
            "text": "Ranked #{} of {} applicants for this role".format(
                pool["rank"], pool["total"]),
        })

    if weak:
        facts.append({
            "icon": "circle-exclamation",
            "text": "{} skill(s) are listed but unsupported by any project or role".format(len(weak)),
            "warn": True,
        })

    probes = []
    for skill in weak:
        probes.append({
            "skill": skill["name"],
            "kind": "unsupported",
            "text": "Listed on the resume with nothing behind it. Ask for a specific "
                    "project or task where they used it, and what was hard about it.",
        })
    for skill in partial:
        probes.append({
            "skill": skill["name"],
            "kind": "partial",
            "text": "Only indirect evidence ({}). Probe how deep this goes.".format(
                (skill.get("evidence") or "related work")[:90]),
        })
    for skill in missing:
        shared = ""
        if pool:
            info = (pool.get("rarity") or {}).get(normalize_skill(skill["name"]))
            if info and info["common"]:
                shared = (" {} of {} applicants also lack it, so consider whether the "
                          "requirement is realistic.".format(info["lacking"], info["total"]))
        probes.append({
            "skill": skill["name"],
            "kind": "missing",
            "text": "Not evidenced anywhere in the resume. Confirm whether the role "
                    "genuinely needs it on day one." + shared,
        })

    title, blurb = SCREENING_LEVELS[level]
    return {
        "level": level,
        "title": title,
        "blurb": blurb,
        "facts": facts,
        "probes": probes,
        "counts": {
            "demonstrated": len(analysis["matched_skills"]),
            "partial": len(partial),
            "unsupported": len(weak),
            "missing": len(missing),
        },
    }


def build_skill_pool_context(job_id, candidate_id, required_skills):
    """How this candidate compares with everyone else who applied.

    A missing skill means something different when nobody else has it either -
    that points at the job specification, not the candidate.
    """
    rows = CandidateResult.query.filter_by(job_id=job_id).all()
    parsed = []
    for row in rows:
        if not row.breakdown_json:
            continue
        try:
            data = json.loads(row.breakdown_json)
        except ValueError:
            continue
        parsed.append((row, {normalize_skill(k): float(v or 0)
                             for k, v in (data.get('matched_skills') or {}).items()}))

    if not parsed:
        return None

    rarity = {}
    for skill in required_skills:
        key = normalize_skill(skill)
        lacking = sum(1 for _, conf in parsed if conf.get(key, 0) < BAND_PARTIAL)
        rarity[key] = {
            'lacking': lacking,
            'total': len(parsed),
            'share': int((lacking / float(len(parsed))) * 100),
            'common': lacking > len(parsed) / 2.0,
        }

    ordered = sorted(parsed, key=lambda pair: pair[0].score or 0, reverse=True)
    rank = next((i + 1 for i, (row, _) in enumerate(ordered)
                 if row.candidate_id == candidate_id), None)

    scores = [row.score or 0 for row, _ in parsed]
    return {
        'rank': rank,
        'total': len(parsed),
        'rarity': rarity,
        'pool_average': round(sum(scores) / len(scores), 1),
    }


def build_pool_analytics(results):
    """Aggregate view of an applicant pool.

    Four counters told HR nothing about the shape of the pool. This answers the
    questions a recruiter actually has: is 72 a good score here, are the scores
    clustered, and is one requirement failing everybody (which usually means the
    requirement is unrealistic, not that every applicant is weak).
    """
    scores = [r.score for r in results if r.score is not None]
    if not scores:
        return None

    bands = [
        {"label": "0-20", "low": 0, "high": 20, "count": 0},
        {"label": "20-40", "low": 20, "high": 40, "count": 0},
        {"label": "40-60", "low": 40, "high": 60, "count": 0},
        {"label": "60-80", "low": 60, "high": 80, "count": 0},
        {"label": "80-100", "low": 80, "high": 101, "count": 0},
    ]
    for score in scores:
        for band in bands:
            if band["low"] <= score < band["high"]:
                band["count"] += 1
                break

    tallest = max(band["count"] for band in bands) or 1
    for band in bands:
        band["height"] = int((band["count"] / float(tallest)) * 100)

    missing = {}
    partial = {}
    graded = 0
    for row in results:
        if not row.breakdown_json:
            continue
        try:
            matched = (json.loads(row.breakdown_json).get("matched_skills") or {})
        except ValueError:
            continue
        graded += 1
        for skill, confidence in matched.items():
            value = float(confidence or 0)
            if value <= 0:
                missing[skill] = missing.get(skill, 0) + 1
            elif value < 0.6:
                partial[skill] = partial.get(skill, 0) + 1

    gaps = []
    for skill, count in sorted(missing.items(), key=lambda kv: -kv[1]):
        gaps.append({
            "skill": skill.title(),
            "missing": count,
            "weak": partial.get(skill, 0),
            "share": int((count / float(graded)) * 100) if graded else 0,
        })

    ordered = sorted(scores, reverse=True)
    middle = len(ordered) // 2
    median = (ordered[middle] if len(ordered) % 2
              else (ordered[middle - 1] + ordered[middle]) / 2.0)

    return {
        "bands": bands,
        "gaps": gaps[:6],
        "graded": graded,
        "average": round(sum(scores) / len(scores), 1),
        "median": round(median, 1),
        "spread": round(max(scores) - min(scores), 1),
        "scored_by": sorted({(r.scored_by or "local") for r in results}),
    }


# -------------------- SCORING PROGRESS --------------------

# AI Sort used to run inside the request, so the browser sat on a blank page
# with no feedback for however long the LLM took. Scoring now runs on a worker
# thread and the page polls this registry.
_scoring_progress = {}
_scoring_lock = threading.Lock()


def set_progress(job_id, **fields):
    with _scoring_lock:
        state = _scoring_progress.setdefault(job_id, {})
        state.update(fields)


def get_progress(job_id):
    with _scoring_lock:
        return dict(_scoring_progress.get(job_id, {}))


def applications_needing_scoring(job_id):
    """Applications with no result yet, plus any whose resume file changed."""
    result_exists = exists().where(
        and_(
            CandidateResult.job_id == JobApplication.job_id,
            CandidateResult.candidate_id == JobApplication.candidate_id
        )
    )
    todo = JobApplication.query.filter(
        JobApplication.job_id == job_id,
        JobApplication.status == 'Active',
        ~result_exists
    ).all()

    stale = []
    for entry in JobApplication.query.filter_by(job_id=job_id, status='Active').all():
        existing = CandidateResult.query.filter_by(
            job_id=job_id, candidate_id=entry.candidate_id
        ).first()
        if existing and existing.resume_file != entry.resume_file:
            stale.append(entry)
            db.session.delete(existing)

    if stale:
        db.session.commit()
        todo.extend(stale)
    return todo


def run_scoring_job(job_id):
    """Score every outstanding application for a job. Runs on a worker thread."""
    with app.app_context():
        try:
            job = JobAnalysis.query.get(job_id)
            if job is None:
                set_progress(job_id, state='error', message='Job not found')
                return

            job_skills = parse_job_skills(job.required_skills)
            job_text = build_job_text(job.job_title, job_skills, job.min_experience)

            pending = []
            for entry in applications_needing_scoring(job_id):
                path = os.path.join(UPLOAD_DIR, entry.resume_file)
                if os.path.exists(path):
                    pending.append((entry, extract_text(path)))

            set_progress(job_id, total=len(pending), done=0, state='running', message='')
            if not pending:
                set_progress(job_id, state='done')
                return

            scores = score_resumes_bulk(
                [(entry.candidate_name, text) for entry, text in pending],
                job_text, job_skills, job.min_experience, job.job_title,
                on_progress=lambda n: set_progress(job_id, done=n)
            )

            for (entry, resume_text), (final_score, score_breakdown) in zip(pending, scores):
                skills = extract_skills_comprehensive(resume_text)
                education = extract_education(resume_text)

                persisted_breakdown = {
                    'semantic': score_breakdown['semantic'],
                    'tfidf': score_breakdown['tfidf'],
                    'skills': score_breakdown['skills'],
                    'experience': score_breakdown['experience'],
                    'education': score_breakdown['education'],
                    'years': score_breakdown['years'],
                    'education_level': score_breakdown['education_level'],
                    'matched_skills': score_breakdown['matched_skills'],
                    'evidence': score_breakdown.get('evidence', {}),
                    'summary': score_breakdown.get('summary', ''),
                    'all_skills': score_breakdown.get('all_skills', []),
                }

                db.session.add(CandidateResult(
                    job_id=job.id,
                    candidate_id=entry.candidate_id,
                    candidate_name=entry.candidate_name,
                    score=final_score,
                    skills=", ".join(skills),
                    education=" | ".join(education),
                    experience=score_breakdown['years'],
                    resume_file=entry.resume_file,
                    status="Pending",
                    breakdown_json=json.dumps(persisted_breakdown),
                    scored_by=score_breakdown.get('scored_by', 'local')
                ))

            db.session.commit()
            set_progress(job_id, state='done', done=len(pending))
        except Exception as exc:
            db.session.rollback()
            print("[SCORING] job {} failed: {}".format(job_id, exc))
            set_progress(job_id, state='error', message=str(exc)[:200])


# LLM calls are network-bound and independent, so they overlap well. Kept
# modest because free-tier Gemini quotas are per-minute.
SCORING_CONCURRENCY = int(os.environ.get("SCORING_CONCURRENCY", "4"))


def score_resumes_bulk(items, job_text, job_skills, min_experience, job_title,
                       on_progress=None):
    """Score many resumes at once. items = [(candidate_name, resume_text), ...]

    Results come back in the same order. Falls back to a plain loop when the LLM
    is not configured: local scoring takes ~0.2s, so threads would only add
    overhead.
    """
    done = {'n': 0}
    progress_lock = threading.Lock()

    def mark_done():
        if on_progress is None:
            return
        with progress_lock:
            done['n'] += 1
            count = done['n']
        on_progress(count)

    extractor = get_llm_extractor()
    if not extractor.available() or len(items) < 2:
        results = []
        for name, text in items:
            results.append(
                score_resume(text, job_text, job_skills, min_experience, job_title, name)
            )
            mark_done()
        return results

    def local_only(name, text):
        score, breakdown = calculate_final_score_improved(
            text, job_text, job_skills, min_experience, job_title
        )
        breakdown['scored_by'] = 'local'
        return score, breakdown

    def with_llm(item):
        name, text = item
        scored = score_resume(text, job_text, job_skills, min_experience, job_title, name)
        mark_done()
        return scored

    # Two-stage funnel. With LLM_TOP_N set, every resume is ranked locally first
    # (~0.2s each), then only the strongest N get an LLM call. For 200 applicants
    # with LLM_TOP_N=25 that is 25 API calls instead of 200. Set it to 0 (the
    # default) to send every resume to the LLM.
    shortlist_size = int(os.environ.get("LLM_TOP_N", "0"))
    targets = list(range(len(items)))
    results = [None] * len(items)

    if shortlist_size and len(items) > shortlist_size:
        local_scores = [local_only(name, text) for name, text in items]
        ranked = sorted(range(len(items)), key=lambda i: local_scores[i][0], reverse=True)
        targets = ranked[:shortlist_size]
        for i in ranked[shortlist_size:]:
            results[i] = local_scores[i]
            mark_done()
        print("[SCORING] LLM_TOP_N={}: {} of {} candidates sent to the LLM".format(
            shortlist_size, len(targets), len(items)))

    workers = max(1, min(SCORING_CONCURRENCY, len(targets)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, scored in zip(targets, pool.map(with_llm, [items[i] for i in targets])):
            results[index] = scored
    return results


def score_resume(resume_text, job_text, job_skills, min_experience, job_title,
                 candidate_name=None):
    """Score one resume, preferring LLM extraction when it is configured.

    With no ANTHROPIC_API_KEY the app stays fully offline and uses the local
    embedding + keyword pipeline, so nothing breaks without network access.
    """
    extractor = get_llm_extractor()
    if extractor.available():
        return extractor.score(
            resume_text, job_title, job_skills, min_experience, candidate_name
        )

    score, breakdown = calculate_final_score_improved(
        resume_text, job_text, job_skills, min_experience, job_title
    )
    breakdown['scored_by'] = 'local'
    return score, breakdown


# -------------------- ENHANCED SKILL GAP ANALYSIS FUNCTIONS --------------------

# Confidence bands for LLM-assessed skills. "Weak" is the important one: it is
# where a resume lists a skill with nothing behind it, which keyword matching
# scores identically to a genuinely demonstrated skill.
BAND_STRONG = 0.80
BAND_PARTIAL = 0.50
BAND_WEAK = 0.20


def confidence_band(confidence):
    if confidence >= BAND_STRONG:
        return "strong"
    if confidence >= BAND_PARTIAL:
        return "partial"
    if confidence >= BAND_WEAK:
        return "weak"
    return "absent"


BAND_LABEL = {
    "strong": "Demonstrated",
    "partial": "Partial evidence",
    "weak": "Claimed, unsupported",
    "absent": "Not found",
}


def analyze_skill_gap(resume_text, job_skills, llm_confidences=None, llm_evidence=None):
    """Per-skill gap analysis for one candidate.

    When the LLM has assessed this resume its per-skill confidences are used
    directly. Keyword matching is the fallback for offline scoring only.

    This matters: on a resume that simply lists every required skill with no
    supporting evidence, keyword matching reports a 100% match while the LLM
    reports 24%. The ranking page uses the LLM figure, so a keyword-based gap
    page contradicted the ranking it was meant to explain.
    """
    resume_lower = resume_text.lower()
    job_skills_lower = [s for s in (normalize_skill(s) for s in job_skills) if s]
    synonyms = SKILL_DATABASE['alternatives']
    llm_confidences = {normalize_skill(k): v for k, v in (llm_confidences or {}).items()}
    llm_evidence = {normalize_skill(k): v for k, v in (llm_evidence or {}).items()}
    using_llm = bool(llm_confidences)

    matched_skills = []
    missing_skills = []
    partial_matches = []
    skill_gaps_detailed = []
    confidences = []

    for skill in job_skills_lower:
        skill_data = {
            'name': skill.title(),
            'level': 'beginner',
            'confidence': 0,
            'band': 'absent',
            'band_label': BAND_LABEL['absent'],
            'evidence': llm_evidence.get(skill, ''),
            'priority': classify_skill_priority(skill),
            'alternatives': SKILL_DATABASE['alternatives'].get(skill, []),
            'learning_time': SKILL_DATABASE['learning_time'].get(
                skill, {'beginner': 50, 'intermediate': 100, 'expert': 250}
            ),
            'learning_time_is_estimate': skill not in SKILL_DATABASE['learning_time'],
            'learning_path': SKILL_DATABASE['learning_paths'].get(skill, {}),
            'category': classify_skill_category(skill),
        }

        if using_llm:
            score = max(0.0, min(1.0, float(llm_confidences.get(skill, 0) or 0)))
            band = confidence_band(score)
            skill_data['confidence'] = round(score * 100)
            skill_data['band'] = band
            skill_data['band_label'] = BAND_LABEL[band]
            skill_data['proficiency_score'] = round(score * 100)
            skill_data['level'] = {
                'strong': 'advanced', 'partial': 'intermediate',
                'weak': 'beginner', 'absent': 'beginner',
            }[band]
            confidences.append(score)
        else:
            hit = match_skill_variant(skill, resume_lower)
            synonym_hit = None
            if not hit:
                for variant in skill_variants(skill):
                    for syn in synonyms.get(variant, []):
                        found = match_skill_variant(syn, resume_lower)
                        if found:
                            synonym_hit = found
                            break
                    if synonym_hit:
                        break

            if hit:
                score = 1.0 if hit == skill else 0.95
                skill_data['level'] = determine_proficiency(resume_lower, hit)
                skill_data['proficiency_score'] = calculate_proficiency_score(resume_text, hit)
            elif synonym_hit:
                score = 0.7
                skill_data['name'] = "{} (via {})".format(skill.title(), synonym_hit)
                skill_data['level'] = determine_proficiency(resume_lower, synonym_hit)
                skill_data['proficiency_score'] = round(
                    calculate_proficiency_score(resume_text, synonym_hit) * 0.8)
            else:
                score = 0.0

            band = confidence_band(score)
            skill_data['confidence'] = round(score * 100)
            skill_data['band'] = band
            skill_data['band_label'] = BAND_LABEL[band]
            confidences.append(score)

        if skill_data['band'] == 'strong':
            matched_skills.append(skill_data)
        elif skill_data['band'] in ('partial', 'weak'):
            partial_matches.append(skill_data)
        else:
            skill_data['gap_impact'] = 3 if skill_data['priority'] == 'critical' else 1
            missing_skills.append(skill_data['name'])
            skill_gaps_detailed.append(skill_data)

    total_required = len(job_skills_lower)
    if confidences:
        # The mean confidence, which is what the ranking page scores on. A count
        # of matched skills would disagree with the score beside it.
        match_percentage = round((sum(confidences) / len(confidences)) * 100)
    else:
        match_percentage = 0

    critical_gaps = [s for s in skill_gaps_detailed if s['priority'] == 'critical']

    return {
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'partial_matches': partial_matches,
        'skill_gaps_detailed': skill_gaps_detailed,
        'critical_gaps': critical_gaps,
        'match_percentage': match_percentage,
        'total_required': total_required,
        'total_matched': len(matched_skills),
        'total_learning_time': estimate_total_learning_time(skill_gaps_detailed),
        'assessed_by': 'llm' if using_llm else 'keyword matching',
        'weak_claims': [s for s in partial_matches if s['band'] == 'weak'],
        'priority_breakdown': {
            'critical': len([s for s in skill_gaps_detailed if s['priority'] == 'critical']),
            'important': len([s for s in skill_gaps_detailed if s['priority'] == 'important']),
            'nice_to_have': len([s for s in skill_gaps_detailed if s['priority'] == 'nice_to_have']),
        },
    }


def classify_skill_priority(skill):
    """Priority of a skill the HR user listed as a requirement.

    Every skill reaching this function was typed into the job posting, so it is
    required by definition - "nice to have" is not an honest label for it. The
    curated critical list still promotes the high-demand ones; everything else
    is important rather than optional. Previously "data structures",
    "algorithms" and "databases" all came back as nice-to-have on a Software
    Engineer posting that explicitly required them.
    """
    skill_lower = normalize_skill(skill)
    for variant in skill_variants(skill_lower):
        if variant in SKILL_DATABASE['priority']['critical']:
            return 'critical'
    return 'important'


def classify_skill_category(skill):
    skill_lower = skill.lower()
    for category, skills in SKILL_DATABASE.items():
        if category in ['technical', 'soft', 'domain'] and skill_lower in skills:
            return category
    return 'technical'


def calculate_proficiency_score(resume_text, skill):
    """0-100 confidence that the candidate really uses this skill."""
    skill_norm = normalize_skill(skill)
    lowered = resume_text.lower()

    # Token matches, not substring counts: resume.count("java") treated every
    # "JavaScript" as evidence of Java.
    matches = list(re.finditer(skill_pattern(skill_norm), lowered))
    if not matches:
        return 0

    expertise_score = 0
    best_years = 0

    for match in matches:
        context = sentence_around(lowered, match.start(), match.end())
        best_years = max(best_years, years_beside(lowered, match.start(), match.end()))

        if any(word in context for word in BEGINNER_WORDS):
            expertise_score += 5
        elif any(word in context for word in EXPERT_WORDS):
            expertise_score += 25
        elif any(word in context for word in ADVANCED_WORDS):
            expertise_score += 15

    if best_years:
        expertise_score += min(best_years * 5, 30)

    base_score = min(len(matches) * 10, 40)
    return min(base_score + expertise_score, 100)


def estimate_total_learning_time(skill_gaps):
    """Rough training time to close the gaps.

    Only a handful of skills have researched hour counts; everything else falls
    back to a flat default. `estimated` reports how many of the figures are that
    fallback, so the UI can say so instead of implying precision it does not have.
    """
    total_hours = 0
    estimated = 0
    for gap in skill_gaps:
        total_hours += gap['learning_time'].get('intermediate', 100)
        if gap.get('learning_time_is_estimate'):
            estimated += 1

    total_days = total_hours / 4.0  # assumes ~4 focused study hours per day
    return {
        'hours': total_hours,
        'days': round(total_days),
        'weeks': round(total_days / 5, 1),
        'estimated': estimated,
        'total': len(skill_gaps),
        'is_rough': estimated > 0,
    }


def determine_skill_level(resume_text, skill):
    exp_indicators = {
        'expert': ['expert', 'advanced', '5+ years', '5 years', '6 years', '7 years', '8 years', '9 years', '10 years', 'senior', 'lead', 'architect'],
        'intermediate': ['intermediate', '3 years', '4 years', '2+ years', 'experienced', 'proficient'],
        'beginner': ['beginner', 'basic', 'familiar', '1 year', '2 years', 'entry', 'junior', 'learning']
    }
    
    skill_pos = resume_text.find(skill)
    if skill_pos == -1:
        return 'beginner'
    
    start = max(0, skill_pos - 100)
    end = min(len(resume_text), skill_pos + 100)
    context = resume_text[start:end]
    
    for level, indicators in exp_indicators.items():
        for indicator in indicators:
            if indicator in context:
                return level
    
    years_match = re.search(r'(\d+)\s*\+?\s*years?.*?' + re.escape(skill), resume_text)
    if years_match:
        years = int(years_match.group(1))
        if years >= 5:
            return 'expert'
        elif years >= 3:
            return 'intermediate'
    
    return 'beginner'

# -------------------- ROUTES --------------------

@app.route("/")
def index():
    check_and_close_expired_jobs()
    return render_template("index.html")


@app.route("/admin-login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        user = HRUser.query.filter_by(username=request.form["username"]).first()
        if user and check_password_hash(user.password_hash, request.form["password"]):
            session["hr_logged_in"] = True
            return redirect("/hr-dashboard")
        flash("Invalid username or password")
    return render_template("login.html")


@app.route("/hr-dashboard")
def hr_dashboard():
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")

    check_and_close_expired_jobs()
    
    jobs = JobAnalysis.query.order_by(JobAnalysis.created_at.desc()).all()
    
    jobs_with_meta = []
    for job in jobs:
        days_remaining = get_days_remaining(job)
        jobs_with_meta.append({
            'job': job,
            'days_remaining': days_remaining if days_remaining is not None else -999,
            'has_deadline': job.deadline is not None,
            'status_badge': get_job_status_badge(job)
        })
    
    upcoming_interviews = InterviewSchedule.query.filter(
        InterviewSchedule.status == 'Scheduled',
        InterviewSchedule.interview_date > datetime.now()
    ).count()
    
    return render_template("hr_dashboard.html", 
                         jobs_with_meta=jobs_with_meta,
                         upcoming_interviews=upcoming_interviews)


@app.route("/create-job", methods=["GET", "POST"])
def create_job():
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")

    if request.method == "POST":
        deadline_str = request.form.get("deadline")
        deadline = datetime.strptime(deadline_str, "%Y-%m-%d").date() if deadline_str else None
        
        db.session.add(JobAnalysis(
            job_title=request.form["job_title"],
            required_skills=request.form["required_skills"],
            min_experience=int(request.form["min_experience"]),
            deadline=deadline,
            status="Active"
        ))
        db.session.commit()
        flash("Job created successfully")
        return redirect("/hr-dashboard")

    return render_template("create_job.html", today=date.today().isoformat())


@app.route("/hr/close-job/<int:job_id>", methods=["POST"])
def close_job_manually(job_id):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")
    
    job = JobAnalysis.query.get_or_404(job_id)
    job.status = 'Closed'
    db.session.commit()
    
    flash(f"Job '{job.job_title}' has been closed")
    return redirect("/hr-dashboard")


@app.route("/hr/reopen-job/<int:job_id>", methods=["POST"])
def reopen_job(job_id):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")
    
    job = JobAnalysis.query.get_or_404(job_id)
    
    new_deadline = date.today() + timedelta(days=7)
    job.deadline = new_deadline
    job.status = 'Active'
    db.session.commit()
    
    flash(f"Job '{job.job_title}' reopened until {new_deadline}")
    return redirect("/hr-dashboard")


@app.route("/hr/delete-job/<int:job_id>", methods=["POST"])
def delete_job(job_id):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")
    
    job = JobAnalysis.query.get_or_404(job_id)
    title = job.job_title

    try:
        # Interviews must go first. They have a NOT NULL job_id and a backref to
        # JobAnalysis, so leaving them behind makes SQLAlchemy try to orphan them
        # with job_id=NULL, which the schema rejects with an IntegrityError.
        InterviewSchedule.query.filter_by(job_id=job_id).delete()
        JobApplication.query.filter_by(job_id=job_id).delete()
        CandidateResult.query.filter_by(job_id=job_id).delete()

        db.session.delete(job)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        print("[DELETE JOB] {} failed: {}".format(job_id, exc))
        flash("Could not delete '{}'. Nothing was changed.".format(title), "warning")
        return redirect("/hr-dashboard")

    flash(f"Job '{title}' has been deleted")
    return redirect("/hr-dashboard")


@app.route("/candidate-login", methods=["GET", "POST"])
def candidate_login():
    if request.method == "POST":
        candidate = CandidateUser.query.filter_by(username=request.form["username"]).first()
        if candidate and check_password_hash(candidate.password_hash, request.form["password"]):
            session["candidate_logged_in"] = True
            session["candidate_id"] = candidate.id
            session["candidate_name"] = candidate.name
            return redirect("/candidate-dashboard")
        flash("Invalid username or password")
    return render_template("candidate_login.html")


@app.route("/candidate-register", methods=["GET", "POST"])
def candidate_register():
    if request.method == "POST":
        if request.form["password"] != request.form["confirm_password"]:
            flash("Passwords do not match")
            return redirect("/candidate-register")

        try:
            db.session.add(CandidateUser(
                name=request.form["name"],
                phone=request.form["phone"],
                email=request.form["email"],
                username=request.form["username"],
                password_hash=generate_password_hash(request.form["password"])
            ))
            db.session.commit()
            flash("Account created successfully")
            return redirect("/candidate-login")
        except Exception as e:
            db.session.rollback()
            flash("Username or email already exists")
            return redirect("/candidate-register")

    return render_template("candidate_register.html")


@app.route("/candidate-dashboard")
def candidate_dashboard():
    if not session.get("candidate_logged_in"):
        return redirect("/candidate-login")

    check_and_close_expired_jobs()
    
    candidate_id = session.get("candidate_id")
    
    applications = db.session.query(
        JobAnalysis.job_title,
        CandidateResult.score,
        CandidateResult.status,
        JobApplication.id,
        JobApplication.withdrawn_at,
        JobApplication.status.label('app_status')
    ).join(
        JobApplication, JobApplication.job_id == JobAnalysis.id
    ).outerjoin(
        CandidateResult, 
        (CandidateResult.job_id == JobAnalysis.id) & 
        (CandidateResult.candidate_id == candidate_id)
    ).filter(
        JobApplication.candidate_id == candidate_id
    ).order_by(JobApplication.applied_at.desc()).all()

    applied_job_ids = db.session.query(JobApplication.job_id).filter(
        JobApplication.candidate_id == candidate_id,
        JobApplication.status != 'Withdrawn'
    ).all()
    applied_job_ids = [j[0] for j in applied_job_ids]

    jobs = JobAnalysis.query.filter(
        JobAnalysis.status == 'Active',
        (JobAnalysis.deadline >= date.today()) | (JobAnalysis.deadline == None)
    ).order_by(JobAnalysis.created_at.desc()).all()

    jobs_with_status = []
    for job in jobs:
        job.already_applied = job.id in applied_job_ids
        jobs_with_status.append(job)

    return render_template(
        "candidate_dashboard.html",
        jobs=jobs_with_status,
        applications=applications,
        today=date.today()
    )


@app.route("/withdraw-application/<int:application_id>", methods=["POST"])
def withdraw_application(application_id):
    if not session.get("candidate_logged_in"):
        return redirect("/candidate-login")
    
    application = JobApplication.query.get_or_404(application_id)
    
    if application.candidate_id != session.get("candidate_id"):
        flash("Unauthorized action")
        return redirect("/candidate-dashboard")
    
    existing_result = CandidateResult.query.filter_by(
        job_id=application.job_id,
        candidate_id=application.candidate_id
    ).first()
    
    if existing_result and existing_result.status in ['Accepted', 'Rejected']:
        flash("Cannot withdraw - decision already made")
        return redirect("/candidate-dashboard")
    
    application.withdrawn_at = datetime.utcnow()
    application.status = "Withdrawn"
    
    if existing_result and existing_result.status == 'Pending':
        db.session.delete(existing_result)
    
    db.session.commit()
    
    flash("Application withdrawn successfully")
    return redirect("/candidate-dashboard")


@app.route("/apply/<int:job_id>", methods=["GET", "POST"])
def apply_job(job_id):
    if not session.get("candidate_logged_in"):
        return redirect("/candidate-login")

    job = JobAnalysis.query.get_or_404(job_id)
    
    if job.status == 'Closed':
        flash("This job posting has been closed")
        return redirect("/candidate-dashboard")
    
    if job.deadline and job.deadline < date.today():
        flash("The deadline for this job has passed")
        return redirect("/candidate-dashboard")

    candidate = CandidateUser.query.get(session["candidate_id"])

    if request.method == "POST":
        already_applied = JobApplication.query.filter(
            JobApplication.job_id == job.id,
            JobApplication.candidate_id == candidate.id,
            JobApplication.status != 'Withdrawn'
        ).first()
        if already_applied:
            flash("You have already applied for this position")
            return redirect("/candidate-dashboard")

        file = request.files.get("resume")
        if not file or file.filename == "":
            flash("Please upload resume")
            return redirect(request.url)

        stored_name = build_resume_filename(file.filename, candidate.id, job.id)
        if stored_name is None:
            flash("Unsupported file type. Please upload a PDF, DOCX or image.")
            return redirect(request.url)

        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file.save(os.path.join(UPLOAD_DIR, stored_name))

        db.session.add(JobApplication(
            job_id=job.id,
            candidate_id=candidate.id,
            candidate_name=candidate.name,
            resume_file=stored_name
        ))
        db.session.commit()

        flash("Application submitted successfully!")
        return redirect("/candidate-dashboard")

    return render_template("apply_job.html", job=job, candidate=candidate)


@app.route("/hr/applications/<int:job_id>")
def hr_view_applications(job_id):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")

    job = JobAnalysis.query.get_or_404(job_id)
    applications = JobApplication.query.filter_by(job_id=job_id, status='Active').all()
    return render_template("hr_applications.html", job=job, applications=applications)


@app.route("/hr/decision/<int:result_id>/<string:decision>", methods=["POST"])
def hr_decision(result_id, decision):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")

    result = CandidateResult.query.get_or_404(result_id)
    job = JobAnalysis.query.get_or_404(result.job_id)
    
    candidate = CandidateUser.query.get(result.candidate_id)
    if not candidate:
        flash("Candidate not found")
        return redirect("/hr-dashboard")

    if decision.lower() == "accept":
        result.status = "Accepted"
        flash(f"{result.candidate_name} has been accepted")
        
        email_sent = send_application_status_email(
            candidate_email=candidate.email,
            candidate_name=result.candidate_name,
            job_title=job.job_title,
            status="Accepted"
        )
        
        if email_sent:
            flash(f"Acceptance email sent to {candidate.email}")
        else:
            flash("Failed to send email notification", "warning")
            
    elif decision.lower() == "reject":
        result.status = "Rejected"
        flash(f"{result.candidate_name} has been rejected")
        
        email_sent = send_application_status_email(
            candidate_email=candidate.email,
            candidate_name=result.candidate_name,
            job_title=job.job_title,
            status="Rejected"
        )
        
        if email_sent:
            flash(f"Rejection email sent to {candidate.email}")
        else:
            flash("Failed to send email notification", "warning")
    else:
        flash("Invalid decision")
        return redirect("/hr-dashboard")

    db.session.commit()
    return redirect(f"/hr/sort/{result.job_id}")


@app.route("/hr/bulk-reject/<int:job_id>", methods=["POST"])
def bulk_reject_remaining(job_id):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")
    
    selected_ids = request.form.getlist('candidate_ids')
    
    if selected_ids:
        candidates = CandidateResult.query.filter(
            CandidateResult.id.in_(selected_ids),
            CandidateResult.status == 'Pending'
        ).all()
    else:
        candidates = CandidateResult.query.filter_by(
            job_id=job_id,
            status='Pending'
        ).all()
    
    rejected_count = 0
    email_failures = 0
    for candidate in candidates:
        candidate.status = 'Rejected'
        
        candidate_user = CandidateUser.query.get(candidate.candidate_id)
        if candidate_user:
            job = JobAnalysis.query.get(job_id)
            sent = send_application_status_email(
                candidate_email=candidate_user.email,
                candidate_name=candidate.candidate_name,
                job_title=job.job_title,
                status="Rejected"
            )
            if not sent:
                email_failures += 1
        rejected_count += 1
    
    db.session.commit()
    
    if rejected_count > 0:
        flash(f"Rejected {rejected_count} candidate(s)")
        if email_failures:
            flash(f"{email_failures} rejection email(s) could not be sent", "warning")
    else:
        flash("No pending candidates to reject")
    
    return redirect(f"/hr/sort/{job_id}")


@app.route("/hr/skill-gap/<int:result_id>")
def skill_gap_analysis(result_id):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")
    
    result = CandidateResult.query.get_or_404(result_id)
    job = JobAnalysis.query.get_or_404(result.job_id)
    
    job_skills = parse_job_skills(job.required_skills)
    
    resume_path = os.path.join(UPLOAD_DIR, result.resume_file)
    resume_text = extract_text(resume_path) if os.path.exists(resume_path) else ""
    
    stored_for_analysis = json.loads(result.breakdown_json) if result.breakdown_json else {}
    analysis = analyze_skill_gap(
        resume_text,
        job_skills,
        llm_confidences=stored_for_analysis.get('matched_skills'),
        llm_evidence=stored_for_analysis.get('evidence'),
    )
    
    # The LLM already recorded, per required skill, the resume text that
    # justified its confidence. It was stored and never shown.
    stored = json.loads(result.breakdown_json) if result.breakdown_json else {}
    evidence = {k.lower(): v for k, v in (stored.get('evidence') or {}).items() if v}

    # Skills the candidate brings that the job never asked for.
    required_lower = {normalize_skill(s) for s in job_skills}
    # Prefer the LLM's full skill list; fall back to the extractor's column so
    # candidates scored before all_skills was persisted still show something.
    demonstrated = stored.get('all_skills') or [
        s.strip() for s in (result.skills or '').split(',') if s.strip()
    ]
    extra_skills = sorted({
        s.strip() for s in demonstrated
        if s and normalize_skill(s) not in required_lower
    })

    pool = build_skill_pool_context(job.id, result.candidate_id, job_skills)

    screening = build_screening_summary(
        analysis, pool, result.experience or 0, job.min_experience or 0)

    return render_template("skill_gap.html",
                         screening=screening,
                         evidence=evidence,
                         extra_skills=extra_skills[:18],
                         pool=pool,
                         assessed_by=analysis['assessed_by'],
                         weak_claims=analysis['weak_claims'],
                         partial_matches_detail=analysis['partial_matches'],
                         ai_summary=stored.get('summary', ''),
                         scored_by=result.scored_by or 'local',
                         candidate_name=result.candidate_name,
                         job_title=job.job_title,
                         job_id=job.id,
                         experience=result.experience or 0,
                         matched_skills=analysis['matched_skills'],
                         missing_skills=analysis['missing_skills'],
                         partial_matches=analysis['partial_matches'],
                         skill_gaps_detailed=analysis['skill_gaps_detailed'],
                         critical_gaps=analysis['critical_gaps'],
                         match_percentage=analysis['match_percentage'],
                         total_learning_time=analysis['total_learning_time'],
                         priority_breakdown=analysis['priority_breakdown'])


@app.route("/hr/schedule-interview/<int:result_id>", methods=["GET", "POST"])
def schedule_interview(result_id):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")
    
    result = CandidateResult.query.get_or_404(result_id)
    job = JobAnalysis.query.get_or_404(result.job_id)
    candidate = CandidateUser.query.get_or_404(result.candidate_id)
    
    if result.status != "Accepted":
        flash("Can only schedule interviews for accepted candidates")
        return redirect(f"/hr/sort/{result.job_id}")
    
    if request.method == "POST":
        interview_date_str = request.form.get("interview_date")
        interview_type = request.form.get("interview_type", "Video Call")
        interview_link = request.form.get("interview_link", "")
        location = request.form.get("location", "")
        notes = request.form.get("notes", "")
        
        try:
            interview_date = datetime.strptime(interview_date_str, "%Y-%m-%dT%H:%M")
            
            existing = InterviewSchedule.query.filter_by(
                job_id=job.id,
                candidate_id=candidate.id
            ).first()
            
            if existing:
                existing.interview_date = interview_date
                existing.interview_type = interview_type
                existing.interview_link = interview_link
                existing.location = location
                existing.notes = notes
                existing.status = "Scheduled"
                is_update = True
            else:
                new_interview = InterviewSchedule(
                    job_id=job.id,
                    candidate_id=candidate.id,
                    candidate_name=candidate.name,
                    candidate_email=candidate.email,
                    interview_date=interview_date,
                    interview_type=interview_type,
                    interview_link=interview_link,
                    location=location,
                    notes=notes
                )
                db.session.add(new_interview)
                is_update = False
            
            db.session.commit()
            
            interview_details = {
                'date': interview_date,
                'type': interview_type,
                'link': interview_link,
                'location': location,
                'notes': notes
            }
            
            email_sent = send_interview_email(
                candidate_email=candidate.email,
                candidate_name=candidate.name,
                job_title=job.job_title,
                interview_details=interview_details,
                is_update=is_update
            )
            
            if email_sent:
                flash(f"Interview {'updated' if is_update else 'scheduled'} and email sent to {candidate.email}")
            else:
                flash(f"Interview {'updated' if is_update else 'scheduled'} but email failed to send", "warning")
            
            return redirect(f"/hr/sort/{result.job_id}")
            
        except Exception as e:
            flash(f"Error scheduling interview: {str(e)}")
            return redirect(request.url)
    
    existing_interview = InterviewSchedule.query.filter_by(
        job_id=job.id,
        candidate_id=candidate.id
    ).first()
    
    default_date = ""
    if existing_interview:
        default_date = existing_interview.interview_date.strftime("%Y-%m-%dT%H:%M")
    else:
        tomorrow = datetime.now() + timedelta(days=1)
        default_date = tomorrow.replace(hour=10, minute=0).strftime("%Y-%m-%dT%H:%M")
    
    return render_template("schedule_interview.html",
                         result=result,
                         job=job,
                         candidate=candidate,
                         existing_interview=existing_interview,
                         default_date=default_date)


@app.route("/hr/interviews")
def view_interviews():
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")
    
    interviews = InterviewSchedule.query.options(
        joinedload(InterviewSchedule.job)
    ).order_by(InterviewSchedule.interview_date.asc()).all()
    
    now = datetime.now()
    
    upcoming = [i for i in interviews if i.status == "Scheduled" and i.interview_date > now]
    past = [i for i in interviews if i.status == "Completed" or (i.status == "Scheduled" and i.interview_date <= now)]
    cancelled = [i for i in interviews if i.status == "Cancelled"]
    
    return render_template("interviews.html",
                         upcoming=upcoming,
                         past=past,
                         cancelled=cancelled)


@app.route("/hr/cancel-interview/<int:interview_id>", methods=["POST"])
def cancel_interview(interview_id):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")
    
    interview = InterviewSchedule.query.get_or_404(interview_id)
    interview.status = "Cancelled"
    db.session.commit()
    
    subject = f"Interview Cancelled: {interview.candidate_name}"
    body = f"Your interview scheduled for {interview.interview_date} has been cancelled."
    send_email(interview.candidate_email, subject, body)
    
    flash("Interview cancelled and candidate notified")
    return redirect("/hr/interviews")


@app.route("/hr/complete-interview/<int:interview_id>", methods=["POST"])
def complete_interview(interview_id):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")
    
    interview = InterviewSchedule.query.get_or_404(interview_id)
    interview.status = "Completed"
    db.session.commit()
    
    flash("Interview marked as completed")
    return redirect("/hr/interviews")


@app.route("/hr/sort/<int:job_id>")
def hr_sort_resumes(job_id):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")

    job = JobAnalysis.query.get_or_404(job_id)

    progress = get_progress(job_id)
    if progress.get('state') == 'running':
        # A worker is already scoring this job - show the progress page.
        return render_template("sorting.html", job=job, progress=progress)

    outstanding = applications_needing_scoring(job_id)
    if outstanding:
        set_progress(job_id, total=len(outstanding), done=0, state='running', message='')
        threading.Thread(target=run_scoring_job, args=(job_id,), daemon=True).start()
        return render_template("sorting.html", job=job, progress=get_progress(job_id))

    if progress.get('state') == 'error':
        flash("Scoring failed: {}".format(progress.get('message', 'unknown error')), "warning")
        set_progress(job_id, state='idle')


    db_results = CandidateResult.query.filter_by(job_id=job_id).order_by(CandidateResult.score.desc()).all()
    
    analytics = build_pool_analytics(db_results)

    pending_count = sum(1 for r in db_results if r.status == 'Pending')
    accepted_count = sum(1 for r in db_results if r.status == 'Accepted')
    rejected_count = sum(1 for r in db_results if r.status == 'Rejected')
    
    # Calculate top score with rounding to 2 decimal places
    top_score = round(db_results[0].score, 2) if db_results else 0
    
    ranked_results = []
    for rank, r in enumerate(db_results, start=1):
        display_score = round(r.score, 2)
        
        # Real per-component scores, recorded at scoring time. Rows created
        # before this column existed have none - show a prompt rather than
        # inventing numbers from the total.
        stored = json.loads(r.breakdown_json) if r.breakdown_json else None
        has_breakdown = stored is not None
        breakdown = {
            key: round(stored.get(key, 0), 1) if has_breakdown else 0
            for key in ('semantic', 'tfidf', 'skills', 'experience', 'education')
        }
        
        ranked_results.append({
            "id": r.id,
            "rank": rank,
            "name": r.candidate_name,
            "score": display_score,
            "breakdown": breakdown,
            "has_breakdown": has_breakdown,
            "scored_by": r.scored_by or "local",
            "summary": (stored or {}).get("summary", ""),
            "evidence": (stored or {}).get("evidence", {}),
            "breakdown_confidence": (stored or {}).get("matched_skills", {}),
            "skills": r.skills.split(", ") if r.skills else [],
            "education": r.education.split(" | ") if r.education else [],
            "experience": r.experience,
            "resume_file": r.resume_file,
            "status": r.status
        })

    return render_template("results.html", 
                         results=ranked_results, 
                         job_title=job.job_title,
                         min_experience=job.min_experience,
                         job_id=job_id,
                         analytics=analytics,
                         pending_count=pending_count,
                         accepted_count=accepted_count,
                         rejected_count=rejected_count,
                         top_score=top_score)  # Pass rounded top score


@app.route("/hr/sort-progress/<int:job_id>")
def hr_sort_progress(job_id):
    """Polled by sorting.html while a scoring job runs."""
    if not session.get("hr_logged_in"):
        return jsonify({"state": "unauthorised"}), 403
    progress = get_progress(job_id)
    return jsonify({
        "state": progress.get("state", "idle"),
        "done": progress.get("done", 0),
        "total": progress.get("total", 0),
        "message": progress.get("message", ""),
    })


# -------------------- SHORTLIST EXPORT --------------------

PDF_MARGIN = 54
PDF_BOTTOM = 780


class ShortlistPage(object):
    """Cursor over a PDF being written top-to-bottom, adding pages as needed."""

    def __init__(self, doc):
        self.doc = doc
        self.page = doc.new_page()
        self.y = PDF_MARGIN

    def space(self, amount):
        self.y += amount

    def rule(self, colour=(0.85, 0.87, 0.91)):
        if self.y > PDF_BOTTOM:
            self.new_page()
        self.page.draw_line(
            fitz.Point(PDF_MARGIN, self.y), fitz.Point(541, self.y),
            color=colour, width=0.7)
        self.y += 12

    def new_page(self):
        self.page = self.doc.new_page()
        self.y = PDF_MARGIN

    def text(self, body, size=10, font="helv", colour=(0.12, 0.16, 0.22),
             indent=0, leading=1.35, gap=4):
        if not body:
            return
        width = 487 - indent
        needed = fitz.get_text_length(body, fontname=font, fontsize=size)
        lines = max(1, int(needed / max(width, 1)) + 1)
        height = lines * size * leading + 6

        if self.y + height > PDF_BOTTOM:
            self.new_page()

        rect = fitz.Rect(PDF_MARGIN + indent, self.y,
                         PDF_MARGIN + indent + width, self.y + height + 30)
        self.page.insert_textbox(rect, body, fontsize=size, fontname=font,
                                 color=colour, lineheight=leading)
        self.y += height + gap


def build_shortlist_pdf(job, accepted, total_candidates):
    """Render the accepted candidates as a PDF and return the bytes."""
    doc = fitz.open()
    cursor = ShortlistPage(doc)

    cursor.text("Candidate Shortlist", size=22, font="hebo", colour=(0.06, 0.09, 0.16))
    cursor.text(job.job_title, size=13, colour=(0.29, 0.33, 0.41), gap=2)
    cursor.text("Generated {} | {} of {} applicants accepted".format(
        datetime.now().strftime("%d %b %Y, %H:%M"), len(accepted), total_candidates),
        size=9, colour=(0.58, 0.62, 0.69))
    cursor.space(6)
    cursor.rule()

    if not accepted:
        cursor.text("No candidates have been accepted for this role yet.",
                    size=11, colour=(0.45, 0.49, 0.55))
    for index, entry in enumerate(accepted, start=1):
        result = entry["result"]
        cursor.text("{}. {}".format(index, result.candidate_name),
                    size=13, font="hebo", colour=(0.06, 0.09, 0.16), gap=2)

        meta = "Match {}%".format(round(result.score or 0, 1))
        if result.experience is not None:
            meta += "  |  {} year(s) experience".format(result.experience)
        if entry["email"]:
            meta += "  |  {}".format(entry["email"])
        if entry["phone"]:
            meta += "  |  {}".format(entry["phone"])
        cursor.text(meta, size=9, colour=(0.39, 0.45, 0.55))

        if entry["summary"]:
            cursor.text(entry["summary"], size=9.5, colour=(0.22, 0.26, 0.33))

        if result.skills:
            cursor.text("Skills: " + result.skills, size=9, colour=(0.32, 0.36, 0.44))
        if result.education:
            cursor.text("Education: " + result.education, size=9, colour=(0.32, 0.36, 0.44))

        if entry["interview"]:
            cursor.text("Interview: {} ({})".format(
                entry["interview"].interview_date.strftime("%d %b %Y, %I:%M %p"),
                entry["interview"].interview_type),
                size=9, colour=(0.09, 0.42, 0.24))

        cursor.space(6)
        cursor.rule((0.93, 0.94, 0.96))

    cursor.space(4)
    cursor.text("Scores come from automated resume screening and are an aid to "
                "shortlisting, not a hiring decision.",
                size=8, colour=(0.62, 0.65, 0.71))

    data = doc.tobytes()
    doc.close()
    return data


@app.route("/hr/shortlist/<int:job_id>")
def hr_shortlist(job_id):
    """Download the accepted candidates for a job as a PDF."""
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")

    job = JobAnalysis.query.get_or_404(job_id)
    results = CandidateResult.query.filter_by(job_id=job_id).order_by(
        CandidateResult.score.desc()).all()
    accepted_results = [r for r in results if r.status == "Accepted"]

    if not accepted_results:
        flash("No candidates have been accepted for this job yet.", "warning")
        return redirect("/hr/sort/{}".format(job_id))

    accepted = []
    for result in accepted_results:
        candidate = CandidateUser.query.get(result.candidate_id)
        stored = {}
        if result.breakdown_json:
            try:
                stored = json.loads(result.breakdown_json)
            except ValueError:
                stored = {}
        accepted.append({
            "result": result,
            "email": candidate.email if candidate else "",
            "phone": candidate.phone if candidate else "",
            "summary": stored.get("summary", ""),
            "interview": InterviewSchedule.query.filter_by(
                job_id=job_id, candidate_id=result.candidate_id
            ).filter(InterviewSchedule.status != "Cancelled").first(),
        })

    pdf = build_shortlist_pdf(job, accepted, len(results))
    filename = "shortlist_{}_{}.pdf".format(
        secure_filename(job.job_title) or "job",
        datetime.now().strftime("%Y%m%d"))

    return send_file(io.BytesIO(pdf), mimetype="application/pdf",
                     as_attachment=True, download_name=filename)


@app.route("/view_resume/<filename>")
def view_resume(filename):
    if not session.get("hr_logged_in"):
        return redirect("/admin-login")
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


def init_db():
    """Create tables and seed the initial HR account.

    Runs on import, not only under __main__, so `gunicorn app:app` starts
    against a real schema.
    """
    with app.app_context():
        db.create_all()

        if not HRUser.query.first():
            seed_password = os.environ.get("HR_ADMIN_PASSWORD", "admin123")
            db.session.add(HRUser(
                username=os.environ.get("HR_ADMIN_USERNAME", "hradmin"),
                password_hash=generate_password_hash(seed_password)
            ))
            db.session.commit()
            print("[INIT] Seeded HR admin account")


init_db()


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1")