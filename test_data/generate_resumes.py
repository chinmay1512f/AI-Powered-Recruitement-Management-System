"""Generate realistic test resumes as PDFs.

These are deliberately designed to exercise the cases where keyword matching
fails and LLM extraction should win:

  * skills shown through PROJECTS rather than named ("LRU cache" -> data structures)
  * experience as DATE RANGES rather than "N years"
  * ABBREVIATIONS resumes really use ("Postgres", "JS", "K8s")
  * a KEYWORD STUFFER with no substance, to check the scorer is not fooled
  * a LONG resume with the good part at the BOTTOM, to check the truncation fix

All names, companies and universities are fictional.

Run:  python test_data/generate_resumes.py
"""
import os

import fitz

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resumes")

RESUMES = {
    # ---------------------------------------------------------------- 1
    # Strong. Never says "data structures" or "algorithms" - shows them.
    # Says "Postgres", not "databases". Dates, not "N years".
    # Hobbies deliberately near the top to test the truncation fix.
    "01_meera_krishnan": """MEERA KRISHNAN
Backend Engineer

INTERESTS
Long-distance running, classical guitar, competitive chess (state level, 2019).
Volunteer instructor at a weekend coding club for school students since 2021.
Member of the university photography society. Enjoys hiking and cooking.

LANGUAGES
English (fluent), Hindi (fluent), Tamil (native), German (basic).

PROFILE
Backend engineer who enjoys systems work and performance tuning.

EXPERIENCE

Backend Engineer, Northwind Systems
Jun 2022 - Present
Built and maintained the order-processing service handling roughly 4,000
requests per minute. Rewrote the pricing engine, cutting p99 latency from
840ms to 120ms by replacing repeated linear scans with an indexed lookup and
an in-memory LRU cache with time-based eviction. Designed the Postgres schema
for the settlements module, including partitioning by month and a set of
covering indexes that removed three sequential scans from the nightly job.
Wrote the migration tooling in Python. Owned the on-call rotation for the
service.

Software Engineering Intern, Bluepeak Labs
Jan 2022 - May 2022
Implemented a shortest-path routing feature for the internal logistics tool
using Dijkstra over a weighted graph of roughly 12,000 nodes. Reduced average
route computation from 2.1s to 180ms by switching to a binary heap for the
frontier. Wrote the service in Java with a Spring Boot HTTP layer.

PROJECTS

Distributed key-value store (personal, 2023)
Implemented a Raft-based replicated store in Java. Log compaction, leader
election, snapshotting. Wrote a deterministic simulation harness to reproduce
split-brain scenarios.

Text index for a 2GB corpus (personal, 2021)
Built an inverted index with a trie for prefix queries and a custom
skip-list for postings intersection. Benchmarked against a naive scan.

EDUCATION
B.Tech Computer Science, Ashwood Institute of Technology, 2022
Relevant coursework: Design and Analysis of Algorithms, Database Management
Systems, Operating Systems, Distributed Systems. CGPA 8.7/10.

TECHNICAL
Java, Python, Postgres, Redis, Docker, Kubernetes, Git, Linux, gRPC.
""",

    # ---------------------------------------------------------------- 2
    # Strong AND explicit. Both scorers should rate this highly.
    "02_arjun_desai": """ARJUN DESAI
Software Engineer

SUMMARY
Software engineer with 4 years of experience in Python and Java backend
development, with a strong foundation in data structures, algorithms and
databases.

TECHNICAL SKILLS
Languages: Python, Java, SQL
Core: Data Structures, Algorithms, Databases, Object-Oriented Design
Tools: PostgreSQL, MySQL, Docker, Git, Jenkins, AWS

EXPERIENCE

Software Engineer, Vertex Retail Technologies
Aug 2021 - Present
Develop and maintain inventory microservices in Java and Python. Designed the
relational schema for the stock-reservation service. Optimised a reporting
query from 14s to 400ms through index redesign and query restructuring.
Mentored two junior engineers on algorithms and code review practice.

Junior Developer, Sandhurst Software
Jul 2020 - Jul 2021
Maintained a Python ETL pipeline processing 60GB of daily transaction data.
Added retry and idempotency handling. Wrote the unit and integration test
suites.

PROJECTS
Scheduling optimiser using a greedy algorithm with a priority queue, reducing
warehouse pick times by 22% in simulation.

EDUCATION
B.E. Computer Engineering, Marchmont University, 2020. CGPA 8.2/10.
Coursework: Data Structures, Algorithms, DBMS, Computer Networks.
""",

    # ---------------------------------------------------------------- 3
    # Keyword stuffer. Every required term present, zero evidence.
    # Keyword matching should love this. Good extraction should not.
    "03_rohit_bhalla": """ROHIT BHALLA
Software Engineer

SKILLS
Data Structures, Algorithms, Databases, Python, Java, Data Structures and
Algorithms, Database Management, Python Programming, Java Programming, SQL,
Problem Solving, Object Oriented Programming, Coding, Software Development,
Data Structures, Algorithms, Databases, Python, Java.

OBJECTIVE
Seeking a challenging position in a reputed organisation where I can utilise
my skills in data structures, algorithms, databases, Python and Java to
contribute to organisational growth while enhancing my professional
capabilities.

EXPERIENCE
Fresher.

EDUCATION
B.Sc Information Technology, Calder Institute, 2024.

CERTIFICATIONS
Online certificate in Python. Online certificate in Java. Online certificate
in Data Structures.

HOBBIES
Reading, music, cricket.
""",

    # ---------------------------------------------------------------- 4
    # Real experience but only as date ranges. Local scorer reads 0 years.
    "04_priya_raghavan": """PRIYA RAGHAVAN
Platform Engineer

EXPERIENCE

Senior Platform Engineer, Halcyon Data
Mar 2019 - Present
Own the data platform serving 40 internal teams. Migrated the batch layer from
cron-driven scripts to an orchestrated DAG, cutting failed overnight runs from
roughly 12 per month to under 1. Built the schema registry in Python. Manage
the Postgres fleet: replication, failover drills, index maintenance, and query
review for teams shipping new features.

Platform Engineer, Halcyon Data
Mar 2017 - Feb 2019
Built internal tooling in Python and Go. Wrote the deployment pipeline.

Software Developer, Ridgeline Consulting
Aug 2015 - Feb 2017
Java backend work for client projects in insurance and logistics. Built a
claims-matching service using a custom scoring index over 3M records.

EDUCATION
M.Tech Computer Science, Brambleton University, 2015
B.Tech Information Technology, Brambleton University, 2013

SKILLS
Python, Go, Java, Postgres, Kafka, Airflow, Terraform, Linux.
""",

    # ---------------------------------------------------------------- 5
    # Everything in abbreviations. Keyword matching sees none of it.
    "05_sameer_qureshi": """SAMEER QURESHI
Full Stack Developer

TECH
JS, TS, React, Node, Postgres, Mongo, K8s, Docker, GCP, CI/CD, ML basics.

WORK

Full Stack Developer, Copperfield Apps
Feb 2021 - Present
Own three services end to end. Moved the reporting store from Mongo to
Postgres after write contention issues; designed the new schema and the
backfill. Wrote the backfill tooling in Py. Run our K8s manifests and the
GH Actions pipelines.

Developer, Toft Media
Jun 2019 - Jan 2021
Built the CMS front end in React and the API in Node. Added a caching layer
that cut average response time by half.

PROJECTS
Recommendation prototype using collaborative filtering on a 400k-row ratings
set. Wrote the similarity computation from scratch before switching to a
library implementation.

EDUCATION
BCA, Elmsworth College, 2019.
""",

    # ---------------------------------------------------------------- 6
    # Fresher, no jobs, but genuine algorithmic depth via contests.
    "06_ananya_bose": """ANANYA BOSE
Computer Science Graduate

EDUCATION
B.Tech Computer Science and Engineering, Thornbury Institute, 2025
CGPA 9.1/10
Coursework: Advanced Data Structures, Design and Analysis of Algorithms,
Database Systems, Compiler Design, Machine Learning.

COMPETITIVE PROGRAMMING
Candidate Master on a major competitive programming site (peak rating 1974).
Solved 1,400+ problems, concentrated in graph theory, dynamic programming and
segment trees. Team placed 14th in the regional ICPC round, 2024.

PROJECTS

Query engine for CSV files (2024)
Wrote a small SQL subset interpreter in Python: tokenizer, parser producing an
AST, and an executor supporting SELECT, WHERE, JOIN and GROUP BY. Implemented
hash join and sort-merge join and benchmarked both across input sizes.

Route planner (2024)
Java implementation of A* with a landmark-based heuristic over an OpenStreetMap
extract of roughly 300,000 nodes. Compared against plain Dijkstra.

Autocomplete service (2023)
Trie with frequency-ranked completions, served over a small HTTP API.

INTERNSHIP
Software Intern, Pinehaven Tech, May 2024 - Jul 2024
Wrote Python scripts to validate data exports. Added test coverage.

SKILLS
Python, Java, C++, SQL, Git.
""",

    # ---------------------------------------------------------------- 7
    # CSE, but manual-QA focused. Some Java, no algorithms or schema work.
    "07_nikhil_pawar": """NIKHIL PAWAR
QA Engineer

PROFILE
Quality assurance engineer focused on manual and regression testing for web
applications.

EXPERIENCE

QA Engineer, Bellhaven Software
Feb 2021 - Present
Own the regression suite for a mid-sized web product. Write and execute test
cases, log defects, and verify fixes across three release trains per quarter.
Automated roughly 40 smoke tests using Selenium WebDriver with Java. Coordinate
UAT sessions with client stakeholders. Maintain the test-case repository and
the defect triage board.

Junior QA Analyst, Marlowe Systems
Aug 2019 - Jan 2021
Executed manual test plans for a banking portal. Performed cross-browser and
cross-device verification. Raised and tracked defects.

SKILLS
Manual testing, regression testing, test case design, Selenium, Java (basic),
JIRA, Postman, SQL (basic SELECT queries for data verification), Agile.

EDUCATION
B.E. Computer Science, Ferngate Institute of Engineering, 2019. CGPA 7.4/10.

CERTIFICATIONS
ISTQB Foundation Level, 2020.
""",

    # ---------------------------------------------------------------- 8
    # CSE, ML specialist. Heavy Python, some SQL, little Java or DSA.
    "08_ritika_chandra": """RITIKA CHANDRA
Machine Learning Engineer

EXPERIENCE

Machine Learning Engineer, Solstice Analytics
Jan 2022 - Present
Build and deploy demand-forecasting models for retail clients. Own the feature
pipeline: pulls from a Snowflake warehouse, transforms in Python with pandas,
and writes features to a Postgres feature store. Cut model training time from
6 hours to 40 minutes by rewriting the aggregation step to avoid repeated
full-table scans. Deployed four models to production behind a FastAPI service.

Data Science Intern, Solstice Analytics
Jun 2021 - Dec 2021
Built churn-prediction prototypes. Ran the A/B evaluation for the first
production model.

PROJECTS
Image classification pipeline on a 60k-image dataset using PyTorch, with a
custom data loader and augmentation stack. Reached 94% top-1 on the held-out
split.

SKILLS
Python, pandas, NumPy, scikit-learn, PyTorch, TensorFlow, SQL, Postgres,
FastAPI, Docker, MLflow, Airflow.

EDUCATION
M.Tech Data Science, Ravensmoor University, 2021
B.Tech Computer Science, Ravensmoor University, 2019
Coursework: Machine Learning, Statistics, Database Systems, Data Structures.
""",

    # ---------------------------------------------------------------- 9
    # CSE, embedded systems. C/C++ and hardware. No Python, Java or databases.
    "09_devendra_shetty": """DEVENDRA SHETTY
Embedded Systems Engineer

EXPERIENCE

Embedded Software Engineer, Corvid Instruments
Apr 2020 - Present
Develop firmware for industrial sensor modules on ARM Cortex-M4. Implemented
the CAN bus communication stack and a bootloader supporting over-the-air
updates. Reduced power consumption 31% by reworking the sleep-state machine.
Write in C with selected C++ for the driver abstraction layer. Debug with JTAG
and a logic analyser.

Firmware Intern, Corvid Instruments
Sep 2019 - Mar 2020
Wrote device drivers for SPI and I2C peripherals. Built a test harness for
the sensor calibration routine.

PROJECTS
RTOS scheduler implemented from scratch for a university project: round-robin
and priority-based scheduling with a fixed-size task control block pool.

SKILLS
C, C++, ARM assembly, FreeRTOS, CAN, SPI, I2C, UART, Git, Keil, JTAG debugging,
oscilloscopes, KiCad.

EDUCATION
B.Tech Electronics and Computer Engineering, Halloway Institute, 2019
Coursework: Microprocessors, Embedded Systems, Operating Systems, Digital Design.
""",

    # ---------------------------------------------------------------- 10
    # CSE, genuinely strong, but an entirely different stack.
    "10_farah_siddiqui": """FARAH SIDDIQUI
Frontend Engineer

EXPERIENCE

Senior Frontend Engineer, Wisteria Interactive
Sep 2020 - Present
Lead the design-system team. Rebuilt the component library in TypeScript and
React, cutting bundle size 38% and standardising accessibility across 60+
components. Introduced visual regression testing. Mentor three engineers.

Frontend Developer, Quillon Studio
Jul 2018 - Aug 2020
Built marketing sites and a customer portal in React. Improved Lighthouse
performance scores from the 50s to the 90s across the main funnel.

PROJECTS
Canvas-based data visualisation library handling 100k points at 60fps using
quadtree spatial indexing for hit testing.

SKILLS
TypeScript, JavaScript, React, Next.js, CSS, HTML, Webpack, Jest, Figma.

EDUCATION
B.Des Interaction Design, Whitmore College, 2018
""",
}


def write_pdf(path, text):
    doc = fitz.open()
    page = doc.new_page()
    margin, top, width = 56, 60, 483
    y = top

    for line in text.split("\n"):
        if y > 760:                      # continue on a new page
            page = doc.new_page()
            y = top
        if not line.strip():
            y += 7
            continue
        # crude heading detection: short, all caps
        stripped = line.strip()
        is_heading = stripped.isupper() and len(stripped) < 46
        size = 13 if is_heading else 9.5
        font = "helv" if not is_heading else "hebo"
        page.insert_textbox(
            fitz.Rect(margin, y, margin + width, y + 40),
            line, fontsize=size, fontname=font,
        )
        y += 17 if is_heading else 12

    doc.save(path)
    doc.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, text in RESUMES.items():
        path = os.path.join(OUT_DIR, name + ".pdf")
        write_pdf(path, text)
        print("  wrote {}  ({} chars of text)".format(os.path.basename(path), len(text)))
    print("\n{} resumes in {}".format(len(RESUMES), OUT_DIR))


if __name__ == "__main__":
    main()
