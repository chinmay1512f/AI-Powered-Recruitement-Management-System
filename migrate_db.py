"""Idempotent schema migration for resume_system.db.

Adds the columns introduced alongside real score-breakdown persistence, and
reports any resumes that were overwritten by the old upload naming scheme.

Run with:  python migrate_db.py
"""
import os
import sqlite3

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resume_system.db")

NEW_COLUMNS = [
    ("candidate_result", "breakdown_json", "TEXT"),
    ("candidate_result", "scored_by", "VARCHAR(20) DEFAULT 'local'"),
]


def existing_columns(cur, table):
    return {row[1] for row in cur.execute("PRAGMA table_info({})".format(table))}


def main():
    if not os.path.exists(DB):
        print("No database at {} - nothing to migrate.".format(DB))
        return

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    added = 0
    for table, column, coltype in NEW_COLUMNS:
        if column in existing_columns(cur, table):
            print("  ok      {}.{} already present".format(table, column))
            continue
        cur.execute("ALTER TABLE {} ADD COLUMN {} {}".format(table, column, coltype))
        print("  added   {}.{}".format(table, column))
        added += 1

    conn.commit()

    # Report collisions caused by the old "save under the browser filename" code.
    dupes = cur.execute(
        """
        SELECT resume_file, COUNT(*) AS n, GROUP_CONCAT(candidate_name, ' | ')
        FROM job_application
        GROUP BY resume_file
        HAVING n > 1
        """
    ).fetchall()

    if dupes:
        print("\n  WARNING - these applicants share one resume file (an upload was overwritten):")
        for filename, count, names in dupes:
            print("    {} <- {} applicants: {}".format(filename, count, names))
        print("  The overwritten file is not recoverable. Those candidates must re-apply")
        print("  before their scores mean anything.")

    conn.close()
    print("\nMigration complete ({} column(s) added).".format(added))


if __name__ == "__main__":
    main()
