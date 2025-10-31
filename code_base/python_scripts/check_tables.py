"""
Simple check script used by Airflow DAG to print table names and row counts.
Writes output to stdout (Airflow logs will capture it).
"""

import os
from sqlalchemy import create_engine, text

DB_URL = os.environ.get("CAPSTONE_DB_URL") or "postgresql://db_user:db_password@localhost:5000/db"

engine = create_engine(DB_URL)

def main():
    with engine.connect() as conn:
        print("Listing tables and row counts (public schema):")
        rows = conn.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)).fetchall()
        names = [r[0] for r in rows]
        if not names:
            print("  (no tables found in public schema)")
            return
        for t in names:
            try:
                cnt = conn.execute(text(f'SELECT count(*) FROM public."{t}"')).scalar()
            except Exception as e:
                cnt = f"ERROR: {e}"
            print(f" - {t}: {cnt}")

if __name__ == "__main__":
    main()
