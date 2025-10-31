"""
Airflow DAG to orchestrate Excel -> Postgres loading.

Drop this file into:
    /home/anmol_ubuntu/repos/capstone/airflow/dags/orchestrator.py

Notes:
- Uses Airflow 2.6+ style: use `schedule` (not schedule_interval) and a static start_date.
- Schedules every 15 minutes via cron "*/15 * * * *".
- Adjust PROJECT_ROOT, VENV_PATH, DB connection and paths as required for your environment.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import os
from pathlib import Path

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator

# -----------------------
# Configuration (edit as needed)
# -----------------------
PROJECT_ROOT = Path("/opt/airflow")
PY_SCRIPTS = PROJECT_ROOT / "python_scripts"
LOADER_SCRIPT = PY_SCRIPTS / "upload_stg_files.py"
CHECK_SCRIPT = PY_SCRIPTS / "check_tables.py"
MAPPING_JSON = PY_SCRIPTS / "table_mapping.json"
VENV_PATH = PROJECT_ROOT / ".venv"  # if Airflow can't access this venv, install deps in Airflow env

# Postgres connection details (matches your docker-compose)
DB_USER = "db_user"
DB_PASS = "db_password"
DB_HOST = "db"
DB_PORT = "5432"
DB_NAME = "db"
DB_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Execution defaults
DEFAULT_ARGS = {
    "owner": "capstone",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

# How long a single task is allowed to run before Airflow marks it failed
TASK_TIMEOUT = timedelta(hours=2)

# -----------------------
# Helper: build python command (activates venv if present)
# -----------------------
def build_python_cmd(script_path: Path, extra_args: str = "") -> str:
    script = str(script_path)
    if VENV_PATH and VENV_PATH.exists():
        activate = f"source {str(VENV_PATH)}/bin/activate"
        python = "python"
        return f"{activate} && {python} {script} {extra_args}"
    else:
        return f"python3 {script} {extra_args}"


# -----------------------
# DAG
# -----------------------
with DAG(
    dag_id="capstone_orchestrator",
    default_args=DEFAULT_ARGS,
    description="Load Excel files into Postgres (per-sheet mapping) and verify counts",
    schedule="*/15 * * * *",                 # every 15 minutes (modern Airflow uses `schedule`)
    start_date=datetime(2025, 9, 23),        # static start_date (use a past date)
    catchup=False,
    max_active_runs=1,
    tags=["capstone", "etl"],
) as dag:

    start = EmptyOperator(task_id="start")

    loader_cmd = build_python_cmd(
        LOADER_SCRIPT,
        extra_args=(
            f'--db-url "{DB_URL}" '
            f'--files-dir "{PROJECT_ROOT / "files"}" '
            f'--mapping "{MAPPING_JSON}"'
        ),
    )

    run_loader = BashOperator(
        task_id="run_loader",
        bash_command=loader_cmd,
        env={"CAPSTONE_DB_URL": DB_URL, "PYTHONUNBUFFERED": "1"},
        execution_timeout=TASK_TIMEOUT,
        retries=2,
    )

    check_cmd = build_python_cmd(CHECK_SCRIPT)
    run_check = BashOperator(
        task_id="run_post_load_check",
        bash_command=check_cmd,
        env={"CAPSTONE_DB_URL": DB_URL, "PYTHONUNBUFFERED": "1"},
        execution_timeout=timedelta(minutes=15),
    )

    success = EmptyOperator(task_id="success")

    start >> run_loader >> run_check >> success
