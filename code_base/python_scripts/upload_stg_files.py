#!/usr/bin/env python3
"""
load_to_postgres.py

Loads Excel sheets into Postgres. When a JSON mapping is supplied, only files and sheets
present in the mapping are processed. Mapping must be per-sheet only (no file-level defaults).

Mapping schema (per-sheet only):
{
  "Inventory.xlsx": {
    "Sheet1": {
      "if_exists": "append",
      "incremental": true,
      "pk": ["inventory_id"]
    }
  }
}

Defaults if a sheet option is missing:
  - if_exists -> "replace"
  - incremental -> false

Place this file in:
  /home/anmol_ubuntu/repos/capstone/python_scripts/load_to_postgres.py
"""

import json
import hashlib
import re
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine.base import Engine
from tqdm import tqdm


# ----------------------------
# Utilities
# ----------------------------
def sanitize_identifier(s: str, max_len: int = 63) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = re.sub(r'_+', '_', s)
    s = s.strip('_')
    if not s:
        s = "table"
    if len(s) > max_len:
        s = s[:max_len].rstrip('_')
    return s


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    newcols = []
    for c in df.columns:
        if isinstance(c, str):
            nc = re.sub(r'\s+', '_', c.strip())
            nc = re.sub(r'[^A-Za-z0-9_]', '', nc)
            nc = nc.lower()
            if nc == '':
                nc = 'col'
            newcols.append(nc)
        else:
            newcols.append(str(c))
    # make unique
    seen = {}
    for i, c in enumerate(newcols):
        if c in seen:
            seen[c] += 1
            newcols[i] = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
    df.columns = newcols
    return df


def compute_row_hash(df: pd.DataFrame) -> pd.Series:
    def row_hash(row):
        parts = []
        for v in row:
            if pd.isna(v):
                parts.append("<NA>")
            elif hasattr(v, "isoformat"):
                try:
                    parts.append(v.isoformat())
                except Exception:
                    parts.append(str(v))
            else:
                parts.append(str(v))
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return df.apply(row_hash, axis=1)


def create_engine_from_url(db_url: str) -> Engine:
    return create_engine(db_url, pool_pre_ping=True)


# ----------------------------
# DB helpers
# ----------------------------
def _table_exists(engine: Engine, schema: str, table: str) -> bool:
    q = text("""
        SELECT EXISTS (
          SELECT 1 FROM information_schema.tables
          WHERE table_schema = :schema AND table_name = :table
        )
    """)
    with engine.begin() as conn:
        return bool(conn.execute(q, {"schema": schema, "table": table}).scalar())


def _drop_table_if_exists(engine: Engine, schema: str, table: str):
    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "{schema}"."{table}" CASCADE;'))


def _get_table_rowcount(engine: Engine, schema: str, table: str) -> int:
    q = text(f'SELECT count(*) FROM "{schema}"."{table}"')
    with engine.begin() as conn:
        return int(conn.execute(q).scalar() or 0)


def _get_table_columns_from_df(engine: Engine, schema: str, table: str) -> List[str]:
    q = text("""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
        ORDER BY ordinal_position
    """)
    with engine.begin() as conn:
        rows = conn.execute(q, {"schema": schema, "table": table}).fetchall()
    return [r[0] for r in rows]


# ----------------------------
# staging / insert helpers
# ----------------------------
def _write_df_to_temp_table(engine: Engine, df: pd.DataFrame, temp_table: str, schema: str, chunksize: int):
    df.to_sql(name=temp_table, con=engine, schema=schema, if_exists='replace', index=False, method='multi', chunksize=chunksize)


def _insert_new_rows_by_pk(engine: Engine, schema: str, target_table: str, temp_table: str, pk_cols: List[str]) -> int:
    pk_conditions = " AND ".join([f"t.\"{col}\" = s.\"{col}\"" for col in pk_cols])
    col_list = _get_table_columns_from_df(engine, schema, temp_table)
    cols_projection = ", ".join([f"s.\"{c}\"" for c in col_list])

    insert_sql = f"""
    WITH inserted AS (
      INSERT INTO "{schema}"."{target_table}" ({', '.join(['"' + c + '"' for c in col_list])})
      SELECT {cols_projection}
      FROM "{schema}"."{temp_table}" s
      WHERE NOT EXISTS (
        SELECT 1 FROM "{schema}"."{target_table}" t WHERE {pk_conditions}
      )
      RETURNING 1
    )
    SELECT count(*) as cnt FROM inserted;
    """
    with engine.begin() as conn:
        res = conn.execute(text(insert_sql))
        cnt = res.scalar() or 0
    return int(cnt)


def _insert_new_rows_by_hash(engine: Engine, schema: str, target_table: str, temp_table: str, hash_col: str = "_row_hash") -> int:
    with engine.begin() as conn:
        conn.execute(text(f"""
            ALTER TABLE "{schema}"."{target_table}"
            ADD COLUMN IF NOT EXISTS "{hash_col}" text;
        """))
    col_list = _get_table_columns_from_df(engine, schema, temp_table)
    cols_projection = ", ".join([f"s.\"{c}\"" for c in col_list])

    insert_sql = f"""
    WITH inserted AS (
      INSERT INTO "{schema}"."{target_table}" ({', '.join(['"' + c + '"' for c in col_list])})
      SELECT {cols_projection}
      FROM "{schema}"."{temp_table}" s
      WHERE NOT EXISTS (
        SELECT 1 FROM "{schema}"."{target_table}" t WHERE t."{hash_col}" = s."{hash_col}"
      )
      RETURNING 1
    )
    SELECT count(*) as cnt FROM inserted;
    """
    with engine.begin() as conn:
        res = conn.execute(text(insert_sql))
        cnt = res.scalar() or 0
    return int(cnt)


# ----------------------------
# Main loader (mapping per-sheet only)
# ----------------------------
def load_excel_dir_to_postgres(
    db_url: str,
    files_dir: str = "/home/anmol_ubuntu/repos/capstone/files",
    pattern: str = "*.xls*",
    schema: str = "public",
    default_if_exists: str = "replace",
    chunksize: int = 5000,
    verbose: bool = True,
    mapping: Optional[Dict] = None,
):
    engine = create_engine_from_url(db_url)
    files_dir = Path(files_dir)

    # If mapping provided, only process files listed in mapping (allow stem or exact name)
    if mapping:
        candidate = set(mapping.keys())
        all_files = []
        for p in files_dir.glob(pattern):
            if not p.is_file():
                continue
            if p.name in candidate or p.stem in candidate:
                all_files.append(p)
        all_files = sorted(all_files)
    else:
        # If no mapping provided, process all excel files (but user requested mapping; still support fallback)
        all_files = sorted([p for p in files_dir.glob(pattern) if p.is_file()])

    if not all_files:
        raise FileNotFoundError(f"No Excel files found to process in {files_dir} (mapping used: {bool(mapping)})")

    summary = []

    for file_path in all_files:
        # pick mapping entry by exact name or stem
        map_entry = {}
        if mapping:
            if file_path.name in mapping:
                map_entry = mapping[file_path.name]
            elif file_path.stem in mapping:
                map_entry = mapping[file_path.stem]
            else:
                # Shouldn't happen due to selection above, but skip defensively
                if verbose:
                    print(f"Skipping {file_path.name} — not in mapping")
                continue

        try:
            if verbose:
                print(f"\nProcessing file: {file_path.name}")
            xls = pd.ExcelFile(file_path, engine='openpyxl')
            sheet_names = xls.sheet_names

            for sheet in tqdm(sheet_names, desc=f"  sheets in {file_path.name}", unit="sheet"):
                try:
                    # sheet options must be present under map_entry if mapping provided
                    sheet_map = map_entry.get(sheet, {}) if map_entry else {}
                    # If mapping provided and sheet not present: skip it (mapping-only behavior)
                    if mapping and not sheet_map:
                        if verbose:
                            print(f"    - skipping sheet '{sheet}' (not present in mapping for this file)")
                        summary.append((file_path.name, sheet, 0, "skipped-not-in-mapping"))
                        continue

                    # defaults per sheet
                    sheet_if_exists = sheet_map.get("if_exists", default_if_exists)
                    sheet_incremental = bool(sheet_map.get("incremental", False))
                    pk_cols = sheet_map.get("pk")  # may be None
                    use_hash_for_sheet = bool(sheet_map.get("use_hash", False))

                    df = pd.read_excel(xls, sheet_name=sheet)
                    if df.empty:
                        if verbose:
                            print(f"    - sheet '{sheet}' empty → skipping")
                        summary.append((file_path.name, sheet, 0, "skipped-empty"))
                        continue

                    df = sanitize_columns(df)

                    base_name = sanitize_identifier(file_path.stem)
                    sheet_s = sanitize_identifier(sheet)
                    target_table = sanitize_identifier(f"{base_name}__{sheet_s}")

                    if not sheet_incremental or sheet_if_exists == "replace":
                        if verbose:
                            print(f"    → writing table '{schema}.{target_table}' (mode: {sheet_if_exists}) rows={len(df)}")
                        df.to_sql(name=target_table, con=engine, schema=schema, if_exists=sheet_if_exists, index=False, method='multi', chunksize=chunksize)
                        summary.append((file_path.name, sheet, len(df), "loaded-full"))
                        continue

                    # ---------- Incremental flow ----------
                    # Decide whether to use pk-based or hash-based incremental:
                    use_hash = use_hash_for_sheet or (pk_cols is None)
                    if use_hash:
                        hash_col = "_row_hash"
                        if hash_col in df.columns:
                            hash_col = "_row_hash_auto"
                        df[hash_col] = compute_row_hash(df)
                        temp_table = f"staging__{target_table}"
                        if verbose:
                            print(f"    → incremental (hash) -> target: {schema}.{target_table}, staging: {schema}.{temp_table} rows={len(df)}")
                        _write_df_to_temp_table(engine, df, temp_table, schema, chunksize)

                        if not _table_exists(engine, schema, target_table):
                            if verbose:
                                print(f"      target {target_table} doesn't exist → creating and loading all rows (initial load)")
                            with engine.begin() as conn:
                                conn.execute(text(f'CREATE TABLE "{schema}"."{target_table}" (LIKE "{schema}"."{temp_table}" INCLUDING ALL);'))
                                conn.execute(text(f'INSERT INTO "{schema}"."{target_table}" SELECT * FROM "{schema}"."{temp_table}";'))
                            summary.append((file_path.name, sheet, len(df), "created-and-loaded"))
                        else:
                            inserted = _insert_new_rows_by_hash(engine, schema, target_table, temp_table, hash_col=hash_col)
                            summary.append((file_path.name, sheet, inserted, "incremental-hash"))
                        _drop_table_if_exists(engine, schema, temp_table)

                    else:
                        # PK-based incremental: ensure pk names match sanitized df columns
                        missing = [c for c in pk_cols if c not in df.columns]
                        if missing:
                            if verbose:
                                print(f"    ! PK columns {missing} not found in sanitized columns -> fallback to hash incremental")
                            df["_row_hash"] = compute_row_hash(df)
                            temp_table = f"staging__{target_table}"
                            _write_df_to_temp_table(engine, df, temp_table, schema, chunksize)
                            if not _table_exists(engine, schema, target_table):
                                with engine.begin() as conn:
                                    conn.execute(text(f'CREATE TABLE "{schema}"."{target_table}" (LIKE "{schema}"."{temp_table}" INCLUDING ALL);'))
                                    conn.execute(text(f'INSERT INTO "{schema}"."{target_table}" SELECT * FROM "{schema}"."{temp_table}";'))
                                summary.append((file_path.name, sheet, len(df), "created-and-loaded-fallback-hash"))
                            else:
                                inserted = _insert_new_rows_by_hash(engine, schema, target_table, temp_table, hash_col="_row_hash")
                                summary.append((file_path.name, sheet, inserted, "incremental-hash-fallback"))
                            _drop_table_if_exists(engine, schema, temp_table)
                        else:
                            temp_table = f"staging__{target_table}"
                            if verbose:
                                print(f"    → incremental (pk: {pk_cols}) -> target: {schema}.{target_table}, staging: {schema}.{temp_table} rows={len(df)}")
                            _write_df_to_temp_table(engine, df, temp_table, schema, chunksize)
                            if not _table_exists(engine, schema, target_table):
                                if verbose:
                                    print(f"      target {target_table} doesn't exist → creating and loading all rows (initial load)")
                                with engine.begin() as conn:
                                    conn.execute(text(f'CREATE TABLE "{schema}"."{target_table}" (LIKE "{schema}"."{temp_table}" INCLUDING ALL);'))
                                    conn.execute(text(f'INSERT INTO "{schema}"."{target_table}" SELECT * FROM "{schema}"."{temp_table}";'))
                                summary.append((file_path.name, sheet, len(df), "created-and-loaded"))
                            else:
                                inserted = _insert_new_rows_by_pk(engine, schema, target_table, temp_table, pk_cols)
                                summary.append((file_path.name, sheet, inserted, "incremental-pk"))
                            _drop_table_if_exists(engine, schema, temp_table)

                except Exception as e_sheet:
                    summary.append((file_path.name, sheet, 0, f"error:{str(e_sheet)}"))
                    if verbose:
                        print(f"    ! Error loading sheet '{sheet}': {e_sheet}")

        except Exception as e_file:
            if verbose:
                print(f"! Error processing file {file_path.name}: {e_file}")
            summary.append((file_path.name, None, 0, f"error:{str(e_file)}"))

    # Summary
    if verbose:
        print("\nSummary:")
        for fname, sheet, rows, status in summary:
            sname = sheet if sheet is not None else "<file-level>"
            print(f" - {fname} | {sname:20} | rows={rows:6} | {status}")

    return summary


# ----------------------------
# CLI
# ----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Load Excel sheets into Postgres using per-sheet-only JSON mapping.")
    parser.add_argument("--db-url", required=True, help="SQLAlchemy DB URL (postgresql://user:pass@host:5432/db)")
    parser.add_argument("--files-dir", default="/home/anmol_ubuntu/repos/capstone/files", help="Directory with Excel files")
    parser.add_argument("--mapping", help="Path to JSON mapping file (required if you want mapping-driven behavior)")
    parser.add_argument("--schema", default="public", help="Target DB schema")
    parser.add_argument("--if-exists", default="replace", choices=["replace", "append", "fail"], help="Default if_exists when sheet option omitted")
    parser.add_argument("--chunksize", type=int, default=5000, help="Rows per insert chunk")
    args = parser.parse_args()

    mapping = None
    if args.mapping:
        with open(args.mapping, "r", encoding="utf-8") as fh:
            mapping = json.load(fh)

    load_excel_dir_to_postgres(
        db_url=args.db_url,
        files_dir=args.files_dir,
        schema=args.schema,
        default_if_exists=args.if_exists,
        chunksize=args.chunksize,
        verbose=True,
        mapping=mapping
    )


if __name__ == "__main__":
    main()
