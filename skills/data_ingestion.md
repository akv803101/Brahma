# Skill: Data Ingestion

## Purpose
Ingest data from any source, validate integrity, and save a clean raw snapshot ready for downstream processing.

---

## Step 1 — Detect Source Type

Classify the data source string into one of four categories:

| Category   | Indicators                                                                 |
|------------|----------------------------------------------------------------------------|
| FILE       | `.csv`, `.xlsx`, `.xls`, `.parquet`, `.json`, `.xml`, `.tsv`, local path   |
| DATABASE   | `postgresql://`, `mysql://`, `sqlite://`, `mssql://`, connection string    |
| CLOUD      | `bigquery://`, `snowflake://`, `s3://`, `az://`, `gs://`, Google Sheets URL|
| API        | `http://`, `https://`, `graphql`, REST endpoint                            |

Detection logic (Python):
```python
def detect_source_type(source: str) -> str:
    source_lower = source.lower().strip()
    file_exts = ['.csv', '.xlsx', '.xls', '.parquet', '.json', '.xml', '.tsv']
    db_prefixes = ['postgresql://', 'postgres://', 'mysql://', 'sqlite://', 'mssql://']
    cloud_prefixes = ['bigquery://', 'snowflake://', 's3://', 'az://', 'gs://', 'docs.google.com/spreadsheets']
    api_prefixes = ['http://', 'https://']

    if any(source_lower.endswith(ext) for ext in file_exts) or ('/' in source and '://' not in source):
        return 'FILE'
    if any(source_lower.startswith(p) for p in db_prefixes):
        return 'DATABASE'
    if any(p in source_lower for p in cloud_prefixes):
        return 'CLOUD'
    if any(source_lower.startswith(p) for p in api_prefixes):
        return 'API'
    return 'FILE'  # default fallback
```

---

## Step 2 — Write and Execute Loading Code

### FILE
```python
import pandas as pd
from pathlib import Path

def load_file(source: str) -> pd.DataFrame:
    path = Path(source)
    ext = path.suffix.lower()
    loaders = {
        '.csv':     lambda: pd.read_csv(source, encoding='utf-8', on_bad_lines='warn'),
        '.tsv':     lambda: pd.read_csv(source, sep='\t', encoding='utf-8', on_bad_lines='warn'),
        '.xlsx':    lambda: pd.read_excel(source, engine='openpyxl'),
        '.xls':     lambda: pd.read_excel(source, engine='xlrd'),
        '.parquet': lambda: pd.read_parquet(source),
        '.json':    lambda: pd.read_json(source),
        '.xml':     lambda: pd.read_xml(source),
    }
    loader = loaders.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file type: {ext}")
    try:
        return loader()
    except UnicodeDecodeError:
        # Fallback encoding
        if ext in ['.csv', '.tsv']:
            return pd.read_csv(source, encoding='latin-1', on_bad_lines='warn')
        raise
```

### DATABASE
```python
import pandas as pd
from sqlalchemy import create_engine, text

def load_database(connection_string: str, query: str) -> pd.DataFrame:
    engine = create_engine(connection_string)
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    return df

# Usage: load_database("postgresql://user:pass@host:5432/dbname", "SELECT * FROM table_name")
# Usage: load_database("mysql+pymysql://user:pass@host:3306/dbname", "SELECT * FROM table_name")
# Usage: load_database("sqlite:///path/to/file.db", "SELECT * FROM table_name")
```

### CLOUD
```python
import pandas as pd

def load_cloud(source: str, **kwargs) -> pd.DataFrame:
    source_lower = source.lower()

    # BigQuery
    if 'bigquery://' in source_lower:
        from google.cloud import bigquery
        project = kwargs.get('project')
        query = kwargs.get('query')
        client = bigquery.Client(project=project)
        return client.query(query).to_dataframe()

    # Snowflake
    elif 'snowflake://' in source_lower:
        from sqlalchemy import create_engine
        engine = create_engine(source)
        query = kwargs.get('query', 'SELECT * FROM table_name')
        return pd.read_sql(query, engine)

    # S3
    elif source_lower.startswith('s3://'):
        import s3fs
        fs = s3fs.S3FileSystem()
        ext = source.split('.')[-1].lower()
        with fs.open(source) as f:
            if ext == 'csv':    return pd.read_csv(f)
            if ext == 'parquet': return pd.read_parquet(f)
            if ext == 'json':   return pd.read_json(f)
        raise ValueError(f"Unsupported S3 file type: {ext}")

    # Azure Blob
    elif source_lower.startswith('az://'):
        from azure.storage.blob import BlobServiceClient
        conn_str = kwargs.get('connection_string')
        container, blob = source.replace('az://', '').split('/', 1)
        client = BlobServiceClient.from_connection_string(conn_str)
        data = client.get_container_client(container).download_blob(blob).readall()
        import io
        ext = blob.split('.')[-1].lower()
        if ext == 'csv':     return pd.read_csv(io.BytesIO(data))
        if ext == 'parquet': return pd.read_parquet(io.BytesIO(data))
        raise ValueError(f"Unsupported Azure Blob file type: {ext}")

    # GCS
    elif source_lower.startswith('gs://'):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        ext = source.split('.')[-1].lower()
        with fs.open(source) as f:
            if ext == 'csv':    return pd.read_csv(f)
            if ext == 'parquet': return pd.read_parquet(f)
        raise ValueError(f"Unsupported GCS file type: {ext}")

    # Google Sheets
    elif 'docs.google.com/spreadsheets' in source_lower:
        sheet_id = source.split('/d/')[1].split('/')[0]
        export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        return pd.read_csv(export_url)

    raise ValueError(f"Unrecognised CLOUD source: {source}")
```

### API
```python
import pandas as pd
import requests

def load_api(source: str, **kwargs) -> pd.DataFrame:
    headers = kwargs.get('headers', {})
    params  = kwargs.get('params', {})
    auth    = kwargs.get('auth', None)
    json_path = kwargs.get('json_path', None)   # dot-path to records, e.g. "data.items"

    # GraphQL
    if kwargs.get('graphql_query'):
        payload = {'query': kwargs['graphql_query']}
        resp = requests.post(source, json=payload, headers=headers, auth=auth)
        resp.raise_for_status()
        data = resp.json()
    else:
        resp = requests.get(source, headers=headers, params=params, auth=auth)
        resp.raise_for_status()
        data = resp.json()

    # Navigate nested JSON path if provided
    if json_path:
        for key in json_path.split('.'):
            data = data[key]

    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame([data])
    raise ValueError("API response could not be converted to a DataFrame.")
```

---

## Step 3 — Print Dataset Profile

```python
def print_profile(df: pd.DataFrame, source: str, source_type: str):
    print("=" * 60)
    print("  DATASET PROFILE")
    print("=" * 60)
    print(f"  Source      : {source}")
    print(f"  Source Type : {source_type}")
    print(f"  Rows        : {df.shape[0]:,}")
    print(f"  Columns     : {df.shape[1]:,}")
    print("-" * 60)
    print("  COLUMN DTYPES")
    print("-" * 60)
    for col, dtype in df.dtypes.items():
        null_pct = df[col].isna().mean() * 100
        print(f"  {col:<35} {str(dtype):<15} nulls: {null_pct:.1f}%")
    print("-" * 60)
    print("  FIRST 5 ROWS")
    print("-" * 60)
    print(df.head().to_string(index=True))
    print("=" * 60)
```

---

## Step 4 — Run 5 Integrity Checks

```python
def run_integrity_checks(df: pd.DataFrame) -> list[dict]:
    checks = []

    # Check 1: Zero rows
    checks.append({
        'check':  'Zero Rows',
        'passed': len(df) > 0,
        'detail': f"{len(df):,} rows found" if len(df) > 0 else "FATAL: DataFrame is empty"
    })

    # Check 2: >50% empty columns
    empty_cols = [c for c in df.columns if df[c].isna().mean() > 0.5]
    checks.append({
        'check':  '>50% Empty Columns',
        'passed': len(empty_cols) == 0,
        'detail': f"No columns >50% null" if not empty_cols else f"WARNING: {empty_cols}"
    })

    # Check 3: Unparsed date columns
    import re
    date_pattern = re.compile(r'date|time|dt|year|month|day', re.IGNORECASE)
    unparsed_dates = [
        c for c in df.columns
        if date_pattern.search(c) and df[c].dtype == object
    ]
    checks.append({
        'check':  'Unparsed Date Columns',
        'passed': len(unparsed_dates) == 0,
        'detail': f"No unparsed date columns" if not unparsed_dates
                  else f"WARNING: {unparsed_dates} — dtype is object, not datetime"
    })

    # Check 4: Duplicate primary key candidates
    potential_id_cols = [c for c in df.columns if re.search(r'\bid\b|_id$|^id', c, re.IGNORECASE)]
    dup_details = []
    for col in potential_id_cols:
        n_dups = df[col].duplicated().sum()
        if n_dups > 0:
            dup_details.append(f"{col}: {n_dups:,} duplicates")
    checks.append({
        'check':  'Duplicate Key Candidates',
        'passed': len(dup_details) == 0,
        'detail': "No duplicate keys found" if not dup_details else f"WARNING: {dup_details}"
    })

    # Check 5: Encoding / non-UTF-8 artifacts in string columns
    artifact_pattern = re.compile(r'[^\x00-\x7F\u00C0-\u024F]')
    encoding_issues = []
    for col in df.select_dtypes(include='object').columns:
        sample = df[col].dropna().astype(str).head(500)
        hits = sample.apply(lambda x: bool(artifact_pattern.search(x))).sum()
        if hits > 0:
            encoding_issues.append(f"{col}: {hits} suspect values")
    checks.append({
        'check':  'Encoding Issues',
        'passed': len(encoding_issues) == 0,
        'detail': "No encoding artifacts detected" if not encoding_issues
                  else f"WARNING: {encoding_issues}"
    })

    return checks


def print_integrity_report(checks: list[dict]):
    print("\n" + "=" * 60)
    print("  INTEGRITY CHECKS")
    print("=" * 60)
    all_passed = True
    for c in checks:
        status = "PASS" if c['passed'] else "FAIL"
        if not c['passed']:
            all_passed = False
        print(f"  [{status}]  {c['check']}")
        print(f"         {c['detail']}")
    print("-" * 60)
    print(f"  Overall: {'ALL CHECKS PASSED' if all_passed else 'ISSUES DETECTED — REVIEW ABOVE'}")
    print("=" * 60)
```

---

## Step 5 — Save Raw Loaded Data

```python
import os

def save_raw(df: pd.DataFrame, output_path: str = "outputs/data/raw_loaded.parquet"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False, engine='pyarrow')
    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n  Saved raw data → {output_path}  ({size_kb:.1f} KB)")
```

---

## Step 6 — DATA INGESTION COMPLETE Report

```python
import datetime

def print_completion_report(df: pd.DataFrame, source: str, source_type: str,
                             checks: list[dict], output_path: str):
    passed = sum(1 for c in checks if c['passed'])
    total  = len(checks)
    ts     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 60)
    print("  ✅  DATA INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Timestamp     : {ts}")
    print(f"  Source        : {source}")
    print(f"  Source Type   : {source_type}")
    print(f"  Rows Loaded   : {df.shape[0]:,}")
    print(f"  Columns       : {df.shape[1]:,}")
    print(f"  Integrity     : {passed}/{total} checks passed")
    print(f"  Raw Snapshot  : {output_path}")
    print("=" * 60)
    print("  Brahma is ready for EDA.\n")
```

---

## Master Orchestrator

```python
def run_data_ingestion(source: str, **kwargs) -> pd.DataFrame:
    print(f"\nBrahma: Detecting source type for → {source}")
    source_type = detect_source_type(source)
    print(f"Brahma: Source classified as [{source_type}]")

    # Load
    if source_type == 'FILE':
        df = load_file(source)
    elif source_type == 'DATABASE':
        query = kwargs.get('query', 'SELECT * FROM table_name')
        df = load_database(source, query)
    elif source_type == 'CLOUD':
        df = load_cloud(source, **kwargs)
    elif source_type == 'API':
        df = load_api(source, **kwargs)
    else:
        raise ValueError(f"Unknown source type: {source_type}")

    output_path = kwargs.get('output_path', 'outputs/data/raw_loaded.parquet')

    # Profile
    print_profile(df, source, source_type)

    # Integrity checks
    checks = run_integrity_checks(df)
    print_integrity_report(checks)

    # Save
    save_raw(df, output_path)

    # Final report
    print_completion_report(df, source, source_type, checks, output_path)

    return df
```

---

## Usage Examples

```python
# CSV file
df = run_data_ingestion("data/customers.csv")

# Excel file
df = run_data_ingestion("data/sales_report.xlsx")

# PostgreSQL
df = run_data_ingestion(
    "postgresql://user:password@localhost:5432/mydb",
    query="SELECT * FROM transactions WHERE created_at > '2024-01-01'"
)

# BigQuery
df = run_data_ingestion(
    "bigquery://my-project",
    project="my-project",
    query="SELECT * FROM `my_project.my_dataset.my_table` LIMIT 10000"
)

# S3
df = run_data_ingestion("s3://my-bucket/data/events.parquet")

# REST API
df = run_data_ingestion(
    "https://api.example.com/v1/records",
    headers={"Authorization": "Bearer TOKEN"},
    json_path="data.items"
)

# Google Sheets
df = run_data_ingestion("https://docs.google.com/spreadsheets/d/SHEET_ID/edit")
```
