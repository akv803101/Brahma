import os
import json
import importlib.util
import anthropic

SKILLS_DIR  = "skills"
AGENTS_DIR  = "agents"
CLAUDE_MD   = "CLAUDE.md"

STAGE_SCRIPTS = [
    ("stage3_eda",       "EDA"),
    ("stage4_features",  "Features"),
    ("stage6_train",     "Train"),
    ("stage7_evaluate",  "Evaluate"),
    ("stage8_validate",  "Validate"),
    ("stage9_ensemble",  "Ensemble"),
    ("stage10_uat",      "UAT"),
    ("stage11_deploy",   "Deploy"),
]


class BrahmaEngine:

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("anthropic_api_key")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Go to Streamlit Cloud → your app → Settings → Secrets and add:\n"
                "[secrets]\nANTHROPIC_API_KEY = 'sk-ant-...'"
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.system_prompt = self._build_system_prompt()

    # ── Load all .md files into a single system prompt ──────────────────────
    def _load_md(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def _build_system_prompt(self) -> str:
        parts = []

        # CLAUDE.md — Brahma identity
        claude_md = self._load_md(CLAUDE_MD)
        if claude_md:
            parts.append(f"# BRAHMA IDENTITY\n{claude_md}")

        # All agent .md files
        if os.path.isdir(AGENTS_DIR):
            for fname in sorted(os.listdir(AGENTS_DIR)):
                if fname.endswith(".md"):
                    content = self._load_md(os.path.join(AGENTS_DIR, fname))
                    if content:
                        parts.append(f"# AGENT: {fname}\n{content}")

        # All skill .md files
        if os.path.isdir(SKILLS_DIR):
            for fname in sorted(os.listdir(SKILLS_DIR)):
                if fname.endswith(".md"):
                    content = self._load_md(os.path.join(SKILLS_DIR, fname))
                    if content:
                        parts.append(f"# SKILL: {fname}\n{content}")

        # Web app context — tells Brahma it's running in Streamlit
        parts.append("""
# DEPLOYMENT CONTEXT
You are running as a public web application via Streamlit.
The user has provided their goal and data source through a web form.
You do NOT need to ask for the goal or data source — they are already provided below.
Proceed directly with your activation protocol, echo your understanding, then confirm.
Because this is a web app, format your responses clearly with sections and line breaks.
Do not use terminal-style box drawing characters — use plain text headers instead.
""")

        return "\n\n---\n\n".join(parts)

    # ── Build the masked data source description for Claude ─────────────────
    def _describe_source(self, config: dict, masked: dict) -> str:
        src_type = config.get("type", "unknown")

        if src_type == "file":
            return f"FILE UPLOAD — filename: {config.get('filename', 'unknown')}"

        if src_type == "snowflake":
            return (
                f"SNOWFLAKE — account: {masked.get('account')} | "
                f"warehouse: {masked.get('warehouse')} | "
                f"database: {masked.get('database')} | "
                f"schema: {masked.get('schema')} | "
                f"table/query: {masked.get('table_or_query')} | "
                f"user: {masked.get('user')} | password: {masked.get('password')}"
            )

        if src_type in ("postgresql", "mysql"):
            return (
                f"{src_type.upper()} — host: {masked.get('host')} | "
                f"port: {masked.get('port')} | "
                f"database: {masked.get('database')} | "
                f"user: {masked.get('user')} | password: {masked.get('password')} | "
                f"table/query: {masked.get('table_or_query')}"
            )

        if src_type == "bigquery":
            return (
                f"BIGQUERY — project: {masked.get('project')} | "
                f"dataset: {masked.get('dataset')} | "
                f"table/query: {masked.get('table_or_query')} | "
                f"service account: {masked.get('credentials_json')}"
            )

        if src_type == "s3":
            return (
                f"AWS S3 — bucket: {masked.get('bucket')} | "
                f"key: {masked.get('key')} | "
                f"region: {masked.get('region')} | "
                f"format: {masked.get('file_format')} | "
                f"access key: {masked.get('access_key')} | "
                f"secret: {masked.get('secret_key')}"
            )

        if src_type == "azure_blob":
            return (
                f"AZURE BLOB — account: {masked.get('account')} | "
                f"container: {masked.get('container')} | "
                f"blob: {masked.get('blob')} | "
                f"key: {masked.get('key')} | "
                f"format: {masked.get('file_format')}"
            )

        if src_type == "gcs":
            return (
                f"GOOGLE CLOUD STORAGE — bucket: {masked.get('bucket')} | "
                f"path: {masked.get('path')} | "
                f"format: {masked.get('file_format')} | "
                f"service account: {masked.get('credentials_json')}"
            )

        if src_type == "google_sheets":
            return (
                f"GOOGLE SHEETS — url: {masked.get('url')} | "
                f"tab: {masked.get('tab')} | "
                f"service account: {masked.get('credentials_json')}"
            )

        if src_type == "rest_api":
            return (
                f"REST API — url: {masked.get('url')} | "
                f"method: {masked.get('method')} | "
                f"json path: {masked.get('json_path')} | "
                f"api key: {masked.get('api_key')}"
            )

        if src_type == "sqlite":
            return (
                f"SQLITE — path: {masked.get('path')} | "
                f"table/query: {masked.get('table_or_query')}"
            )

        return f"DATA SOURCE TYPE: {src_type}"

    # ── Build connection code for the actual pipeline scripts ────────────────
    def _build_connection_code(self, config: dict) -> str:
        src = config.get("type", "")

        if src == "file":
            path = config.get("temp_path", "data/data.csv")
            ext  = config.get("filename", "data.csv").split(".")[-1].lower()
            readers = {
                "csv": f"pd.read_csv(r'{path}')",
                "xlsx": f"pd.read_excel(r'{path}')",
                "xls": f"pd.read_excel(r'{path}')",
                "json": f"pd.read_json(r'{path}')",
                "parquet": f"pd.read_parquet(r'{path}')",
            }
            return (
                "import pandas as pd\n"
                f"df = {readers.get(ext, f'pd.read_csv(r\"{path}\")')}"
            )

        if src == "snowflake":
            return (
                "import snowflake.connector, pandas as pd\n"
                "conn = snowflake.connector.connect(\n"
                f"    account='{config['account']}',\n"
                f"    user='{config['user']}',\n"
                f"    password='{config['password']}',\n"
                f"    warehouse='{config['warehouse']}',\n"
                f"    database='{config['database']}',\n"
                f"    schema='{config['schema']}',\n"
                f"    role='{config.get('role', '')}'\n"
                ")\n"
                f"df = pd.read_sql(\"{config['table_or_query']}\", conn)\n"
                "conn.close()"
            )

        if src == "postgresql":
            return (
                "import pandas as pd\nfrom sqlalchemy import create_engine\n"
                f"engine = create_engine('postgresql://{config['user']}:{config['password']}"
                f"@{config['host']}:{config['port']}/{config['database']}')\n"
                f"df = pd.read_sql(\"{config['table_or_query']}\", engine)"
            )

        if src == "mysql":
            return (
                "import pandas as pd\nfrom sqlalchemy import create_engine\n"
                f"engine = create_engine('mysql+pymysql://{config['user']}:{config['password']}"
                f"@{config['host']}:{config['port']}/{config['database']}')\n"
                f"df = pd.read_sql(\"{config['table_or_query']}\", engine)"
            )

        if src == "bigquery":
            return (
                "import pandas as pd\nfrom google.cloud import bigquery\n"
                "import json, tempfile, os\n"
                "creds_dict = json.loads(r'''" + config.get("credentials_json", "{}") + "''')\n"
                "with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:\n"
                "    json.dump(creds_dict, f); creds_path = f.name\n"
                "os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path\n"
                f"client = bigquery.Client(project='{config['project']}')\n"
                f"df = client.query(\"{config['table_or_query']}\").to_dataframe()"
            )

        if src == "s3":
            return (
                "import pandas as pd, boto3, io\n"
                f"s3 = boto3.client('s3', region_name='{config['region']}',\n"
                f"    aws_access_key_id='{config['access_key']}',\n"
                f"    aws_secret_access_key='{config['secret_key']}')\n"
                f"obj = s3.get_object(Bucket='{config['bucket']}', Key='{config['key']}')\n"
                f"df = pd.read_{config.get('file_format','csv')}(io.BytesIO(obj['Body'].read()))"
            )

        if src == "azure_blob":
            return (
                "import pandas as pd, io\n"
                "from azure.storage.blob import BlobServiceClient\n"
                f"client = BlobServiceClient(account_url='https://{config['account']}.blob.core.windows.net',\n"
                f"    credential='{config['key']}')\n"
                f"blob = client.get_blob_client(container='{config['container']}', blob='{config['blob']}')\n"
                "data = blob.download_blob().readall()\n"
                f"df = pd.read_{config.get('file_format','csv')}(io.BytesIO(data))"
            )

        if src == "gcs":
            return (
                "import pandas as pd, io, json, tempfile, os\n"
                "from google.cloud import storage\n"
                "creds_dict = json.loads(r'''" + config.get("credentials_json", "{}") + "''')\n"
                "with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:\n"
                "    json.dump(creds_dict, f); creds_path = f.name\n"
                "os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path\n"
                f"client = storage.Client()\n"
                f"bucket = client.bucket('{config['bucket']}')\n"
                f"blob = bucket.blob('{config['path']}')\n"
                "data = blob.download_as_bytes()\n"
                f"df = pd.read_{config.get('file_format','csv')}(io.BytesIO(data))"
            )

        if src == "google_sheets":
            return (
                "import pandas as pd, gspread, json\n"
                "from google.oauth2.service_account import Credentials\n"
                "creds_dict = json.loads(r'''" + config.get("credentials_json", "{}") + "''')\n"
                "scopes = ['https://www.googleapis.com/auth/spreadsheets.readonly']\n"
                "creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)\n"
                "gc = gspread.authorize(creds)\n"
                f"sh = gc.open_by_url('{config['url']}')\n"
                f"ws = sh.worksheet('{config.get('tab','Sheet1')}')\n"
                "df = pd.DataFrame(ws.get_all_records())"
            )

        if src == "rest_api":
            headers_str = config.get("headers", "{}")
            body_str    = config.get("body", "{}")
            return (
                "import pandas as pd, requests, json\n"
                f"headers = {headers_str or '{}'}\n"
                f"if '{config.get('api_key','')}': headers['Authorization'] = f\"Bearer {config.get('api_key','')}\"\n"
                f"resp = requests.{config.get('method','get').lower()}(\n"
                f"    '{config.get('url','')}',\n"
                f"    headers=headers,\n"
                f"    json={body_str or 'None'}\n"
                ")\n"
                "data = resp.json()\n"
                f"path = '{config.get('json_path','')}'\n"
                "for key in [k for k in path.split('.') if k]: data = data[key]\n"
                "df = pd.DataFrame(data if isinstance(data, list) else [data])"
            )

        if src == "sqlite":
            return (
                "import pandas as pd, sqlite3\n"
                f"conn = sqlite3.connect(r'{config['path']}')\n"
                f"df = pd.read_sql(\"{config['table_or_query']}\", conn)\n"
                "conn.close()"
            )

        return "import pandas as pd\ndf = pd.DataFrame()"

    # ── Run a stage Python script ────────────────────────────────────────────
    def _run_stage(self, script_name: str) -> str:
        script_path = f"{script_name}.py"
        if not os.path.exists(script_path):
            return f"[{script_name}] Script not found — skipping."
        try:
            spec   = importlib.util.spec_from_file_location(script_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return f"[{script_name}] Completed successfully."
        except Exception as e:
            return f"[{script_name}] Error: {str(e)}"

    # ── Inject connection code into stage3_eda.py at runtime ────────────────
    def _inject_connection(self, connection_config: dict) -> None:
        conn_code = self._build_connection_code(connection_config)
        conn_code_tagged = f"# AUTO-INJECTED BY BRAHMA ENGINE\n{conn_code}\n# END INJECTION\n\n"

        for script_name, _ in STAGE_SCRIPTS:
            path = f"{script_name}.py"
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            # Only inject if not already injected
            if "AUTO-INJECTED BY BRAHMA ENGINE" not in content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(conn_code_tagged + content)

    # ── Main run method — streams to Streamlit ───────────────────────────────
    def run(self, goal: str, connection_config: dict, masked_config: dict):
        source_description = self._describe_source(connection_config, masked_config)

        # Inject real connection code into stage scripts
        self._inject_connection(connection_config)

        user_message = (
            f"Wake Up Brahma\n\n"
            f"GOAL: {goal}\n\n"
            f"DATA SOURCE: {source_description}\n\n"
            f"Run the full pipeline. After each stage completes, announce it clearly "
            f"with a header like 'STAGE 3 — EDA COMPLETE' before moving to the next."
        )

        # Phase 1 — Brahma's understanding + confirmation (streamed)
        with self.client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_message}]
        ) as stream:
            for text in stream.text_stream:
                yield text, -1

        yield "\n\n---\n\n", -1

        # Phase 2 — Run each stage script in sequence
        for i, (script_name, stage_label) in enumerate(STAGE_SCRIPTS):
            yield f"\nSTAGE {i+3} — {stage_label.upper()} RUNNING...\n", i
            result = self._run_stage(script_name)
            yield f"{result}\n", i

        yield "\n\n---\n\nPIPELINE COMPLETE. All outputs written to outputs/\n", len(STAGE_SCRIPTS)
