import streamlit as st
import os
import tempfile
from brahma_engine import BrahmaEngine

st.set_page_config(
    page_title="Brahma — The Creator Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background: #0a0a0a;
        color: #e8e8e8;
    }

    .main-header {
        text-align: center;
        padding: 48px 0 32px 0;
    }

    .brahma-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 52px;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: 0.08em;
        margin: 0;
    }

    .brahma-sub {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 14px;
        color: #666;
        letter-spacing: 0.15em;
        margin-top: 8px;
    }

    .brahma-tagline {
        font-size: 16px;
        color: #888;
        margin-top: 16px;
        font-style: italic;
    }

    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        color: #555;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-bottom: 12px;
        margin-top: 28px;
    }

    .credential-warning {
        background: #1a1200;
        border: 1px solid #3d2e00;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 13px;
        color: #888;
        margin-bottom: 16px;
        font-family: 'IBM Plex Mono', monospace;
    }

    .stage-badge {
        display: inline-block;
        background: #111;
        border: 1px solid #333;
        border-radius: 4px;
        padding: 3px 10px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        color: #888;
        margin: 4px 4px 4px 0;
    }

    .stage-badge.active {
        border-color: #f59e0b;
        color: #f59e0b;
    }

    .stage-badge.done {
        border-color: #22c55e;
        color: #22c55e;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stSelectbox"] select {
        background: #111 !important;
        border: 1px solid #2a2a2a !important;
        color: #e8e8e8 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 13px !important;
        border-radius: 6px !important;
    }

    div[data-testid="stTextInput"] input[type="password"] {
        letter-spacing: 0.2em;
    }

    .stButton > button {
        background: #f59e0b !important;
        color: #000 !important;
        border: none !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 600 !important;
        letter-spacing: 0.1em !important;
        font-size: 14px !important;
        padding: 12px 32px !important;
        border-radius: 6px !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        background: #d97706 !important;
    }

    .output-block {
        background: #0f0f0f;
        border: 1px solid #1f1f1f;
        border-radius: 8px;
        padding: 20px 24px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        color: #ccc;
        line-height: 1.8;
        margin-top: 16px;
        white-space: pre-wrap;
    }

    .pipeline-banner {
        background: #0d1a0d;
        border: 1px solid #1a3a1a;
        border-radius: 8px;
        padding: 20px 24px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        color: #22c55e;
        margin-top: 16px;
    }

    .divider {
        border: none;
        border-top: 1px solid #1a1a1a;
        margin: 32px 0;
    }

    .source-chip {
        display: inline-block;
        background: #111;
        border: 1px solid #222;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 12px;
        color: #666;
        margin: 3px;
        font-family: 'IBM Plex Mono', monospace;
    }

    [data-testid="stFileUploader"] {
        background: #111 !important;
        border: 1px dashed #333 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <p class="brahma-title">⚡ BRAHMA</p>
    <p class="brahma-sub">THE CREATOR INTELLIGENCE</p>
    <p class="brahma-tagline">"Tell me your goal and your data source. Nothing else is required."</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── SESSION STATE ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "confirmed" not in st.session_state:
    st.session_state.confirmed = False
if "brahma_understanding" not in st.session_state:
    st.session_state.brahma_understanding = None


# ── INPUT AREA ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.2, 1], gap="large")

with col_left:
    st.markdown('<p class="section-label">01 — Your Goal</p>', unsafe_allow_html=True)
    goal = st.text_area(
        label="goal",
        label_visibility="hidden",
        placeholder='e.g. "Predict which customers will churn next month"\n"Segment our retail customers by purchase behaviour"\n"Forecast demand for next quarter across SKUs"',
        height=120,
        key="goal_input"
    )

    st.markdown('<p class="section-label">02 — Data Source</p>', unsafe_allow_html=True)
    source_type = st.selectbox(
        label="source_type",
        label_visibility="hidden",
        options=[
            "Upload a file (CSV / Excel / JSON / Parquet)",
            "Snowflake",
            "PostgreSQL",
            "MySQL",
            "BigQuery",
            "AWS S3",
            "Azure Blob Storage",
            "Google Cloud Storage",
            "Google Sheets",
            "REST API",
            "SQLite (file path)",
        ],
        key="source_type"
    )

with col_right:
    st.markdown('<p class="section-label">Supported Sources</p>', unsafe_allow_html=True)
    for chip in ["CSV", "Excel", "Parquet", "JSON", "Snowflake", "PostgreSQL",
                 "MySQL", "BigQuery", "S3", "Azure Blob", "GCS", "Google Sheets", "REST API"]:
        st.markdown(f'<span class="source-chip">{chip}</span>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("""
    <div class="credential-warning">
        🔒 All credentials are masked in the UI and never logged or stored.
        Connection strings are held in session memory only and discarded on page close.
    </div>
    """, unsafe_allow_html=True)


# ── DYNAMIC CREDENTIAL FORM ───────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<p class="section-label">03 — Connection Details</p>', unsafe_allow_html=True)

connection_config = {}
uploaded_file = None

if source_type == "Upload a file (CSV / Excel / JSON / Parquet)":
    uploaded_file = st.file_uploader(
        "Drop your file here",
        type=["csv", "xlsx", "xls", "json", "parquet"],
        label_visibility="visible"
    )
    if uploaded_file:
        connection_config["type"] = "file"
        connection_config["filename"] = uploaded_file.name

elif source_type == "Snowflake":
    c1, c2 = st.columns(2)
    with c1:
        sf_account   = st.text_input("Account identifier", placeholder="xyz12345.us-east-1")
        sf_user      = st.text_input("Username")
        sf_warehouse = st.text_input("Warehouse", placeholder="COMPUTE_WH")
        sf_database  = st.text_input("Database", placeholder="PROD_DB")
    with c2:
        sf_password  = st.text_input("Password", type="password")
        sf_role      = st.text_input("Role (optional)", placeholder="ANALYST")
        sf_schema    = st.text_input("Schema", placeholder="PUBLIC")
        sf_table     = st.text_input("Table or SQL query", placeholder="CUSTOMER_EVENTS or SELECT * FROM ...")
    connection_config = {
        "type": "snowflake",
        "account": sf_account, "user": sf_user,
        "password": sf_password, "warehouse": sf_warehouse,
        "database": sf_database, "schema": sf_schema,
        "role": sf_role, "table_or_query": sf_table
    }

elif source_type == "PostgreSQL":
    c1, c2 = st.columns(2)
    with c1:
        pg_host = st.text_input("Host", placeholder="localhost or IP")
        pg_port = st.text_input("Port", value="5432")
        pg_db   = st.text_input("Database name")
    with c2:
        pg_user     = st.text_input("Username")
        pg_password = st.text_input("Password", type="password")
        pg_table    = st.text_input("Table or SQL query")
    connection_config = {
        "type": "postgresql",
        "host": pg_host, "port": pg_port, "database": pg_db,
        "user": pg_user, "password": pg_password,
        "table_or_query": pg_table
    }

elif source_type == "MySQL":
    c1, c2 = st.columns(2)
    with c1:
        my_host = st.text_input("Host", placeholder="localhost")
        my_port = st.text_input("Port", value="3306")
        my_db   = st.text_input("Database name")
    with c2:
        my_user     = st.text_input("Username")
        my_password = st.text_input("Password", type="password")
        my_table    = st.text_input("Table or SQL query")
    connection_config = {
        "type": "mysql",
        "host": my_host, "port": my_port, "database": my_db,
        "user": my_user, "password": my_password,
        "table_or_query": my_table
    }

elif source_type == "BigQuery":
    bq_project  = st.text_input("GCP Project ID")
    bq_dataset  = st.text_input("Dataset")
    bq_table    = st.text_input("Table or SQL query")
    bq_creds    = st.text_area(
        "Service Account JSON (paste contents)",
        height=100,
        placeholder='{"type": "service_account", "project_id": "...", ...}'
    )
    connection_config = {
        "type": "bigquery",
        "project": bq_project, "dataset": bq_dataset,
        "table_or_query": bq_table, "credentials_json": bq_creds
    }

elif source_type == "AWS S3":
    c1, c2 = st.columns(2)
    with c1:
        s3_bucket    = st.text_input("Bucket name")
        s3_key       = st.text_input("File path (key)", placeholder="data/customers.csv")
        s3_region    = st.text_input("Region", placeholder="us-east-1")
    with c2:
        s3_access    = st.text_input("Access Key ID", type="password")
        s3_secret    = st.text_input("Secret Access Key", type="password")
        s3_filetype  = st.selectbox("File format", ["csv", "parquet", "json", "excel"])
    connection_config = {
        "type": "s3",
        "bucket": s3_bucket, "key": s3_key, "region": s3_region,
        "access_key": s3_access, "secret_key": s3_secret,
        "file_format": s3_filetype
    }

elif source_type == "Azure Blob Storage":
    c1, c2 = st.columns(2)
    with c1:
        az_account   = st.text_input("Storage account name")
        az_container = st.text_input("Container name")
        az_blob      = st.text_input("Blob path", placeholder="data/customers.parquet")
    with c2:
        az_key       = st.text_input("Account key or SAS token", type="password")
        az_format    = st.selectbox("File format", ["csv", "parquet", "json", "excel"])
    connection_config = {
        "type": "azure_blob",
        "account": az_account, "container": az_container,
        "blob": az_blob, "key": az_key, "file_format": az_format
    }

elif source_type == "Google Cloud Storage":
    c1, c2 = st.columns(2)
    with c1:
        gcs_bucket  = st.text_input("Bucket name")
        gcs_path    = st.text_input("File path", placeholder="data/customers.csv")
        gcs_format  = st.selectbox("File format", ["csv", "parquet", "json", "excel"])
    with c2:
        gcs_creds   = st.text_area(
            "Service Account JSON",
            height=120,
            placeholder='{"type": "service_account", ...}'
        )
    connection_config = {
        "type": "gcs",
        "bucket": gcs_bucket, "path": gcs_path,
        "file_format": gcs_format, "credentials_json": gcs_creds
    }

elif source_type == "Google Sheets":
    gs_url   = st.text_input("Google Sheet URL", placeholder="https://docs.google.com/spreadsheets/d/...")
    gs_tab   = st.text_input("Sheet tab name (optional)", placeholder="Sheet1")
    gs_creds = st.text_area(
        "Service Account JSON",
        height=100,
        placeholder='{"type": "service_account", ...}'
    )
    connection_config = {
        "type": "google_sheets",
        "url": gs_url, "tab": gs_tab, "credentials_json": gs_creds
    }

elif source_type == "REST API":
    c1, c2 = st.columns(2)
    with c1:
        api_url    = st.text_input("Endpoint URL", placeholder="https://api.example.com/data")
        api_method = st.selectbox("Method", ["GET", "POST"])
        api_path   = st.text_input("JSON path to data", placeholder="data.records or leave blank for root")
    with c2:
        api_key    = st.text_input("API Key / Bearer Token (if required)", type="password")
        api_headers = st.text_area("Extra headers (JSON)", height=80, placeholder='{"X-Custom": "value"}')
        api_body    = st.text_area("Request body (POST only, JSON)", height=80, placeholder='{"filters": {}}')
    connection_config = {
        "type": "rest_api",
        "url": api_url, "method": api_method, "json_path": api_path,
        "api_key": api_key, "headers": api_headers, "body": api_body
    }

elif source_type == "SQLite (file path)":
    sqlite_path  = st.text_input("File path on server", placeholder="/data/local.db")
    sqlite_table = st.text_input("Table name or SQL query")
    connection_config = {
        "type": "sqlite",
        "path": sqlite_path, "table_or_query": sqlite_table
    }


# ── RUN BUTTON ────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)

ready = bool(goal and (uploaded_file or any(
    v for k, v in connection_config.items()
    if k not in ["type", "file_format", "method"] and v
)))

run_col, _ = st.columns([1, 2])
with run_col:
    run_clicked = st.button(
        "⚡  WAKE UP BRAHMA",
        disabled=not ready,
        key="run_brahma"
    )


# ── PIPELINE OUTPUT ───────────────────────────────────────────────────────────
if run_clicked and ready:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">04 — Brahma Running</p>', unsafe_allow_html=True)

    # ── Save uploaded file to temp location ─────────────────────────────────
    temp_path = None
    if uploaded_file:
        suffix = "." + uploaded_file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name
        connection_config["temp_path"] = temp_path

    # ── Mask credentials for display ────────────────────────────────────────
    masked_config = {
        k: ("••••••••" if any(s in k for s in ["password", "key", "secret", "token", "json", "creds"]) else v)
        for k, v in connection_config.items()
    }

    # ── Stage tracker ────────────────────────────────────────────────────────
    stages = ["EDA", "Features", "Train", "Evaluate", "Validate", "Ensemble", "UAT", "Deploy"]
    stage_placeholder = st.empty()

    def render_stages(active_idx):
        badges = ""
        for i, s in enumerate(stages):
            if i < active_idx:
                cls = "done"
            elif i == active_idx:
                cls = "active"
            else:
                cls = ""
            badges += f'<span class="stage-badge {cls}">{s}</span>'
        stage_placeholder.markdown(badges, unsafe_allow_html=True)

    render_stages(-1)

    # ── Brahma response stream ────────────────────────────────────────────────
    output_area = st.empty()
    full_response = ""

    engine = BrahmaEngine()

    with st.spinner(""):
        for chunk, stage_idx in engine.run(goal, connection_config, masked_config):
            full_response += chunk
            output_area.markdown(
                f'<div class="output-block">{full_response}</div>',
                unsafe_allow_html=True
            )
            if stage_idx >= 0:
                render_stages(stage_idx)

    render_stages(len(stages))

    # ── Outputs section ───────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">05 — Outputs</p>', unsafe_allow_html=True)

    out_col1, out_col2, out_col3 = st.columns(3)

    with out_col1:
        charts_dir = "outputs/charts"
        if os.path.exists(charts_dir):
            chart_files = []
            for root, _, files in os.walk(charts_dir):
                for f in files:
                    if f.endswith(".png"):
                        chart_files.append(os.path.join(root, f))
            if chart_files:
                st.markdown("**Charts**")
                for cf in chart_files[:8]:
                    st.image(cf, use_column_width=True)

    with out_col2:
        models_dir = "outputs/models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
            if model_files:
                st.markdown("**Models**")
                for mf in model_files:
                    mpath = os.path.join(models_dir, mf)
                    with open(mpath, "rb") as mfile:
                        st.download_button(
                            label=f"⬇ {mf}",
                            data=mfile,
                            file_name=mf,
                            mime="application/octet-stream",
                            key=f"dl_{mf}"
                        )

    with out_col3:
        data_dir = "outputs/data"
        if os.path.exists(data_dir):
            data_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
            if data_files:
                st.markdown("**Data**")
                for df_name in data_files:
                    df_path = os.path.join(data_dir, df_name)
                    with open(df_path, "rb") as df_file:
                        st.download_button(
                            label=f"⬇ {df_name}",
                            data=df_file,
                            file_name=df_name,
                            mime="text/csv",
                            key=f"dl_{df_name}"
                        )

    # ── Cleanup temp file ─────────────────────────────────────────────────────
    if temp_path and os.path.exists(temp_path):
        os.unlink(temp_path)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; font-family:'IBM Plex Mono',monospace; font-size:12px; color:#333; padding-bottom:40px;">
    BRAHMA © 2026 · IntelliBridge · Built by Aakash Verma
</div>
""", unsafe_allow_html=True)
