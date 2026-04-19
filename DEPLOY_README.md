# BRAHMA — DEPLOYMENT GUIDE

## Files to add to your GitHub repo root

```
app.py              ← Streamlit frontend (the web UI)
brahma_engine.py    ← Backend: loads .md files, calls Claude API, runs stages
requirements.txt    ← All Python dependencies
.streamlit/
  secrets.toml      ← Your API key (never commit this — Streamlit Cloud handles it)
```

---

## Step 1 — Add files to your repo

Copy app.py, brahma_engine.py, requirements.txt into the ROOT of your Brahma repo
(same level as CLAUDE.md, agents/, skills/, stage3_eda.py etc.)

---

## Step 2 — Add your Anthropic API key

In Streamlit Cloud (NOT in the file — see Step 4 for why):

Go to your app settings → Secrets → paste this:

```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

DO NOT commit a secrets.toml to GitHub. Streamlit Cloud injects it securely.

---

## Step 3 — Deploy on Streamlit Cloud

1. Go to streamlit.io/cloud
2. Sign in with GitHub
3. Click "New app"
4. Repository: akv803101/Brahma
5. Branch: main
6. Main file path: app.py
7. Click "Deploy"

Streamlit gives you a URL like: brahma.streamlit.app

---

## Step 4 — Credential security model

How credentials are protected in this deployment:

| Layer              | Protection                                      |
|--------------------|------------------------------------------------|
| Anthropic API key  | Streamlit Cloud secrets — never in code        |
| DB passwords       | type="password" fields — masked in UI          |
| Service account JSON | Never logged, held in session memory only   |
| Connection strings | Built in memory, never written to disk         |
| Temp files         | Deleted immediately after pipeline completes   |

What this deployment does NOT protect against:
- If you share your Streamlit URL publicly, anyone can use your Anthropic API key
  (their requests = your bill)

Recommended for public deployment:
Add a simple password gate by adding this to the top of app.py:

```python
import streamlit as st
password = st.text_input("Access password", type="password")
if password != st.secrets.get("APP_PASSWORD", ""):
    st.stop()
```

Then add APP_PASSWORD = "your-cohort-password" to Streamlit secrets.

---

## Step 5 — Test locally first

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-your-key-here
streamlit run app.py
```

Opens at: http://localhost:8501

---

## Folder structure after adding these files

```
Brahma/
├── app.py                    ← NEW: Streamlit frontend
├── brahma_engine.py          ← NEW: Backend engine
├── requirements.txt          ← NEW: Updated dependencies
├── CLAUDE.md                 ← Existing: Brahma identity
├── agents/                   ← Existing: Agent .md files
├── skills/                   ← Existing: Skill .md files
├── stage3_eda.py             ← Existing: Stage scripts
├── stage4_features.py
├── stage6_train.py
├── stage7_evaluate.py
├── stage8_validate.py
├── stage9_ensemble.py
├── stage10_uat.py
├── stage11_deploy.py
├── dashboard.py
├── data/                     ← Existing: Sample data
└── outputs/                  ← Existing: Pipeline outputs
```
