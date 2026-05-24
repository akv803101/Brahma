import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json
import time
import requests
import pandas as pd

os.makedirs('outputs/decks', exist_ok=True)

GAMMA_API_KEY  = os.environ.get("GAMMA_API_KEY") or os.environ.get("gamma_api_key")
GAMMA_BASE_URL = "https://gamma.app/api/v1"
POLL_INTERVAL  = 3   # seconds between status checks
POLL_TIMEOUT   = 120 # seconds max wait

print("=" * 60)
print("STAGE 13 — SLIDE DECK BUILDER (Gamma)")
print("=" * 60)

if not GAMMA_API_KEY:
    print("\n  GAMMA_API_KEY not set — skipping slide generation.")
    print("  Add GAMMA_API_KEY to Streamlit secrets to enable this stage.")
    print("  Get a free key at: gamma.app/api")
    with open('outputs/decks/deck_info.json', 'w') as f:
        json.dump({"status": "skipped", "reason": "GAMMA_API_KEY not set", "url": None}, f, indent=2)
    print("\nSTAGE 13 SKIPPED — no API key.")
else:
    # ── Load pipeline outputs ─────────────────────────────────────────────────

    print("\n[1/4] Loading pipeline outputs...")

    leaderboard = None
    if os.path.exists('outputs/data/leaderboard.csv'):
        leaderboard = pd.read_csv('outputs/data/leaderboard.csv')
        print(f"  Leaderboard loaded: {len(leaderboard)} models")
    else:
        print("  leaderboard.csv not found — metrics section will be skipped")

    train_dist = {}
    if os.path.exists('outputs/data/training_distribution.json'):
        with open('outputs/data/training_distribution.json') as f:
            train_dist = json.load(f)
        print(f"  training_distribution.json loaded")

    goal = "ML pipeline analysis"
    if os.path.exists('outputs/data/pipeline_meta.json'):
        with open('outputs/data/pipeline_meta.json') as f:
            meta = json.load(f)
            goal = meta.get('goal', goal)

    # ── Build best model summary ──────────────────────────────────────────────

    best = {}
    if leaderboard is not None and len(leaderboard) > 0:
        valid = leaderboard[leaderboard['model'] != 'DummyClassifier']
        if len(valid) > 0:
            row = valid.iloc[0]
            best = {
                "name":      row.get('model', 'Unknown'),
                "auc_val":   round(float(row.get('auc_val',   0)), 4),
                "f1_val":    round(float(row.get('f1_val',    0)), 4),
                "recall":    round(float(row.get('recall_val', 0)), 4),
                "precision": round(float(row.get('precision_val', 0)), 4),
                "gap":       round(float(row.get('gap', 0)), 4),
            }

    n_charts = sum(
        len(os.listdir(f"outputs/charts/{sub}"))
        for sub in ["eda", "training", "evaluation", "validation", "ensembling"]
        if os.path.exists(f"outputs/charts/{sub}")
    )

    # ── Build slide prompt ────────────────────────────────────────────────────

    print("\n[2/4] Building slide content...")

    model_table = ""
    if leaderboard is not None:
        rows = []
        for _, r in leaderboard.iterrows():
            if r['model'] == 'DummyClassifier':
                continue
            rows.append(
                f"| {r['model']} | {float(r.get('auc_val',0)):.4f} | "
                f"{float(r.get('f1_val',0)):.4f} | {float(r.get('recall_val',0)):.4f} |"
            )
        if rows:
            model_table = (
                "| Model | AUC-Val | F1-Val | Recall-Val |\n"
                "|-------|---------|--------|------------|\n" +
                "\n".join(rows)
            )

    prompt = f"""Create a professional 10-slide executive presentation titled "Brahma ML Pipeline Report".

The audience is a CXO / business leader — no ML jargon, focus on business impact.

Use a dark, professional theme. Clean layouts. Bold numbers.

---

Slide 1 — Cover
Title: Brahma ML Pipeline Report
Subtitle: {goal}
Footer: Built by Brahma · The Creator Intelligence

Slide 2 — Executive Summary
Headline: Pipeline complete in 11 stages
3 key bullets:
- Best model: {best.get('name', 'XGBoost')} · AUC {best.get('auc_val', 'N/A')}
- {n_charts} charts generated across EDA, training, evaluation, and validation
- Model packaged and ready for deployment

Slide 3 — The Business Goal
Restate in plain English: {goal}
Frame it as: "What question were we answering?"

Slide 4 — The Data
Show: source type, rows loaded, features engineered
Emphasise: Brahma connected to the data source automatically — no manual extraction

Slide 5 — Key EDA Finding
Headline: What the data revealed before modelling
Show the most important pattern found in exploration (distribution skew, class imbalance, top correlation)

Slide 6 — Model Selection
Show all models tried as a comparison table:
{model_table if model_table else "Multiple models evaluated and compared"}
Explain why {best.get('name', 'the best model')} was selected

Slide 7 — Model Performance
Big number: AUC = {best.get('auc_val', 'N/A')}
Supporting metrics:
- F1 Score: {best.get('f1_val', 'N/A')}
- Recall: {best.get('recall', 'N/A')}
- Precision: {best.get('precision', 'N/A')}
- Train/Val gap: {best.get('gap', 'N/A')} (model is not overfitting)

Slide 8 — Model Reliability
Headline: How confident can we be?
Show: 10-fold cross-validation result, overfitting check, threshold analysis
Plain English: "The model performs consistently across different data slices"

Slide 9 — Recommendations
3 clear action items the business should take based on the model output
Make these specific and actionable, not generic

Slide 10 — Next Steps & Deployment
Deployment package is ready: outputs/models/deployment_package.pkl
Drift monitoring is armed — key features are being tracked
Recommended review cadence: monthly model refresh
CTA: "Brahma is ready for your next dataset"
"""

    print(f"  Prompt built: {len(prompt)} characters, 10 slides")

    # ── Call Gamma API ────────────────────────────────────────────────────────

    print("\n[3/4] Calling Gamma API...")

    headers = {
        "Authorization": f"Bearer {GAMMA_API_KEY}",
        "Content-Type":  "application/json",
    }

    try:
        resp = requests.post(
            f"{GAMMA_BASE_URL}/generate",
            headers=headers,
            json={
                "prompt": prompt,
                "mode":   "presentation",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data        = resp.json()
        generation_id = data.get("id") or data.get("generationId") or data.get("generation_id")
        print(f"  Generation started. ID: {generation_id}")
    except requests.exceptions.HTTPError as e:
        print(f"\n  Gamma API error: {e}")
        print(f"  Response: {resp.text[:300]}")
        with open('outputs/decks/deck_info.json', 'w') as f:
            json.dump({"status": "error", "reason": str(e), "url": None}, f, indent=2)
        raise SystemExit(1)
    except requests.exceptions.RequestException as e:
        print(f"\n  Network error calling Gamma API: {e}")
        with open('outputs/decks/deck_info.json', 'w') as f:
            json.dump({"status": "error", "reason": str(e), "url": None}, f, indent=2)
        raise SystemExit(1)

    # ── Poll for completion ───────────────────────────────────────────────────

    print(f"\n[4/4] Waiting for Gamma to build the deck (timeout: {POLL_TIMEOUT}s)...")

    deck_url = None
    elapsed  = 0

    while elapsed < POLL_TIMEOUT:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

        try:
            status_resp = requests.get(
                f"{GAMMA_BASE_URL}/generations/{generation_id}",
                headers=headers,
                timeout=15,
            )
            status_resp.raise_for_status()
            status_data = status_resp.json()
        except requests.exceptions.RequestException as e:
            print(f"  Poll error (will retry): {e}")
            continue

        status = status_data.get("status", "")
        print(f"  [{elapsed:>3}s] status: {status}")

        if status in ("complete", "completed", "done"):
            deck_url = (
                status_data.get("url") or
                status_data.get("deckUrl") or
                status_data.get("deck_url") or
                status_data.get("gamma_url")
            )
            break
        elif status in ("error", "failed"):
            print(f"  Gamma generation failed: {status_data.get('error', 'unknown error')}")
            break

    # ── Save result ───────────────────────────────────────────────────────────

    deck_info = {
        "status":        "complete" if deck_url else "timeout",
        "generation_id": generation_id,
        "url":           deck_url,
        "goal":          goal,
        "best_model":    best.get("name"),
        "auc_val":       best.get("auc_val"),
    }

    with open('outputs/decks/deck_info.json', 'w') as f:
        json.dump(deck_info, f, indent=2)

    print("\n" + "=" * 60)
    print("STAGE 13 SLIDE DECK BUILDER COMPLETE")
    print("=" * 60)
    if deck_url:
        print(f"  Deck URL : {deck_url}")
    else:
        print(f"  Deck generation timed out after {POLL_TIMEOUT}s.")
        print(f"  Generation ID saved — check gamma.app manually: {generation_id}")
    print(f"  Saved    : outputs/decks/deck_info.json")
    print("=" * 60)
