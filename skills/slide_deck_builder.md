# SKILL: Slide Deck Builder (Gamma)

## Purpose
Generates a 10-slide executive presentation from completed pipeline outputs via the Gamma API.
Triggered at Stage 13, after deployment packaging is complete.

## Inputs Read
| File | Contents used |
|------|--------------|
| `outputs/data/leaderboard.csv` | Model names, AUC, F1, recall, precision, train/val gap |
| `outputs/data/training_distribution.json` | Dataset shape and feature stats |
| `outputs/data/pipeline_meta.json` | Original goal string (if saved) |

## Output
| File | Contents |
|------|---------|
| `outputs/decks/deck_info.json` | `{ status, generation_id, url, goal, best_model, auc_val }` |

## Slide Structure (10 slides)
1. Cover — pipeline title + goal
2. Executive Summary — best model, chart count, deployment status
3. The Business Goal — plain-English restatement
4. The Data — source type, rows, features engineered
5. Key EDA Finding — top pattern from exploration
6. Model Selection — comparison table, rationale
7. Model Performance — big AUC number + supporting metrics
8. Model Reliability — cross-validation, overfitting gap
9. Recommendations — 3 specific, actionable business steps
10. Next Steps & Deployment — deployment package, drift monitoring, CTA

## Required Secret
```toml
GAMMA_API_KEY = "your-gamma-api-key"
```
Get a key at: gamma.app/api

## Behaviour When Key Is Missing
Stage 13 skips gracefully — writes `{ "status": "skipped" }` to `deck_info.json` and prints
a clear message. The rest of the pipeline is unaffected.

## API Flow
1. POST `https://gamma.app/api/v1/generate` — start generation, receive `generation_id`
2. Poll `GET https://gamma.app/api/v1/generations/{id}` every 3s (120s timeout)
3. On `status: complete` — extract deck URL, save to `deck_info.json`
