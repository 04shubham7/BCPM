# Project Documentation — Shyamati

This document captures the in-depth design for the training pipeline, API, frontend, and PDF/reporting system. It complements the high-level README.

## End-to-end workflow
1. Train models from `data.csv` using `train_model.py` (sklearn) or `train_dl.py` (optional DL). Export:
	- `model_pipeline.joblib`, `model_metadata.json`, optional `model_pipeline_stacking.joblib`, plots in `artifacts/`
2. Start FastAPI (`app/main.py`): lazy-load artifacts; detect DL runtime; expose endpoints.
3. Start Next.js (`frontend/`): dev rewrites proxy API calls; pages: Home, Learn, Demo.
4. User makes predictions on Demo; API streams a per-prediction PDF; awareness PDF available via `/awareness`.

## Dataset
- Input: tabular features; target column `diagnosis` (B/M or 0/1). Auto-drops `id` and any `Unnamed:*` columns.
- Split: stratified train/test; metrics logged in `model_metadata.json`.

## Training & preprocessing
- Imputer (median) → StandardScaler → SelectKBest (k tuned) → estimator
- Candidates: LogisticRegression, RandomForest, HistGradientBoosting; optional XGBoost/LightGBM
- Search: GridSearchCV with 5-fold CV; metadata persisted with best params and per-model comparison

## Explainability
- If `shap` installed, store `artifacts/shap_summary.png`. The streamed prediction report will include SHAP when present.

## Backend (FastAPI)
Routes:
- `GET /health` — service check
- `POST /predict` — `{ features, model_type, explain }`
- `POST /report` — streams PDF with ROC/PR/Confusion and metrics
- `GET /awareness?lang=en|hi|mr` — multilingual awareness PDF, cached per day
- `GET /models` — availability of sklearn/stacking/DL and DL runtime status
- `GET /sample` — canonical input for the demo

Implementation details:
- DL runtime detection is lazy; UI only enables DL if both `dl_model.h5` exists and TF is importable
- ReportLab is pinned to avoid rendering differences; PDFs generated in-memory and streamed
- Awareness PDF includes inline vector illustrations; caching avoids regenerating within a day per language

## Frontend (Next.js)
- Dark purple theme with Tailwind; smooth scrolling; section fade-in utilities
- Components: `VideoEmbed` (nocookie embeds + external link), `SiteFooter` (useful links)
- Pages: Home (spotlight video + features), Learn (guide with FAQ, awareness download), Demo (prediction form, report open)
- Dev rewrite: `/api/awareness` → backend `/awareness` (ensure the API is up on :8000)

## PDF generation
- `generate_pdf.py` creates `report.pdf` for the homepage “Download Example Report” link
- Structure includes: title page, executive summary, performance tables, configuration, model comparison, visualizations, key findings, technical details, clinical implications, conclusion
- New: “Project Workflow” section (matches README) and compact styles to avoid blank space

## Running locally
```powershell
python -m uvicorn app.main:APP --reload --port 8000
cd frontend; npm run dev
```

## Tests
Run `python test_report.py` for basic PDF build checks. Add more tests under `tests/` or reuse `scripts/` utilities.

## Deployment
- Pin library versions; prefer Docker/compose (optional)
- Serve Next.js behind a reverse proxy; put FastAPI behind a WSGI/ASGI server (uvicorn/gunicorn)

## Accessibility & i18n
- Large contrast, scalable fonts, reduced motion-friendly animations
- Awareness PDFs available in English, Hindi, Marathi (extendable)

## Known limitations / future work
- More robust unit tests and CI
- Model registry and experiment tracking
- Optional CNN path for image data (out of scope for tabular dataset)

