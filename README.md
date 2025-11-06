# Breast Cancer Prediction - Training Pipeline

This small project contains a reproducible training script to build a breast cancer prediction model from `data.csv`.

Files added:
- `train_model.py` — training script: loads `data.csv`, preprocesses data, runs GridSearchCV, evaluates, and saves the best pipeline to `model_pipeline.joblib` and metadata to `model_metadata.json`.
- `requirements.txt` — minimal dependencies for running the script.

How to run
1. Create a Python environment (recommended):

   python -m venv .venv
   .venv\Scripts\Activate.ps1  # PowerShell on Windows

2. Install dependencies:

   pip install -r requirements.txt

3. Place `data.csv` in the repository root (same folder as this README). Then run:

   python train_model.py

Outputs
- `model_pipeline.joblib` — saved sklearn Pipeline (imputer/scaler/selector/classifier). Load this with `joblib.load()` for inference.
- `model_metadata.json` — training metadata and metrics.

Next steps (FastAPI + Next.js integration)
- Create a FastAPI service that loads `model_pipeline.joblib` at startup and exposes a `/predict` endpoint that accepts JSON arrays (feature values) and returns predictions and probabilities.
- Build a Next.js frontend that calls the FastAPI endpoint and shows results in a modern UI.

FastAPI quick start

1. Start the API (from repo root):

   uvicorn app.main:APP --reload --port 8000

2. Health check:

   GET http://127.0.0.1:8000/health

3. Example prediction (JSON body):

   POST http://127.0.0.1:8000/predict
   Body: {"features": [x1, x2, ..., x30]}

The included `app/main.py` is a minimal FastAPI server that loads the saved pipeline lazily and exposes `/predict` and `/health`.

Deep learning (optional)

If you'd rather train a small deep network instead of the sklearn pipeline, there's an optional `train_dl.py` which trains a Keras MLP on the tabular CSV. TensorFlow is not included by default because it's large; to use it install `tensorflow` (or `tensorflow-cpu`) in your environment and run:

    python train_dl.py

Note: CNNs are most appropriate when your input is images. Since `data.csv` is tabular, a simple MLP is recommended. If you have medical image data later, I can add a CNN training pipeline as well.

Frontend (Next.js) quick example

Here's an example fetch snippet your Next.js frontend can use to call the FastAPI endpoint:

```js
// pages/api/predict.js (example client-side call)
async function predict(features) {
   const res = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features })
   });
   return res.json();
}
```

I can scaffold a minimal Next.js app (with a form that sends feature values and displays results) if you want — tell me and I'll add it to the repo.

Notes
- The training script uses SelectKBest feature selection; you can switch to PCA or keep all features easily by editing `train_model.py`.
- For deployment, pin versions exactly and consider building a Docker container.
