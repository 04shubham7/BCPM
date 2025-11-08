from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import json
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import lru_cache
import base64

APP = FastAPI(title="Breast Cancer Prediction API")

# basic logger (uvicorn will integrate)
logger = logging.getLogger("uvicorn.error")

# Development CORS policy (open). Restrict this in production.
# Allow frontend origins (local dev + Vercel). You can also set FRONTEND_ORIGIN env var to override.
import os
_frontend_origin = os.getenv('FRONTEND_ORIGIN')
_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
if _frontend_origin:
    _origins.append(_frontend_origin)
else:
    # broad default for hosted preview URLs on Vercel (https)
    _origins += ["https://*.vercel.app", "https://vercel.app"]

APP.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@APP.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return structured JSON when request body validation fails (helps the frontend)."""
    logger.exception("Request validation error: %s", exc)
    return JSONResponse(status_code=422, content={"error": "validation_error", "detail": exc.errors(), "body": exc.body})


@APP.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all JSON handler so frontend always gets JSON errors (keeps HTTPException semantics)."""
    # Preserve normal HTTPExceptions so they keep their status codes/details
    if isinstance(exc, StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, content={"error": "http_exception", "detail": str(exc.detail)})
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": str(exc)})

# TensorFlow/Keras will be imported lazily inside _get_dl_model() to avoid heavy startup costs
tf = None
keras = None

# Check if tensorflow is importable without importing it (avoid heavy startup cost)
import importlib.util
_tf_spec = importlib.util.find_spec('tensorflow')
DL_RUNTIME_AVAILABLE = _tf_spec is not None

try:
    import shap
except Exception:
    shap = None

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model_pipeline.joblib"
DL_MODEL_PATH = ROOT / "dl_model.h5"
SCALER_PATH = ROOT / "dl_scaler.joblib"
META_PATH = ROOT / "model_metadata.json"
DATA_PATH = ROOT / "data.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"


class PredictRequest(BaseModel):
    features: List[float]
    # select which model to use for prediction: 'sklearn' (default) or 'dl' or 'stacking'
    model_type: Optional[str] = None
    # optionally return an inline explanation (PNG base64) with the prediction
    explain: Optional[bool] = False


def _get_model():
    """Lazily load the model on first request (safer for test clients and server start)."""
    global MODEL
    if 'MODEL' in globals():
        return MODEL
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Run training first.")
    MODEL = joblib.load(MODEL_PATH)
    return MODEL


def _get_dl_model():
    """Load Keras model if available."""
    global DL_MODEL
    # If runtime not available, provide a clear message (don't try to import)
    if not DL_RUNTIME_AVAILABLE:
        raise RuntimeError('TensorFlow/Keras runtime not available. To enable DL support install TensorFlow (or tensorflow-cpu) in your environment: pip install tensorflow-cpu')

    if 'DL_MODEL' in globals():
        return DL_MODEL
    if not DL_MODEL_PATH.exists():
        raise RuntimeError(f"DL model file not found at {DL_MODEL_PATH}. Run DL training first.")
    # import tensorflow lazily to avoid importing large libraries at app startup
    global tf, keras
    try:
        import tensorflow as _tf
        from tensorflow import keras as _keras
        tf = _tf
        keras = _keras
    except Exception as e:
        raise RuntimeError('TensorFlow/Keras import failed at runtime: ' + str(e))
    DL_MODEL = keras.models.load_model(str(DL_MODEL_PATH))
    return DL_MODEL


@lru_cache(maxsize=1)
def _get_dl_scaler():
    """Load persisted StandardScaler for DL preprocessing, if available.
    Returns the scaler or None when not present.
    """
    try:
        if SCALER_PATH.exists():
            return joblib.load(SCALER_PATH)
    except Exception:
        pass
    return None


def _get_model_by_name(name: str):
    """Return a model object for a given name: 'sklearn'|'dl'|'stacking'."""
    name = (name or 'sklearn').lower()
    if name == 'dl':
        # if runtime isn't available, _get_dl_model will raise with a helpful message
        return _get_dl_model(), 'dl'
    if name == 'stacking':
        stacking_path = ROOT / 'model_pipeline_stacking.joblib'
        if not stacking_path.exists():
            raise RuntimeError('Stacking model not found')
        return joblib.load(stacking_path), 'sklearn'
    # default: sklearn pipeline
    return _get_model(), 'sklearn'


# cached renderer: cache PNG bytes for (plot_type, model_name) combos
@lru_cache(maxsize=64)
def _render_plot_bytes(plot_type: str, model_name: str, for_pdf: bool = False) -> bytes:
    """Render plot to PNG bytes for caching. This function must use only hashable args.
    It rebuilds the train/test split and computes metrics similarly to /plot.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(DATA_PATH)
    cols_to_drop = [c for c in df.columns if c.startswith('Unnamed')]
    if 'id' in df.columns:
        cols_to_drop.append('id')
    if 'diagnosis' not in df.columns:
        raise RuntimeError("data.csv missing 'diagnosis' for plotting")
    X = df.drop(columns=cols_to_drop + ['diagnosis'])
    y = df['diagnosis']
    # coerce target to binary numeric (many upstream notebooks use 'M'/'B')
    try:
        # try numeric conversion first
        y = pd.to_numeric(y)
    except Exception:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        # Ensure binary label for plotting metrics
        if len(le.classes_) != 2:
            raise RuntimeError("diagnosis column must be binary for plotting")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model_obj, mtype = _get_model_by_name(model_name)
    if mtype == 'dl':
        X_test_in = X_test.to_numpy()
        try:
            scaler = _get_dl_scaler()
            if scaler is not None:
                X_test_in = scaler.transform(X_test_in)
        except Exception:
            # if scaling fails for any reason, fall back to raw
            pass
    else:
        X_test_in = X_test

    try:
        import matplotlib as mpl
        # If the PNG is intended for embedding in the PDF, use a light-on-white
        # style (black text on white background) so it prints and displays
        # correctly in PDF viewers. For on-screen inline images (UI), use the
        # dark theme settings used earlier.
        if for_pdf:
            mpl.rcParams.update({
                "text.color": "#111827",
                "axes.labelcolor": "#111827",
                "xtick.color": "#111827",
                "ytick.color": "#111827",
                "axes.edgecolor": "#374151",
                "axes.facecolor": "white",
                "figure.facecolor": "white",
            })
        else:
            mpl.rcParams.update({
                "text.color": "#EDE9FE",
                "axes.labelcolor": "#EDE9FE",
                "xtick.color": "#EDE9FE",
                "ytick.color": "#EDE9FE",
                "axes.edgecolor": "#8b5cf6",
                "axes.facecolor": "none",
                "figure.facecolor": "none",
            })

        if mtype == 'dl':
            preds = model_obj.predict(X_test_in)
            try:
                probs = preds.flatten()
                labels = (probs >= 0.5).astype(int)
            except Exception:
                probs = None
                labels = preds
        else:
            labels = model_obj.predict(X_test_in)
            try:
                probs = model_obj.predict_proba(X_test_in)[:, 1]
            except Exception:
                try:
                    probs = model_obj.decision_function(X_test_in)
                except Exception:
                    probs = None
    except Exception as e:
        # Try a safe fallback: if the model failed due to feature-count mismatch, try using the
        # primary sklearn pipeline (MODEL_PATH) if available. This happens when a saved object
        # is the raw classifier trained on selected features (k-best) rather than a full pipeline.
        err = str(e)
        if 'feature' in err or 'features' in err or 'X has' in err:
            try:
                fb = _get_model()
                labels = fb.predict(X_test)
                try:
                    probs = fb.predict_proba(X_test)[:, 1]
                except Exception:
                    try:
                        probs = fb.decision_function(X_test)
                    except Exception:
                        probs = None
            except Exception:
                raise RuntimeError(f'Prediction failed: {e}')
        else:
            raise RuntimeError(f'Prediction failed: {e}')

    buf = BytesIO()
    plt.figure(figsize=(6, 5))
    try:
        if plot_type == 'roc':
            if probs is None:
                raise RuntimeError('No probability scores available for ROC')
            from sklearn.metrics import roc_curve, roc_auc_score
            fpr, tpr, _ = roc_curve(y_test, probs)
            auc = roc_auc_score(y_test, probs)
            # larger figure for PDFs
            if for_pdf:
                plt.figure(figsize=(9, 7))
            plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
            plt.plot([0, 1], [0, 1], '--', color='gray')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()

        elif plot_type in ('pr', 'precision_recall'):
            if probs is None:
                raise RuntimeError('No probability scores available for PR curve')
            from sklearn.metrics import precision_recall_curve, average_precision_score
            precision, recall, _ = precision_recall_curve(y_test, probs)
            ap = average_precision_score(y_test, probs)
            if for_pdf:
                plt.figure(figsize=(9, 7))
            plt.plot(recall, precision, label=f'AP = {ap:.3f}', linewidth=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()

        elif plot_type == 'confusion':
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, labels)
            import seaborn as sns
            if for_pdf:
                plt.figure(figsize=(8, 6))
                annot_kws = {"color": "black"}
                cmap = 'Blues'
            else:
                annot_kws = {"color": "white"}
                cmap = 'Blues'
            # Ensure annotation text is visible on the chosen background
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, annot_kws=annot_kws)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')

        elif plot_type in ('fi', 'feature_importances'):
            # Feature importance not available for DL models and ensemble stacking
            if mtype == 'dl':
                raise RuntimeError('Feature importance not supported for deep learning models. Consider using SHAP or integrated gradients instead.')
            if model_name == 'stacking':
                raise RuntimeError('Feature importance not directly available for stacking ensemble. Use individual base models instead.')
            
            fi = None
            try:
                clf = None
                if hasattr(model_obj, 'named_steps') and 'clf' in model_obj.named_steps:
                    clf = model_obj.named_steps['clf']
                elif hasattr(model_obj, 'estimators_'):
                    clf = model_obj
                if clf is not None:
                    if hasattr(clf, 'feature_importances_'):
                        fi = clf.feature_importances_
                    elif hasattr(clf, 'coef_'):
                        fi = np.abs(clf.coef_).ravel()
            except Exception:
                fi = None
            if fi is None:
                raise RuntimeError('No feature importances or coefficients available for this model')
            feat_names = list(X_test.columns)
            try:
                selector = model_obj.named_steps.get('selector') if hasattr(model_obj, 'named_steps') else None
                if selector is not None and hasattr(selector, 'get_support'):
                    mask = selector.get_support()
                    names = [n for n, m in zip(feat_names, mask) if m]
                else:
                    names = feat_names
            except Exception:
                names = feat_names
            if for_pdf:
                plt.figure(figsize=(10, max(4, len(names)*0.25)))
            else:
                plt.figure(figsize=(8, max(4, len(names)*0.2)))
            import seaborn as sns
            sns.barplot(x=fi, y=names)
            plt.title('Feature importances')

        else:
            raise RuntimeError('Unknown plot type')

        # finalize and save the figure for all plot types
        plt.tight_layout()
        # For PDF embedding prefer an opaque white background. For UI
        # inline images prefer transparent so they blend into the dark UI.
        if for_pdf:
            plt.savefig(buf, format='png', dpi=300, transparent=False)
        else:
            plt.savefig(buf, format='png', dpi=300, transparent=True)
        plt.close()
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        plt.close()
        raise


# Mount static endpoints so the frontend can fetch artifacts/report during development.
if ARTIFACTS_DIR.exists():
    APP.mount("/files/artifacts", StaticFiles(directory=str(ARTIFACTS_DIR)), name="artifacts")
# mount root files (report.pdf, model_metadata.json) under /files
APP.mount("/files", StaticFiles(directory=str(ROOT)), name="files")


@APP.get("/health")
def health():
    return {"status": "ok"}


@APP.get("/metadata")
def metadata():
    if not META_PATH.exists():
        raise HTTPException(status_code=404, detail="metadata not found")
    with open(META_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


@APP.get("/sample")
def sample():
    """Return a single sample row from `data.csv` (features only) to help populate the demo UI."""
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="data.csv not found")
    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    # drop unnamed/id and target if present
    cols_to_drop = [c for c in df.columns if c.startswith('Unnamed')]
    if 'id' in df.columns:
        cols_to_drop.append('id')
    if 'diagnosis' in df.columns:
        sample_row = df.drop(columns=cols_to_drop + ['diagnosis']).iloc[0].tolist()
        feature_names = list(df.drop(columns=cols_to_drop + ['diagnosis']).columns)
    else:
        sample_row = df.drop(columns=cols_to_drop).iloc[0].tolist()
        feature_names = list(df.drop(columns=cols_to_drop).columns)
    return {"features": sample_row, "feature_names": feature_names}


@APP.post("/predict")
def predict(req: PredictRequest):
    # choose model type/name
    # Respect an explicit model_type selection from the client. Default to 'sklearn'
    # unless the client explicitly requests 'stacking' or 'dl'. Previously the code
    # auto-preferred 'stacking' which caused the frontend to display stacking
    # results even when the user selected 'sklearn'. Change default to 'sklearn'.
    requested = (req.model_type or '').lower() if req.model_type else ''
    if requested == '' or requested == 'sklearn':
        model_type = 'sklearn'
    else:
        model_type = requested

    # If client explicitly requested DL but runtime is unavailable, try a safe fallback.
    # If a DL model file exists but TensorFlow isn't installed, fallback to the sklearn pipeline
    # and add a warning for the client. If no DL model file exists, return a clear 400.
    fallback_from_dl = False
    fallback_warning = None
    if model_type == 'dl' and not DL_RUNTIME_AVAILABLE:
        if DL_MODEL_PATH.exists():
            fallback_from_dl = True
            fallback_warning = 'DL runtime not available; falling back to the sklearn pipeline. To enable DL support install TensorFlow in your environment: pip install tensorflow-cpu'
            model_type = 'sklearn'
        else:
            avail = {"sklearn": MODEL_PATH.exists(), "dl_file": DL_MODEL_PATH.exists(), "dl_runtime": bool(DL_RUNTIME_AVAILABLE), "stacking": (ROOT / 'model_pipeline_stacking.joblib').exists()}
            raise HTTPException(status_code=400, detail={
                'message': "DL runtime not available and no DL model file was found",
                'install': 'pip install tensorflow-cpu  # or tensorflow',
                'available': avail
            })
    arr = np.array(req.features).reshape(1, -1)

    # load model by name (supports 'sklearn', 'dl', 'stacking')
    try:
        model_obj, mtype = _get_model_by_name(model_type)
    except Exception as e:
        avail = {"sklearn": MODEL_PATH.exists(), "dl": DL_MODEL_PATH.exists(), "stacking": (ROOT / 'model_pipeline_stacking.joblib').exists()}
        raise HTTPException(status_code=400, detail={
            'message': f"Unknown or unavailable model_type '{req.model_type or model_type}'",
            'available': avail,
            'error': str(e)
        })

    if mtype == 'dl':
        if keras is None:
            raise HTTPException(status_code=500, detail='DL model support not available (tensorflow not installed)')
        DL = model_obj
        try:
            arr_in = arr
            try:
                scaler = _get_dl_scaler()
                if scaler is not None:
                    arr_in = scaler.transform(arr_in)
                else:
                    # note missing scaler for transparency
                    fallback_warning = (fallback_warning + '; ' if fallback_warning else '') + 'dl_scaler_missing: using unscaled inputs for DL inference'
            except Exception:
                pass
            pred_prob = float(DL.predict(arr_in)[0].flatten()[-1])
        except Exception:
            # model may return single probability
            try:
                arr_in2 = arr
                try:
                    scaler = _get_dl_scaler()
                    if scaler is not None:
                        arr_in2 = scaler.transform(arr_in2)
                    else:
                        fallback_warning = (fallback_warning + '; ' if fallback_warning else '') + 'dl_scaler_missing: using unscaled inputs for DL inference'
                except Exception:
                    pass
                pred_prob = float(DL.predict(arr_in2)[0])
            except Exception:
                pred_prob = None
        pred_label = int(pred_prob >= 0.5) if pred_prob is not None else None
        # attempt to create a PDF report for this prediction
        try:
            # try to infer feature names from data.csv for nicer report
            import pandas as _pd
            _df = _pd.read_csv(DATA_PATH)
            _cols_to_drop = [c for c in _df.columns if c.startswith('Unnamed')]
            if 'id' in _df.columns:
                _cols_to_drop.append('id')
            if 'diagnosis' in _df.columns:
                _feature_names = list(_df.drop(columns=_cols_to_drop + ['diagnosis']).columns)
            else:
                _feature_names = list(_df.drop(columns=_cols_to_drop).columns)
        except Exception:
            _feature_names = None

        resp = {"prediction": pred_label, "probability": pred_prob, "model_type": 'dl'}
        try:
            fname = generate_pdf_report(list(req.features), _feature_names, resp, 'dl')
            resp['report_url'] = f"/files/reports/{fname}"
        except Exception:
            # don't fail the prediction if report generation fails
            logger.exception('Report generation failed for DL prediction')
        return resp

    # sklearn-like model
    MODEL = model_obj
    # validate input length against model if available
    expected = None
    if hasattr(MODEL, 'n_features_in_'):
        expected = MODEL.n_features_in_
    else:
        try:
            expected = MODEL.named_steps['imputer'].n_features_in_
        except Exception:
            expected = None

    # If feature count mismatches, try safe adaptation (truncate if extra), otherwise return helpful error
    warning = None
    if expected is not None and arr.shape[1] != expected:
        if arr.shape[1] > expected:
            # truncate extra submitted features (frontend may have included extra columns)
            warning = f'truncated_input_from_{arr.shape[1]}_to_{expected}'
            arr = arr[:, :expected]
        else:
            # fewer features than expected -> explicit error with details
            raise HTTPException(status_code=400, detail={
                'message': f'Input must have {expected} features',
                'expected': expected,
                'got': arr.shape[1]
            })

    try:
        pred = MODEL.predict(arr)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Model prediction failed: {e}')

    prob = None
    try:
        prob = float(MODEL.predict_proba(arr)[:, 1][0])
    except Exception:
        prob = None

    resp = {"prediction": int(pred), "probability": prob, "model_type": model_type}
    if warning is not None:
        resp['warning'] = warning

    # If we earlier fell back from an explicit DL request, communicate that to the client
    try:
        if fallback_from_dl and fallback_warning:
            # append or set a warning field
            prev = resp.get('warning')
            if prev:
                resp['warning'] = f"{prev}; {fallback_warning}"
            else:
                resp['warning'] = fallback_warning
    except Exception:
        pass

    # optionally produce a small local explanation image (base64 PNG)
    if getattr(req, 'explain', False):
        try:
            import pandas as pd
            # prepare training data to get feature names and transformations
            df = pd.read_csv(DATA_PATH)
            cols_to_drop = [c for c in df.columns if c.startswith('Unnamed')]
            if 'id' in df.columns:
                cols_to_drop.append('id')
            if 'diagnosis' in df.columns:
                X_all = df.drop(columns=cols_to_drop + ['diagnosis'])
            else:
                X_all = df.drop(columns=cols_to_drop)

            feature_names = list(X_all.columns)

            # get preprocessor (pipeline without classifier) if available
            preproc = None
            try:
                if isinstance(MODEL, joblib.numpy_pickle.Pickler) or hasattr(MODEL, 'named_steps'):
                    preproc = MODEL[:-1]
                else:
                    preproc = None
            except Exception:
                try:
                    preproc = MODEL[:-1]
                except Exception:
                    preproc = None

            # build a DataFrame for the single sample with original feature names
            sample_df = pd.DataFrame([req.features], columns=feature_names)
            # apply preproc transform if available
            if preproc is not None:
                try:
                    sample_trans = preproc.transform(sample_df)
                except Exception:
                    # try numpy
                    sample_trans = preproc.transform(sample_df.values)
            else:
                sample_trans = np.array(req.features).reshape(1, -1)

            # obtain classifier (clf)
            clf = None
            if hasattr(MODEL, 'named_steps') and 'clf' in MODEL.named_steps:
                clf = MODEL.named_steps['clf']
            else:
                clf = MODEL

            # try to produce a small bar chart of contributions or importances
            buf = BytesIO()
            import matplotlib.pyplot as _plt
            _plt.figure(figsize=(6, 4))
            produced = False
            try:
                if hasattr(clf, 'coef_'):
                    coefs = np.array(clf.coef_).ravel()
                    vals = np.array(sample_trans).ravel()
                    # match length
                    if len(coefs) == len(vals):
                        contrib = coefs * vals
                        names = feature_names
                    else:
                        # if selector changed features, try to map
                        try:
                            selector = MODEL.named_steps.get('selector') if hasattr(MODEL, 'named_steps') else None
                            if selector is not None and hasattr(selector, 'get_support'):
                                mask = selector.get_support()
                                names = [n for n, m in zip(feature_names, mask) if m]
                                contrib = coefs * vals
                            else:
                                names = [f'f{i}' for i in range(len(contrib))]
                        except Exception:
                            names = [f'f{i}' for i in range(len(contrib))]
                    # plot top contributions
                    importances = pd.Series(contrib, index=names).abs().sort_values(ascending=False).head(12)
                    import seaborn as sns
                    sns.barplot(x=importances.values, y=importances.index)
                    _plt.title('Top local feature contributions (abs)')
                    produced = True
                elif hasattr(clf, 'feature_importances_'):
                    importances = np.array(clf.feature_importances_)
                    # map to feature names if selector present
                    try:
                        selector = MODEL.named_steps.get('selector') if hasattr(MODEL, 'named_steps') else None
                        if selector is not None and hasattr(selector, 'get_support'):
                            mask = selector.get_support()
                            names = [n for n, m in zip(feature_names, mask) if m]
                        else:
                            names = feature_names
                    except Exception:
                        names = feature_names
                    import pandas as pd
                    s = pd.Series(importances, index=names).sort_values(ascending=False).head(12)
                    import seaborn as sns
                    sns.barplot(x=s.values, y=s.index)
                    _plt.title('Feature importances (global)')
                    produced = True
            except Exception:
                produced = False

            if produced:
                _plt.tight_layout()
                _plt.savefig(buf, format='png', dpi=300)
                _plt.close()
                buf.seek(0)
                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                resp['explanation'] = f"data:image/png;base64,{b64}"
        except Exception as e:
            # explanation generation failed; return without explanation
            resp['explanation_error'] = str(e)

    # After assembling response, attempt to generate a PDF report. If PDF
    # generation fails, do not fail the prediction — just log the error.
    try:
        try:
            import pandas as _pd
            _df = _pd.read_csv(DATA_PATH)
            _cols_to_drop = [c for c in _df.columns if c.startswith('Unnamed')]
            if 'id' in _df.columns:
                _cols_to_drop.append('id')
            if 'diagnosis' in _df.columns:
                _feature_names = list(_df.drop(columns=_cols_to_drop + ['diagnosis']).columns)
            else:
                _feature_names = list(_df.drop(columns=_cols_to_drop).columns)
        except Exception:
            _feature_names = None
        fname = generate_pdf_report(list(req.features), _feature_names, resp, model_type)
        resp['report_url'] = f"/files/reports/{fname}"
    except Exception:
        logger.exception('Report generation failed for prediction')

    return resp


def _ensure_reports_dir():
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        # best-effort: if directory can't be created, downstream will error
        pass
    return REPORTS_DIR


def generate_pdf_report(features, feature_names, result, model_type: str):
    """Generate a PDF report summarizing the input features, prediction
    and performance plots. Returns the filename (relative to /files/reports).
    This uses ReportLab and embeds PNGs produced by _render_plot_bytes.
    """
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from datetime import datetime

    _ensure_reports_dir()
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    safe_model = (model_type or 'sklearn').replace('/', '_')
    fname = f"report_{ts}_{safe_model}.pdf"
    out_path = REPORTS_DIR / fname

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('title', parent=styles['Heading1'], alignment=1, textColor=colors.HexColor('#8b5cf6'))
    normal = styles['Normal']

    from reportlab.platypus import PageBreak
    doc = SimpleDocTemplate(str(out_path), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=72, bottomMargin=54)
    elems = []

    # cover/logo if available
    logo_path = None
    for p in [
        ROOT / 'logo.png',
        ROOT / 'logo.jpg',
        ROOT / 'frontend' / 'public' / 'logo.png',
        ROOT / 'frontend' / 'public' / 'logo.jpg',
        ROOT / 'reports' / 'logo.png',
        ROOT / 'artifacts' / 'logo.png'
    ]:
        try:
            if p.exists():
                logo_path = p
                break
        except Exception:
            continue

    if logo_path is not None:
        try:
            rl_logo = RLImage(str(logo_path), width=2.5*inch, height=2.5*inch)
            elems.append(Spacer(1, 18))
            elems.append(rl_logo)
            elems.append(Spacer(1, 12))
        except Exception:
            elems.append(Spacer(1, 36))
    else:
        elems.append(Spacer(1, 36))

    elems.append(Paragraph('Breast Cancer Prediction Report', title_style))
    elems.append(Spacer(1, 8))
    elems.append(Paragraph(f'Generated: {datetime.utcnow().isoformat()} UTC', styles['Normal']))
    elems.append(Spacer(1, 12))

    # Input features table
    elems.append(Paragraph('Input features', styles['Heading3']))
    elems.append(Spacer(1, 6))
    table_data = [['Feature', 'Value']]
    if feature_names and len(feature_names) == len(features):
        for n, v in zip(feature_names, features):
            table_data.append([str(n), str(v)])
    else:
        for i, v in enumerate(features):
            table_data.append([f'f{i+1}', str(v)])

    tbl = Table(table_data, colWidths=[3.0*inch, 3.0*inch])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4c1d95')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elems.append(tbl)
    elems.append(Spacer(1, 12))

    # For each available model, include a brief prediction summary and
    # embed all standard plots (roc, confusion, pr, fi) when possible.
    available_models = []
    try:
        if MODEL_PATH.exists():
            available_models.append('sklearn')
    except Exception:
        pass
    try:
        if (ROOT / 'model_pipeline_stacking.joblib').exists():
            available_models.append('stacking')
    except Exception:
        pass
    try:
        if DL_MODEL_PATH.exists():
            available_models.append('dl')
    except Exception:
        pass

    plots = [('ROC Curve', 'roc'), ('Confusion Matrix', 'confusion'), ('Precision-Recall', 'pr'), ('Feature Importances', 'fi')]

    import numpy as _np
    for m in available_models:
        elems.append(Paragraph(f'Model: {m}', styles['Heading2']))
        # try to compute a prediction/probability for the provided single sample
        try:
            model_obj, mtype = _get_model_by_name(m)
            pred_val = None
            prob_val = None
            try:
                arr = _np.array(features).reshape(1, -1)
                if mtype == 'dl':
                    try:
                        arr_in = arr
                        try:
                            scaler = _get_dl_scaler()
                            if scaler is not None:
                                arr_in = scaler.transform(arr_in)
                        except Exception:
                            pass
                        preds = model_obj.predict(arr_in)
                        # flatten to probability if possible
                        try:
                            prob_val = float(preds.flatten()[-1])
                        except Exception:
                            prob_val = float(preds[0])
                        pred_val = int(prob_val >= 0.5) if prob_val is not None else None
                    except Exception:
                        pred_val = None
                        prob_val = None
                else:
                    try:
                        pred_val = int(model_obj.predict(arr)[0])
                    except Exception:
                        pred_val = None
                    try:
                        prob_val = float(model_obj.predict_proba(arr)[:, 1][0])
                    except Exception:
                        try:
                            prob_val = float(model_obj.decision_function(arr)[0])
                        except Exception:
                            prob_val = None
            except Exception:
                pred_val = None
                prob_val = None
        except Exception as e:
            elems.append(Paragraph(f'Could not load model {m}: {e}', normal))
            elems.append(Spacer(1, 6))
            continue

        # summary table for this model
        summ = [['Field', 'Value'], ['Model', str(m)], ['Prediction', str(pred_val)], ['Probability', str(prob_val)]]
        summ_tbl = Table(summ, colWidths=[2.5*inch, 3.5*inch])
        summ_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4c1d95')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        elems.append(summ_tbl)
        elems.append(Spacer(1, 8))

        for title, ptype in plots:
            try:
                img_bytes = _render_plot_bytes(ptype, m, for_pdf=True)
                if img_bytes:
                    elems.append(Paragraph(title, styles['Heading3']))
                    elems.append(Spacer(1, 6))
                    bio = BytesIO(img_bytes)
                    rlimg = RLImage(bio, width=6.0*inch, height=4.0*inch)
                    elems.append(rlimg)
                    elems.append(Spacer(1, 12))
            except Exception as e:
                elems.append(Paragraph(f"{title} not available for {m}: {e}", normal))
                elems.append(Spacer(1, 6))

    def _header_footer(canvas_obj, doc_obj):
        try:
            canvas_obj.saveState()
            canvas_obj.setFont('Helvetica', 9)
            canvas_obj.setFillColor(colors.HexColor('#6b21a8'))
            # header
            header_text = f"Breast Cancer Report - {safe_model}"
            canvas_obj.drawString(36, doc_obj.pagesize[1] - 36, header_text)
            # footer: page number and timestamp
            from datetime import datetime as _dt
            footer_text = f"{_dt.utcnow().strftime('%Y-%m-%d %H:%M UTC')} - Page {doc_obj.page}"
            canvas_obj.drawRightString(doc_obj.pagesize[0] - 36, 36, footer_text)
            canvas_obj.restoreState()
        except Exception:
            pass

    try:
        doc.build(elems, onFirstPage=_header_footer, onLaterPages=_header_footer)
    except Exception as e:
        # If PDF generation fails, remove partial file if present and re-raise
        try:
            if out_path.exists():
                out_path.unlink()
        except Exception:
            pass
        raise

    return fname


def generate_pdf_report_bytes(features, feature_names, model_type: str):
    """Generate a PDF report (in-memory) containing inputs and all model
    plots/summaries, and return the raw PDF bytes. This is suitable for
    streaming directly to the client without saving to disk.
    """
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from datetime import datetime

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('title', parent=styles['Heading1'], alignment=1, textColor=colors.HexColor('#8b5cf6'))
    normal = styles['Normal']

    from reportlab.platypus import PageBreak
    bio_out = BytesIO()
    # increase top/bottom margins to make room for header/footer
    doc = SimpleDocTemplate(bio_out, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=72, bottomMargin=54)
    elems = []

    # try to find a logo image to use on the cover page
    logo_path = None
    for p in [
        ROOT / 'logo.png',
        ROOT / 'logo.jpg',
        ROOT / 'frontend' / 'public' / 'logo.png',
        ROOT / 'frontend' / 'public' / 'logo.jpg',
        ROOT / 'reports' / 'logo.png',
        ROOT / 'artifacts' / 'logo.png'
    ]:
        try:
            if p.exists():
                logo_path = p
                break
        except Exception:
            continue

    if logo_path is not None:
        try:
            rl_logo = RLImage(str(logo_path), width=2.5*inch, height=2.5*inch)
            elems.append(Spacer(1, 18))
            elems.append(rl_logo)
            elems.append(Spacer(1, 12))
        except Exception:
            elems.append(Spacer(1, 36))
    else:
        elems.append(Spacer(1, 36))

    # Title and intro
    elems.append(Paragraph('Breast Cancer Prediction Report', title_style))
    elems.append(Spacer(1, 8))
    elems.append(Paragraph(f'Generated: {datetime.utcnow().isoformat()} UTC', styles['Normal']))
    elems.append(Spacer(1, 12))

    elems.append(Paragraph('Introduction', styles['Heading2']))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph('This report summarizes the provided input features, the model outputs and performance visualizations. Each model available on the system is evaluated using the same input sample and the corresponding plots (ROC, Confusion Matrix, Precision-Recall curve and Feature Importances where available) are embedded below.', normal))
    elems.append(Spacer(1, 12))

    # Input features (single section)
    elems.append(Paragraph('Input features', styles['Heading3']))
    elems.append(Spacer(1, 6))
    table_data = [['Feature', 'Value']]
    if feature_names and len(feature_names) == len(features):
        for n, v in zip(feature_names, features):
            table_data.append([str(n), str(v)])
    else:
        for i, v in enumerate(features):
            table_data.append([f'f{i+1}', str(v)])

    tbl = Table(table_data, colWidths=[3.0*inch, 3.0*inch])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4c1d95')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elems.append(tbl)
    elems.append(Spacer(1, 12))

    # Detect available models and then for each model include: description, output, plots
    available_models = []
    try:
        if MODEL_PATH.exists():
            available_models.append('sklearn')
    except Exception:
        pass
    try:
        if (ROOT / 'model_pipeline_stacking.joblib').exists():
            available_models.append('stacking')
    except Exception:
        pass
    try:
        if DL_MODEL_PATH.exists():
            available_models.append('dl')
    except Exception:
        pass

    plots = [('ROC Curve', 'roc'), ('Confusion Matrix', 'confusion'), ('Precision-Recall', 'pr'), ('Feature Importances', 'fi')]

    import numpy as _np
    for m in available_models:
        elems.append(Paragraph(f'Model: {m}', styles['Heading2']))
        elems.append(Spacer(1, 6))

        # Short description (generic) about the model
        if m == 'sklearn':
            elems.append(Paragraph('Model description: Scikit-learn based pipeline which includes preprocessing and a trained classifier. It expects the input features to match the trained feature set and outputs a binary prediction and confidence score.', normal))
        elif m == 'stacking':
            elems.append(Paragraph('Model description: Stacking ensemble that combines multiple base learners into a meta-classifier to improve predictive performance. Feature importances may not be directly available for the ensemble.', normal))
        elif m == 'dl':
            elems.append(Paragraph('Model description: Deep learning model (Keras/TensorFlow). Outputs probability scores; feature importances are not natively available.', normal))
        elems.append(Spacer(1, 8))

        # Output: prediction for this model
        try:
            model_obj, mtype = _get_model_by_name(m)
            pred_val = None
            prob_val = None
            try:
                arr = _np.array(features).reshape(1, -1)
                if mtype == 'dl':
                    try:
                        arr_in = arr
                        try:
                            scaler = _get_dl_scaler()
                            if scaler is not None:
                                arr_in = scaler.transform(arr_in)
                        except Exception:
                            pass
                        preds = model_obj.predict(arr_in)
                        try:
                            prob_val = float(preds.flatten()[-1])
                        except Exception:
                            prob_val = float(preds[0])
                        pred_val = int(prob_val >= 0.5) if prob_val is not None else None
                    except Exception:
                        pred_val = None
                        prob_val = None
                else:
                    try:
                        pred_val = int(model_obj.predict(arr)[0])
                    except Exception:
                        pred_val = None
                    try:
                        prob_val = float(model_obj.predict_proba(arr)[:, 1][0])
                    except Exception:
                        try:
                            prob_val = float(model_obj.decision_function(arr)[0])
                        except Exception:
                            prob_val = None
            except Exception:
                pred_val = None
                prob_val = None

            out_table = [['Field', 'Value'], ['Model', str(m)], ['Prediction', str(pred_val)], ['Probability', str(prob_val)]]
            out_tbl = Table(out_table, colWidths=[2.5*inch, 3.5*inch])
            out_tbl.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4c1d95')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ]))
            elems.append(out_tbl)
            elems.append(Spacer(1, 8))
        except Exception as e:
            elems.append(Paragraph(f'Could not compute output for {m}: {e}', normal))
            elems.append(Spacer(1, 6))

        # Plots for this model
        for title, ptype in plots:
            try:
                img_bytes = _render_plot_bytes(ptype, m, for_pdf=True)
                if img_bytes:
                    elems.append(Paragraph(title, styles['Heading3']))
                    elems.append(Spacer(1, 6))
                    bio = BytesIO(img_bytes)
                    rlimg = RLImage(bio, width=6.0*inch, height=4.0*inch)
                    elems.append(rlimg)
                    elems.append(Spacer(1, 12))
            except Exception as e:
                elems.append(Paragraph(f"{title} not available for {m}: {e}", normal))
                elems.append(Spacer(1, 6))

    # End of report
    elems.append(Spacer(1, 12))
    elems.append(Paragraph('End of report', styles['Normal']))

    def _header_footer_bytes(canvas_obj, doc_obj):
        try:
            canvas_obj.saveState()
            canvas_obj.setFont('Helvetica', 9)
            canvas_obj.setFillColor(colors.HexColor('#6b21a8'))
            header_text = f"Breast Cancer Report - {(model_type or 'sklearn')}"
            canvas_obj.drawString(36, doc_obj.pagesize[1] - 36, header_text)
            from datetime import datetime as _dt
            footer_text = f"{_dt.utcnow().strftime('%Y-%m-%d %H:%M UTC')} - Page {doc_obj.page}"
            canvas_obj.drawRightString(doc_obj.pagesize[0] - 36, 36, footer_text)
            canvas_obj.restoreState()
        except Exception:
            pass

    try:
        doc.build(elems, onFirstPage=_header_footer_bytes, onLaterPages=_header_footer_bytes)
    except Exception:
        raise

    bio_out.seek(0)
    return bio_out.getvalue()


def generate_awareness_pdf_bytes(lang: str = 'en') -> bytes:
    """Generate a printable educational PDF about breast cancer awareness.
    This mirrors the frontend Learn page content in a printer-friendly layout.
    """
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Flowable
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.graphics.shapes import Drawing, Circle, Rect, Polygon
    from reportlab.graphics import renderPDF

    texts = {
        'en': {
            'title': 'Understanding Breast Cancer',
            'intro': ('Breast cancer happens when some cells in the breast grow faster than they should and form a lump (tumor). '
                      'Some tumors are benign (not dangerous). Some are malignant (can spread). Early finding saves lives.'),
            'symptoms': 'Common Signs & Symptoms',
            'symptom_list': [
                'New lump in the breast or underarm',
                'Thickening or swelling of part of the breast',
                'Irritation or dimpling of breast skin',
                'Redness or flaky skin in the nipple area',
                'Pulling in of the nipple',
                'Any change in size or shape',
            ],
            'symptom_note': 'Note: These signs can have other causes. Only a doctor can tell for sure.',
            'screening': 'How Screening Works',
            'screen_steps': [
                ('Self-Awareness', 'Know your normal. If anything changes, talk to a doctor.'),
                ('Clinical Exam', 'A health worker checks and feels for changes.'),
                ('Imaging', 'Tests like mammogram, ultrasound, or MRI help see inside.'),
            ],
            'screen_note': 'Screening does not prevent cancer, but it helps find it early when it is easier to treat.',
            'risk': 'Reducing Risk',
            'risk_list': ['Stay Active', 'Eat Balanced', 'Limit Alcohol', 'Do not Smoke'],
            'app': 'How This App Helps',
            'app_p': 'This demo shows how AI can support doctors with a second opinion on data. It does not replace a doctor.',
            'disclaimer': 'Disclaimer: For education only. Not medical advice.',
            'resources': 'Helpful Resources',
            'legend': 'Icon Legend',
            'links': [
                ('WHO: Breast Cancer Facts', 'https://www.who.int/news-room/fact-sheets/detail/breast-cancer'),
                ('NCI: Types of Breast Cancer', 'https://www.cancer.gov/types/breast'),
                ('CDC: Basics About Breast Cancer', 'https://www.cdc.gov/cancer/breast/basic_info/index.htm'),
            ],
            'faq': [
                ('Does a lump always mean cancer?', 'No. Many lumps are benign. But any new lump should be checked by a healthcare professional.'),
                ('Can men get breast cancer?', 'Yes, men can get breast cancer too, though it is less common than in women.'),
                ('What age should screening start?', 'It depends on national guidelines and personal risk. Talk to your doctor about when to begin and how often.'),
                ('Does screening hurt?', 'Some tests (like mammograms) can be uncomfortable but are quick. The benefits of early detection are significant.'),
            ],
            'legend_items': [
                'Healthy Cell / Abnormal Cell',
                'Benign Lump / Malignant Tumor',
                'Step markers 1-3 indicate self-awareness, clinical exam, imaging',
            ],
        },
        'hi': {
            'title': 'स्तन कैंसर को समझें',
            'intro': 'स्तन की कुछ कोशिकाएँ सामान्य से तेज़ बढ़ती हैं और एक गांठ (ट्यूमर) बना सकती हैं। कुछ ट्यूमर सौम्य होते हैं, कुछ घातक। जल्दी पता चलना जीवन बचाता है।',
            'symptoms': 'सामान्य संकेत और लक्षण',
            'symptom_list': ['स्तन या बगल में नई गांठ', 'स्तन के किसी हिस्से में सूजन/गाढ़ापन', 'त्वचा में जलन या गड्ढ़े पड़ना', 'निप्पल क्षेत्र में लालिमा या परतदार त्वचा', 'निप्पल का अंदर की ओर मुड़ना', 'आकार या स्वरूप में कोई बदलाव'],
            'symptom_note': 'नोट: इन संकेतों के अन्य कारण भी हो सकते हैं। निश्चित रूप से डॉक्टर ही बता सकते हैं।',
            'screening': 'स्क्रीनिंग कैसे काम करती है',
            'screen_steps': [('स्वयं-जागरूकता', 'अपने सामान्य को जानें। कोई बदलाव हो तो डॉक्टर से मिलें।'), ('क्लिनिकल जांच', 'स्वास्थ्यकर्मी बदलावों की जाँच करते हैं।'), ('इमेजिंग', 'मैमोग्राम/अल्ट्रासाउंड/MRI जैसे परीक्षण अंदर देखने में मदद करते हैं।')],
            'screen_note': 'स्क्रीनिंग कैंसर को रोकती नहीं है, लेकिन उसे जल्दी ढूँढने में मदद करती है।',
            'risk': 'जोखिम कैसे कम करें',
            'risk_list': ['सक्रिय रहें', 'संतुलित भोजन', 'शराब सीमित करें', 'धूम्रपान न करें'],
            'app': 'यह ऐप कैसे मदद करता है',
            'app_p': 'यह डेमो दिखाता है कि एआई कैसे डॉक्टरों को डेटा पर दूसरी राय देने में मदद कर सकता है। यह डॉक्टर का स्थान नहीं लेता।',
            'disclaimer': 'अस्वीकरण: केवल शिक्षा के लिए। चिकित्सीय सलाह नहीं।',
            'resources': 'सहायक संसाधन',
            'legend': 'चिन्ह (आइकन) मार्गदर्शिका',
            'links': [('WHO: स्तन कैंसर तथ्य', 'https://www.who.int/news-room/fact-sheets/detail/breast-cancer'), ('NCI: स्तन कैंसर प्रकार', 'https://www.cancer.gov/types/breast'), ('CDC: स्तन कैंसर के बारे में', 'https://www.cdc.gov/cancer/breast/basic_info/index.htm')],
            'faq': [
                ('क्या हर गांठ कैंसर होती है?', 'नहीं। कई गांठें सौम्य होती हैं। फिर भी नई गांठ दिखे तो स्वास्थ्यकर्मी से जाँच कराएँ।'),
                ('क्या पुरुषों को भी स्तन कैंसर हो सकता है?', 'हाँ, पुरुषों में भी हो सकता है, हालांकि महिलाओं की तुलना में कम होता है।'),
                ('स्क्रीनिंग कब शुरू करनी चाहिए?', 'यह दिशानिर्देश और व्यक्तिगत जोखिम पर निर्भर करता है। अपने डॉक्टर से समय और आवृत्ति पूछें।'),
                ('क्या स्क्रीनिंग में दर्द होता है?', 'कुछ जाँचें असुविधाजनक हो सकती हैं पर जल्दी हो जाती हैं। जल्दी पता चलने के लाभ महत्वपूर्ण हैं।'),
            ],
            'legend_items': [
                'स्वस्थ कोशिका / असामान्य कोशिका',
                'सौम्य गांठ / घातक ट्यूमर',
                'स्टेप 1-3: स्वयं-जागरूकता, क्लिनिकल जाँच, इमेजिंग',
            ],
        },
        'mr': {
            'title': 'स्तनाचा कर्करोग समजून घ्या',
            'intro': 'स्तनातील काही पेशी सामान्यपेक्षा जलद वाढतात आणि गाठ (ट्यूमर) तयार होऊ शकते. काही सौम्य, काही घातक. लवकर निदान जीव वाचवते.',
            'symptoms': 'सामान्य लक्षणे',
            'symptom_list': ['स्तन/बगल येथे नवी गाठ', 'स्तनाच्या भागात सूज/जाडपणा', 'त्वचेवर खळगे/दुमडी', 'निपलभोवती लालसर/सोलणारी त्वचा', 'निपल आत ओढले जाणे', 'आकार/आकृतीत बदल'],
            'symptom_note': 'टीप: या लक्षणांची इतर कारणेही असू शकतात. निश्चित सांगू शकतो तो डॉक्टरच.',
            'screening': 'स्क्रीनिंग कसे होते',
            'screen_steps': [('स्वतःची जाणीव', 'आपली सामान्य अवस्था ओळखा. बदल जाणवला तर डॉक्टरांकडे जा.'), ('क्लिनिकल तपासणी', 'आरोग्यकर्मी स्पर्शाने तपासतात.'), ('इमेजिंग', 'मॅमोग्रॅम/अल्ट्रासाऊंड/MRI तपास मदत करतात.')],
            'screen_note': 'स्क्रीनिंग कर्करोग रोखत नाही; पण लवकर शोधण्यास मदत करते.',
            'risk': 'जोखीम कशी कमी करावी',
            'risk_list': ['सक्रिय राहा', 'संतुलित आहार', 'मद्यपान कमी करा', 'धूम्रपान टाळा'],
            'app': 'हा ॲप कशी मदत करतो',
            'app_p': 'हा डेमो दाखवतो की एआय डॉक्टरांना डेटा-आधारित दुसरे मत देण्यात कशी मदत करू शकते. हा डॉक्टरची जागा घेत नाही.',
            'disclaimer': 'सूचना: शैक्षणिक हेतूसाठी. वैद्यकीय सल्ला नाही.',
            'resources': 'उपयुक्त स्त्रोत',
            'legend': 'चिन्हांची स्पष्टीकरणे',
            'links': [('WHO: स्तन कर्करोग तथ्य', 'https://www.who.int/news-room/fact-sheets/detail/breast-cancer'), ('NCI: स्तन कर्करोग प्रकार', 'https://www.cancer.gov/types/breast'), ('CDC: मूलभूत माहिती', 'https://www.cdc.gov/cancer/breast/basic_info/index.htm')],
            'faq': [
                ('प्रत्येक गाठ कर्करोग असते का?', 'नाही. अनेक गाठी सौम्य असतात. तरीही नवी गाठ दिसल्यास आरोग्यतज्ज्ञांकडून तपासणी करा.'),
                ('पुरुषांनाही स्तनाचा कर्करोग होतो का?', 'हो, पुरुषांमध्येही होऊ शकतो; पण महिलांच्या तुलनेत कमी सामान्य आहे.'),
                ('स्क्रीनिंग कधी सुरू करावी?', 'हे राष्ट्रीय मार्गदर्शक तत्त्वे आणि वैयक्तिक जोखमीवर अवलंबून असते. आपल्या डॉक्टरांना विचारा.'),
                ('स्क्रीनिंगमध्ये वेदना होतात का?', 'काही चाचण्या (उदा. मॅमोग्रॅम) अस्वस्थ वाटू शकतात; पण लवकर होतात. लवकर निदानाचे फायदे महत्वाचे आहेत.'),
            ],
            'legend_items': [
                'निरोगी पेशी / असामान्य पेशी',
                'सौम्य गाठ / घातक ट्यूमर',
                'पायरी 1-3: स्वयं-जाणीव, क्लिनिकल तपासणी, इमेजिंग',
            ],
        },
    }

    t = texts.get(lang, texts['en'])
    bio_out = BytesIO()
    doc = SimpleDocTemplate(bio_out, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=48, bottomMargin=48)
    styles = getSampleStyleSheet()
    title = ParagraphStyle('title', parent=styles['Heading1'], textColor=colors.HexColor('#4c1d95'), alignment=1)
    h2 = styles['Heading2']
    normal = styles['Normal']
    small = ParagraphStyle('small', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#374151'))

    elems = []
    elems.append(Paragraph(t['title'], title))
    elems.append(Spacer(1, 8))
    elems.append(Paragraph(t['intro'], normal))
    elems.append(Spacer(1, 12))

    # Inline vector illustrations (to match on-screen visuals)
    class DrawingFlowable(Flowable):
        def __init__(self, drawing, width, height):
            super().__init__()
            self.drawing = drawing
            self._w = width
            self._h = height
        def wrap(self, availWidth, availHeight):
            return self._w, self._h
        def draw(self):
            renderPDF.draw(self.drawing, self.canv, 0, 0)

    def make_healthy_cell():
        d = Drawing(120, 90)
        d.add(Circle(60, 45, 34, fillColor=colors.HexColor('#10b981'), strokeColor=colors.white, strokeWidth=1))
        d.add(Circle(60, 45, 12, fillColor=colors.white, strokeColor=colors.HexColor('#065f46'), strokeWidth=1))
        return d

    def make_abnormal_cell():
        d = Drawing(120, 90)
        d.add(Circle(60, 45, 34, fillColor=colors.HexColor('#f59e0b'), strokeColor=colors.white, strokeWidth=1))
        # off-center nucleus
        d.add(Circle(75, 55, 12, fillColor=colors.white, strokeColor=colors.HexColor('#92400e'), strokeWidth=1))
        return d

    def make_benign_lump():
        d = Drawing(120, 90)
        d.add(Rect(20, 20, 80, 50, rx=18, ry=18, fillColor=colors.HexColor('#3b82f6'), strokeColor=colors.white, strokeWidth=1))
        return d

    def make_malignant_tumor():
        d = Drawing(120, 90)
        star = Polygon(points=[60,75, 80,55, 95,45, 80,35, 60,15, 40,35, 25,45, 40,55],
                        fillColor=colors.HexColor('#ef4444'), strokeColor=colors.white, strokeWidth=1)
        d.add(star)
        return d

    illus_cells = [
        ('Healthy Cell', make_healthy_cell()),
        ('Abnormal Cell', make_abnormal_cell()),
        ('Benign Lump', make_benign_lump()),
        ('Malignant Tumor', make_malignant_tumor()),
    ]

    # Build a 2x2 grid of drawings with labels
    cell_flowables = []
    for label, drawing in illus_cells:
        df = DrawingFlowable(drawing, 120, 90)
        cell_flowables.append([df, Paragraph(label, small)])

    # Arrange in two columns
    illus_table = Table([
        [cell_flowables[0], cell_flowables[1]],
        [cell_flowables[2], cell_flowables[3]],
    ], colWidths=[3*inch, 3*inch])
    illus_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    elems.append(illus_table)
    elems.append(Spacer(1, 12))

    elems.append(Paragraph(t['symptoms'], h2))
    for s in t['symptom_list']:
        elems.append(Paragraph(f"• {s}", normal))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph(t['symptom_note'], small))
    elems.append(Spacer(1, 12))

    elems.append(Paragraph(t['screening'], h2))
    for title_step, desc in t['screen_steps']:
        elems.append(Paragraph(f"• <b>{title_step}</b>: {desc}", normal))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph(t['screen_note'], small))
    elems.append(Spacer(1, 12))

    elems.append(Paragraph(t['risk'], h2))
    elems.append(Paragraph('• ' + '; '.join(t['risk_list']), normal))
    elems.append(Spacer(1, 12))

    elems.append(Paragraph(t['app'], h2))
    elems.append(Paragraph(t['app_p'], normal))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph(t['disclaimer'], small))
    elems.append(Spacer(1, 12))

    elems.append(Paragraph(t['resources'], h2))
    for label, url in t['links']:
        elems.append(Paragraph(f"• <a href='{url}' color='blue'>{label}</a>", normal))

    # FAQ
    elems.append(Spacer(1, 12))
    elems.append(PageBreak())
    elems.append(Paragraph('FAQ', h2))
    for q, a in t.get('faq', []):
        elems.append(Paragraph(f"<b>{q}</b>", normal))
        elems.append(Paragraph(a, small))
        elems.append(Spacer(1, 6))

    # Icon legend
    elems.append(Spacer(1, 12))
    elems.append(Paragraph(t.get('legend', 'Icon Legend'), h2))
    for item in t.get('legend_items', []):
        elems.append(Paragraph(f"• {item}", normal))

    def _header_footer(canvas_obj, doc_obj):
        try:
            canvas_obj.saveState()
            canvas_obj.setFont('Helvetica', 9)
            canvas_obj.setFillColor(colors.HexColor('#6b21a8'))
            canvas_obj.drawString(36, doc_obj.pagesize[1]-30, 'Breast Cancer Awareness')
            canvas_obj.drawRightString(doc_obj.pagesize[0]-36, 24, f"Page {doc_obj.page}")
            canvas_obj.restoreState()
        except Exception:
            pass

    doc.build(elems, onFirstPage=_header_footer, onLaterPages=_header_footer)
    bio_out.seek(0)
    return bio_out.getvalue()


@APP.get('/awareness')
def awareness(lang: Optional[str] = 'en'):
    """Return an educational breast cancer PDF (multi-language)."""
    try:
        # Disk cache per day and language
        from datetime import datetime
        _ensure_reports_dir()
        day = datetime.utcnow().strftime('%Y%m%d')
        safe_lang = (lang or 'en').lower()
        out_path = REPORTS_DIR / f"awareness_{safe_lang}_{day}.pdf"
        if out_path.exists():
            return StreamingResponse(out_path.open('rb'), media_type='application/pdf', headers={"Content-Disposition": f"attachment; filename=breast_cancer_guide_{safe_lang}.pdf"})

        pdf = generate_awareness_pdf_bytes(lang=safe_lang)
        try:
            with open(out_path, 'wb') as f:
                f.write(pdf)
        except Exception:
            pass
        headers = {"Content-Disposition": f"attachment; filename=breast_cancer_guide_{safe_lang}.pdf"}
        return StreamingResponse(BytesIO(pdf), media_type='application/pdf', headers=headers)
    except Exception as e:
        logger.exception('Awareness PDF generation failed: %s', e)
        raise HTTPException(status_code=500, detail=str(e))


@APP.get("/models")
def models_list():
    """List available persisted models (sklearn/dl)."""
    out = {
        "sklearn": MODEL_PATH.exists(),
        # whether a DL model file exists on disk
        "dl_file": DL_MODEL_PATH.exists(),
        # whether DL runtime (tensorflow) is available in this environment
        "dl_runtime": bool(DL_RUNTIME_AVAILABLE),
    }
    # stacking model
    out['stacking'] = (ROOT / 'model_pipeline_stacking.joblib').exists()
    return out


@APP.get('/plot')
def plot(type: str = 'roc', model: str = 'sklearn', inline: bool = False, request: Request = None):
    """Dynamically generate a PNG plot (roc, pr, confusion, fi, shap) for the selected model.
    Query params: type (roc|pr|confusion|fi|shap), model (sklearn|dl|stacking)
    """
    # load data and recreate the train/test split used in training
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail='data.csv not found')
    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    cols_to_drop = [c for c in df.columns if c.startswith('Unnamed')]
    if 'id' in df.columns:
        cols_to_drop.append('id')
    if 'diagnosis' not in df.columns:
        raise HTTPException(status_code=400, detail="data.csv must contain 'diagnosis' column")
    X = df.drop(columns=cols_to_drop + ['diagnosis'])
    y = df['diagnosis']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Support inline base64 JSON response when requested by the client
    # Use query param `inline=1` or `inline=true` to receive a JSON response
    # {"image": "data:image/png;base64,..."}. FastAPI will populate the
    # `inline` boolean automatically from the query string.

    try:
        img_bytes = _render_plot_bytes(type, model)
        if inline:
            b64 = base64.b64encode(img_bytes).decode('utf-8')
            return JSONResponse(content={"image": f"data:image/png;base64,{b64}"})
        return StreamingResponse(BytesIO(img_bytes), media_type='image/png')
    except RuntimeError as e:
        logger.exception('Plot generation runtime error: %s', e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception('Plot generation unexpected error: %s', e)
        raise HTTPException(status_code=500, detail=str(e))


@APP.post('/report')
def report(req: PredictRequest):
    """Generate a PDF report in-memory for the provided features and return it
    as an application/pdf stream. This endpoint mirrors the logic used by the
    frontend 'Download Reports' action and does not persist files to disk.
    """
    try:
        # infer feature names if available in data.csv
        try:
            import pandas as pd
            df = pd.read_csv(DATA_PATH)
            cols_to_drop = [c for c in df.columns if c.startswith('Unnamed')]
            if 'id' in df.columns:
                cols_to_drop.append('id')
            if 'diagnosis' in df.columns:
                feature_names = list(df.drop(columns=cols_to_drop + ['diagnosis']).columns)
            else:
                feature_names = list(df.drop(columns=cols_to_drop).columns)
        except Exception:
            feature_names = None

        pdf_bytes = generate_pdf_report_bytes(list(req.features), feature_names, req.model_type or 'sklearn')
        headers = {"Content-Disposition": "attachment; filename=report.pdf"}
        return StreamingResponse(BytesIO(pdf_bytes), media_type='application/pdf', headers=headers)
    except Exception as e:
        logger.exception('Report generation failed: %s', e)
        raise HTTPException(status_code=500, detail=str(e))
