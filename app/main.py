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
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000", "*"],
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

try:
    import shap
except Exception:
    shap = None

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model_pipeline.joblib"
DL_MODEL_PATH = ROOT / "dl_model.h5"
META_PATH = ROOT / "model_metadata.json"
DATA_PATH = ROOT / "data.csv"
ARTIFACTS_DIR = ROOT / "artifacts"


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
    except Exception:
        raise RuntimeError('TensorFlow/Keras not available in this environment')
    DL_MODEL = keras.models.load_model(str(DL_MODEL_PATH))
    return DL_MODEL


def _get_model_by_name(name: str):
    """Return a model object for a given name: 'sklearn'|'dl'|'stacking'."""
    name = (name or 'sklearn').lower()
    if name == 'dl':
        return _get_dl_model(), 'dl'
    if name == 'stacking':
        stacking_path = ROOT / 'model_pipeline_stacking.joblib'
        if not stacking_path.exists():
            raise RuntimeError('Stacking model not found')
        return joblib.load(stacking_path), 'sklearn'
    # default: sklearn pipeline
    return _get_model(), 'sklearn'


# cached renderer: cache PNG bytes for (plot_type, model_name) combos
@lru_cache(maxsize=32)
def _render_plot_bytes(plot_type: str, model_name: str) -> bytes:
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
    else:
        X_test_in = X_test

    try:
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
            plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
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
            plt.plot(recall, precision, label=f'AP = {ap:.3f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()

        elif plot_type == 'confusion':
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, labels)
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
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
            plt.figure(figsize=(8, max(4, len(names)*0.2)))
            import seaborn as sns
            sns.barplot(x=fi, y=names)
            plt.title('Feature importances')

        else:
            raise RuntimeError('Unknown plot type')

        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150)
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
    # prefer stacking when available and no explicit model requested
    requested = (req.model_type or '').lower() if req.model_type else ''
    if requested in ('', 'sklearn'):
        # prefer stacking if exists
        if (ROOT / 'model_pipeline_stacking.joblib').exists():
            model_type = 'stacking'
        else:
            model_type = 'sklearn'
    else:
        model_type = requested
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
            pred_prob = float(DL.predict(arr)[0].flatten()[-1])
        except Exception:
            # model may return single probability
            try:
                pred_prob = float(DL.predict(arr)[0])
            except Exception:
                pred_prob = None
        pred_label = int(pred_prob >= 0.5) if pred_prob is not None else None
        return {"prediction": pred_label, "probability": pred_prob, "model_type": 'dl'}

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
                _plt.savefig(buf, format='png', dpi=150)
                _plt.close()
                buf.seek(0)
                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                resp['explanation'] = f"data:image/png;base64,{b64}"
        except Exception as e:
            # explanation generation failed; return without explanation
            resp['explanation_error'] = str(e)

    return resp


@APP.get("/models")
def models_list():
    """List available persisted models (sklearn/dl)."""
    out = {"sklearn": MODEL_PATH.exists(), "dl": DL_MODEL_PATH.exists()}
    # stacking model
    out['stacking'] = (ROOT / 'model_pipeline_stacking.joblib').exists()
    return out


@APP.get('/plot')
def plot(type: str = 'roc', model: str = 'sklearn'):
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

    try:
        img_bytes = _render_plot_bytes(type, model)
        return StreamingResponse(BytesIO(img_bytes), media_type='image/png')
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
