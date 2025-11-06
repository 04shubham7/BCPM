"""train_model.py
Standalone training script for breast cancer prediction.

Usage:
    python train_model.py

This script will:
 - load data.csv from the notebook directory
 - clean and preprocess features
 - build a sklearn Pipeline with imputer, scaler, feature selector (optional) and classifier
 - run a small GridSearchCV across LogisticRegression and RandomForest
 - evaluate on a held-out test set and print metrics
 - save the best pipeline to `model_pipeline.joblib`

The saved pipeline is ready to be loaded for inference by FastAPI or other services.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime
import sklearn as sk

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.metrics import precision_recall_curve, average_precision_score
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, RocCurveDisplay
import seaborn as sns
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone

try:
    import shap
except Exception:
    shap = None


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data.csv"
MODEL_OUT = ROOT / "model_pipeline.joblib"
META_OUT = ROOT / "model_metadata.json"
ARTIFACTS = ROOT / "artifacts"


RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    # Drop any unnamed index columns commonly found in CSVs
    cols_to_drop = [c for c in df.columns if c.startswith("Unnamed")] 
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # If 'id' column exists, drop it
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Ensure target encoding
    if 'diagnosis' in df.columns:
        if df['diagnosis'].dtype == object:
            # Map M/B to 1/0
            df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    return df


def build_pipeline():
    # Basic pipeline: imputer -> scaler -> optional feature selection -> classifier
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_classif, k=12)),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    return pipe


def search_and_train(X, y, pipeline):
    # Build a param grid where each estimator gets only its valid params.
    param_grid = []

    # Logistic Regression grid
    param_grid.append({
        'selector__k': [8, 12],
        'clf': [LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)],
        'clf__C': [0.1, 1.0, 10.0],
    })

    # RandomForest grid
    param_grid.append({
        'selector__k': [8, 12],
        'clf': [RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)],
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 6],
    })

    # HistGradientBoosting grid (different params)
    param_grid.append({
        'selector__k': [8, 12],
        'clf': [HistGradientBoostingClassifier(random_state=RANDOM_STATE)],
        'clf__max_iter': [100, 200],
        'clf__max_depth': [None, 6],
    })

    # Optional: include XGBoost if available
    try:
        from xgboost import XGBClassifier
        param_grid.append({
            'selector__k': [8, 12],
            'clf': [XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=1)],
            'clf__n_estimators': [100],
            'clf__max_depth': [3, 6],
            'clf__learning_rate': [0.1, 0.01],
        })
    except Exception:
        # xgboost not installed — skip
        pass
    # Optional: include LightGBM if available
    try:
        from lightgbm import LGBMClassifier
        param_grid.append({
            'selector__k': [8, 12],
            'clf': [LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1)],
            'clf__n_estimators': [100],
            'clf__max_depth': [3, 6],
            'clf__learning_rate': [0.1, 0.01],
        })
    except Exception:
        # lightgbm not installed — skip
        pass

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    gs.fit(X, y)
    return gs


def _get_candidate_clfs():
    """Return a list of (name, class, param_grid) for candidate classifiers available in the env."""
    cand = []
    # Logistic Regression
    cand.append((
        'logreg',
        LogisticRegression,
        {'C': [0.1, 1.0, 10.0], 'max_iter': [1000], 'random_state': [RANDOM_STATE]}
    ))

    # RandomForest
    cand.append((
        'rf',
        RandomForestClassifier,
        {'n_estimators': [100, 200], 'max_depth': [None, 6], 'random_state': [RANDOM_STATE], 'n_jobs': [-1]}
    ))

    # HistGradientBoosting
    cand.append((
        'hgb',
        HistGradientBoostingClassifier,
        {'max_iter': [100, 200], 'max_depth': [None, 6], 'random_state': [RANDOM_STATE]}
    ))

    # Optional XGBoost
    try:
        from xgboost import XGBClassifier
        cand.append((
            'xgb',
            XGBClassifier,
            {'n_estimators': [100], 'max_depth': [3, 6], 'learning_rate': [0.1, 0.01], 'random_state': [RANDOM_STATE]}
        ))
    except Exception:
        pass

    # Optional LightGBM
    try:
        from lightgbm import LGBMClassifier
        cand.append((
            'lgb',
            LGBMClassifier,
            {'n_estimators': [100], 'max_depth': [3, 6], 'learning_rate': [0.1, 0.01], 'random_state': [RANDOM_STATE]}
        ))
    except Exception:
        pass

    return cand


def per_model_search_and_eval(X_train_t, y_train, X_test_t, y_test, artifacts_dir: Path, preproc_k=None):
    """Run GridSearchCV per candidate classifier on preprocessed features and save per-model ROC/PR curves."""
    results = {}
    candidates = _get_candidate_clfs()
    for name, cls, grid in candidates:
        try:
            print(f"Running GridSearch for {name}...")
            # instantiate a fresh estimator (GridSearch will set params)
            est = cls()
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            gs = GridSearchCV(estimator=est, param_grid=grid, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=0)
            gs.fit(X_train_t, y_train)
            best = gs.best_estimator_
            # evaluate
            y_pred = best.predict(X_test_t)
            try:
                y_prob = best.predict_proba(X_test_t)[:, 1]
            except Exception:
                try:
                    y_prob = best.decision_function(X_test_t)
                except Exception:
                    y_prob = None

            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred)),
                'recall': float(recall_score(y_test, y_pred)),
                'f1': float(f1_score(y_test, y_pred)),
            }
            if y_prob is not None:
                try:
                    metrics['roc_auc'] = float(roc_auc_score(y_test, y_prob))
                except Exception:
                    pass

            # save curves
            try:
                if y_prob is not None:
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    plt.figure(figsize=(6, 5))
                    plt.plot(fpr, tpr, label=f'{name} (AUC = {metrics.get("roc_auc",0):.3f})')
                    plt.plot([0, 1], [0, 1], '--', color='gray')
                    plt.xlabel('FPR')
                    plt.ylabel('TPR')
                    plt.title(f'ROC Curve - {name}')
                    plt.legend()
                    plt.tight_layout()
                    (artifacts_dir / f'roc_{name}.png').parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(artifacts_dir / f'roc_{name}.png', dpi=150)
                    plt.close()

                    precision, recall, _ = precision_recall_curve(y_test, y_prob)
                    ap = average_precision_score(y_test, y_prob)
                    plt.figure(figsize=(6, 5))
                    plt.plot(recall, precision, label=f'AP = {ap:.3f}')
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.title(f'Precision-Recall - {name}')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(artifacts_dir / f'pr_{name}.png', dpi=150)
                    plt.close()
            except Exception as e:
                print(f"Warning: failed to save curves for {name}:", e)

            results[name] = {
                'best_params': gs.best_params_,
                'metrics': metrics,
                'estimator_class': str(cls),
            }
            # keep fitted estimator instance for stacking
            results[name]['fitted_estimator'] = best
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")
    return results


def build_and_eval_stacking(per_model_results, X_train_t, y_train, X_test_t, y_test, artifacts_dir: Path):
    """Build a stacking classifier from provided fitted estimators (re-instantiate with best params), evaluate, optionally calibrate, and save artifacts."""
    estimators = []
    for name, info in per_model_results.items():
        try:
            cls_name = info.get('estimator_class')
            fitted = info.get('fitted_estimator')
            if fitted is None:
                continue
            # create an unfitted clone with same class and params
            base = clone(fitted)
            estimators.append((name, base))
        except Exception:
            continue

    if not estimators:
        print('No base estimators available for stacking. Skipping stacking.')
        return None

    try:
        stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), n_jobs=-1, passthrough=False)
        stack.fit(X_train_t, y_train)

        # Optionally calibrate
        try:
            calib = CalibratedClassifierCV(stack, cv=3)
            calib.fit(X_train_t, y_train)
            use_model = calib
            model_name = 'stacking_calibrated'
        except Exception:
            use_model = stack
            model_name = 'stacking'

        # evaluate
        y_pred = use_model.predict(X_test_t)
        try:
            y_prob = use_model.predict_proba(X_test_t)[:, 1]
        except Exception:
            try:
                y_prob = use_model.decision_function(X_test_t)
            except Exception:
                y_prob = None

        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
        }
        if y_prob is not None:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_test, y_prob))
            except Exception:
                pass

        # save curves
        try:
            if y_prob is not None:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, label=f'Stacking (AUC = {metrics.get("roc_auc",0):.3f})')
                plt.plot([0, 1], [0, 1], '--', color='gray')
                plt.xlabel('FPR')
                plt.ylabel('TPR')
                plt.title('ROC Curve - Stacking')
                plt.legend()
                plt.tight_layout()
                plt.savefig(artifacts_dir / f'roc_stacking.png', dpi=150)
                plt.close()

                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                ap = average_precision_score(y_test, y_prob)
                plt.figure(figsize=(6, 5))
                plt.plot(recall, precision, label=f'AP = {ap:.3f}')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall - Stacking')
                plt.legend()
                plt.tight_layout()
                plt.savefig(artifacts_dir / f'pr_stacking.png', dpi=150)
                plt.close()
        except Exception as e:
            print('Warning: failed to save stacking curves:', e)

        return {'model_name': model_name, 'metrics': metrics, 'fitted_model': use_model}
    except Exception as e:
        print('Stacking build failed:', e)
        return None


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = None
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        # Some classifiers may not have predict_proba
        pass

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
    }
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_prob)

    # Save evaluation artifacts
    try:
        ARTIFACTS.mkdir(exist_ok=True)
        # Confusion matrix heatmap
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        cm_path = ARTIFACTS / 'confusion_matrix.png'
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150)
        plt.close()

        # ROC curve
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f'ROC (AUC = {metrics.get("roc_auc", 0):.3f})')
            plt.plot([0, 1], [0, 1], '--', color='gray')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            roc_path = ARTIFACTS / 'roc_curve.png'
            plt.tight_layout()
            plt.savefig(roc_path, dpi=150)
            plt.close()

            # Precision-Recall curve + average precision
            try:
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                ap = average_precision_score(y_test, y_prob)
                plt.figure(figsize=(6, 5))
                plt.plot(recall, precision, label=f'AP = {ap:.3f}')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend(loc='lower left')
                pr_path = ARTIFACTS / 'precision_recall_curve.png'
                plt.tight_layout()
                plt.savefig(pr_path, dpi=150)
                plt.close()
            except Exception:
                pass

        # Feature importances (if available)
        try:
            if hasattr(model.named_steps['clf'], 'feature_importances_'):
                importances = model.named_steps['clf'].feature_importances_
                # selector may reduce features; get selected feature indices
                selector = model.named_steps.get('selector')
                if selector is not None and hasattr(selector, 'get_support'):
                    mask = selector.get_support()
                    feat_idx = np.where(mask)[0]
                    feat_names = [f'feat_{i}' for i in feat_idx]
                    imp_vals = importances
                else:
                    feat_names = [f'feat_{i}' for i in range(len(importances))]
                    imp_vals = importances
                plt.figure(figsize=(8, 4))
                sns.barplot(x=imp_vals, y=feat_names)
                plt.title('Feature importances')
                fi_path = ARTIFACTS / 'feature_importances.png'
                plt.tight_layout()
                plt.savefig(fi_path, dpi=150)
                plt.close()
        except Exception:
            pass
    except Exception as e:
        print('Warning: could not save artifacts:', e)

    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Metrics:\n", json.dumps(metrics, indent=2))
    # save classification report text to artifact
    try:
        ARTIFACTS.mkdir(exist_ok=True)
        report_text = classification_report(y_test, y_pred)
        with open(ARTIFACTS / 'classification_report.txt', 'w', encoding='utf-8') as rf:
            rf.write(report_text)
    except Exception:
        pass
    return metrics


def save_artifact(best_estimator, metadata: dict):
    joblib.dump(best_estimator, MODEL_OUT)
    with open(META_OUT, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def main():
    assert DATA_PATH.exists(), f"{DATA_PATH} not found. Put your data.csv in the same folder as this script."

    df = load_data(DATA_PATH)
    df = clean_and_prepare(df)

    if 'diagnosis' not in df.columns:
        raise RuntimeError("The dataset must contain a 'diagnosis' column as target.")

    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline()
    print("Starting GridSearchCV training... this may take a few minutes depending on your CPU")
    gs = search_and_train(X_train, y_train, pipeline)

    print(f"Best params: {gs.best_params_}")
    best = gs.best_estimator_

    print("Evaluating best model on test set...")
    metrics = evaluate_model(best, X_test, y_test)

    # SHAP explainability (optional)
    if shap is not None:
        try:
            ARTIFACTS.mkdir(exist_ok=True)
            # transform training data through pipeline without classifier
            try:
                preproc = best[:-1]
                X_train_trans = preproc.transform(X_train)
            except Exception:
                X_train_trans = X_train

            clf = best.named_steps.get('clf') if isinstance(best, Pipeline) else None
            if clf is not None:
                # Use TreeExplainer when possible for tree models, otherwise generic Explainer
                try:
                    explainer = shap.Explainer(clf, X_train_trans)
                    shap_values = explainer(X_train_trans)

                    # summary plot
                    plt.figure(figsize=(8, 6))
                    shap.summary_plot(shap_values, X_train_trans, show=False)
                    spath = ARTIFACTS / 'shap_summary.png'
                    plt.tight_layout()
                    plt.savefig(spath, dpi=150)
                    plt.close()

                    # dependence plot for top feature
                    try:
                        vals = np.abs(shap_values.values).mean(axis=0)
                        top_idx = int(np.argmax(vals))
                        plt.figure(figsize=(6, 5))
                        shap.dependence_plot(top_idx, shap_values.values, X_train_trans, show=False)
                        dpath = ARTIFACTS / 'shap_dependence_top.png'
                        plt.tight_layout()
                        plt.savefig(dpath, dpi=150)
                        plt.close()
                    except Exception:
                        pass
                except Exception as e:
                    print('SHAP explainer failed:', e)
        except Exception as e:
            print('Warning: SHAP generation failed:', e)

    # make best_params JSON-serializable by converting estimator objects to strings
    def _serialize_params(d: dict):
        out = {}
        for k, v in d.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                out[k] = v
            else:
                try:
                    out[k] = str(v)
                except Exception:
                    out[k] = repr(v)
        return out

    metadata = {
        'trained_at': datetime.utcnow().isoformat() + 'Z',
        'random_state': RANDOM_STATE,
        'best_params': _serialize_params(gs.best_params_),
        'metrics': metrics,
        'scikit_learn_version': sk.__version__,
    }

    # Additional model comparisons and stacking (run on preprocessed features)
    try:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        # derive a preprocessor (pipeline without classifier)
        try:
            preproc = best[:-1]
            # Transform train/test
            X_train_trans = preproc.transform(X_train)
            X_test_trans = preproc.transform(X_test)
            # if selector k is available use it for naming
            selector_k = None
            try:
                selector_k = gs.best_params_.get('selector__k')
            except Exception:
                selector_k = None

            per_model_results = per_model_search_and_eval(X_train_trans, y_train, X_test_trans, y_test, ARTIFACTS, preproc_k=selector_k)
            metadata['per_model_comparison'] = {}
            # serialize per-model results (drop fitted objects)
            for mname, minfo in (per_model_results or {}).items():
                serial = {k: v for k, v in minfo.items() if k != 'fitted_estimator'}
                metadata['per_model_comparison'][mname] = serial

            stacking_info = None
            try:
                stacking_info = build_and_eval_stacking(per_model_results or {}, X_train_trans, y_train, X_test_trans, y_test, ARTIFACTS)
            except Exception as e:
                print('Stacking step failed:', e)

            if stacking_info is not None:
                # save stacking model separately
                stack_out = ROOT / 'model_pipeline_stacking.joblib'
                try:
                    joblib.dump(stacking_info.get('fitted_model'), stack_out)
                    metadata['stacking_model_file'] = str(stack_out.name)
                    metadata['stacking'] = stacking_info.get('metrics')
                except Exception as e:
                    print('Could not save stacking model:', e)
        except Exception as e:
            print('Could not run model comparison/stacking:', e)
    except Exception as e:
        print('Warning while preparing artifacts:', e)

    save_artifact(best, metadata)
    print(f"Saved pipeline to {MODEL_OUT} and metadata to {META_OUT}")


if __name__ == '__main__':
    main()
