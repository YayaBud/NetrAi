"""
NetrAi Classifier — XGBoost Classifier (v2)
=============================================
Takes the 256-D feature vectors (128-D VIB1 ⊕ 128-D VIB2) and trains
THREE INDEPENDENT BINARY XGBoost classifiers — one per disease.

Why 3 binary classifiers instead of 1 multiclass:
  - Disease labels are NOT mutually exclusive. A patient can have both
    DR and Glaucoma simultaneously. Softmax forces mutual exclusion and
    would suppress a valid Glaucoma signal when DR features are "louder".
  - Each classifier optimises its own AUC independently.
  - Clinically, you want separate confidence scores per disease —
    not a probability that must sum to 1 across diseases.

Architecture:
    256-D input → XGBoost_DR   → P(DR)   ∈ [0, 1]
    256-D input → XGBoost_Glauc → P(Glauc) ∈ [0, 1]
    256-D input → XGBoost_PM   → P(PM)   ∈ [0, 1]

Run:
    python -m classifier xgboost --config classifier/config.yaml
"""

import os
import json
import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb

from .extract import load_features
from .utils   import (
    load_config,
    setup_logging,
    compute_classification_metrics,
    print_metrics,
)

CLASS_NAMES = ["DR", "Glaucoma", "PM"]
FEAT_DIM    = 256   # 128-D VIB1 ⊕ 128-D VIB2


# ---------------------------------------------------------------------------
# Single binary XGBoost wrapper
# ---------------------------------------------------------------------------

class BinaryXGBoost:
    """
    Binary XGBoost classifier for one disease.

    Input:  (N, 256) float32 feature vectors
    Output: P(disease) ∈ [0, 1] — sigmoid of log-odds
    """

    def __init__(self, cfg: dict, disease_name: str):
        xc = cfg['xgboost']
        self.disease_name = disease_name
        self.params = {
            "objective":        "binary:logistic",
            "eval_metric":      "auc",
            "max_depth":        xc['max_depth'],
            "learning_rate":    xc['learning_rate'],
            "subsample":        xc['subsample'],
            "colsample_bytree": xc['colsample_bytree'],
            "min_child_weight": xc['min_child_weight'],
            "gamma":            xc['gamma'],
            "reg_alpha":        xc['reg_alpha'],
            "reg_lambda":       xc['reg_lambda'],
            "seed":             xc['seed'],
            "device":           xc.get('device', 'cpu'),
            "tree_method":      xc.get('tree_method', 'hist'),
            "nthread":          xc.get('n_jobs', -1),
            "verbosity":        1,
        }
        self.n_estimators          = xc['n_estimators']
        self.early_stopping_rounds = xc['early_stopping_rounds']
        self.booster: Optional[xgb.Booster] = None

    def train(
        self,
        X_train: np.ndarray,    # (N, 256)
        y_train: np.ndarray,    # (N,) binary 0/1
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        logger=None,
    ) -> None:
        fnames  = _feature_names()
        dtrain  = xgb.DMatrix(X_train, label=y_train, feature_names=fnames)
        dval    = xgb.DMatrix(X_val,   label=y_val,   feature_names=fnames)
        evals_log: dict = {}

        log = logger.info if logger else print
        log(
            f"  [{self.disease_name}] XGBoost binary training: "
            f"N_pos_train={int(y_train.sum())}  N_neg_train={int((1-y_train).sum())}  "
            f"N_val={len(y_val)}  n_estimators={self.n_estimators}"
        )

        self.booster = xgb.train(
            params                = self.params,
            dtrain                = dtrain,
            num_boost_round       = self.n_estimators,
            evals                 = [(dtrain, "train"), (dval, "val")],
            evals_result          = evals_log,
            early_stopping_rounds = self.early_stopping_rounds,
            verbose_eval          = 100,
        )

        best_round = self.booster.best_iteration
        best_auc   = self.booster.best_score
        log(f"  [{self.disease_name}] done — best_round={best_round}  best_val_auc={best_auc:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns (N,) sigmoid probabilities."""
        assert self.booster is not None, f"[{self.disease_name}] Not trained yet."
        dm = xgb.DMatrix(X, feature_names=_feature_names())
        return self.booster.predict(dm).astype(np.float32)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns (N,) binary predictions at given threshold."""
        return (self.predict_proba(X) >= threshold).astype(np.int32)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "booster":      self.booster,
                "params":       self.params,
                "disease_name": self.disease_name,
            }, f)
        print(f"  [{self.disease_name}] XGBoost saved → {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.booster      = data["booster"]
        self.params       = data.get("params", self.params)
        self.disease_name = data.get("disease_name", self.disease_name)
        print(f"  [{self.disease_name}] XGBoost loaded ← {path}")

    def shap_importance(
        self,
        X:           np.ndarray,
        n_samples:   int = 500,
        output_path: Optional[str] = None,
    ) -> dict:
        """SHAP feature importance for this binary classifier."""
        try:
            import shap
        except ImportError:
            print("  Install shap: pip install shap")
            return {}

        idx  = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        X_s  = X[idx]

        explainer   = shap.TreeExplainer(self.booster)
        shap_values = explainer.shap_values(X_s)   # (N, 256)
        mean_abs    = np.abs(shap_values).mean(0)
        names       = _feature_names()
        result      = {n: float(v) for n, v in zip(names, mean_abs)}

        if output_path:
            with open(output_path, "w") as f:
                json.dump(
                    dict(sorted(result.items(), key=lambda x: -x[1])), f, indent=2
                )
            print(f"  [{self.disease_name}] SHAP saved → {output_path}")

        top = sorted(result.items(), key=lambda x: -x[1])[:10]
        print(f"  [{self.disease_name}] Top-10 SHAP features:")
        for name, val in top:
            print(f"    {name:30s} {val:.4f}")

        return result


# ---------------------------------------------------------------------------
# Three-classifier wrapper
# ---------------------------------------------------------------------------

class NetrAiXGBoost:
    """
    Wraps 3 independent BinaryXGBoost classifiers.
    Provides a unified train / predict_proba / save / load interface.
    """

    def __init__(self, cfg: dict):
        self.classifiers = {
            name: BinaryXGBoost(cfg, name) for name in CLASS_NAMES
        }
        self.cfg = cfg

    def train(
        self,
        X_train:    np.ndarray,    # (N, 256)
        y_train_vec: np.ndarray,   # (N, 3)  multi-hot float labels
        X_val:      np.ndarray,
        y_val_vec:  np.ndarray,
        logger=None,
    ) -> None:
        """Trains all 3 binary classifiers independently."""
        for i, (name, clf) in enumerate(self.classifiers.items()):
            clf.train(
                X_train = X_train,
                y_train = y_train_vec[:, i].astype(np.float32),
                X_val   = X_val,
                y_val   = y_val_vec[:, i].astype(np.float32),
                logger  = logger,
            )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns (N, 3) — independent probabilities [P(DR), P(Glauc), P(PM)].
        Each column is independent; they do NOT sum to 1.
        """
        probs = [clf.predict_proba(X) for clf in self.classifiers.values()]
        return np.stack(probs, axis=1).astype(np.float32)   # (N, 3)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Returns (N, 3) binary predictions.
        For single-class-per-image inference, use argmax on predict_proba instead.
        """
        return (self.predict_proba(X) >= threshold).astype(np.int32)

    def evaluate(
        self,
        X:      np.ndarray,
        y_vec:  np.ndarray,   # (N, 3) multi-hot
        y_int:  np.ndarray,   # (N,) integer class index
        logger=None,
        label:  str = "eval",
    ) -> dict:
        """Evaluates all 3 classifiers + reports overall argmax accuracy."""
        log = logger.info if logger else print
        log(f"\n  ── {label} ──")

        proba = self.predict_proba(X)   # (N, 3)

        results = {}
        for i, name in enumerate(CLASS_NAMES):
            # Per-disease binary metrics
            y_bin  = y_vec[:, i].astype(np.int32)
            y_pred = (proba[:, i] >= 0.5).astype(np.int32)
            try:
                from sklearn.metrics import roc_auc_score, average_precision_score
                auc = roc_auc_score(y_bin, proba[:, i])
                ap  = average_precision_score(y_bin, proba[:, i])
            except Exception:
                auc = ap = float("nan")
            acc = float((y_bin == y_pred).mean())
            log(f"    {name:10s}  AUC={auc:.4f}  AP={ap:.4f}  Acc@0.5={acc:.4f}")
            results[name] = {"auc": auc, "ap": ap, "acc": acc}

        # Overall: argmax of 3 independent probabilities → single-class accuracy
        preds_int = proba.argmax(axis=1).astype(np.int32)
        overall   = float((preds_int == y_int).mean())
        log(f"    Overall argmax accuracy: {overall:.4f}")
        results['overall_acc'] = overall

        return results

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        for name, clf in self.classifiers.items():
            clf.save(os.path.join(dir_path, f"xgb_{name}.pkl"))

    def load(self, dir_path: str) -> None:
        for name, clf in self.classifiers.items():
            clf.load(os.path.join(dir_path, f"xgb_{name}.pkl"))

    def shap_importance(
        self,
        X:          np.ndarray,
        output_dir: Optional[str] = None,
    ) -> dict:
        results = {}
        for name, clf in self.classifiers.items():
            out = os.path.join(output_dir, f"shap_{name}.json") if output_dir else None
            results[name] = clf.shap_importance(X, output_path=out)
        return results


# ---------------------------------------------------------------------------
# Feature name helper (SHAP readability)
# ---------------------------------------------------------------------------

def _feature_names() -> list[str]:
    """
    Human-readable names for all 256 feature dimensions.
        [0:128]   → vib1_z_000 ... vib1_z_127   (custom SegFormer heads stream)
        [128:256] → vib2_z_000 ... vib2_z_127   (RETFound stream)
    """
    names = []
    for i in range(128):
        names.append(f"vib1_z_{i:03d}")   # custom heads (DR + Glauc + PM)
    for i in range(128):
        names.append(f"vib2_z_{i:03d}")   # RETFound
    return names


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(
        description="Phase 2: Train 3 binary XGBoost classifiers on 256-D vectors"
    )
    parser.add_argument("--config",    default="classifier/config.yaml")
    parser.add_argument("--shap",      action="store_true")
    args = parser.parse_args(args)

    cfg    = load_config(args.config)
    logger = setup_logging(cfg['paths']['checkpoint_dir'], name="xgboost")

    p = cfg['paths']

    # ---- Load feature matrices ----
    logger.info("Loading 256-D feature vectors...")
    X_train, y_train_vec, y_train_int, _ = load_features(p['features_dir'], "train")
    X_val,   y_val_vec,   y_val_int,   _ = load_features(p['features_dir'], "val")
    logger.info(
        f"  Train: {X_train.shape}  Val: {X_val.shape}  "
        f"(feature_dim={X_train.shape[1]} expected=256)"
    )
    assert X_train.shape[1] == FEAT_DIM, \
        f"Expected 256-D features, got {X_train.shape[1]}. Re-run extract first."

    # ---- Train 3 classifiers ----
    clf = NetrAiXGBoost(cfg)
    clf.train(X_train, y_train_vec, X_val, y_val_vec, logger=logger)

    # ---- Evaluate ----
    train_mets = clf.evaluate(X_train, y_train_vec, y_train_int, logger=logger, label="TRAIN")
    val_mets   = clf.evaluate(X_val,   y_val_vec,   y_val_int,   logger=logger, label="VAL")

    # Save results
    results_path = os.path.join(p['checkpoint_dir'], "xgboost_results.json")
    with open(results_path, "w") as f:
        json.dump({"train": train_mets, "val": val_mets}, f, indent=2)
    logger.info(f"Results saved → {results_path}")

    # ---- Save all 3 models ----
    xgb_dir = os.path.join(p['checkpoint_dir'], "xgboost")
    clf.save(xgb_dir)

    # ---- SHAP ----
    if args.shap:
        shap_dir = os.path.join(p['checkpoint_dir'], "shap")
        clf.shap_importance(X_val, output_dir=shap_dir)

    logger.info("XGBoost training complete ✓")


if __name__ == "__main__":
    main()
