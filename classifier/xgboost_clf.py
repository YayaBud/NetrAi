"""
NetrAi Classifier — XGBoost Classifier
=========================================
Takes the 1793-D feature vectors (769 SegFormer + 1024 RETFound) and
trains / evaluates / persists an XGBoost multi-class classifier.

Why XGBoost over an MLP:
  - Tabular supremacy: gradient boosted trees empirically beat MLPs on
    fixed-dimension 1D vectors.
  - Column sampling (colsample_bytree): randomly hides dimensions from
    individual trees → hard to overfit on 1793 medical features.
  - Shrinkage (learning_rate): each tree's contribution is scaled down,
    making the ensemble robust to noisy minority-class signals.
  - SHAP explainability: tree SHAP values are exact and fast →
    clinically meaningful feature importance for free.

Softmax over 3 log-odd stacks → P(DR), P(Glaucoma), P(PM).

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


# ---------------------------------------------------------------------------
# XGBoost wrapper
# ---------------------------------------------------------------------------

class NetrAiXGBoost:
    """
    Wraps XGBoost with:
      - 3-class softmax output
      - Early stopping on validation log-loss
      - SHAP feature importance export
      - Checkpoint save / load
    """

    def __init__(self, cfg: dict):
        xc = cfg['xgboost']
        self.params = {
            "objective":          xc['objective'],           # multi:softprob
            "num_class":          xc['num_class'],
            "eval_metric":        xc['eval_metric'],
            "max_depth":          xc['max_depth'],
            "learning_rate":      xc['learning_rate'],
            "subsample":          xc['subsample'],
            "colsample_bytree":   xc['colsample_bytree'],
            "min_child_weight":   xc['min_child_weight'],
            "gamma":              xc['gamma'],
            "reg_alpha":          xc['reg_alpha'],
            "reg_lambda":         xc['reg_lambda'],
            "seed":               xc['seed'],
            "device":             xc.get('device', 'cpu'),
            "tree_method":        xc.get('tree_method', 'hist'),
            "nthread":            xc.get('n_jobs', -1),
            "verbosity":          1,
        }
        self.n_estimators            = xc['n_estimators']
        self.early_stopping_rounds   = xc['early_stopping_rounds']
        self.booster: Optional[xgb.Booster] = None

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        logger=None,
    ) -> None:
        """
        Trains the XGBoost booster with early stopping on val log-loss.

        Args:
            X_train / X_val  (N, 1793) float32 arrays
            y_train / y_val  (N,) int32 class indices {0,1,2}
        """
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=_feature_names(X_train.shape[1]))
        dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=_feature_names(X_val.shape[1]))

        evals      = [(dtrain, "train"), (dval, "val")]
        evals_log: dict = {}

        log = logger.info if logger else print
        log(
            f"XGBoost training: X_train={X_train.shape}  X_val={X_val.shape}  "
            f"n_estimators={self.n_estimators}  "
            f"early_stopping={self.early_stopping_rounds}"
        )

        self.booster = xgb.train(
            params               = self.params,
            dtrain               = dtrain,
            num_boost_round      = self.n_estimators,
            evals                = evals,
            evals_result         = evals_log,
            early_stopping_rounds= self.early_stopping_rounds,
            verbose_eval         = 50,
        )

        best_round = self.booster.best_iteration
        best_score = self.booster.best_score
        log(f"XGBoost done — best_round={best_round}  best_val_mlogloss={best_score:.4f}")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns (N, 3) softmax probabilities [P(DR), P(Glaucoma), P(PM)].
        """
        assert self.booster is not None, "Model not trained yet — call train() or load()."
        dm   = xgb.DMatrix(X, feature_names=_feature_names(X.shape[1]))
        prob = self.booster.predict(dm)
        if prob.ndim == 1:
            # XGBoost sometimes returns flat array — reshape
            prob = prob.reshape(-1, int(self.params["num_class"]))
        return prob.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns (N,) integer class predictions."""
        return self.predict_proba(X).argmax(axis=1).astype(np.int32)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X:      np.ndarray,
        y:      np.ndarray,
        logger=None,
        label:  str = "eval",
    ) -> dict:
        proba = self.predict_proba(X)
        preds = proba.argmax(axis=1)
        mets  = compute_classification_metrics(y, preds, y_prob=proba)
        log   = logger.info if logger else print
        log(f"\n  ── {label} ──")
        print_metrics(mets, logger=logger)
        return mets

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Saves booster + params as a pickle."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"booster": self.booster, "params": self.params}, f)
        print(f"  XGBoost model saved → {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.booster = data["booster"]
        self.params  = data.get("params", self.params)
        print(f"  XGBoost model loaded ← {path}")

    # ------------------------------------------------------------------
    # SHAP feature importance
    # ------------------------------------------------------------------

    def shap_importance(
        self,
        X:         np.ndarray,
        n_samples: int = 500,
        output_path: Optional[str] = None,
    ) -> dict:
        """
        Computes mean absolute SHAP values per feature dimension.
        Returns dict: {"feature_name": mean_abs_shap, ...}
        """
        try:
            import shap
        except ImportError:
            print("  [SHAP] Install shap: pip install shap")
            return {}

        assert self.booster is not None

        # Subsample for speed
        idx = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        X_s = X[idx]

        explainer   = shap.TreeExplainer(self.booster)
        shap_values = explainer.shap_values(X_s)  # (N, 1793, 3)

        # Mean absolute over samples and classes
        mean_abs = np.abs(shap_values).mean(axis=(0, 2)) if shap_values.ndim == 3 else np.abs(shap_values).mean(0)
        names    = _feature_names(X.shape[1])
        result   = {n: float(v) for n, v in zip(names, mean_abs)}

        if output_path:
            with open(output_path, "w") as f:
                json.dump(dict(sorted(result.items(), key=lambda x: -x[1])), f, indent=2)
            print(f"  SHAP importance saved → {output_path}")

        # Print top 20
        top = sorted(result.items(), key=lambda x: -x[1])[:20]
        print("  Top-20 SHAP features:")
        for name, val in top:
            print(f"    {name:30s} {val:.4f}")

        return result


# ---------------------------------------------------------------------------
# Feature name helper (for XGBoost / SHAP readability)
# ---------------------------------------------------------------------------

def _feature_names(total_dim: int) -> list[str]:
    """
    Assigns human-readable names to the 1793 dimensions.
        [0:384]   → segformer_pathA_XXX  (Path A raw context)
        [384:768] → segformer_vib_XXX    (Path B VIB μ)
        [768]     → global_anomaly_score
        [769:1793]→ retfound_XXX         (RETFound [CLS] embedding)
    """
    names = []
    for i in range(384):
        names.append(f"segformer_pathA_{i:03d}")
    for i in range(384):
        names.append(f"segformer_vib_{i:03d}")
    names.append("global_anomaly_score")
    for i in range(1024):
        names.append(f"retfound_{i:04d}")
    return names[:total_dim]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(description="Train XGBoost on 1793-D feature vectors")
    parser.add_argument("--config",    default="classifier/config.yaml")
    parser.add_argument("--shap",      action="store_true", help="Run SHAP analysis after training")
    args = parser.parse_args(args)

    cfg    = load_config(args.config)
    logger = setup_logging(cfg['paths']['checkpoint_dir'], name="xgboost")

    p = cfg['paths']

    # ---- Load feature matrices ----
    logger.info("Loading features...")
    X_train, y_train, train_stems = load_features(p['features_dir'], prefix="train")
    X_val,   y_val,   val_stems   = load_features(p['features_dir'], prefix="val")
    logger.info(f"  Train: {X_train.shape}  Val: {X_val.shape}")

    # ---- Train ----
    clf = NetrAiXGBoost(cfg)
    clf.train(X_train, y_train, X_val, y_val, logger=logger)

    # ---- Evaluate ----
    train_mets = clf.evaluate(X_train, y_train, logger=logger, label="TRAIN")
    val_mets   = clf.evaluate(X_val,   y_val,   logger=logger, label="VAL")

    # Save metrics
    results_path = os.path.join(p['checkpoint_dir'], "xgboost_results.json")
    with open(results_path, "w") as f:
        json.dump({"train": train_mets, "val": val_mets}, f, indent=2)
    logger.info(f"Results saved → {results_path}")

    # ---- Save model ----
    model_path = os.path.join(p['checkpoint_dir'], "xgboost_model.pkl")
    clf.save(model_path)

    # ---- SHAP ----
    if args.shap:
        shap_path = os.path.join(p['checkpoint_dir'], "shap_importance.json")
        clf.shap_importance(X_val, output_path=shap_path)

    logger.info("XGBoost training complete ✓")


if __name__ == "__main__":
    main()
