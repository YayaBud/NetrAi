"""
NetrAi Classifier — CLI Entry Point
=====================================
Usage:
    python -m classifier <command> [options]

Commands:
    cache-retfound   Pre-compute and save RETFound embeddings for the dataset
    train            Train the NetrAiEncoder (SegFormer phase)
    extract          Extract 1793-D feature vectors using trained encoder
    xgboost          Train XGBoost on extracted feature vectors
    infer            Run full inference on a single image

Examples:
    # Step 1: pre-compute RETFound cache (one-time, before training)
    python -m classifier cache-retfound --config classifier/config.yaml

    # Step 2: train SegFormer
    python -m classifier train --config classifier/config.yaml

    # Step 3: extract 1793-D feature vectors
    python -m classifier extract --config classifier/config.yaml

    # Step 4: train XGBoost
    python -m classifier xgboost --config classifier/config.yaml --shap

    # Inference on a new image
    python -m classifier infer --config classifier/config.yaml \\
                               --image patient_001.jpg \\
                               --anomaly patient_001_anomaly.png
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="NetrAi Classifier — Retinal Disease Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ---------- cache-retfound ----------
    p_cache = sub.add_parser("cache-retfound", help="Pre-compute RETFound embedding cache")
    p_cache.add_argument("--config",    default="classifier/config.yaml")
    p_cache.add_argument("--overwrite", action="store_true")

    # ---------- train ----------
    p_train = sub.add_parser("train", help="Train NetrAiEncoder (SegFormer phase)")
    p_train.add_argument("--config",  default="classifier/config.yaml")
    p_train.add_argument("--resume",  default=None)
    p_train.add_argument("--log-dir", default=None)

    # ---------- extract ----------
    p_extract = sub.add_parser("extract", help="Extract 1793-D feature vectors")
    p_extract.add_argument("--config",     default="classifier/config.yaml")
    p_extract.add_argument("--checkpoint", default=None)
    p_extract.add_argument("--splits",     nargs="+", default=["train", "val"])

    # ---------- xgboost ----------
    p_xgb = sub.add_parser("xgboost", help="Train XGBoost classifier")
    p_xgb.add_argument("--config", default="classifier/config.yaml")
    p_xgb.add_argument("--shap",   action="store_true")

    # ---------- infer ----------
    p_infer = sub.add_parser("infer", help="Run inference on a single image")
    p_infer.add_argument("--config",         default="classifier/config.yaml")
    p_infer.add_argument("--image",          required=True)
    p_infer.add_argument("--anomaly",        default=None)
    p_infer.add_argument("--segformer-ckpt", default=None)
    p_infer.add_argument("--xgboost-ckpt",   default=None)
    p_infer.add_argument("--load-retfound",  action="store_true")

    args = parser.parse_args()

    # ---- Dispatch ----
    if args.command == "cache-retfound":
        from .retfound import precompute_retfound_cache
        from .utils    import load_config
        import os
        cfg = load_config(args.config)
        p   = cfg['paths']
        precompute_retfound_cache(
            image_dirs   = [
                os.path.join(p['data_dir'], "train"),
                os.path.join(p['data_dir'], "val"),
            ],
            cache_dir    = p['retfound_cache_dir'],
            weights_path = p.get('retfound_weights'),
            batch_size   = cfg['retfound']['cache_batch_size'],
            overwrite    = args.overwrite,
        )

    elif args.command == "train":
        from .train import main as train_main
        train_main([
            "--config",  args.config,
            *(["--resume",  args.resume]  if args.resume  else []),
            *(["--log-dir", args.log_dir] if args.log_dir else []),
        ])

    elif args.command == "extract":
        from .extract import main as extract_main
        extract_main([
            "--config", args.config,
            *(["--checkpoint", args.checkpoint] if args.checkpoint else []),
            "--splits", *args.splits,
        ])

    elif args.command == "xgboost":
        from .xgboost_clf import main as xgb_main
        xgb_main([
            "--config", args.config,
            *(["--shap"] if args.shap else []),
        ])

    elif args.command == "infer":
        from .inference import main as infer_main
        cli_args = ["--config", args.config, "--image", args.image]
        if args.anomaly:
            cli_args += ["--anomaly", args.anomaly]
        if args.segformer_ckpt:
            cli_args += ["--segformer-ckpt", args.segformer_ckpt]
        if args.xgboost_ckpt:
            cli_args += ["--xgboost-ckpt", args.xgboost_ckpt]
        if args.load_retfound:
            cli_args += ["--load-retfound"]
        infer_main(cli_args)


if __name__ == "__main__":
    main()
