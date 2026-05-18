"""Allow `python -m diffusion` as an alternative entry point."""
from .train import main
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()
main(config_path=args.config)
