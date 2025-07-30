#!/usr/bin/env python3
# MIT License
# 
# Copyright (c) 2020 centerforaisafety
# Copyright (c) 2025 Kazuma Matsumoto
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
HLE Rubric Evaluation Script

This script evaluates model predictions using a 5-point rubric with OpenAI o3-mini.
It loads predictions from predictions/hle_{model}.json and evaluates them against 
the HLE dataset using the rubric criteria in utils/evaluation_rubric.yaml.

Usage:
    python rubric_evaluate.py --config conf/config_rubric_evaluation.yaml
    python rubric_evaluate.py --model deepseek-reasoner --judge o3-mini-2025-01-31
"""

import argparse
import hydra
from omegaconf import DictConfig
from hle_benchmark._configs import Config
from hle_benchmark.rubric_evaluation import main

@hydra.main(version_base=None, config_path="conf", config_name="config_rubric_evaluation")
def hydra_main(cfg: DictConfig) -> None:
    """Hydra main function"""
    args = Config(**cfg)
    main(args)

def cli_main():
    """CLI main function for direct usage"""
    parser = argparse.ArgumentParser(description="Evaluate HLE predictions using rubric criteria")
    parser.add_argument("--model", type=str, 
                       help="Model name (used to locate predictions file)")
    parser.add_argument("--predictions-file", type=str,
                       help="Direct path to predictions JSON file (overrides --model)")
    parser.add_argument("--judge", type=str, default="o3-mini-2025-01-31",
                       help="Judge model for evaluation")
    parser.add_argument("--dataset", type=str, default="cais/hle",
                       help="HuggingFace dataset name")
    parser.add_argument("--num-workers", type=int, default=5,
                       help="Number of concurrent API workers")
    parser.add_argument("--config", type=str, help="Path to config file (optional)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.config and not args.model and not args.predictions_file:
        parser.error("Must specify either --config, --model, or --predictions-file")
    
    if args.config:
        # Use Hydra with specified config
        with hydra.initialize(config_path="conf", version_base=None):
            cfg = hydra.compose(config_name=args.config.replace("conf/", "").replace(".yaml", ""))
            hydra_args = Config(**cfg)
            main(hydra_args)
    else:
        # Determine model name and predictions file
        if args.predictions_file:
            import os
            # Extract model name from predictions file path
            model_name = os.path.basename(args.predictions_file).replace("hle_", "").replace(".json", "")
            predictions_file = args.predictions_file
        else:
            model_name = args.model
            predictions_file = f"predictions/hle_{args.model}.json"
        
        # Use CLI arguments directly
        config_dict = {
            "model": model_name,
            "judge": args.judge,
            "dataset": args.dataset,
            "num_workers": args.num_workers,
            "provider": "openai",  # Always use OpenAI for rubric evaluation
            "base_url": "https://api.openai.com/v1",
            "max_tokens": 2048,
            "max_completion_tokens": 2048,
            "reasoning": False,
            "max_samples": 0,  # Required field
            "predictions_file": predictions_file,  # Add custom predictions file path
        }
        config_args = Config(**config_dict)
        main(config_args)

if __name__ == "__main__":
    try:
        # Try Hydra first (if run with config file)
        hydra_main()
    except:
        # Fall back to CLI mode
        cli_main()