"""Small synthetic CRAFT-GBM demonstration; not a paper benchmark.

Endogenous mode returns the training-loss patience endpoint F_tau. This differs
from lightgbm.train_with_validation_stopping(), which selects a validation-best
prefix. The selector is native; lightgbm.craft supplies the paper's stopping rule.
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
from lightgbm.craft import CraftResult, TrainingLossController, train_craft
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("endogenous", "exogenous"), default="endogenous")
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    features = rng.normal(size=(512, 12))
    labels = (features[:, 0] + .5 * features[:, 1] + rng.normal(size=512) > 0).astype(int)
    endogenous = args.mode == "endogenous"
    params = {"objective": "binary", "feature_fraction": 1.0 if endogenous else .5,
              "learning_rate": .05, "num_leaves": 7, "min_data_in_leaf": 10,
              "adaptive_feature_confidence_delta": .05,
              "adaptive_feature_calibration_reference_count": 300 if endogenous else 1000}
    result = train_craft(params, lgb.Dataset(features, label=labels),
                         num_boost_round=None if endogenous else 30)
    print(json.dumps({"mode": args.mode, "stop_reason": result.stop_reason,
                      "completed_rounds": result.booster.current_iteration(),
                      "initial_training_loss": result.training_losses[0],
                      "final_training_loss": result.training_losses[-1]}))


if __name__ == "__main__":
    main()
