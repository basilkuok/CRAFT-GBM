"""Small synthetic CRAFT-GBM demonstration; not a paper benchmark.

Endogenous mode returns the training-loss patience endpoint F_tau. This differs
from lightgbm.train_with_validation_stopping(), which selects a validation-best
prefix. The selector is native; the paper's training-loss stopping rule is here.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math

import lightgbm as lgb
import numpy as np


class TrainingLossController:
    """Paper stopping: epsilon=.001, patience=10, strict anchor improvements."""

    def __init__(self, initial_loss: float):
        self._validate_loss(initial_loss)
        self.threshold = .001 * initial_loss
        if not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("initial training loss must give a positive finite threshold")
        self.losses = [float(initial_loss)]
        self.anchor = None
        self.last_reset_round = None
        self.stopped = False

    @staticmethod
    def _validate_loss(loss):
        if not math.isfinite(float(loss)) or loss < 0:
            raise ValueError("training loss must be finite and nonnegative")

    def observe(self, completed_round: int, loss: float) -> bool:
        if self.stopped:
            raise RuntimeError("cannot observe after the patience endpoint")
        if completed_round != len(self.losses):
            raise ValueError("completed rounds must be consecutive")
        self._validate_loss(loss)
        self.losses.append(float(loss))
        if completed_round == 2:
            self.anchor = min(self.losses)
            self.last_reset_round = 2
        elif completed_round >= 3:
            if loss < self.anchor - self.threshold:
                self.anchor = float(loss)
                self.last_reset_round = completed_round
            self.stopped = completed_round - self.last_reset_round >= 10
        return self.stopped


@dataclass
class CraftResult:
    booster: lgb.Booster
    training_losses: list[float]
    stop_reason: str


def train_craft(params, train_set, *, num_boost_round=None) -> CraftResult:
    """Fit a fresh model, using endogenous patience or an exogenous fixed horizon.

    Use feature_fraction=1 for endogenous selection and omit num_boost_round.
    Use 0<feature_fraction<1 with num_boost_round for exogenous selection.
    Only the paper's binary/multiclass log loss and regression RMSE are supported
    by this example. Native unsupported configurations fail explicitly.
    """
    settings = dict(params)
    for name in ("num_iterations", "feature_fraction"):
        aliases = lgb.basic._ConfigAliases.get(name)
        if name == "feature_fraction":
            aliases = aliases - {name}
        if any(alias in settings for alias in aliases):
            raise ValueError(f"use canonical feature_fraction and num_boost_round, not {name} aliases")
    settings.setdefault("adaptive_feature_selection", True)
    settings.setdefault("force_col_wise", True)
    settings.setdefault("num_threads", 2)
    settings.setdefault("verbosity", -1)
    if settings["adaptive_feature_selection"] is not True:
        raise ValueError("this example requires adaptive_feature_selection=True")
    fraction = float(settings.get("feature_fraction", 1.0))
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("feature_fraction must be in (0,1]")
    endogenous = fraction == 1.0
    if endogenous and num_boost_round is not None:
        raise ValueError("endogenous training does not accept an external round limit")
    if not endogenous and (isinstance(num_boost_round, bool)
                            or not isinstance(num_boost_round, int) or num_boost_round < 1):
        raise ValueError("exogenous training requires a positive integer num_boost_round")
    metric = {"binary": "binary_logloss", "multiclass": "multi_logloss",
              "regression": "rmse"}.get(settings.get("objective"))
    if metric is None or settings.get("metric", metric) != metric:
        raise ValueError("use binary/binary_logloss, multiclass/multi_logloss, or regression/rmse")
    settings["metric"] = metric
    # This fork's native API uses objective initialization and dataset weights.
    # The same training dataset is intentional: no held-out data guide stopping.
    if lgb.validation_stopping._has_init_score(train_set) or train_set._predictor is not None:
        raise ValueError("init_score and warm-start predictors are unsupported")
    train_set._update_params(settings)
    _, initial = lgb.validation_stopping._native_initial_state(settings, train_set, train_set)
    controller = TrainingLossController(float(initial))
    losses = [float(initial)]
    booster = lgb.Booster(settings, train_set=train_set)
    booster.set_train_data_name("training")
    while True:
        before = booster.current_iteration()
        finished = booster.update()
        completed = booster.current_iteration()
        if completed != before + 1:
            raise RuntimeError("native training did not add one complete boosting round")
        metrics = booster.eval_train()
        if (len(metrics) != 1 or metrics[0].dataset_name != "training"
                or metrics[0].metric_name != metric or metrics[0].maximize):
            raise RuntimeError("expected exactly one native training-loss metric")
        loss = float(metrics[0].metric_value)
        TrainingLossController._validate_loss(loss)
        losses.append(loss)
        if endogenous and controller.observe(completed, loss):
            return CraftResult(booster, losses, "training_loss_patience")
        if not endogenous and completed == num_boost_round:
            return CraftResult(booster, losses, "fixed_horizon")
        if finished:
            raise RuntimeError("native structural termination before the requested endpoint")
        if endogenous and completed >= 10002:
            raise RuntimeError("training-loss patience did not satisfy its 10002-round bound")


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
