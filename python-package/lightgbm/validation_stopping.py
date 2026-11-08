# coding: utf-8
"""Authoritative validation stopping for fixed-round LightGBM learners.

This module intentionally owns the single-round update loop.  It keeps the
exact-best validation checkpoint separate from the anchor used by the fixed
improvement threshold, and it does not use LightGBM's native early stopping.
"""

import copy
import ctypes
import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .basic import (
    Booster,
    Dataset,
    EvalResult,
    _ConfigAliases,
    _LIB,
    _choose_param_value,
    _c_str,
    _param_dict_to_str,
    _safe_call,
)

__all__ = [
    "AdaptiveTrainingResult",
    "AdaptiveTrainingTiming",
    "InitialPredictor",
    "RoundTimingBlock",
    "ValidationStoppingController",
    "ValidationStoppingDecision",
    "train_with_validation_stopping",
]


_EPSILON_NUMERATOR = 1
_EPSILON_DENOMINATOR = 1000
_EPSILON = _EPSILON_NUMERATOR / _EPSILON_DENOMINATOR
_PATIENCE = 10
_STOPPING_BOUND_COUNT = _PATIENCE * _EPSILON_DENOMINATOR // _EPSILON_NUMERATOR
_THEORETICAL_MAX_ROUNDS = 2 + _STOPPING_BOUND_COUNT


@dataclass(frozen=True)
class ValidationStoppingDecision:
    """State returned after one completed boosting round."""

    should_stop: bool
    stop_reason: Optional[str]
    patience_satisfied: bool
    round_cap_reached: bool


@dataclass(frozen=True)
class RoundTimingBlock:
    """Low-overhead timing totals for one consecutive block of rounds."""

    first_round: int
    last_round: int
    native_update_ns: int
    validation_evaluation_ns: int
    controller_ns: int


@dataclass(frozen=True)
class AdaptiveTrainingTiming:
    """Nanosecond timing receipt produced identically for every learner.

    Aggregate native time includes a terminal update attempt, if one occurs;
    per-round vectors contain only successfully committed boosting rounds.
    """

    total_function_ns: int
    initial_validation_evaluation_ns: int
    native_update_ns: int
    validation_evaluation_ns: int
    controller_ns: int
    selected_prefix_materialization_ns: int
    per_round_native_update_ns: Tuple[int, ...]
    per_round_validation_evaluation_ns: Tuple[int, ...]
    per_round_controller_ns: Tuple[int, ...]

    def round_blocks(self, block_size: int = 25) -> Tuple[RoundTimingBlock, ...]:
        """Aggregate per-round timings into consecutive blocks (25 by default)."""
        if isinstance(block_size, bool) or not isinstance(block_size, int):
            raise TypeError("block_size must be an integer")
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        num_rounds = len(self.per_round_native_update_ns)
        if (
            len(self.per_round_validation_evaluation_ns) != num_rounds
            or len(self.per_round_controller_ns) != num_rounds
        ):
            raise RuntimeError("per-round timing vectors are not aligned")
        blocks = []
        for first_index in range(0, num_rounds, block_size):
            last_index = min(first_index + block_size, num_rounds)
            blocks.append(
                RoundTimingBlock(
                    first_round=first_index + 1,
                    last_round=last_index,
                    native_update_ns=sum(self.per_round_native_update_ns[first_index:last_index]),
                    validation_evaluation_ns=sum(
                        self.per_round_validation_evaluation_ns[first_index:last_index]
                    ),
                    controller_ns=sum(self.per_round_controller_ns[first_index:last_index]),
                )
            )
        return tuple(blocks)


class ValidationStoppingController:
    """Fixed ``epsilon=0.001``, ``patience=10`` validation-loss controller.

    The controller accepts finite nonnegative *raw* validation losses.  Its
    anchor threshold is fixed once as ``0.001 * initial_loss``.  Exact-best
    checkpoint selection uses strict improvement independently of the anchor.
    """

    epsilon = _EPSILON
    patience = _PATIENCE
    stopping_bound_count = _STOPPING_BOUND_COUNT
    theoretical_max_rounds = _THEORETICAL_MAX_ROUNDS

    def __init__(self, initial_loss: float, max_boosting_rounds: Optional[int] = None) -> None:
        self._validate_max_rounds(max_boosting_rounds)
        self._validate_loss(initial_loss, "initial_loss")

        self.initial_loss = float(initial_loss)
        self.max_boosting_rounds = max_boosting_rounds
        self.absolute_threshold = self.epsilon * self.initial_loss
        if self.initial_loss > 0.0 and (
            not math.isfinite(self.absolute_threshold) or self.absolute_threshold <= 0.0
        ):
            raise ValueError("0.001 * initial_loss must be finite and strictly positive; numerical underflow occurred")

        self.completed_rounds = 0
        self.best_checkpoint = 0
        self.best_loss = self.initial_loss
        self.anchor_loss: Optional[float] = None
        self.last_reset_round: Optional[int] = None
        self.patience_satisfied = False
        self.round_cap_reached = False
        self.stop_reason: Optional[str] = None
        self.losses = [self.initial_loss]

        if self.initial_loss == 0.0:
            self.stop_reason = "zero_initial_validation_loss"

    @staticmethod
    def _validate_max_rounds(max_boosting_rounds: Optional[int]) -> None:
        if max_boosting_rounds is None:
            return
        if isinstance(max_boosting_rounds, bool) or not isinstance(max_boosting_rounds, int):
            raise TypeError("max_boosting_rounds must be None or an integer")
        if max_boosting_rounds < 1:
            raise ValueError("max_boosting_rounds must be at least 1")

    @staticmethod
    def _validate_loss(loss: float, name: str) -> None:
        if isinstance(loss, bool) or not isinstance(loss, (int, float, np.integer, np.floating)):
            raise TypeError(f"{name} must be a real scalar")
        if not math.isfinite(float(loss)) or float(loss) < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")

    @property
    def should_stop(self) -> bool:
        """Whether no further boosting round may be evaluated."""
        return self.stop_reason is not None

    def observe(self, completed_round: int, loss: float) -> ValidationStoppingDecision:
        """Commit the validation loss for the next completed boosting round."""
        if self.should_stop:
            raise RuntimeError("cannot observe another loss after the stopping controller has stopped")
        if completed_round != self.completed_rounds + 1:
            raise ValueError(
                f"completed_round must be consecutive; expected {self.completed_rounds + 1}, got {completed_round}"
            )
        self._validate_loss(loss, "validation loss")
        loss = float(loss)

        self.completed_rounds = completed_round
        self.losses.append(loss)
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_checkpoint = completed_round

        if completed_round == 2:
            self.anchor_loss = self.best_loss
            self.last_reset_round = 2
        elif completed_round >= 3:
            assert self.anchor_loss is not None
            assert self.last_reset_round is not None
            if self.anchor_loss - loss > self.absolute_threshold:
                self.anchor_loss = loss
                self.last_reset_round = completed_round
            self.patience_satisfied = completed_round - self.last_reset_round >= self.patience

        self.round_cap_reached = (
            self.max_boosting_rounds is not None and completed_round >= self.max_boosting_rounds
        )
        if self.patience_satisfied:
            self.stop_reason = "validation_patience"
        elif self.round_cap_reached:
            self.stop_reason = "max_rounds"

        return ValidationStoppingDecision(
            should_stop=self.should_stop,
            stop_reason=self.stop_reason,
            patience_satisfied=self.patience_satisfied,
            round_cap_reached=self.round_cap_reached,
        )

    def mark_native_termination(self) -> ValidationStoppingDecision:
        """Record structural native termination without calling it a plateau."""
        if self.should_stop:
            raise RuntimeError("cannot replace an existing stopping decision")
        self.stop_reason = "native_termination"
        return ValidationStoppingDecision(True, self.stop_reason, False, self.round_cap_reached)


@dataclass(frozen=True)
class InitialPredictor:
    """Explicit training-derived predictor before any boosting tree is added."""

    objective: str
    response: Tuple[float, ...]
    raw_score: Tuple[float, ...]

    def _num_rows(self, data: Any) -> int:
        if isinstance(data, Dataset):
            return data.construct().num_data()
        shape = getattr(data, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
        if isinstance(data, (str, Path)):
            raise TypeError("file-path prediction is unavailable for an initial predictor; provide in-memory data")
        try:
            return len(data)
        except TypeError as exc:
            raise TypeError("cannot determine the number of prediction rows") from exc

    def predict(self, data: Any, *, raw_score: bool = False, **kwargs: Any) -> np.ndarray:
        """Return the constant initial prediction for every row in ``data``."""
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported initial-predictor options: {unknown}")
        num_rows = self._num_rows(data)
        values = np.asarray(self.raw_score if raw_score else self.response, dtype=np.float64)
        if len(values) == 1:
            return np.full(num_rows, values[0], dtype=np.float64)
        return np.broadcast_to(values, (num_rows, len(values))).copy()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "model_type": "lightgbm_initial_predictor",
            "objective": self.objective,
            "response": list(self.response),
            "raw_score": list(self.raw_score),
            "boosting_rounds": 0,
        }

    def save_model(self, filename: Union[str, Path]) -> None:
        """Export this zero-tree predictor without invoking native zero-iteration semantics."""
        Path(filename).write_text(json.dumps(self.to_dict(), sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class AdaptiveTrainingResult:
    """Executed training checkpoint and selected prediction checkpoint.

    ``full_booster`` is the in-memory object at the executed endpoint and is
    retained for audit only. Portable continuation is unsupported because the
    matched controller, RNG, learner, validation monitor, and data state are not
    serialized as one checkpoint.
    """

    full_booster: Optional[Booster]
    predictor: Union[Booster, InitialPredictor]
    executed_rounds: int
    selected_checkpoint: int
    initial_validation_loss: float
    selected_validation_loss: float
    epsilon: float
    patience: int
    stopping_bound_count: int
    absolute_threshold: float
    anchor_loss: Optional[float]
    last_reset_round: Optional[int]
    terminal_patience_counter: Optional[int]
    validation_losses: Tuple[float, ...]
    stop_reason: str
    patience_satisfied: bool
    round_cap_reached: bool
    uncapped_endpoint_censored: bool
    max_boosting_rounds: Optional[int]
    max_boosting_rounds_provenance: str
    portable_resumption_supported: bool
    timing: AdaptiveTrainingTiming

    def predict(self, data: Any, **kwargs: Any) -> np.ndarray:
        """Predict with the selected checkpoint, including checkpoint zero."""
        return self.predictor.predict(data, **kwargs)


def _single_string_param(params: Dict[str, Any], name: str) -> str:
    value = params.get(name)
    if not isinstance(value, str) or not value.strip() or "," in value:
        raise ValueError(f"adaptive validation stopping requires one explicit string-valued {name}")
    return value.strip().lower()


def _has_init_score(dataset: Dataset) -> bool:
    if dataset.init_score is not None:
        return True
    if dataset._handle is None:
        return False
    return dataset.get_init_score() is not None


def _native_initial_state(
    params: Dict[str, Any], train_set: Dataset, evaluation_set: Dataset
) -> Tuple[InitialPredictor, float]:
    """Read native objective initialization and metric without creating a learner."""
    objective = _single_string_param(params, "objective")
    metric = _single_string_param(params, "metric")
    if objective in {"regression", "regression_l2", "l2", "mean_squared_error", "mse"}:
        objective = "regression"
    if (objective, metric) not in {
        ("binary", "binary_logloss"),
        ("multiclass", "multi_logloss"),
        ("regression", "rmse"),
    }:
        raise ValueError(
            "metric/objective must be binary_logloss/binary, multi_logloss/multiclass, or rmse/L2 regression"
        )
    num_outputs = 1
    if objective == "multiclass":
        num_class = params.get("num_class")
        if isinstance(num_class, bool) or not isinstance(num_class, (int, np.integer)) or int(num_class) < 2:
            raise ValueError("multiclass objective requires integer num_class >= 2")
        num_outputs = int(num_class)
    train_set.construct()
    evaluation_set.construct()
    if _has_init_score(train_set) or _has_init_score(evaluation_set):
        raise ValueError("init_score is not supported because it cannot be embedded in a portable zero-tree predictor")
    raw_scores = (ctypes.c_double * num_outputs)()
    responses = (ctypes.c_double * num_outputs)()
    out_len = ctypes.c_int64()
    loss = ctypes.c_double()
    _safe_call(
        _LIB.LGBM_DatasetGetNativeInitialPrediction(
            train_set._handle,
            evaluation_set._handle,
            _c_str(_param_dict_to_str(params)),
            ctypes.c_int64(num_outputs),
            ctypes.byref(out_len),
            raw_scores,
            responses,
            ctypes.byref(loss),
        )
    )
    if out_len.value != num_outputs:
        raise RuntimeError("native initial predictor returned an unexpected number of outputs")
    if not all(math.isfinite(value) for value in (*raw_scores, *responses)):
        raise ValueError("native initial predictions must be finite")
    ValidationStoppingController._validate_loss(loss.value, "initial_loss")
    return InitialPredictor(objective, tuple(responses), tuple(raw_scores)), loss.value


def _validate_params(params: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(params)
    for alias in _ConfigAliases.get("num_iterations"):
        if alias in result:
            raise ValueError(
                f"'{alias}' conflicts with max_boosting_rounds; remove it and use the dedicated optional argument"
            )
    for alias in _ConfigAliases.get("early_stopping_round"):
        if alias in result:
            value = result.pop(alias)
            if value not in (None, 0, "0"):
                raise ValueError(
                    "native early stopping conflicts with the authoritative validation-stopping controller"
                )
    if "early_stopping_min_delta" in result:
        value = result.pop("early_stopping_min_delta")
        if value not in (None, 0, 0.0, "0", "0.0"):
            raise ValueError("early_stopping_min_delta conflicts with the fixed validation-stopping controller")

    result = _choose_param_value("objective", result, None)
    if callable(result["objective"]):
        raise ValueError("custom objectives are not supported by adaptive validation stopping")
    result = _choose_param_value("boosting", result, "gbdt")
    if str(result["boosting"]).lower() not in {"gbdt", "goss"}:
        raise ValueError("adaptive validation stopping supports only GBDT/GOSS boosting")
    result = _choose_param_value("device_type", result, "cpu")
    if str(result["device_type"]).lower() != "cpu":
        raise ValueError("adaptive validation stopping currently supports only native CPU training")

    _single_string_param(result, "objective")
    _single_string_param(result, "metric")
    return result


def _single_validation_loss(results: list[EvalResult], validation_name: str, metric: str) -> float:
    if len(results) != 1:
        raise RuntimeError("adaptive validation stopping requires exactly one native validation metric result")
    result = results[0]
    if result.dataset_name != validation_name or result.metric_name != metric:
        observed = (result.dataset_name, result.metric_name)
        expected = (validation_name, metric)
        raise RuntimeError(f"unexpected validation result {observed}; expected {expected}")
    if result.maximize:
        raise ValueError("the stopping metric must be a loss for which lower values are better")
    ValidationStoppingController._validate_loss(result.metric_value, "validation loss")
    return float(result.metric_value)


def _selected_predictor(
    full_booster: Optional[Booster], initial_predictor: InitialPredictor, checkpoint: int
) -> Union[Booster, InitialPredictor]:
    if checkpoint == 0:
        return initial_predictor
    if full_booster is None:
        raise RuntimeError("a positive selected checkpoint requires a fitted Booster")
    model = Booster(model_str=full_booster.model_to_string(num_iteration=checkpoint))
    model.best_iteration = checkpoint
    return model


def train_with_validation_stopping(
    params: Dict[str, Any],
    train_set: Dataset,
    validation_set: Dataset,
    *,
    max_boosting_rounds: Optional[int] = None,
    validation_name: str = "validation",
) -> AdaptiveTrainingResult:
    """Train with fixed validation stopping and return the exact-best prefix.

    ``validation_set`` must be a stopping-only split distinct from ``train_set``;
    final-test outcomes must never be supplied here.  With no explicit cap, the
    deterministic theorem bound of 10002 rounds is only an implementation safety
    bound and is not reported as a user-specified maximum.
    """
    function_start_ns = perf_counter_ns()
    if not isinstance(params, dict):
        raise TypeError("params must be a dictionary")
    if not isinstance(train_set, Dataset) or not isinstance(validation_set, Dataset):
        raise TypeError("train_set and validation_set must be Dataset instances")
    if validation_set is train_set:
        raise ValueError("validation_set must be distinct from train_set")
    if not isinstance(validation_name, str) or not validation_name or validation_name == "training":
        raise ValueError("validation_name must be a nonempty name other than 'training'")
    ValidationStoppingController._validate_max_rounds(max_boosting_rounds)
    effective_params = _validate_params(params)
    metric = _single_string_param(effective_params, "metric")
    if metric not in {"binary_logloss", "multi_logloss", "rmse"}:
        raise ValueError("supported stopping metrics are binary_logloss, multi_logloss, and rmse")
    if _has_init_score(train_set) or _has_init_score(validation_set):
        raise ValueError("init_score is not supported because it cannot be embedded in a portable zero-tree predictor")
    if train_set._predictor is not None or validation_set._predictor is not None:
        raise ValueError("warm-start Dataset predictors are not supported")

    execution_limit = min(max_boosting_rounds or _THEORETICAL_MAX_ROUNDS, _THEORETICAL_MAX_ROUNDS)
    effective_params["num_iterations"] = execution_limit
    reduced_valid = validation_set._update_params(effective_params).set_reference(train_set)
    train_set._update_params(effective_params)
    booster: Optional[Booster] = None
    native_update_total_ns = 0
    per_round_native_ns = []
    per_round_validation_ns = []
    per_round_controller_ns = []
    try:
        train_set.construct()
        reduced_valid.construct()
        # Text and binary Dataset sidecars are loaded only during construction.
        if _has_init_score(train_set) or _has_init_score(reduced_valid):
            raise ValueError("init_score is not supported because it cannot be embedded in a portable zero-tree predictor")
        initial_validation_start_ns = perf_counter_ns()
        initial_predictor, initial_loss = _native_initial_state(effective_params, train_set, reduced_valid)
        initial_validation_ns = perf_counter_ns() - initial_validation_start_ns
        controller_start_ns = perf_counter_ns()
        controller = ValidationStoppingController(initial_loss, max_boosting_rounds)
        controller_initialization_ns = perf_counter_ns() - controller_start_ns

        if not controller.should_stop:
            booster = Booster(params=effective_params, train_set=train_set)
            booster.add_valid(reduced_valid, validation_name)
            for _ in range(execution_limit):
                before = booster.current_iteration()
                native_start_ns = perf_counter_ns()
                produced_empty_tree = booster.update()
                native_elapsed_ns = perf_counter_ns() - native_start_ns
                native_update_total_ns += native_elapsed_ns
                per_round_native_ns.append(native_elapsed_ns)
                after = booster.current_iteration()
                if after == before:
                    per_round_native_ns.pop()
                    controller_start_ns = perf_counter_ns()
                    controller.mark_native_termination()
                    controller_initialization_ns += perf_counter_ns() - controller_start_ns
                    break
                if after != before + 1:
                    raise RuntimeError(f"native update committed {after - before} rounds; expected exactly one")

                validation_start_ns = perf_counter_ns()
                validation_loss = _single_validation_loss(booster.eval_valid(), validation_name, metric)
                per_round_validation_ns.append(perf_counter_ns() - validation_start_ns)
                controller_start_ns = perf_counter_ns()
                decision = controller.observe(after, validation_loss)
                per_round_controller_ns.append(perf_counter_ns() - controller_start_ns)
                if decision.should_stop:
                    break
                if produced_empty_tree:
                    controller_start_ns = perf_counter_ns()
                    controller.mark_native_termination()
                    per_round_controller_ns[-1] += perf_counter_ns() - controller_start_ns
                    break
            else:
                if max_boosting_rounds is None:
                    raise AssertionError("uncapped validation stopping exceeded its deterministic 10002-round bound")
                raise AssertionError("the adaptive loop reached an unclassified execution limit")

        if controller.stop_reason is None:
            raise AssertionError("adaptive training ended without a classified stopping reason")
        prefix_start_ns = perf_counter_ns()
        selected = _selected_predictor(booster, initial_predictor, controller.best_checkpoint)
        prefix_ns = perf_counter_ns() - prefix_start_ns
    finally:
        train_set._reverse_update_params()
        reduced_valid._reverse_update_params()

    timing = AdaptiveTrainingTiming(
        total_function_ns=perf_counter_ns() - function_start_ns,
        initial_validation_evaluation_ns=initial_validation_ns,
        native_update_ns=native_update_total_ns,
        validation_evaluation_ns=initial_validation_ns + sum(per_round_validation_ns),
        controller_ns=controller_initialization_ns + sum(per_round_controller_ns),
        selected_prefix_materialization_ns=prefix_ns,
        per_round_native_update_ns=tuple(per_round_native_ns),
        per_round_validation_evaluation_ns=tuple(per_round_validation_ns),
        per_round_controller_ns=tuple(per_round_controller_ns),
    )
    return AdaptiveTrainingResult(
        full_booster=booster,
        predictor=selected,
        executed_rounds=controller.completed_rounds,
        selected_checkpoint=controller.best_checkpoint,
        initial_validation_loss=initial_loss,
        selected_validation_loss=controller.best_loss,
        epsilon=controller.epsilon,
        patience=controller.patience,
        stopping_bound_count=controller.stopping_bound_count,
        absolute_threshold=controller.absolute_threshold,
        anchor_loss=controller.anchor_loss,
        last_reset_round=controller.last_reset_round,
        terminal_patience_counter=(
            None
            if controller.last_reset_round is None
            else controller.completed_rounds - controller.last_reset_round
        ),
        validation_losses=tuple(controller.losses),
        stop_reason=controller.stop_reason,
        patience_satisfied=controller.patience_satisfied,
        round_cap_reached=controller.round_cap_reached,
        uncapped_endpoint_censored=controller.stop_reason == "max_rounds" and not controller.patience_satisfied,
        max_boosting_rounds=max_boosting_rounds,
        max_boosting_rounds_provenance=(
            "omitted" if max_boosting_rounds is None else "user_explicit"
        ),
        portable_resumption_supported=False,
        timing=timing,
    )
