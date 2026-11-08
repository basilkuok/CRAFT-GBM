"""Focused public-release checks; no benchmark datasets or performance claims."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from train_craft import TrainingLossController, train_craft


class CraftTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(73)
        self.x = rng.normal(size=(256, 8))
        self.y = (self.x[:, 0] + rng.normal(size=256) > 0).astype(int)

    def params(self, objective="binary", fraction=.5):
        return {"objective": objective, "feature_fraction": fraction,
                "num_threads": 2, "num_leaves": 7, "min_data_in_leaf": 10,
                "learning_rate": .15, "verbosity": -1}

    def test_training_patience_threshold_and_validation_difference(self):
        training = TrainingLossController(1000.0)
        validation = lgb.ValidationStoppingController(1000.0)
        losses = [999.0, 998.0, 997.0] + [998.0] * 9
        for round_number, loss in enumerate(losses, 1):
            stop = training.observe(round_number, loss)
            validation.observe(round_number, loss)
            self.assertEqual(stop, round_number == 12)
        # Exactly threshold-sized improvement at round 3 does not reset.
        self.assertEqual(training.last_reset_round, 2)
        self.assertEqual(validation.best_checkpoint, 3)
        self.assertEqual(len(training.losses) - 1, 12)
        with self.assertRaises(RuntimeError):
            training.observe(13, 998.0)
        for initial in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(initial=initial), self.assertRaises(ValueError):
                TrainingLossController(initial)
        reset = TrainingLossController(1000.0)
        for round_number, loss in enumerate([999., 998., 996.5], 1):
            self.assertFalse(reset.observe(round_number, loss))
        self.assertEqual(reset.last_reset_round, 3)
        with self.assertRaises(ValueError):
            reset.observe(5, 996.5)

    def test_fixed_budget_objectives_and_prediction_round_trip(self):
        for objective in ("binary", "regression", "multiclass"):
            with self.subTest(objective=objective):
                params = self.params(objective)
                labels = self.y
                if objective == "regression":
                    labels = self.x[:, 0] + .3 * self.x[:, 1]
                elif objective == "multiclass":
                    labels = np.argmax(self.x[:, :3], axis=1)
                    params["num_class"] = 3
                result = train_craft(params, lgb.Dataset(self.x, label=labels), num_boost_round=5)
                self.assertEqual(result.stop_reason, "fixed_horizon")
                self.assertEqual(result.booster.current_iteration(), 5)
                self.assertEqual(result.booster.num_trees(), 15 if objective == "multiclass" else 5)
                self.assertEqual(result.booster.adaptive_feature_selected_count_history(),
                                 [8, 8, 4, 4, 4])
                restored = lgb.Booster(model_str=result.booster.model_to_string())
                np.testing.assert_array_equal(result.booster.predict(self.x, num_threads=2),
                                              restored.predict(self.x, num_threads=2))

    def test_endogenous_returns_completed_patience_endpoint(self):
        result = train_craft(self.params(fraction=1.), lgb.Dataset(self.x, label=self.y))
        completed = result.booster.current_iteration()
        self.assertEqual(result.stop_reason, "training_loss_patience")
        self.assertEqual(len(result.training_losses), completed + 1)
        self.assertLessEqual(completed, 10002)
        self.assertGreaterEqual(completed, 12)
        counts = result.booster.adaptive_feature_selected_count_history()
        self.assertEqual(len(counts), completed)
        for round_number, selected in enumerate(counts, 1):
            if round_number <= 20:
                self.assertEqual(selected, 8)
            else:
                self.assertTrue(1 <= selected <= 8)
        # The returned model includes all ten patience rounds, with no trimming.
        anchor = min(result.training_losses[:3])
        last_reset = 2
        for r, loss in enumerate(result.training_losses[3:], 3):
            if loss < anchor - .001 * result.training_losses[0]:
                anchor, last_reset = loss, r
        self.assertEqual(completed - last_reset, 10)

    def test_external_horizon_and_no_progress_fail_explicitly(self):
        dataset = lgb.Dataset(self.x, label=self.y)
        with self.assertRaises(ValueError):
            train_craft(self.params(fraction=1.), dataset, num_boost_round=20)
        with self.assertRaises(ValueError):
            train_craft(self.params(fraction=.5), dataset)
        with patch.object(lgb.validation_stopping, "_native_initial_state", return_value=(None, 1.)), \
                patch.object(lgb, "Booster") as native:
            native.return_value.current_iteration.return_value = 0
            with self.assertRaisesRegex(RuntimeError, "one complete boosting round"):
                train_craft(self.params(fraction=.5), dataset, num_boost_round=5)


if __name__ == "__main__":
    unittest.main()
