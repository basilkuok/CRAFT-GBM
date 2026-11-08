/*!
 * Copyright (c) 2026-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#include <gtest/gtest.h>

#include <LightGBM/config.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/treelearner/adaptive_feature_root_gain.hpp"
#include "../../src/treelearner/adaptive_feature_mcb.hpp"

namespace LightGBM {

// Controlled interval states isolate ranking and diagnostic arithmetic from
// the boundary solver, whose coverage and numerical tests are separate.
struct AdaptiveFeatureMcbTestPeer {
  static void SetConfidenceState(
      AdaptiveFeatureMcb* sampler, const std::vector<double>& means,
      const std::vector<double>& counts, const std::vector<double>& radii) {
    ASSERT_FALSE(sampler->awaiting_update_);
    ASSERT_EQ(means.size(), sampler->unit_count());
    ASSERT_EQ(counts.size(), means.size());
    ASSERT_EQ(radii.size(), means.size());
    for (size_t i = 0; i < means.size(); ++i) {
      sampler->accumulated_gain_[i] = means[i] * counts[i];
    }
    sampler->exposure_count_ = counts;
    sampler->radius_by_unit_ = radii;
  }
};

}  // namespace LightGBM

namespace {

using LightGBM::AdaptiveFeatureRootGain;
using LightGBM::AdaptiveFeatureMcb;
using LightGBM::AdaptiveFeatureMcbTestPeer;
using LightGBM::GammaExponentialMixtureBoundary;
using LightGBM::Config;

Config BaseAdaptiveConfig() {
  Config config;
  config.num_iterations = 600;
  config.feature_fraction = 1.0;
  config.adaptive_feature_selection = true;
  config.adaptive_feature_confidence_delta = 0.05;
  config.adaptive_feature_log_period = 0;
  return config;
}

std::vector<double> ExtremeRewards(const std::vector<int>& selected) {
  std::vector<double> rewards;
  rewards.reserve(selected.size());
  for (int position : selected) {
    rewards.push_back(position == 0 ? 1.0 : 0.0);
  }
  return rewards;
}

void ExpectSelectionMatchesEndogenousRule(
    const AdaptiveFeatureMcb& sampler,
    const std::vector<int>& selected) {
  GammaExponentialMixtureBoundary boundary(
      sampler.log_inverse_alpha(), sampler.empirical_bernstein_rho());
  double threshold = 0.0;
  for (size_t position = 0; position < sampler.unit_count(); ++position) {
    const double exposure_count = sampler.exposure_count()[position];
    double lower_bound = 0.0;
    if (exposure_count > 0.0) {
      const double mean = std::max(
          0.0,
          std::min(
              1.0,
              sampler.accumulated_gain()[position] / exposure_count));
      const double radius =
          boundary.Boundary(sampler.empirical_variance()[position]) / exposure_count;
      lower_bound =
          std::max(0.0, std::min(1.0, mean - radius));
    }
    threshold = std::max(
        threshold, lower_bound);
  }
  EXPECT_DOUBLE_EQ(sampler.comparison_threshold(), threshold);
  std::vector<int> reference_selected;
  for (size_t position = 0; position < sampler.unit_count(); ++position) {
    const double exposure_count = sampler.exposure_count()[position];
    double upper_bound = 1.0;
    if (exposure_count > 0.0) {
      const double mean = std::max(
          0.0,
          std::min(
              1.0,
              sampler.accumulated_gain()[position] / exposure_count));
      const double radius =
          boundary.Boundary(sampler.empirical_variance()[position]) / exposure_count;
      upper_bound =
          std::max(0.0, std::min(1.0, mean + radius));
    }
    const bool expected =
        upper_bound >= threshold;
    const bool actual =
        std::find(
            selected.begin(), selected.end(),
            static_cast<int>(position)) != selected.end();
    EXPECT_EQ(actual, expected || sampler.selection_round() <= 20);
    if (expected) {
      reference_selected.push_back(static_cast<int>(position));
    }
  }
  EXPECT_EQ(sampler.possible_best_count(), reference_selected.size());
  if (sampler.selection_round() <= 20) {
    EXPECT_EQ(selected.size(), sampler.unit_count());
  } else {
    EXPECT_EQ(selected, reference_selected);
  }
}

TEST(AdaptiveFeatureMcbTest,
     ProposalAbortIsDeterministicAndRoundTwoIsAutomaticallyFull) {
  Config config = BaseAdaptiveConfig();
  config.num_iterations = 4;
  AdaptiveFeatureMcb sampler(&config);
  sampler.Reset({50, 10, 40, 20});
  EXPECT_TRUE(sampler.selected_count_history().empty());

  const std::vector<int> first = sampler.SelectNextSubset();
  EXPECT_EQ(first, std::vector<int>({0, 1, 2, 3}));
  EXPECT_TRUE(sampler.awaiting_update());
  EXPECT_EQ(sampler.round(), 0);
  EXPECT_EQ(sampler.selection_round(), 1);
  sampler.AbortPendingSelection();
  EXPECT_TRUE(sampler.selected_count_history().empty());

  EXPECT_EQ(sampler.SelectNextSubset(), first);
  sampler.Update({0.9, 0.0, 0.1, 1.0});
  EXPECT_EQ(sampler.round(), 1);
  EXPECT_EQ(sampler.selected_count_history(), std::vector<int>({4}));
  EXPECT_TRUE(sampler.calibration_complete());
  EXPECT_EQ(sampler.accumulated_gain(), std::vector<double>(4, 0.0));
  EXPECT_EQ(sampler.exposure_count(), std::vector<double>(4, 0.0));
  EXPECT_EQ(sampler.empirical_variance(), std::vector<double>(4, 0.0));

  for (int position = 0; position < 4; ++position) {
    EXPECT_DOUBLE_EQ(sampler.mean(position), 0.0);
    EXPECT_DOUBLE_EQ(sampler.lower_bound(position), 0.0);
    EXPECT_DOUBLE_EQ(sampler.upper_bound(position), 1.0);
    EXPECT_TRUE(std::isinf(sampler.radius(position)));
  }
  const std::vector<int> second = sampler.SelectNextSubset();
  EXPECT_EQ(second, first);
  EXPECT_DOUBLE_EQ(sampler.comparison_threshold(), 0.0);
  ExpectSelectionMatchesEndogenousRule(sampler, second);
  sampler.AbortPendingSelection();
  EXPECT_EQ(sampler.SelectNextSubset(), second);
  sampler.Update({1.0, 0.0, 0.5, 0.25});
  EXPECT_EQ(sampler.round(), 2);
  EXPECT_EQ(sampler.exposure_count(), std::vector<double>(4, 1.0));
  for (int position = 0; position < 4; ++position) {
    EXPECT_TRUE(std::isfinite(sampler.radius(position)));
    EXPECT_DOUBLE_EQ(sampler.lower_bound(position), 0.0);
    EXPECT_DOUBLE_EQ(sampler.upper_bound(position), 1.0);
  }
}

TEST(AdaptiveFeatureMcbTest,
     EndogenousWarmupRetainsAllFeaturesThroughTwentyAndCommitsCoveredUpdates) {
  Config config = BaseAdaptiveConfig();
  config.adaptive_feature_log_period = 1;
  AdaptiveFeatureMcb sampler(&config);
  sampler.Reset({0, 1, 2});
  EXPECT_EQ(sampler.full_feature_warmup_rounds(), 20);
  const std::vector<int> full({0, 1, 2});

  for (int round = 1; round <= 19; ++round) {
    EXPECT_EQ(sampler.SelectNextSubset(), full);
    sampler.Update({1.0, 0.0, 0.25});
    EXPECT_EQ(sampler.exposure_count(), std::vector<double>(3, round - 1));
  }
  EXPECT_EQ(sampler.accumulated_gain(),
            std::vector<double>({18.0, 0.0, 4.5}));
  const auto gains = sampler.accumulated_gain();
  const auto counts = sampler.exposure_count();
  const auto variances = sampler.empirical_variance();
  // Force informative bounds to distinguish full warmup selection from its
  // strict possible-best subset. The diagnostic must not claim all are best.
  AdaptiveFeatureMcbTestPeer::SetConfidenceState(
      &sampler, {1.0, 0.0, 0.25}, counts, {0.01, 0.01, 0.01});
  EXPECT_EQ(sampler.SelectNextSubset(), full);
  EXPECT_EQ(sampler.possible_best_count(), 1);
  EXPECT_TRUE(sampler.possible_best_set_fully_retained());
  EXPECT_DOUBLE_EQ(sampler.comparison_threshold(), 0.99);
  EXPECT_DOUBLE_EQ(sampler.selection_score_cutoff(), 0.01);
  sampler.AbortPendingSelection();
  EXPECT_EQ(sampler.round(), 19);
  EXPECT_EQ(sampler.accumulated_gain(), gains);
  EXPECT_EQ(sampler.exposure_count(), counts);
  EXPECT_EQ(sampler.empirical_variance(), variances);
  EXPECT_EQ(sampler.SelectNextSubset(), full);
  sampler.Update({1.0, 0.0, 0.25});
  EXPECT_EQ(sampler.exposure_count(), std::vector<double>(3, 19.0));
  EXPECT_EQ(sampler.detailed_possible_best_positions(), std::vector<int>({0}));
  EXPECT_NE(sampler.DetailedReceiptLine().find("selection_phase=warmup_full"),
            std::string::npos);
  EXPECT_NE(sampler.SummaryReceiptLine().find("full_feature_warmup_rounds=20"),
            std::string::npos);

  AdaptiveFeatureMcbTestPeer::SetConfidenceState(
      &sampler, {1.0, 0.0, 0.25}, {19.0, 19.0, 19.0}, {0.01, 0.01, 0.01});
  EXPECT_EQ(sampler.SelectNextSubset(), std::vector<int>({0}));
  sampler.AbortPendingSelection();
  EXPECT_EQ(sampler.round(), 20);
  EXPECT_EQ(sampler.exposure_count(), std::vector<double>(3, 19.0));
  EXPECT_EQ(sampler.SelectNextSubset(), std::vector<int>({0}));
  sampler.Update({1.0});
  EXPECT_EQ(sampler.exposure_count(), std::vector<double>({20.0, 19.0, 19.0}));
  EXPECT_EQ(sampler.last_committed_newly_excluded_count(), 2);
  EXPECT_NE(sampler.DetailedReceiptLine().find("selection_phase=adaptive"),
            std::string::npos);
}

TEST(AdaptiveFeatureMcbTest,
     FixedBudgetUsesNativeRoundingAndKeepsTheFirstTwoRoundsFull) {
  Config below_half = BaseAdaptiveConfig();
  below_half.feature_fraction = 0.49;
  AdaptiveFeatureMcb rounded_down(&below_half);
  rounded_down.Reset({50, 10, 40, 20, 30});
  EXPECT_TRUE(rounded_down.fixed_budget_enabled());
  EXPECT_EQ(rounded_down.full_feature_warmup_rounds(), 2);
  EXPECT_EQ(rounded_down.fixed_budget_count(), 2);

  Config at_half = below_half;
  at_half.feature_fraction = 0.5;
  AdaptiveFeatureMcb rounded_up(&at_half);
  rounded_up.Reset({50, 10, 40, 20, 30});
  EXPECT_EQ(rounded_up.fixed_budget_count(), 3);

  Config minimum = below_half;
  minimum.feature_fraction = 1e-12;
  AdaptiveFeatureMcb at_least_one(&minimum);
  at_least_one.Reset({50, 10, 40, 20, 30});
  EXPECT_EQ(at_least_one.fixed_budget_count(), 1);

  Config rounded_to_all = below_half;
  rounded_to_all.feature_fraction = 0.99;
  AdaptiveFeatureMcb all_units(&rounded_to_all);
  all_units.Reset({50, 10, 40, 20, 30});
  EXPECT_TRUE(all_units.fixed_budget_enabled());
  EXPECT_EQ(all_units.fixed_budget_count(), 5);

  Config config = BaseAdaptiveConfig();
  config.num_iterations = 80;
  config.adaptive_feature_confidence_delta = 0.99;
  config.feature_fraction = 0.4;
  AdaptiveFeatureMcb sampler(&config);
  sampler.Reset({50, 10, 40, 20, 30});
  EXPECT_DOUBLE_EQ(sampler.feature_fraction(), 0.4);
  EXPECT_EQ(sampler.fixed_budget_count(), 2);

  const std::vector<int> full({0, 1, 2, 3, 4});
  EXPECT_EQ(sampler.SelectNextSubset(), full);
  sampler.Update(std::vector<double>(5, 0.0));
  EXPECT_EQ(sampler.SelectNextSubset(), full);
  sampler.Update(std::vector<double>(5, 0.5));

  // All bounded interval midpoints and exposure counts tie after the full
  // covered round, so
  // canonical IDs 10 and 20 win; returned positions remain sorted for the
  // downstream mask updater.
  const std::vector<int> third = sampler.SelectNextSubset();
  EXPECT_EQ(third, std::vector<int>({1, 3}));
  EXPECT_EQ(sampler.possible_best_count(), 5);
  EXPECT_FALSE(sampler.possible_best_set_fully_retained());
  EXPECT_DOUBLE_EQ(sampler.selection_score_cutoff(), 0.5);
  sampler.AbortPendingSelection();
  EXPECT_EQ(sampler.SelectNextSubset(), third);
  sampler.Update(std::vector<double>(2, 0.5));

  // The midpoints remain tied at one half; smaller exposure now has priority,
  // followed by canonical ID among the three previously omitted units.
  const std::vector<int> fourth = sampler.SelectNextSubset();
  EXPECT_EQ(fourth, std::vector<int>({2, 4}));
  sampler.Update(std::vector<double>(2, 0.5));
  EXPECT_EQ(sampler.last_committed_selected_count(), 2);
  const std::string receipt = sampler.SummaryReceiptLine();
  EXPECT_NE(receipt.find("[CRAFT-LightGBM]"), std::string::npos);
  EXPECT_NE(receipt.find("selector=confidence_midpoint_top_k"), std::string::npos);
  EXPECT_NE(receipt.find("cardinality_mode=exogenous_fixed"), std::string::npos);
  EXPECT_NE(receipt.find("feature_fraction=0.40000000000000002"),
            std::string::npos);
  EXPECT_NE(receipt.find("fixed_budget_units=2"), std::string::npos);
  EXPECT_NE(receipt.find("fixed_budget_active=true"), std::string::npos);
  EXPECT_NE(
      receipt.find(
          "fixed_budget_tie_break=midpoint_desc_exposure_asc_canonical_id_asc"),
      std::string::npos);
}

TEST(AdaptiveFeatureMcbTest,
     FixedPossibleBestDiagnosticIsLazyExactAndRollbackSafe) {
  Config config = BaseAdaptiveConfig();
  config.num_iterations = 10;
  config.adaptive_feature_confidence_delta = 0.99;
  config.feature_fraction = 0.4;
  AdaptiveFeatureMcb sampler(&config);
  sampler.Reset({50, 10, 40, 20, 30});

  sampler.SelectNextSubset();
  sampler.Update(std::vector<double>(5, 0.0));
  sampler.SelectNextSubset();
  sampler.Update({1.0, 0.75, 0.5, 0.25, 0.0});
  EXPECT_EQ(sampler.possible_best_diagnostic_scan_count(), 0U);

  // The fixed selector ranks midpoints. Its possible-best count is computed
  // exactly only when a diagnostic consumer asks for it.
  const auto third = sampler.SelectNextSubset();
  ASSERT_EQ(third.size(), 2U);
  EXPECT_EQ(sampler.possible_best_diagnostic_scan_count(), 0U);
  sampler.Update(std::vector<double>(third.size(), 0.5));
  EXPECT_EQ(sampler.possible_best_diagnostic_scan_count(), 0U);

  // A committed receipt remains available while a later proposal is pending.
  const auto fourth = sampler.SelectNextSubset();
  const std::string committed_receipt = sampler.SummaryReceiptLine();
  EXPECT_FALSE(committed_receipt.empty());
  EXPECT_EQ(sampler.possible_best_diagnostic_scan_count(), 1U);
  const int pending_possible_best = sampler.possible_best_count();
  EXPECT_GE(pending_possible_best, 1);
  EXPECT_LE(pending_possible_best, 5);
  EXPECT_EQ(sampler.possible_best_diagnostic_scan_count(), 2U);
  EXPECT_EQ(sampler.possible_best_count(), pending_possible_best);
  EXPECT_EQ(sampler.possible_best_diagnostic_scan_count(), 2U);

  sampler.AbortPendingSelection();
  EXPECT_EQ(sampler.possible_best_diagnostic_scan_count(), 2U);
  const int committed_possible_best = sampler.possible_best_count();
  EXPECT_GE(committed_possible_best, 1);
  EXPECT_EQ(sampler.possible_best_diagnostic_scan_count(), 2U);

  // If no pending diagnostic was requested, an accepted round remains lazy
  // and materializes exactly once when its committed value is queried.
  EXPECT_EQ(sampler.SelectNextSubset(), fourth);
  sampler.Update(std::vector<double>(fourth.size(), 0.5));
  EXPECT_EQ(sampler.possible_best_diagnostic_scan_count(), 2U);
  EXPECT_GE(sampler.possible_best_count(), 1);
  EXPECT_EQ(sampler.possible_best_diagnostic_scan_count(), 3U);
  EXPECT_GE(sampler.possible_best_count(), 1);
  EXPECT_EQ(sampler.possible_best_diagnostic_scan_count(), 3U);
}

TEST(AdaptiveFeatureMcbTest,
     PeriodicDetailedReceiptUsesPreSelectionStateAndExactDecisionSets) {
  Config config = BaseAdaptiveConfig();
  config.feature_fraction = 0.5;
  config.adaptive_feature_log_period = 3;
  AdaptiveFeatureMcb sampler(&config);
  sampler.Reset({20, 10});

  sampler.SelectNextSubset();
  sampler.Update({0.0, 0.0});
  EXPECT_FALSE(sampler.detailed_receipt_available());
  sampler.SelectNextSubset();
  sampler.Update({0.75, 0.25});
  EXPECT_FALSE(sampler.detailed_receipt_available());

  // Both intervals are [0,1] after the first covered observation. Canonical ID 10
  // wins the fixed-budget tie, while both units remain possible best.
  EXPECT_EQ(sampler.SelectNextSubset(), std::vector<int>({1}));
  sampler.Update({0.5});
  ASSERT_TRUE(sampler.detailed_receipt_available());
  EXPECT_EQ(sampler.possible_best_diagnostic_scan_count(), 1U);
  EXPECT_EQ(
      sampler.detailed_possible_best_positions(),
      std::vector<int>({0, 1}));
  const std::string receipt = sampler.DetailedReceiptLine();
  EXPECT_NE(receipt.find("iteration=3"), std::string::npos);
  EXPECT_NE(
      receipt.find("diagnostic_state=pre_iteration_selection"),
      std::string::npos);
  EXPECT_NE(
      receipt.find("history_through_iteration=2"),
      std::string::npos);
  EXPECT_NE(
      receipt.find("canonical_unit_ids=[20,10]"),
      std::string::npos);
  EXPECT_NE(
      receipt.find("selected_canonical_unit_ids=[10]"),
      std::string::npos);
  EXPECT_NE(
      receipt.find("possible_best_canonical_unit_ids=[20,10]"),
      std::string::npos);
  EXPECT_NE(
      receipt.find("accumulated_gain_by_unit=[0.75,0.25]"),
      std::string::npos);
  EXPECT_NE(
      receipt.find("exposure_count_by_unit=[1,1]"),
      std::string::npos);
  EXPECT_NE(
      receipt.find("empirical_variance_by_unit=[0.0625,0.0625]"),
      std::string::npos);
  EXPECT_NE(
      receipt.find("lower_confidence_bound_by_unit=[0,0]"),
      std::string::npos);
  EXPECT_NE(
      receipt.find("upper_confidence_bound_by_unit=[1,1]"),
      std::string::npos);
  // The controller has already committed round 3; the receipt deliberately
  // preserves the immutable state that selected round 3.
  EXPECT_EQ(sampler.accumulated_gain(), std::vector<double>({0.75, 0.75}));
  EXPECT_EQ(sampler.exposure_count(), std::vector<double>({1.0, 2.0}));

  const auto fourth = sampler.SelectNextSubset();
  sampler.Update(std::vector<double>(fourth.size(), 0.0));
  EXPECT_FALSE(sampler.detailed_receipt_available());
  EXPECT_EQ(sampler.possible_best_diagnostic_scan_count(), 1U);
}

TEST(AdaptiveFeatureMcbTest, DirectConfigurationRejectsInvalidFractions) {
  for (double invalid_fraction :
       {0.0, -0.1, 1.01, std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    Config config = BaseAdaptiveConfig();
    config.feature_fraction = invalid_fraction;
    EXPECT_THROW(
        {
          AdaptiveFeatureMcb invalid(&config);
          (void)invalid;
        },
        std::runtime_error);
  }
}

TEST(AdaptiveFeatureMcbTest, FixedBudgetSelectsTheLargestNonTiedMidpoint) {
  Config config = BaseAdaptiveConfig();
  config.num_iterations = 3;
  config.adaptive_feature_confidence_delta = 0.99;
  config.feature_fraction = 0.5;
  AdaptiveFeatureMcb sampler(&config);
  sampler.Reset({9, 4});

  sampler.SelectNextSubset();
  sampler.Update({0.0, 0.0});
  sampler.SelectNextSubset();
  sampler.Update({1.0, 0.0});

  // The calibrated confidence sequences initially give both intervals [0,1].
  // Exercise the deterministic exposure tie-break until accumulated evidence
  // separates the two endpoints, recording each unit's fixed reward law.
  const auto midpoint = [&sampler](int position) {
    return 0.5 * (sampler.lower_bound(position) + sampler.upper_bound(position));
  };
  bool observed_non_tied_midpoint = false;
  while (sampler.round() < 40 && !observed_non_tied_midpoint) {
    const auto selected = sampler.SelectNextSubset();
    ASSERT_EQ(selected.size(), 1U);
    sampler.Update({selected.front() == 0 ? 1.0 : 0.0});
    observed_non_tied_midpoint = midpoint(0) > midpoint(1);
  }
  ASSERT_TRUE(observed_non_tied_midpoint);

  const std::vector<int> selected = sampler.SelectNextSubset();
  EXPECT_EQ(selected, std::vector<int>({0}));
  EXPECT_DOUBLE_EQ(
      sampler.selection_score_cutoff(), midpoint(0));
  sampler.Update({0.5});
}

TEST(AdaptiveFeatureMcbTest, FixedBudgetRanksBoundedMidpointNotMeanOrUpperBound) {
  Config config = BaseAdaptiveConfig();
  config.feature_fraction = 0.5;
  AdaptiveFeatureMcb sampler(&config);
  sampler.Reset({0, 1});
  sampler.SelectNextSubset();
  sampler.Update({0.0, 0.0});
  sampler.SelectNextSubset();
  sampler.Update({0.0, 0.0});

  // Empirical mean prefers feature 1 (.15 > .10), but bounded midpoints
  // prefer feature 0 (.20 > .15): [0,.40] versus [.14,.16].
  AdaptiveFeatureMcbTestPeer::SetConfidenceState(
      &sampler, {0.10, 0.15}, {100.0, 100.0}, {0.30, 0.01});
  EXPECT_EQ(sampler.SelectNextSubset(), std::vector<int>({0}));
  EXPECT_DOUBLE_EQ(sampler.selection_score_cutoff(), 0.20);
  sampler.AbortPendingSelection();

  // Upper-bound ranking prefers the saturated interval [0,1], whereas
  // midpoint ranking prefers [.69,.71].
  AdaptiveFeatureMcbTestPeer::SetConfidenceState(
      &sampler, {0.10, 0.70}, {100.0, 100.0}, {2.0, 0.01});
  EXPECT_EQ(sampler.SelectNextSubset(), std::vector<int>({1}));
  EXPECT_DOUBLE_EQ(sampler.selection_score_cutoff(), 0.70);
  sampler.AbortPendingSelection();

  // Upper-end restriction is also reflected in the score: [.60,1] has
  // midpoint .80, not the empirical mean .90.
  AdaptiveFeatureMcbTestPeer::SetConfidenceState(
      &sampler, {0.90, 0.85}, {100.0, 100.0}, {0.30, 0.01});
  EXPECT_EQ(sampler.SelectNextSubset(), std::vector<int>({1}));
  EXPECT_DOUBLE_EQ(sampler.selection_score_cutoff(), 0.85);
  sampler.AbortPendingSelection();
}

TEST(AdaptiveFeatureMcbTest, PossibleBestCountWithinBudgetDoesNotImplyContainment) {
  Config config = BaseAdaptiveConfig();
  config.feature_fraction = 0.5;  // Native rounding gives K=2 for p=3.
  config.adaptive_feature_log_period = 3;
  AdaptiveFeatureMcb sampler(&config);
  sampler.Reset({0, 1, 2});
  sampler.SelectNextSubset();
  sampler.Update({0.0, 0.0, 0.0});
  sampler.SelectNextSubset();
  sampler.Update({0.0, 0.0, 0.0});
  AdaptiveFeatureMcbTestPeer::SetConfidenceState(
      &sampler, {0.80, 0.40, 0.60}, {100.0, 100.0, 100.0},
      {0.01, 0.41, 0.01});

  // Possible best = {0,1}, midpoint top-2 = {0,2}. The old count<=K shortcut
  // would report containment incorrectly.
  EXPECT_EQ(sampler.SelectNextSubset(), std::vector<int>({0, 2}));
  EXPECT_EQ(sampler.possible_best_diagnostic_scan_count(), 0U);
  EXPECT_EQ(sampler.possible_best_count(), 2);
  EXPECT_FALSE(sampler.possible_best_set_fully_retained());
  sampler.AbortPendingSelection();
  EXPECT_EQ(sampler.SelectNextSubset(), std::vector<int>({0, 2}));
  sampler.Update({0.1, 0.1});
  EXPECT_EQ(sampler.possible_best_count(), 2);
  EXPECT_FALSE(sampler.possible_best_set_fully_retained());
  EXPECT_EQ(sampler.detailed_possible_best_positions(), std::vector<int>({0, 1}));
  EXPECT_NE(sampler.DetailedReceiptLine().find(
      "possible_best_set_fully_retained=false"), std::string::npos);
}

TEST(AdaptiveFeatureMcbTest, InclusiveMcbBoundaryRetainsAllTiedUnits) {
  Config config = BaseAdaptiveConfig();
  config.num_iterations = 80;
  config.adaptive_feature_confidence_delta = 0.99;
  AdaptiveFeatureMcb sampler(&config);
  sampler.Reset({40, 10, 30, 20});

  sampler.SelectNextSubset();
  sampler.Update(std::vector<double>(4, 0.0));
  for (int round = 2; round <= 50; ++round) {
    const std::vector<int> selected = sampler.SelectNextSubset();
    EXPECT_EQ(selected, std::vector<int>({0, 1, 2, 3}));
    ExpectSelectionMatchesEndogenousRule(sampler, selected);
    sampler.Update(std::vector<double>(4, 0.5));
  }

  EXPECT_DOUBLE_EQ(sampler.lower_bound(0), sampler.lower_bound(1));
  EXPECT_DOUBLE_EQ(sampler.upper_bound(0), sampler.upper_bound(1));
  EXPECT_EQ(sampler.last_committed_selected_count(), 4);
}

TEST(AdaptiveFeatureMcbTest,
     OmittedFeatureStateFreezesWithoutAutomaticUncertaintyWidening) {
  Config config = BaseAdaptiveConfig();
  config.num_iterations = 80;
  config.adaptive_feature_confidence_delta = 0.99;
  AdaptiveFeatureMcb sampler(&config);
  sampler.Reset({0, 1});

  sampler.SelectNextSubset();
  sampler.Update({0.0, 0.0});

  bool observed_exclusion = false;
  while (sampler.round() < 200 && !observed_exclusion) {
    const std::vector<int> selected = sampler.SelectNextSubset();
    ExpectSelectionMatchesEndogenousRule(sampler, selected);
    ASSERT_NE(
        std::find(selected.begin(), selected.end(), 0), selected.end());
    observed_exclusion =
        std::find(selected.begin(), selected.end(), 1) == selected.end();
    sampler.Update(ExtremeRewards(selected));
  }
  ASSERT_TRUE(observed_exclusion);
  EXPECT_EQ(sampler.last_committed_omitted_count(), 1);
  EXPECT_EQ(sampler.last_committed_newly_excluded_count(), 1);

  const double frozen_gain = sampler.accumulated_gain()[1];
  const double frozen_count = sampler.exposure_count()[1];
  const double frozen_variance = sampler.empirical_variance()[1];
  const double frozen_radius = sampler.radius(1);
  for (int step = 0; step < 12; ++step) {
    const std::vector<int> selected = sampler.SelectNextSubset();
    EXPECT_EQ(selected, std::vector<int>({0}));
    sampler.Update({1.0});
    EXPECT_DOUBLE_EQ(sampler.accumulated_gain()[1], frozen_gain);
    EXPECT_DOUBLE_EQ(sampler.exposure_count()[1], frozen_count);
    EXPECT_DOUBLE_EQ(sampler.empirical_variance()[1], frozen_variance);
    EXPECT_DOUBLE_EQ(sampler.radius(1), frozen_radius);
  }
  EXPECT_EQ(sampler.total_exclusion_events(), 1U);
  EXPECT_EQ(sampler.total_reentered(), 0U);
  const std::string receipt = sampler.SummaryReceiptLine();
  EXPECT_NE(receipt.find("selector=mcb_possible_best"), std::string::npos);
  EXPECT_NE(receipt.find("cumulative_statistics=A_N_V"), std::string::npos);
  EXPECT_NE(
      receipt.find(
          "target=accumulated_average_conditional_transformed_root_gain"),
      std::string::npos);
}

TEST(AdaptiveFeatureMcbTest, PredictableResidualsUsePreUpdateState) {
  Config config = BaseAdaptiveConfig();
  AdaptiveFeatureMcb sampler(&config);
  sampler.Reset({17});
  sampler.SelectNextSubset();
  sampler.Update({1.0});
  EXPECT_EQ(sampler.empirical_variance(), std::vector<double>({0.0}));

  const std::vector<double> rewards = {0.25, 0.75, 0.0, 1.0, 0.5};
  double gain = 0.0;
  double variance = 0.0;
  double count = 0.0;
  for (double reward : rewards) {
    sampler.SelectNextSubset();
    const double prediction = (0.5 + gain) / (1.0 + count);
    variance += (reward - prediction) * (reward - prediction);
    gain += reward;
    count += 1.0;
    sampler.Update({reward});
    EXPECT_DOUBLE_EQ(sampler.empirical_variance()[0], variance);
    EXPECT_DOUBLE_EQ(sampler.accumulated_gain()[0], gain);
    EXPECT_DOUBLE_EQ(sampler.exposure_count()[0], count);
    EXPECT_DOUBLE_EQ(sampler.mean(0), gain / count);
  }
}

TEST(AdaptiveFeatureMcbTest, EqualMeansAndCountsCanHaveDifferentRadii) {
  Config config = BaseAdaptiveConfig();
  AdaptiveFeatureMcb constant(&config);
  AdaptiveFeatureMcb variable(&config);
  constant.Reset({1});
  variable.Reset({1});
  constant.SelectNextSubset();
  variable.SelectNextSubset();
  constant.Update({0.0});
  variable.Update({0.0});
  for (int n = 0; n < 40; ++n) {
    constant.SelectNextSubset();
    variable.SelectNextSubset();
    constant.Update({0.5});
    variable.Update({n % 2 == 0 ? 0.0 : 1.0});
  }
  EXPECT_EQ(constant.accumulated_gain(), variable.accumulated_gain());
  EXPECT_EQ(constant.exposure_count(), variable.exposure_count());
  EXPECT_DOUBLE_EQ(constant.empirical_variance()[0], 0.0);
  EXPECT_GT(variable.empirical_variance()[0], 0.0);
  EXPECT_LT(constant.radius(0), variable.radius(0));
}

TEST(AdaptiveFeatureMcbTest, RadiusInvalidatesWhenCountChangesButVarianceDoesNot) {
  Config config = BaseAdaptiveConfig();
  AdaptiveFeatureMcb sampler(&config);
  sampler.Reset({1});
  sampler.SelectNextSubset();
  sampler.Update({0.0});
  sampler.SelectNextSubset();
  sampler.Update({0.5});
  const double first = sampler.radius(0);
  sampler.SelectNextSubset();
  sampler.Update({0.5});
  EXPECT_DOUBLE_EQ(sampler.empirical_variance()[0], 0.0);
  EXPECT_DOUBLE_EQ(sampler.radius(0), first / 2.0);
  EXPECT_EQ(sampler.radius_cache_size(), 1U);
}

TEST(AdaptiveFeatureMcbTest, AbortAndInvalidObservationPreserveAllCommittedStatistics) {
  Config config = BaseAdaptiveConfig();
  config.feature_fraction = 0.5;
  AdaptiveFeatureMcb sampler(&config);
  sampler.Reset({1, 2});
  sampler.SelectNextSubset();
  sampler.Update({1.0, 0.0});
  sampler.SelectNextSubset();
  sampler.Update({0.75, 0.25});
  const auto gain = sampler.accumulated_gain();
  const auto count = sampler.exposure_count();
  const auto variance = sampler.empirical_variance();
  const auto proposal = sampler.SelectNextSubset();
  EXPECT_THROW(sampler.Update({std::numeric_limits<double>::quiet_NaN()}),
               std::runtime_error);
  sampler.AbortPendingSelection();
  EXPECT_EQ(sampler.accumulated_gain(), gain);
  EXPECT_EQ(sampler.exposure_count(), count);
  EXPECT_EQ(sampler.empirical_variance(), variance);
  EXPECT_EQ(sampler.SelectNextSubset(), proposal);
  sampler.AbortPendingSelection();
}

TEST(AdaptiveFeatureMcbTest, ErrorAllocationAndReferenceCalibrationAreDisclosed) {
  Config config = BaseAdaptiveConfig();
  AdaptiveFeatureMcb sampler(&config);
  sampler.Reset({0, 1, 2, 3, 4, 5, 6, 7});
  EXPECT_NEAR(sampler.log_inverse_alpha(), std::log(16.0 / 0.05), 2e-15);
  EXPECT_EQ(sampler.calibration_reference_count(), 300);
  EXPECT_DOUBLE_EQ(sampler.variance_reference(), 75.0);
  const double c = sampler.variance_reference() / sampler.empirical_bernstein_rho();
  EXPECT_NEAR(c - std::log1p(c), 2.0 * std::log(8.0 / 0.05), 2e-14);
  const std::string receipt = sampler.SummaryReceiptLine();
  for (const auto* field : {
           "confidence_sequence=time_uniform_empirical_bernstein",
           "boundary=gamma_exponential_mixture", "boundary_version=1",
           "rho_calibration=normal_reference_scale",
           "calibration_reference_count=300", "variance_reference=75",
           "cumulative_statistics=A_N_V", "predictor=(0.5+A)/(1+N)",
           "variance_grid_bins=128", "boundary_numerical_failures=0",
           "rho_uses_observed_gains=false", "rho_uses_validation_losses=false",
           "rho_uses_max_rounds=false"}) {
    EXPECT_NE(receipt.find(field), std::string::npos) << field;
  }
  EXPECT_EQ(receipt.find("normal_mixture_rho"), std::string::npos);
  EXPECT_EQ(receipt.find("stopping_bound_reference_minimizer"), std::string::npos);
}

TEST(AdaptiveFeatureMcbTest, ConfiguredHorizonAndLoggingDoNotChangeTheCommonPrefix) {
  Config short_config = BaseAdaptiveConfig();
  short_config.num_iterations = 1;
  Config long_config = short_config;
  long_config.num_iterations = 600;
  long_config.adaptive_feature_log_period = 1;
  AdaptiveFeatureMcb short_run(&short_config);
  AdaptiveFeatureMcb long_run(&long_config);
  short_run.Reset({0, 1, 2});
  long_run.Reset({0, 1, 2});
  EXPECT_DOUBLE_EQ(short_run.empirical_bernstein_rho(), long_run.empirical_bernstein_rho());
  for (int round = 0; round < 40; ++round) {
    const auto selected = short_run.SelectNextSubset();
    EXPECT_EQ(selected, long_run.SelectNextSubset());
    const auto rewards = ExtremeRewards(selected);
    short_run.Update(rewards);
    long_run.Update(rewards);
    EXPECT_EQ(short_run.accumulated_gain(), long_run.accumulated_gain());
    EXPECT_EQ(short_run.exposure_count(), long_run.exposure_count());
    EXPECT_EQ(short_run.empirical_variance(), long_run.empirical_variance());
    EXPECT_EQ(short_run.selected_count_history(), long_run.selected_count_history());
    EXPECT_EQ(short_run.SummaryReceiptLine(), long_run.SummaryReceiptLine());
  }
}

TEST(AdaptiveFeatureMcbTest, ResetReplacesVarianceCachesAndFrozenCalibration) {
  Config config = BaseAdaptiveConfig();
  AdaptiveFeatureMcb sampler(&config);
  EXPECT_THROW((void)sampler.empirical_bernstein_rho(), std::runtime_error);
  sampler.Reset({7});
  const double one_rho = sampler.empirical_bernstein_rho();
  sampler.SelectNextSubset();
  sampler.Update({0.0});
  sampler.SelectNextSubset();
  sampler.Update({1.0});
  EXPECT_GT(sampler.empirical_variance()[0], 0.0);
  (void)sampler.radius(0);
  sampler.Reset({7, 11});
  EXPECT_EQ(sampler.round(), 0);
  EXPECT_EQ(sampler.empirical_variance(), std::vector<double>(2, 0.0));
  EXPECT_EQ(sampler.radius_cache_size(), 2U);
  EXPECT_TRUE(std::isinf(sampler.radius(0)));
  EXPECT_NE(sampler.empirical_bernstein_rho(), one_rho);
  config.adaptive_feature_confidence_delta = 0.1;
  config.adaptive_feature_calibration_reference_count = 600;
  sampler.Configure(&config);
  EXPECT_THROW((void)sampler.SelectNextSubset(), std::runtime_error);
  sampler.Reset({7, 11});
  EXPECT_EQ(sampler.calibration_reference_count(), 600);
  EXPECT_DOUBLE_EQ(sampler.variance_reference(), 150.0);
  EXPECT_NEAR(sampler.empirical_bernstein_rho(), 2.0 * one_rho, 2e-14);
  config.adaptive_feature_calibration_reference_count = 1;
  sampler.SelectNextSubset();
  sampler.Update({0.0, 0.0});
  EXPECT_EQ(sampler.calibration_reference_count(), 600);
}

TEST(AdaptiveFeatureMcbTest, FixedSeedCoverageWitnessUsesPredictableSelectionAndConditionalMeans) {
  constexpr int kReplicates = 8;
  constexpr int kFeatures = 3;
  std::mt19937_64 generator(20260918ULL);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  for (int replicate = 0; replicate < kReplicates; ++replicate) {
    Config config = BaseAdaptiveConfig();
    config.adaptive_feature_confidence_delta = 0.05 / kReplicates;
    config.feature_fraction = 0.67;
    AdaptiveFeatureMcb sampler(&config);
    sampler.Reset({0, 1, 2});
    sampler.SelectNextSubset();
    sampler.Update(std::vector<double>(kFeatures, 0.0));
    std::vector<double> conditional_sum(kFeatures, 0.0);
    std::vector<double> previous(kFeatures, 0.0);
    for (int step = 0; step < 80; ++step) {
      const auto selected = sampler.SelectNextSubset();
      std::vector<double> rewards;
      for (int position : selected) {
        const double conditional_mean = 0.1 + 0.6 * previous[position];
        conditional_sum[position] += conditional_mean;
        previous[position] = uniform(generator) < conditional_mean ? 1.0 : 0.0;
        rewards.push_back(previous[position]);
      }
      sampler.Update(rewards);
      for (int position = 0; position < kFeatures; ++position) {
        const double count = sampler.exposure_count()[position];
        if (count > 0.0) {
          const double target = conditional_sum[position] / count;
          EXPECT_LE(sampler.lower_bound(position), target);
          EXPECT_GE(sampler.upper_bound(position), target);
        }
      }
    }
  }
}

TEST(AdaptiveFeatureMcbTest, GeneratedConfigurationHasLeanInterface) {
  const Config defaults;
  EXPECT_DOUBLE_EQ(defaults.feature_fraction, 1.0);
  EXPECT_DOUBLE_EQ(defaults.adaptive_feature_confidence_delta, 0.05);
  EXPECT_EQ(defaults.adaptive_feature_calibration_reference_count, 300);
  EXPECT_EQ(defaults.adaptive_feature_log_period, 0);

  const Config config(Config::Str2Map(
      "num_iterations=20 "
      "adaptive_feature_selection=true "
      "colsample_bytree=0.4 "
      "adaptive_feature_confidence_delta=0.1 "
      "adaptive_feature_calibration_reference_count=500 "
      "adaptive_feature_log_period=3"));
  EXPECT_TRUE(config.adaptive_feature_selection);
  EXPECT_DOUBLE_EQ(config.feature_fraction, 0.4);
  EXPECT_DOUBLE_EQ(config.adaptive_feature_confidence_delta, 0.1);
  EXPECT_EQ(config.adaptive_feature_calibration_reference_count, 500);
  EXPECT_NE(config.ToString().find(
      "[adaptive_feature_calibration_reference_count: 500]"), std::string::npos);
  EXPECT_EQ(config.adaptive_feature_log_period, 3);

  const auto& parameter_types = Config::ParameterTypes();
  const auto& aliases = Config::parameter2aliases();
  EXPECT_EQ(
      parameter_types.at("adaptive_feature_confidence_delta"), "double");
  EXPECT_EQ(parameter_types.at("adaptive_feature_calibration_reference_count"), "int");
  EXPECT_EQ(parameter_types.at("feature_fraction"), "double");
  EXPECT_NE(
      std::find(
          aliases.at("feature_fraction").begin(),
          aliases.at("feature_fraction").end(), "colsample_bytree"),
      aliases.at("feature_fraction").end());
}

TEST(AdaptiveFeatureMcbTest, ReferenceConfigurationRejectsInvalidNativeInputs) {
  for (const char* value : {"0", "-1", "50.5", "nan", "inf",
                            "2147483648", "4294967297"}) {
    EXPECT_THROW(
        {
          Config config;
          config.Set({{"adaptive_feature_calibration_reference_count", value}});
        },
        std::runtime_error) << value;
  }
  Config invalid = BaseAdaptiveConfig();
  invalid.adaptive_feature_calibration_reference_count = 0;
  EXPECT_THROW((void)AdaptiveFeatureMcb(&invalid), std::runtime_error);
}

TEST(AdaptiveFeatureMcbTest, CalibrationRemainsFiniteForOneUnitAndDeltaNearOne) {
  Config config = BaseAdaptiveConfig();
  config.adaptive_feature_confidence_delta = std::nextafter(1.0, 0.0);
  AdaptiveFeatureMcb sampler(&config);
  sampler.Reset({1});
  EXPECT_TRUE(std::isfinite(sampler.empirical_bernstein_rho()));
  EXPECT_GT(sampler.empirical_bernstein_rho(), 0.0);
  const double expected = GammaExponentialMixtureBoundary::CalibratedRho(
      -std::log(config.adaptive_feature_confidence_delta), 300.0);
  EXPECT_DOUBLE_EQ(sampler.empirical_bernstein_rho(), expected);
}

TEST(AdaptiveFeatureRootGainTest, InitialRootPredicateRejectsReusedLeafZero) {
  EXPECT_TRUE(AdaptiveFeatureRootGain::IsInitialRootSearch(1, 0, -1));
  EXPECT_FALSE(AdaptiveFeatureRootGain::IsInitialRootSearch(2, 0, -1));
  EXPECT_FALSE(AdaptiveFeatureRootGain::IsInitialRootSearch(2, 0, 1));
  EXPECT_FALSE(AdaptiveFeatureRootGain::IsInitialRootSearch(1, 1, -1));
  EXPECT_FALSE(AdaptiveFeatureRootGain::IsInitialRootSearch(1, 0, 1));
}

TEST(AdaptiveFeatureRootGainTest,
     LaggedScaleTransformIsBoundedMonotoneAndScaleEquivariant) {
  EXPECT_DOUBLE_EQ(
      AdaptiveFeatureRootGain::NormalizeOpportunity(0.0, 0.0, 7), 0.0);
  EXPECT_DOUBLE_EQ(
      AdaptiveFeatureRootGain::NormalizeOpportunity(0.25, 0.0, 7), 1.0);
  EXPECT_DOUBLE_EQ(
      AdaptiveFeatureRootGain::NormalizeOpportunity(0.25, 0.5, 7),
      1.0 / 3.0);
  const double lower =
      AdaptiveFeatureRootGain::NormalizeOpportunity(0.5, 2.0, 7);
  const double upper =
      AdaptiveFeatureRootGain::NormalizeOpportunity(4.0, 2.0, 7);
  EXPECT_LT(lower, upper);
  EXPECT_DOUBLE_EQ(
      AdaptiveFeatureRootGain::NormalizeOpportunity(5.0, 10.0, 7),
      AdaptiveFeatureRootGain::NormalizeOpportunity(0.5, 1.0, 7));
  EXPECT_THROW(
      (void)AdaptiveFeatureRootGain::NormalizeOpportunity(-0.01, 1.0, 7),
      std::runtime_error);
}

TEST(AdaptiveFeatureRootGainTest, PositiveMedianInPlaceComputesMedianAndCount) {
  const std::vector<double> source = {0.0, 8.0, 2.0, 4.0};
  std::vector<double> scratch = source;
  size_t in_place_count = 0;
  const double in_place =
      AdaptiveFeatureRootGain::PositiveMedianInPlace(
          &scratch, &in_place_count);
  EXPECT_DOUBLE_EQ(in_place, 4.0);
  EXPECT_EQ(in_place_count, 3U);
  EXPECT_EQ(scratch.size(), 3U);

  scratch = {2.0, 8.0};
  EXPECT_DOUBLE_EQ(
      AdaptiveFeatureRootGain::PositiveMedianInPlace(&scratch), 5.0);
  scratch = {0.0, 0.0};
  EXPECT_DOUBLE_EQ(
      AdaptiveFeatureRootGain::PositiveMedianInPlace(&scratch), 0.0);
  scratch = {1.0, -1.0};
  EXPECT_THROW(
      (void)AdaptiveFeatureRootGain::PositiveMedianInPlace(&scratch),
      std::runtime_error);
}

}  // namespace
