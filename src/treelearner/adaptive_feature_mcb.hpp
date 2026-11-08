/*!
 * Copyright (c) 2026-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_ADAPTIVE_FEATURE_MCB_HPP_
#define LIGHTGBM_SRC_TREELEARNER_ADAPTIVE_FEATURE_MCB_HPP_

#include <LightGBM/config.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "adaptive_feature_eb.hpp"

namespace LightGBM {

// Confidence-calibrated feature allocation. Positions index a fixed
// universe of valid original features.
// Each unit maintains cumulative transformed root gain A, observation count N,
// and the empirical-Bernstein intrinsic time V from predictable residuals.
// With feature_fraction=1, SelectNextSubset() uses the
// endogenous multiple-comparisons-with-the-best (MCB) possible-best set after
// 20 full-feature rounds, if training reaches them. With
// feature_fraction<1, the selected feature set size equals LightGBM's native
// rounded count, with selection by the largest bounded interval midpoints. Update() is
// the sole accepted-iteration commit, and AbortPendingSelection() discards an
// unaccepted proposal.
class AdaptiveFeatureMcb {
 public:
  explicit AdaptiveFeatureMcb(const Config* config) {
    Configure(config);
  }

  void Configure(const Config* config) {
    CHECK_NOTNULL(config);
    boundary_ready_ = false;
    empirical_bernstein_rho_ = std::numeric_limits<double>::quiet_NaN();
    confidence_delta_ = config->adaptive_feature_confidence_delta;
    calibration_reference_count_ =
        config->adaptive_feature_calibration_reference_count;
    log_period_ = config->adaptive_feature_log_period;
    feature_fraction_ = config->feature_fraction;
    ValidateConfiguration();
  }

  void Reset(const std::vector<int>& canonical_unit_ids) {
    boundary_ready_ = false;
    if (canonical_unit_ids.empty()) {
      Log::Fatal(
          "Adaptive selectable-unit selection requires at least one unit.");
    }
    canonical_unit_ids_ = canonical_unit_ids;
    std::vector<int> sorted_ids = canonical_unit_ids_;
    std::sort(sorted_ids.begin(), sorted_ids.end());
    CHECK(
        std::adjacent_find(sorted_ids.begin(), sorted_ids.end()) ==
        sorted_ids.end());
    CHECK_GE(sorted_ids.front(), 0);

    const size_t unit_count = canonical_unit_ids_.size();
    CHECK_LE(
        unit_count,
        static_cast<size_t>(std::numeric_limits<int>::max()));
    const int unit_count_as_int = static_cast<int>(unit_count);
    const double unit_count_as_double = static_cast<double>(unit_count);
    CHECK(std::isfinite(unit_count_as_double));
    fixed_budget_enabled_ = feature_fraction_ < 1.0;
    fixed_budget_count_ = unit_count_as_int;
    if (fixed_budget_enabled_) {
      const double requested_count =
          unit_count_as_double * feature_fraction_;
      // This is ColSampler::GetCnt()'s native round-to-nearest/minimum-one
      // rule. The upper branch avoids an out-of-range conversion only at the
      // theoretical int-sized universe limit, where the rounded count is q.
      fixed_budget_count_ =
          requested_count >= unit_count_as_double - 0.5
              ? unit_count_as_int
              : std::max(Common::RoundInt(requested_count), 1);
    }
    CHECK_GE(fixed_budget_count_, 1);
    CHECK_LE(fixed_budget_count_, unit_count_as_int);

    // Howard et al. (2021), Theorem 4, applies the one-sided exponential
    // boundary in both directions. Allocate delta/(2q) to each tail once,
    // simultaneously over all observation counts of the fixed q units.
    const double log_feature_inverse_alpha =
        std::log(unit_count_as_double) - std::log(confidence_delta_);
    log_inverse_alpha_ = log_feature_inverse_alpha + std::log(2.0);
    CHECK(std::isfinite(log_inverse_alpha_));
    CHECK_GT(log_inverse_alpha_, 0.0);
    empirical_bernstein_rho_ = GammaExponentialMixtureBoundary::CalibratedRho(
        log_feature_inverse_alpha, static_cast<double>(calibration_reference_count_));
    boundary_.Reset(log_inverse_alpha_, empirical_bernstein_rho_);
    numerical_warning_emitted_ = false;

    accumulated_gain_.assign(unit_count, 0.0);
    exposure_count_.assign(unit_count, 0.0);
    empirical_variance_.assign(unit_count, 0.0);
    radius_by_unit_.assign(unit_count, std::numeric_limits<double>::infinity());
    upper_bound_scratch_.assign(unit_count, 1.0);
    midpoint_scratch_.assign(fixed_budget_enabled_ ? unit_count : 0, 0.5);
    last_committed_upper_bounds_.assign(unit_count, 1.0);
    ranked_position_scratch_.resize(unit_count);
    std::iota(
        ranked_position_scratch_.begin(),
        ranked_position_scratch_.end(), 0);
    last_selected_.clear();
    last_committed_selected_.clear();
    last_committed_selected_.reserve(unit_count);
    selected_count_history_.clear();
    selected_count_history_.reserve(1002);

    calibration_complete_ = false;
    round_ = 0;
    pending_round_ = 0;
    total_selected_ = 0;
    total_exclusion_events_ = 0;
    total_reentered_ = 0;
    last_committed_selected_count_ = 0;
    last_committed_omitted_count_ = 0;
    last_committed_newly_excluded_count_ = 0;
    last_committed_reentered_count_ = 0;
    pending_omitted_count_ = 0;
    pending_newly_excluded_count_ = 0;
    pending_reentered_count_ = 0;
    last_committed_comparison_threshold_ = 0.0;
    pending_comparison_threshold_ = 0.0;
    last_committed_selection_score_cutoff_ = fixed_budget_enabled_ ? 0.5 : 1.0;
    pending_selection_score_cutoff_ = last_committed_selection_score_cutoff_;
    last_committed_possible_best_count_ = 0;
    pending_possible_best_count_ = 0;
    last_committed_possible_best_set_fully_retained_ = true;
    pending_possible_best_set_fully_retained_ = true;
    last_committed_possible_best_diagnostic_valid_ = true;
    pending_possible_best_diagnostic_valid_ = true;
    last_committed_detailed_receipt_available_ = false;
    pending_detailed_state_suffix_.clear();
    last_committed_detailed_state_suffix_.clear();
    pending_detailed_possible_best_positions_.clear();
    last_committed_detailed_possible_best_positions_.clear();
#ifdef JAEGER_TEST_HOOKS
    possible_best_diagnostic_scan_count_ = 0;
#endif
    awaiting_update_ = false;
    boundary_ready_ = true;

    Log::Info(
        "[CRAFT-LightGBM] confidence-calibrated selectable-unit "
        "allocation enabled: units=%d, "
        "feature_fraction=%.17g, cardinality_mode=%s, "
        "fixed_budget_units=%d, fixed_budget_start_iteration=3, "
        "familywise_delta=%.17g, one_sided_alpha=%.17g, "
        "calibration_iteration=1, automatic_full_iteration=2, "
        "confidence_sequence=time_uniform_empirical_bernstein, "
        "boundary=gamma_exponential_mixture, boundary_version=1, "
        "coverage_horizon=all_completed_rounds, "
        "rho_calibration=normal_reference_scale, "
        "calibration_reference_count=%d, variance_reference=%.17g, "
        "log_one_sided_inverse_alpha=%.17g, empirical_bernstein_rho=%.17g, "
        "predictor=(0.5+A)/(1+N), variance_grid_bins=128, "
        "rho_uses_observed_gains=false, "
        "rho_uses_validation_losses=false, "
        "rho_uses_max_rounds=false, cumulative_statistics=A_N_V, "
        "full_feature_warmup_rounds=%d",
        unit_count_as_int, feature_fraction_,
        fixed_budget_enabled_ ? "exogenous_fixed" : "endogenous_possible_best",
        fixed_budget_count_,
        confidence_delta_, confidence_delta_ / unit_count_as_double / 2.0,
        calibration_reference_count_, variance_reference(),
        log_inverse_alpha_, empirical_bernstein_rho_,
        full_feature_warmup_rounds());
  }

  const std::vector<int>& SelectNextSubset() {
    EnsureBoundaryReady();
    CHECK(!awaiting_update_);
    CHECK(!canonical_unit_ids_.empty());
    pending_round_ = round_ + 1;
    last_selected_.clear();

    const int unit_count = static_cast<int>(canonical_unit_ids_.size());
    if (pending_round_ <= 2) {
      // The calibration iteration is deliberately full and excluded from the
      // confidence statistics. Iteration 2 is also deliberately full so every
      // unit receives the first covered observation before either selector is
      // allowed to omit a unit.
      last_selected_.resize(unit_count);
      std::iota(last_selected_.begin(), last_selected_.end(), 0);
      pending_comparison_threshold_ = 0.0;
      pending_selection_score_cutoff_ = fixed_budget_enabled_ ? 0.5 : 1.0;
      pending_possible_best_count_ = unit_count;
      pending_possible_best_set_fully_retained_ = true;
      pending_possible_best_diagnostic_valid_ = true;
    } else {
      ComputeConfidenceBounds();
      if (fixed_budget_enabled_) {
        SelectFixedBudgetUnits();
      } else {
        SelectMcbUnits();
        if (pending_round_ <= kEndogenousFullFeatureRounds) {
          // Keep the genuine possible-best diagnostics, but observe every
          // feature throughout the endogenous warmup. Covered A/N/V updates
          // still begin at round 2; warmup never postpones a stopping rule.
          last_selected_.resize(unit_count);
          std::iota(last_selected_.begin(), last_selected_.end(), 0);
          pending_selection_score_cutoff_ = *std::min_element(
              upper_bound_scratch_.begin(), upper_bound_scratch_.end());
        }
      }
    }

    CHECK(!last_selected_.empty());
    CHECK_LE(static_cast<int>(last_selected_.size()), unit_count);
    pending_omitted_count_ =
        unit_count - static_cast<int>(last_selected_.size());
    pending_reentered_count_ = 0;
    pending_newly_excluded_count_ = 0;
    if (round_ > 0) {
      // Both selectors return positions in canonical position order. A
      // two-pointer intersection avoids clearing and probing a q-entry bitmap
      // on every iteration while preserving the exact transition counts.
      CHECK(std::is_sorted(last_selected_.begin(), last_selected_.end()));
      CHECK(std::is_sorted(
          last_committed_selected_.begin(),
          last_committed_selected_.end()));
      size_t current_index = 0;
      size_t previous_index = 0;
      int retained_previous = 0;
      while (current_index < last_selected_.size() &&
             previous_index < last_committed_selected_.size()) {
        const int current = last_selected_[current_index];
        const int previous = last_committed_selected_[previous_index];
        if (current == previous) {
          ++retained_previous;
          ++current_index;
          ++previous_index;
        } else if (current < previous) {
          ++current_index;
        } else {
          ++previous_index;
        }
      }
      pending_reentered_count_ =
          static_cast<int>(last_selected_.size()) - retained_previous;
      pending_newly_excluded_count_ =
          last_committed_selected_count_ - retained_previous;
      CHECK_GE(pending_newly_excluded_count_, 0);
    }
    awaiting_update_ = true;
    return last_selected_;
  }

  void Update(const std::vector<double>& selected_opportunities) {
    CHECK(awaiting_update_);
    CHECK_EQ(selected_opportunities.size(), last_selected_.size());
    CHECK_EQ(pending_round_, round_ + 1);

    for (size_t selected_index = 0;
         selected_index < selected_opportunities.size(); ++selected_index) {
      const double reward = selected_opportunities[selected_index];
      if (!std::isfinite(reward) || reward < 0.0 ||
          reward > 1.0 + kInputTolerance) {
        Log::Fatal(
            "Adaptive selectable-unit selection received invalid bounded "
            "root opportunity %.17g for unit %d in iteration %d.",
            reward,
            canonical_unit_ids_[last_selected_[selected_index]],
            pending_round_);
      }
    }

    const bool capture_detailed_receipt =
        ShouldCaptureDetailedReceipt(pending_round_);
    if (capture_detailed_receipt) {
      PreparePendingDetailedReceipt();
    }

    if (pending_round_ == 1) {
      CHECK_EQ(last_selected_.size(), canonical_unit_ids_.size());
      calibration_complete_ = true;
    } else {
      CHECK(calibration_complete_);
      for (size_t selected_index = 0;
           selected_index < last_selected_.size(); ++selected_index) {
        const int position = last_selected_[selected_index];
        const double reward =
            std::min(1.0, selected_opportunities[selected_index]);
        // Predict before observing reward. This is Howard et al.'s squared
        // prediction-error process, not retrospective sample variance.
        const double prediction =
            (0.5 + accumulated_gain_[position]) /
            (1.0 + exposure_count_[position]);
        const double residual = reward - prediction;
        empirical_variance_[position] += residual * residual;
        accumulated_gain_[position] += reward;
        exposure_count_[position] += 1.0;
        radius_by_unit_[position] = std::numeric_limits<double>::quiet_NaN();
        CHECK(std::isfinite(empirical_variance_[position]));
        CHECK_GE(empirical_variance_[position], 0.0);
        CHECK(std::isfinite(accumulated_gain_[position]));
        CHECK(std::isfinite(exposure_count_[position]));
        CHECK_GE(accumulated_gain_[position], 0.0);
        CHECK_LE(
            accumulated_gain_[position],
            exposure_count_[position] + kInputTolerance);
      }
    }
    const bool committing_fixed_adaptive_round =
        fixed_budget_enabled_ && pending_round_ >= 3;

    round_ = pending_round_;
    pending_round_ = 0;
    total_selected_ += static_cast<uint64_t>(last_selected_.size());
    total_exclusion_events_ +=
        static_cast<uint64_t>(pending_newly_excluded_count_);
    total_reentered_ += static_cast<uint64_t>(pending_reentered_count_);
    last_committed_selected_count_ =
        static_cast<int>(last_selected_.size());
    selected_count_history_.push_back(last_committed_selected_count_);
    CHECK_EQ(selected_count_history_.size(), static_cast<size_t>(round_));
    last_committed_omitted_count_ = pending_omitted_count_;
    last_committed_newly_excluded_count_ =
        pending_newly_excluded_count_;
    last_committed_reentered_count_ = pending_reentered_count_;
    last_committed_comparison_threshold_ = pending_comparison_threshold_;
    last_committed_selection_score_cutoff_ =
        pending_selection_score_cutoff_;
    if (committing_fixed_adaptive_round) {
      last_committed_upper_bounds_.swap(upper_bound_scratch_);
      last_committed_possible_best_count_ = pending_possible_best_count_;
      last_committed_possible_best_set_fully_retained_ =
          pending_possible_best_set_fully_retained_;
      last_committed_possible_best_diagnostic_valid_ =
          pending_possible_best_diagnostic_valid_;
    } else {
      last_committed_possible_best_count_ = pending_possible_best_count_;
      last_committed_possible_best_set_fully_retained_ =
          pending_possible_best_set_fully_retained_;
      last_committed_possible_best_diagnostic_valid_ = true;
    }
    if (capture_detailed_receipt) {
      last_committed_detailed_state_suffix_.swap(
          pending_detailed_state_suffix_);
      last_committed_detailed_possible_best_positions_.swap(
          pending_detailed_possible_best_positions_);
      // swap() moves the previous committed diagnostic buffers into the
      // pending buffers. Release their logical contents only on diagnostic
      // rounds; log_period=0 never touches either string or vector here.
      pending_detailed_state_suffix_.clear();
      pending_detailed_possible_best_positions_.clear();
      last_committed_detailed_receipt_available_ = true;
    } else if (last_committed_detailed_receipt_available_) {
      last_committed_detailed_state_suffix_.clear();
      last_committed_detailed_possible_best_positions_.clear();
      last_committed_detailed_receipt_available_ = false;
    }
    last_committed_selected_.assign(
        last_selected_.begin(), last_selected_.end());
    pending_omitted_count_ = 0;
    pending_newly_excluded_count_ = 0;
    pending_reentered_count_ = 0;
    pending_comparison_threshold_ = 0.0;
    pending_selection_score_cutoff_ = fixed_budget_enabled_ ? 0.5 : 1.0;
    pending_possible_best_count_ = 0;
    pending_possible_best_set_fully_retained_ = true;
    pending_possible_best_diagnostic_valid_ = true;
    awaiting_update_ = false;
    LogRoundReceipt();
  }

  void AbortPendingSelection() noexcept {
    if (!awaiting_update_) {
      return;
    }
    last_selected_.clear();
    pending_round_ = 0;
    pending_possible_best_diagnostic_valid_ = true;
    pending_omitted_count_ = 0;
    pending_newly_excluded_count_ = 0;
    pending_reentered_count_ = 0;
    pending_comparison_threshold_ = 0.0;
    pending_selection_score_cutoff_ = fixed_budget_enabled_ ? 0.5 : 1.0;
    pending_possible_best_count_ = 0;
    pending_possible_best_set_fully_retained_ = true;
    if (!pending_detailed_state_suffix_.empty()) {
      pending_detailed_state_suffix_.clear();
      pending_detailed_possible_best_positions_.clear();
    }
    awaiting_update_ = false;
  }

  int round() const { return round_; }
  int selection_round() const {
    return awaiting_update_ ? pending_round_ : round_;
  }
  double confidence_delta() const { return confidence_delta_; }
  double feature_fraction() const { return feature_fraction_; }
  bool fixed_budget_enabled() const { return fixed_budget_enabled_; }
  int full_feature_warmup_rounds() const {
    return fixed_budget_enabled_ ? 2 : kEndogenousFullFeatureRounds;
  }
  int fixed_budget_count() const { return fixed_budget_count_; }
  bool detailed_logging_enabled() const { return log_period_ > 0; }
  double log_inverse_alpha() const { return log_inverse_alpha_; }
  double empirical_bernstein_rho() const {
    EnsureBoundaryReady();
    return empirical_bernstein_rho_;
  }
  int calibration_reference_count() const { return calibration_reference_count_; }
  double variance_reference() const {
    return static_cast<double>(calibration_reference_count_) / 4.0;
  }
  bool calibration_complete() const { return calibration_complete_; }
  double reward_upper_bound() const { return 1.0; }
  bool awaiting_update() const { return awaiting_update_; }
  const std::vector<int>& last_selected() const { return last_selected_; }
  const std::vector<int>& selected_count_history() const {
    return selected_count_history_;
  }
  bool detailed_receipt_available() const {
    return last_committed_detailed_receipt_available_;
  }
  const std::vector<int>& detailed_possible_best_positions() const {
    CHECK(last_committed_detailed_receipt_available_);
    return last_committed_detailed_possible_best_positions_;
  }
  uint64_t total_selected() const { return total_selected_; }
  uint64_t total_exclusion_events() const {
    return total_exclusion_events_;
  }
  uint64_t total_reentered() const { return total_reentered_; }
  int last_committed_selected_count() const {
    return last_committed_selected_count_;
  }
  int last_committed_omitted_count() const {
    return last_committed_omitted_count_;
  }
  int last_committed_newly_excluded_count() const {
    return last_committed_newly_excluded_count_;
  }
  int last_committed_reentered_count() const {
    return last_committed_reentered_count_;
  }
  double comparison_threshold() const {
    return awaiting_update_ ? pending_comparison_threshold_
                            : last_committed_comparison_threshold_;
  }
  // Minimum selected score: interval midpoint in fixed-budget mode, upper
  // confidence bound in the endogenous possible-best mode.
  double selection_score_cutoff() const {
    return awaiting_update_ ? pending_selection_score_cutoff_
                            : last_committed_selection_score_cutoff_;
  }
  int possible_best_count() const {
    if (awaiting_update_) {
      EnsurePendingPossibleBestDiagnostic();
      return pending_possible_best_count_;
    }
    EnsureLastCommittedPossibleBestDiagnostic();
    return last_committed_possible_best_count_;
  }
  bool possible_best_set_fully_retained() const {
    if (awaiting_update_) {
      EnsurePendingPossibleBestDiagnostic();
      return pending_possible_best_set_fully_retained_;
    }
    EnsureLastCommittedPossibleBestDiagnostic();
    return last_committed_possible_best_set_fully_retained_;
  }
#ifdef JAEGER_TEST_HOOKS
  uint64_t possible_best_diagnostic_scan_count() const {
    return possible_best_diagnostic_scan_count_;
  }
  size_t radius_cache_size() const {
    return radius_by_unit_.size();
  }
#endif

  size_t unit_count() const { return canonical_unit_ids_.size(); }

  const std::vector<double>& accumulated_gain() const {
    return accumulated_gain_;
  }
  const std::vector<double>& exposure_count() const {
    return exposure_count_;
  }
  const std::vector<double>& empirical_variance() const {
    return empirical_variance_;
  }

  double mean(int position) const {
    return Mean(static_cast<size_t>(position));
  }
  double radius(int position) const {
    return Radius(static_cast<size_t>(position));
  }
  double lower_bound(int position) const {
    return LowerBound(static_cast<size_t>(position));
  }
  double upper_bound(int position) const {
    return UpperBound(static_cast<size_t>(position));
  }

  std::string SummaryReceiptLine() const {
    EnsureBoundaryReady();
    EnsureLastCommittedPossibleBestDiagnostic();
    const double selected_ratio =
        canonical_unit_ids_.empty()
            ? 0.0
            : static_cast<double>(last_committed_selected_count_) /
                  static_cast<double>(canonical_unit_ids_.size());
    const int covered_update_count = std::max(0, round_ - 1);
    std::ostringstream output;
    output << std::setprecision(17)
           << "[CRAFT-LightGBM] selectable-unit receipt: iteration="
           << round_
           << ", selected_units=" << last_committed_selected_count_
           << ", eligible_units=" << canonical_unit_ids_.size()
           << ", realized_selected_ratio=" << selected_ratio
           << ", omitted_units=" << last_committed_omitted_count_
           << ", newly_excluded_units="
           << last_committed_newly_excluded_count_
           << ", reentered_units=" << last_committed_reentered_count_
           << ", selected_units_total=" << total_selected_
           << ", exclusion_events_total=" << total_exclusion_events_
           << ", reentered_units_total=" << total_reentered_
           << ", comparison_lower_bound="
           << last_committed_comparison_threshold_
           << ", selection_score_cutoff="
           << last_committed_selection_score_cutoff_
           << ", possible_best_units="
           << last_committed_possible_best_count_
           << ", possible_best_set_fully_retained="
           << (last_committed_possible_best_set_fully_retained_
                   ? "true" : "false")
           << ", feature_fraction=" << feature_fraction_
           << ", fixed_budget_units=" << fixed_budget_count_
           << ", fixed_budget_active="
           << (fixed_budget_enabled_ && round_ >= 3 ? "true" : "false")
           << ", selection_phase="
           << (round_ == 1
                   ? "calibration_full"
                   : (round_ == 2 ? "first_covered_full"
                      : (round_ <= full_feature_warmup_rounds()
                             ? "warmup_full" : "adaptive")))
           << ", selector="
           << (fixed_budget_enabled_
                   ? "confidence_midpoint_top_k"
                   : "mcb_possible_best")
           << ", cardinality_mode="
           << (fixed_budget_enabled_
                   ? "exogenous_fixed"
                   : "endogenous_possible_best")
           << ", bounded_confidence_bounds=true"
           << ", unseen_interval=0_to_1"
           << ", fixed_budget_tie_break="
           << (fixed_budget_enabled_
                   ? "midpoint_desc_exposure_asc_canonical_id_asc"
                   : "not_applicable")
           << ", possible_best_boundary_ties_retained="
           << (last_committed_possible_best_set_fully_retained_
                   ? "true" : "false")
           << ", deferred_commit=true"
           << ", reward_transform=lagged_common_unit_gain_scale"
           << ", unit_reward=root_gain"
           << ", calibration_round_excluded=true"
           << ", reward_upper_bound=1"
           << ", cumulative_statistics=A_N_V"
           << ", target=accumulated_average_conditional_transformed_root_gain"
           << ", covered_update_count=" << covered_update_count
           << ", confidence_sequence=time_uniform_empirical_bernstein"
           << ", boundary=gamma_exponential_mixture, boundary_version=1"
           << ", coverage_horizon=all_completed_rounds"
           << ", familywise_delta=" << confidence_delta_
           << ", log_one_sided_inverse_alpha=" << log_inverse_alpha_
           << ", rho_calibration=normal_reference_scale"
           << ", calibration_reference_count=" << calibration_reference_count_
           << ", variance_reference=" << variance_reference()
           << ", empirical_bernstein_rho=" << empirical_bernstein_rho_
           << ", predictor=(0.5+A)/(1+N), variance_grid_bins=128"
           << ", boundary_numerical_failures=" << boundary_.numerical_failure_count()
           << ", rho_uses_observed_gains=false"
           << ", rho_uses_validation_losses=false"
           << ", rho_uses_max_rounds=false"
           << ", full_feature_warmup_rounds=" << full_feature_warmup_rounds();
    return output.str();
  }

  std::string DetailedReceiptLine() const {
    CHECK(last_committed_detailed_receipt_available_);
    return SummaryReceiptLine() + last_committed_detailed_state_suffix_;
  }

 private:
#ifdef JAEGER_TEST_HOOKS
  friend struct AdaptiveFeatureMcbTestPeer;
#endif
  static constexpr double kInputTolerance = 1e-12;
  static constexpr int kEndogenousFullFeatureRounds = 20;

  void EnsureBoundaryReady() const {
    CHECK(boundary_ready_);
    CHECK(std::isfinite(empirical_bernstein_rho_));
    CHECK_GT(empirical_bernstein_rho_, 0.0);
  }

  void ValidateConfiguration() const {
    CHECK_GT(calibration_reference_count_, 0);
    CHECK(std::isfinite(confidence_delta_));
    CHECK_GT(confidence_delta_, 0.0);
    CHECK_LT(confidence_delta_, 1.0);
    CHECK(std::isfinite(feature_fraction_));
    CHECK_GT(feature_fraction_, 0.0);
    CHECK_LE(feature_fraction_, 1.0);
    CHECK_GE(log_period_, 0);
  }

  double Mean(size_t position) const {
    CHECK_LT(position, accumulated_gain_.size());
    if (exposure_count_[position] <= 0.0) {
      return 0.0;
    }
    const double result =
        accumulated_gain_[position] / exposure_count_[position];
    CHECK(std::isfinite(result));
    CHECK_GE(result, -kInputTolerance);
    CHECK_LE(result, 1.0 + kInputTolerance);
    return std::max(0.0, std::min(1.0, result));
  }

  double Radius(size_t position) const {
    CHECK_LT(position, exposure_count_.size());
    EnsureBoundaryReady();
    if (exposure_count_[position] <= 0.0) {
      return std::numeric_limits<double>::infinity();
    }
    double& cached_radius = radius_by_unit_[position];
    if (std::isnan(cached_radius)) {
      cached_radius = boundary_.Boundary(empirical_variance_[position]) /
          exposure_count_[position];
      if (!std::isfinite(cached_radius) && !numerical_warning_emitted_) {
        Log::Warning("CRAFT empirical-Bernstein boundary could not be evaluated; "
                     "using the conservative full [0,1] interval.");
        numerical_warning_emitted_ = true;
      }
    }
    return cached_radius;
  }

  double LowerBound(size_t position) const {
    CHECK_LT(position, exposure_count_.size());
    if (exposure_count_[position] <= 0.0) {
      return 0.0;
    }
    return std::max(0.0, std::min(1.0, Mean(position) - Radius(position)));
  }

  double UpperBound(size_t position) const {
    CHECK_LT(position, exposure_count_.size());
    if (exposure_count_[position] <= 0.0) {
      return 1.0;
    }
    return std::max(0.0, std::min(1.0, Mean(position) + Radius(position)));
  }

  void ComputeConfidenceBounds() {
    CHECK_EQ(
        upper_bound_scratch_.size(), canonical_unit_ids_.size());
    pending_comparison_threshold_ = 0.0;
    for (size_t position = 0;
         position < canonical_unit_ids_.size(); ++position) {
      double lower_bound = 0.0;
      double upper_bound = 1.0;
      if (exposure_count_[position] > 0.0) {
        // Mean and radius are shared by the two confidence endpoints. The
        // arithmetic and bounding order are identical to LowerBound() and
        // UpperBound(); only the duplicate transcendental work is removed.
        const double mean = Mean(position);
        const double radius = Radius(position);
        lower_bound =
            std::max(0.0, std::min(1.0, mean - radius));
        upper_bound =
            std::max(0.0, std::min(1.0, mean + radius));
      }
      upper_bound_scratch_[position] = upper_bound;
      if (fixed_budget_enabled_) {
        // The bounded interval midpoint need not equal A/N when an endpoint
        // reaches 0 or 1. Cache it once, outside the ranking comparator.
        midpoint_scratch_[position] = 0.5 * (lower_bound + upper_bound);
      }
      pending_comparison_threshold_ = std::max(
          pending_comparison_threshold_, lower_bound);
    }
    CHECK(std::isfinite(pending_comparison_threshold_));
    CHECK_GE(pending_comparison_threshold_, 0.0);
    CHECK_LE(pending_comparison_threshold_, 1.0);
  }

  void SelectMcbUnits() {
    last_selected_.reserve(canonical_unit_ids_.size());
    pending_possible_best_count_ = 0;
    pending_selection_score_cutoff_ = 1.0;
    for (size_t position = 0;
         position < canonical_unit_ids_.size(); ++position) {
      // Inclusive comparison is essential: every tie at the MCB boundary is
      // a possible maximizer and must be retained.
      if (upper_bound_scratch_[position] >=
          pending_comparison_threshold_) {
        last_selected_.push_back(static_cast<int>(position));
        ++pending_possible_best_count_;
        pending_selection_score_cutoff_ = std::min(
            pending_selection_score_cutoff_,
            upper_bound_scratch_[position]);
      }
    }
    CHECK_GE(pending_possible_best_count_, 1);
    CHECK_EQ(
        static_cast<int>(last_selected_.size()),
        pending_possible_best_count_);
    pending_possible_best_set_fully_retained_ = true;
    pending_possible_best_diagnostic_valid_ = true;
  }

  void SelectFixedBudgetUnits() {
    CHECK(fixed_budget_enabled_);
    CHECK_GE(fixed_budget_count_, 1);
    CHECK_LE(
        fixed_budget_count_,
        static_cast<int>(canonical_unit_ids_.size()));
    CHECK_EQ(
        ranked_position_scratch_.size(), canonical_unit_ids_.size());
    std::iota(
        ranked_position_scratch_.begin(),
        ranked_position_scratch_.end(), 0);
    const auto better = [this](int left, int right) {
      const double left_midpoint = midpoint_scratch_[left];
      const double right_midpoint = midpoint_scratch_[right];
      if (left_midpoint != right_midpoint) {
        return left_midpoint > right_midpoint;
      }
      const double left_exposure = exposure_count_[left];
      const double right_exposure = exposure_count_[right];
      if (left_exposure != right_exposure) {
        return left_exposure < right_exposure;
      }
      return canonical_unit_ids_[left] < canonical_unit_ids_[right];
    };
    if (fixed_budget_count_ <
        static_cast<int>(ranked_position_scratch_.size())) {
      std::nth_element(
          ranked_position_scratch_.begin(),
          ranked_position_scratch_.begin() + fixed_budget_count_,
          ranked_position_scratch_.end(), better);
    }
    last_selected_.assign(
        ranked_position_scratch_.begin(),
        ranked_position_scratch_.begin() + fixed_budget_count_);
    pending_selection_score_cutoff_ = 1.0;
    pending_possible_best_count_ = 0;
    pending_possible_best_diagnostic_valid_ = false;
    for (int position : last_selected_) {
      pending_selection_score_cutoff_ = std::min(
          pending_selection_score_cutoff_,
          midpoint_scratch_[position]);
    }
    std::sort(last_selected_.begin(), last_selected_.end());
    pending_possible_best_set_fully_retained_ = false;
  }

  void EnsurePendingPossibleBestDiagnostic() const {
    if (pending_possible_best_diagnostic_valid_) {
      return;
    }
    CHECK(fixed_budget_enabled_);
    CHECK(awaiting_update_);
    ComputePossibleBestDiagnostic(
        upper_bound_scratch_, pending_comparison_threshold_, last_selected_,
        &pending_possible_best_count_,
        &pending_possible_best_set_fully_retained_);
    pending_possible_best_diagnostic_valid_ = true;
  }

  void EnsureLastCommittedPossibleBestDiagnostic() const {
    if (last_committed_possible_best_diagnostic_valid_) {
      return;
    }
    CHECK(fixed_budget_enabled_);
    CHECK_GE(round_, 3);
    ComputePossibleBestDiagnostic(
        last_committed_upper_bounds_,
        last_committed_comparison_threshold_, last_committed_selected_,
        &last_committed_possible_best_count_,
        &last_committed_possible_best_set_fully_retained_);
    last_committed_possible_best_diagnostic_valid_ = true;
  }

  void ComputePossibleBestDiagnostic(
      const std::vector<double>& upper_bounds, double threshold,
      const std::vector<int>& selected, int* count, bool* fully_retained) const {
    CHECK_EQ(upper_bounds.size(), canonical_unit_ids_.size());
    CHECK_NOTNULL(count);
    CHECK_NOTNULL(fully_retained);
#ifdef JAEGER_TEST_HOOKS
    ++possible_best_diagnostic_scan_count_;
#endif
    *count = 0;
    *fully_retained = true;
    size_t selected_index = 0;
    for (size_t position = 0; position < upper_bounds.size(); ++position) {
      const bool is_selected = selected_index < selected.size() &&
          selected[selected_index] == static_cast<int>(position);
      if (is_selected) {
        ++selected_index;
      }
      if (upper_bounds[position] >= threshold) {
        ++*count;
        // Midpoint ranking need not retain all possible-best features even
        // when their count is at most K. Check actual set membership lazily.
        *fully_retained = *fully_retained && is_selected;
      }
    }
    CHECK_EQ(selected_index, selected.size());
    CHECK_GE(*count, 1);
    CHECK_LE(*count, static_cast<int>(canonical_unit_ids_.size()));
  }

  bool ShouldCaptureDetailedReceipt(int round) const {
    return log_period_ > 0 && round % log_period_ == 0;
  }

  void PreparePendingDetailedReceipt() {
    CHECK(awaiting_update_);
    CHECK(ShouldCaptureDetailedReceipt(pending_round_));
    EnsurePendingPossibleBestDiagnostic();

    pending_detailed_possible_best_positions_.clear();
    pending_detailed_possible_best_positions_.reserve(
        canonical_unit_ids_.size());
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;
    lower_bounds.reserve(canonical_unit_ids_.size());
    upper_bounds.reserve(canonical_unit_ids_.size());
    for (size_t position = 0;
         position < canonical_unit_ids_.size(); ++position) {
      const double lower = LowerBound(position);
      const double upper = UpperBound(position);
      lower_bounds.push_back(lower);
      upper_bounds.push_back(upper);
      if (upper >= pending_comparison_threshold_) {
        pending_detailed_possible_best_positions_.push_back(
            static_cast<int>(position));
      }
    }
    CHECK_EQ(
        static_cast<int>(
            pending_detailed_possible_best_positions_.size()),
        pending_possible_best_count_);

    std::ostringstream output;
    output << std::setprecision(17)
           << ", diagnostic_state=pre_iteration_selection"
           << ", history_through_iteration=" << pending_round_ - 1;
    AppendVector(
        &output, "canonical_unit_ids", canonical_unit_ids_);
    std::vector<int> selected_canonical_ids;
    selected_canonical_ids.reserve(last_selected_.size());
    for (int position : last_selected_) {
      selected_canonical_ids.push_back(canonical_unit_ids_[position]);
    }
    AppendVector(
        &output, "selected_canonical_unit_ids",
        selected_canonical_ids);
    std::vector<int> possible_best_canonical_ids;
    possible_best_canonical_ids.reserve(
        pending_detailed_possible_best_positions_.size());
    for (int position : pending_detailed_possible_best_positions_) {
      possible_best_canonical_ids.push_back(
          canonical_unit_ids_[position]);
    }
    AppendVector(
        &output, "possible_best_canonical_unit_ids",
        possible_best_canonical_ids);
    AppendVector(
        &output, "accumulated_gain_by_unit", accumulated_gain_);
    AppendVector(
        &output, "exposure_count_by_unit", exposure_count_);
    AppendVector(
        &output, "empirical_variance_by_unit", empirical_variance_);
    AppendVector(
        &output, "lower_confidence_bound_by_unit", lower_bounds);
    AppendVector(
        &output, "upper_confidence_bound_by_unit", upper_bounds);
    pending_detailed_state_suffix_ = output.str();
  }

  template <typename T>
  static void AppendVector(
      std::ostringstream* output, const char* name,
      const std::vector<T>& values) {
    CHECK_NOTNULL(output);
    *output << ", " << name << "=[";
    for (size_t index = 0; index < values.size(); ++index) {
      if (index > 0) {
        *output << ',';
      }
      *output << values[index];
    }
    *output << ']';
  }

  void LogRoundReceipt() const {
    if (!detailed_logging_enabled()) {
      return;
    }
    const bool is_boundary =
        round_ == 1 || round_ == 2;
    const bool is_periodic =
        log_period_ > 0 && round_ % log_period_ == 0;
    // ColSampler emits one enriched line for periodic diagnostic rounds.
    // Retain the lightweight initialization receipts only when they do not
    // coincide with that diagnostic gate.
    if (is_boundary && !is_periodic) {
      Log::Info("%s", SummaryReceiptLine().c_str());
    }
  }

  double confidence_delta_ = 0.05;
  int calibration_reference_count_ = 300;
  double feature_fraction_ = 1.0;
  bool fixed_budget_enabled_ = false;
  int fixed_budget_count_ = 0;
  int log_period_ = 0;
  double log_inverse_alpha_ = 0.0;
  double empirical_bernstein_rho_ = std::numeric_limits<double>::quiet_NaN();
  GammaExponentialMixtureBoundary boundary_;
  bool boundary_ready_ = false;
  mutable bool numerical_warning_emitted_ = false;

  std::vector<int> canonical_unit_ids_;
  std::vector<double> accumulated_gain_;
  std::vector<double> exposure_count_;
  std::vector<double> empirical_variance_;
  std::vector<double> upper_bound_scratch_;
  std::vector<double> midpoint_scratch_;
  std::vector<double> last_committed_upper_bounds_;
  std::vector<int> ranked_position_scratch_;
  mutable std::vector<double> radius_by_unit_;
  std::vector<int> last_selected_;
  std::vector<int> last_committed_selected_;
  std::vector<int> selected_count_history_;

  bool calibration_complete_ = false;
  int round_ = 0;
  int pending_round_ = 0;
  uint64_t total_selected_ = 0;
  uint64_t total_exclusion_events_ = 0;
  uint64_t total_reentered_ = 0;
  int last_committed_selected_count_ = 0;
  int last_committed_omitted_count_ = 0;
  int last_committed_newly_excluded_count_ = 0;
  int last_committed_reentered_count_ = 0;
  int pending_omitted_count_ = 0;
  int pending_newly_excluded_count_ = 0;
  int pending_reentered_count_ = 0;
  double last_committed_comparison_threshold_ = 0.0;
  double pending_comparison_threshold_ = 0.0;
  double last_committed_selection_score_cutoff_ = 1.0;
  double pending_selection_score_cutoff_ = 1.0;
  mutable int last_committed_possible_best_count_ = 0;
  mutable int pending_possible_best_count_ = 0;
  mutable bool last_committed_possible_best_set_fully_retained_ = true;
  mutable bool pending_possible_best_set_fully_retained_ = true;
  mutable bool last_committed_possible_best_diagnostic_valid_ = true;
  mutable bool pending_possible_best_diagnostic_valid_ = true;
  bool last_committed_detailed_receipt_available_ = false;
  std::string pending_detailed_state_suffix_;
  std::string last_committed_detailed_state_suffix_;
  std::vector<int> pending_detailed_possible_best_positions_;
  std::vector<int> last_committed_detailed_possible_best_positions_;
#ifdef JAEGER_TEST_HOOKS
  mutable uint64_t possible_best_diagnostic_scan_count_ = 0;
#endif
  bool awaiting_update_ = false;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_TREELEARNER_ADAPTIVE_FEATURE_MCB_HPP_
