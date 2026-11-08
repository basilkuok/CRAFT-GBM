/*!
 * Copyright (c) 2020-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2020-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_COL_SAMPLER_HPP_
#define LIGHTGBM_SRC_TREELEARNER_COL_SAMPLER_HPP_

#include <LightGBM/dataset.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/utils/random.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "adaptive_feature_root_gain.hpp"
#include "adaptive_feature_mcb.hpp"

namespace LightGBM {
class ColSampler {
 public:
  explicit ColSampler(const Config* config)
      : fraction_bytree_(config->feature_fraction),
        fraction_bynode_(config->feature_fraction_bynode),
        seed_(config->feature_fraction_seed),
        random_(config->feature_fraction_seed),
        adaptive_enabled_(config->adaptive_feature_selection) {
    if (adaptive_enabled_) {
      adaptive_sampler_.reset(new AdaptiveFeatureMcb(config));
    }
    for (auto constraint : config->interaction_constraints_vector) {
      std::unordered_set<int> constraint_set(constraint.begin(), constraint.end());
      interaction_constraints_.push_back(constraint_set);
    }
  }

  static int GetCnt(size_t total_cnt, double fraction) {
    const int min = std::min(1, static_cast<int>(total_cnt));
    int used_feature_cnt = static_cast<int>(Common::RoundInt(total_cnt * fraction));
    return std::max(used_feature_cnt, min);
  }

  void SetTrainingData(const Dataset* train_data) {
    train_data_ = train_data;
    is_feature_used_.resize(train_data_->num_features(), 1);
    valid_feature_indices_ = train_data->ValidFeatureIndices();
    if (adaptive_enabled_) {
      InitializeAdaptiveSampler();
      return;
    }
    if (fraction_bytree_ >= 1.0f) {
      need_reset_bytree_ = false;
      used_cnt_bytree_ = static_cast<int>(valid_feature_indices_.size());
    } else {
      need_reset_bytree_ = true;
      used_cnt_bytree_ =
          GetCnt(valid_feature_indices_.size(), fraction_bytree_);
    }
    ResetByTree();
  }

  void SetConfig(const Config* config) {
    fraction_bytree_ = config->feature_fraction;
    fraction_bynode_ = config->feature_fraction_bynode;
    adaptive_enabled_ = config->adaptive_feature_selection;
    is_feature_used_.resize(train_data_->num_features(), 1);
    // seed is changed
    if (seed_ != config->feature_fraction_seed) {
      seed_ = config->feature_fraction_seed;
      random_ = Random(seed_);
    }
    if (adaptive_enabled_) {
      adaptive_sampler_.reset(new AdaptiveFeatureMcb(config));
      InitializeAdaptiveSampler();
      return;
    }
    adaptive_sampler_.reset();
    adaptive_unit_inner_features_.clear();
    adaptive_valid_position_by_inner_feature_.clear();
    adaptive_canonical_unit_ids_.clear();
    adaptive_selected_inner_features_.clear();
    adaptive_selected_unit_positions_.clear();
    adaptive_previous_selected_unit_positions_.clear();
    if (fraction_bytree_ >= 1.0f) {
      need_reset_bytree_ = false;
      used_cnt_bytree_ = static_cast<int>(valid_feature_indices_.size());
    } else {
      need_reset_bytree_ = true;
      used_cnt_bytree_ =
          GetCnt(valid_feature_indices_.size(), fraction_bytree_);
    }
    ResetByTree();
  }

  void ResetByTree() {
    if (adaptive_enabled_) {
      Log::Fatal(
          "Adaptive selectable-unit selection must begin at the "
          "boosting-iteration "
          "hook before current row sampling, not inside ResetByTree().");
      return;
    }
    if (need_reset_bytree_) {
      std::memset(is_feature_used_.data(), 0,
                  sizeof(int8_t) * is_feature_used_.size());
      used_feature_indices_ = random_.Sample(
          static_cast<int>(valid_feature_indices_.size()), used_cnt_bytree_);
      int omp_loop_size = static_cast<int>(used_feature_indices_.size());

#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 512) if (omp_loop_size >= 1024)
      for (int i = 0; i < omp_loop_size; ++i) {
        int used_feature = valid_feature_indices_[used_feature_indices_[i]];
        int inner_feature_index = train_data_->InnerFeatureIndex(used_feature);
        is_feature_used_[inner_feature_index] = 1;
      }
    }
  }

  std::vector<int8_t> GetByNode(const Tree* tree, int leaf) {
    // get interaction constraints for current branch
    std::unordered_set<int> allowed_features;
    if (!interaction_constraints_.empty()) {
      std::vector<int> branch_features = tree->branch_features(leaf);
      allowed_features.insert(branch_features.begin(), branch_features.end());
      for (auto constraint : interaction_constraints_) {
        int num_feat_found = 0;
        if (branch_features.size() == 0) {
          allowed_features.insert(constraint.begin(), constraint.end());
        }
        for (int feat : branch_features) {
          if (constraint.count(feat) == 0) {
            break;
          }
          ++num_feat_found;
          if (num_feat_found == static_cast<int>(branch_features.size())) {
            allowed_features.insert(constraint.begin(), constraint.end());
            break;
          }
        }
      }
    }

    std::vector<int8_t> ret(train_data_->num_features(), 0);
    if (fraction_bynode_ >= 1.0f) {
      if (interaction_constraints_.empty()) {
        return std::vector<int8_t>(train_data_->num_features(), 1);
      } else {
        for (int feat : allowed_features) {
          int inner_feat = train_data_->InnerFeatureIndex(feat);
          if (inner_feat >= 0) {
            ret[inner_feat] = 1;
          }
        }
        return ret;
      }
    }
    if (need_reset_bytree_) {
      auto used_feature_cnt = GetCnt(used_feature_indices_.size(), fraction_bynode_);
      std::vector<int>* allowed_used_feature_indices;
      std::vector<int> filtered_feature_indices;
      if (interaction_constraints_.empty()) {
        allowed_used_feature_indices = &used_feature_indices_;
      } else {
        for (int feat_ind : used_feature_indices_) {
          if (allowed_features.count(valid_feature_indices_[feat_ind]) == 1) {
            filtered_feature_indices.push_back(feat_ind);
          }
        }
        used_feature_cnt = std::min(used_feature_cnt, static_cast<int>(filtered_feature_indices.size()));
        allowed_used_feature_indices = &filtered_feature_indices;
      }
      auto sampled_indices = random_.Sample(
          static_cast<int>((*allowed_used_feature_indices).size()), used_feature_cnt);
      int omp_loop_size = static_cast<int>(sampled_indices.size());
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 512) if (omp_loop_size >= 1024)
      for (int i = 0; i < omp_loop_size; ++i) {
        int used_feature =
            valid_feature_indices_[(*allowed_used_feature_indices)[sampled_indices[i]]];
        int inner_feature_index = train_data_->InnerFeatureIndex(used_feature);
        ret[inner_feature_index] = 1;
      }
    } else {
      auto used_feature_cnt =
          GetCnt(valid_feature_indices_.size(), fraction_bynode_);
      std::vector<int>* allowed_valid_feature_indices;
      std::vector<int> filtered_feature_indices;
      if (interaction_constraints_.empty()) {
        allowed_valid_feature_indices = &valid_feature_indices_;
      } else {
        for (int feat : valid_feature_indices_) {
          if (allowed_features.count(feat) == 1) {
            filtered_feature_indices.push_back(feat);
          }
        }
        allowed_valid_feature_indices = &filtered_feature_indices;
        used_feature_cnt = std::min(used_feature_cnt, static_cast<int>(filtered_feature_indices.size()));
      }
      auto sampled_indices = random_.Sample(
          static_cast<int>((*allowed_valid_feature_indices).size()), used_feature_cnt);
      int omp_loop_size = static_cast<int>(sampled_indices.size());
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 512) if (omp_loop_size >= 1024)
      for (int i = 0; i < omp_loop_size; ++i) {
        int used_feature = (*allowed_valid_feature_indices)[sampled_indices[i]];
        int inner_feature_index = train_data_->InnerFeatureIndex(used_feature);
        ret[inner_feature_index] = 1;
      }
    }
    return ret;
  }

  const std::vector<int8_t>& is_feature_used_bytree() const {
    return is_feature_used_;
  }

  bool adaptive_feature_selection_enabled() const {
    return adaptive_enabled_;
  }

  // JAEGER disallows per-node column sampling and interaction constraints, so
  // its node mask is exactly the already-installed per-tree mask. Native
  // callers with the same configuration benefit from the same allocation-free
  // path.
  bool has_trivial_node_mask() const {
    return fraction_bynode_ >= 1.0f && interaction_constraints_.empty();
  }

  const std::vector<int>& adaptive_selected_inner_features() const {
    CHECK(adaptive_enabled_);
    return adaptive_selected_inner_features_;
  }

  const std::vector<int>& adaptive_feature_selected_count_history() const {
    static const std::vector<int> empty;
    if (!adaptive_enabled_) {
      return empty;
    }
    CHECK_NOTNULL(adaptive_sampler_.get());
    return adaptive_sampler_->selected_count_history();
  }

#ifdef JAEGER_TEST_HOOKS
  const AdaptiveFeatureMcb& AdaptiveControllerForTesting() const {
    CHECK(adaptive_enabled_);
    CHECK_NOTNULL(adaptive_sampler_.get());
    return *adaptive_sampler_;
  }

  const std::vector<int>& AdaptiveInnerFeaturesForTesting() const {
    return adaptive_unit_inner_features_;
  }

  const std::vector<int>& AdaptiveCanonicalUnitIdsForTesting() const {
    return adaptive_canonical_unit_ids_;
  }

  const std::vector<int>& AdaptiveSelectedUnitPositionsForTesting() const {
    return adaptive_selected_unit_positions_;
  }

  const std::vector<int>& AdaptiveUsedFeaturePositionsForTesting() const {
    return used_feature_indices_;
  }

  const std::vector<int>& AdaptiveSelectedInnerFeaturesForTesting() const {
    return adaptive_selected_inner_features_;
  }
#endif

  std::string AdaptiveFinalReceiptLine() const {
    if (!adaptive_enabled_) {
      return "";
    }
    CHECK_NOTNULL(adaptive_sampler_.get());
    std::ostringstream output;
    output << adaptive_sampler_->SummaryReceiptLine()
           << ", calibration_model_count="
           << adaptive_calibration_gain_scales_.size()
           << ", calibration_gain_scales=[";
    output << std::setprecision(17);
    for (size_t index = 0;
         index < adaptive_calibration_gain_scales_.size(); ++index) {
      if (index > 0) {
        output << ',';
      }
      output << adaptive_calibration_gain_scales_[index];
    }
    output << "]"
           << ", calibration_positive_unit_gain_counts=[";
    for (size_t index = 0;
         index < adaptive_calibration_positive_unit_gain_counts_.size();
         ++index) {
      if (index > 0) {
        output << ',';
      }
      output << adaptive_calibration_positive_unit_gain_counts_[index];
    }
    output << "]"
           << ", current_gain_scales=[";
    for (size_t index = 0; index < adaptive_gain_scales_.size(); ++index) {
      if (index > 0) {
        output << ',';
      }
      output << adaptive_gain_scales_[index];
    }
    output << "]"
           << ", current_missing_gain_scale_count="
           << static_cast<int>(std::count(
                  adaptive_gain_scales_.begin(),
                  adaptive_gain_scales_.end(), 0.0));
    return output.str();
  }

  std::string AdaptiveDetailedReceiptLine() const {
    CHECK(adaptive_enabled_);
    CHECK_NOTNULL(adaptive_sampler_.get());
    CHECK(adaptive_sampler_->detailed_receipt_available());
    const auto& selected_positions = adaptive_sampler_->last_selected();
    const auto& possible_best_positions =
        adaptive_sampler_->detailed_possible_best_positions();
    CHECK_EQ(
        selected_positions, adaptive_selected_unit_positions_);

    const auto original_features =
        [this](const std::vector<int>& unit_positions) {
          std::vector<int> result;
          for (int unit_position : unit_positions) {
            CHECK_GE(unit_position, 0);
            CHECK_LT(
                static_cast<size_t>(unit_position),
                adaptive_unit_inner_features_.size());
            result.push_back(train_data_->RealFeatureIndex(
                adaptive_unit_inner_features_[unit_position]));
          }
          std::sort(result.begin(), result.end());
          CHECK(
              std::adjacent_find(result.begin(), result.end()) ==
              result.end());
          return result;
        };
    const auto selected_original_features =
        original_features(selected_positions);
    const auto possible_best_original_features =
        original_features(possible_best_positions);
    std::vector<int> direct_efb_groups;
    direct_efb_groups.reserve(selected_original_features.size());
    for (int original_feature : selected_original_features) {
      const int inner_feature =
          train_data_->InnerFeatureIndex(original_feature);
      CHECK_GE(inner_feature, 0);
      direct_efb_groups.push_back(
          train_data_->Feature2Group(inner_feature));
    }
    std::sort(direct_efb_groups.begin(), direct_efb_groups.end());
    direct_efb_groups.erase(
        std::unique(
            direct_efb_groups.begin(), direct_efb_groups.end()),
        direct_efb_groups.end());

    std::ostringstream output;
    output << adaptive_sampler_->DetailedReceiptLine()
           << ", original_feature_mask_available=true"
           << ", original_feature_ids=[";
    for (size_t unit_position = 0;
         unit_position < adaptive_unit_inner_features_.size();
         ++unit_position) {
      if (unit_position > 0) {
        output << ',';
      }
      output << train_data_->RealFeatureIndex(
          adaptive_unit_inner_features_[unit_position]);
    }
    output << "]"
           << ", selected_original_feature_ids=[";
    for (size_t index = 0;
         index < selected_original_features.size(); ++index) {
      if (index > 0) {
        output << ',';
      }
      output << selected_original_features[index];
    }
    output << "]"
           << ", possible_best_original_feature_ids=[";
    for (size_t index = 0;
         index < possible_best_original_features.size(); ++index) {
      if (index > 0) {
        output << ',';
      }
      output << possible_best_original_features[index];
    }
    output << ']'
           << ", direct_efb_group_ids=[";
    for (size_t index = 0; index < direct_efb_groups.size(); ++index) {
      if (index > 0) {
        output << ',';
      }
      output << direct_efb_groups[index];
    }
    output << ']'
           << ", direct_efb_group_count="
           << direct_efb_groups.size()
           << ", direct_efb_group_original_feature_ids=[";
    for (size_t group_position = 0;
         group_position < direct_efb_groups.size();
         ++group_position) {
      if (group_position > 0) {
        output << ',';
      }
      output << '[';
      bool first_member = true;
      for (int inner_feature = 0;
           inner_feature < train_data_->num_features();
           ++inner_feature) {
        if (train_data_->Feature2Group(inner_feature) !=
            direct_efb_groups[group_position]) {
          continue;
        }
        if (!first_member) {
          output << ',';
        }
        output << train_data_->RealFeatureIndex(inner_feature);
        first_member = false;
      }
      CHECK(!first_member);
      output << ']';
    }
    output << ']';
    return output.str();
  }

  std::string AdaptiveCalibrationReceiptLine() const {
    if (!adaptive_enabled_) {
      return "";
    }
    std::ostringstream output;
    output << std::setprecision(17)
           << "[JAEGER-LightGBM v7] calibration gain-scale receipt: iteration=1"
           << ", calibration_model_count="
           << adaptive_calibration_gain_scales_.size()
           << ", calibration_gain_scales=[";
    for (size_t index = 0;
         index < adaptive_calibration_gain_scales_.size(); ++index) {
      if (index > 0) {
        output << ',';
      }
      output << adaptive_calibration_gain_scales_[index];
    }
    output << "]"
           << ", calibration_positive_unit_gain_counts=[";
    for (size_t index = 0;
         index < adaptive_calibration_positive_unit_gain_counts_.size();
         ++index) {
      if (index > 0) {
        output << ',';
      }
      output << adaptive_calibration_positive_unit_gain_counts_[index];
    }
    output << "]"
           << ", calibration_missing_gain_scale_count="
           << static_cast<int>(std::count(
                  adaptive_calibration_gain_scales_.begin(),
                  adaptive_calibration_gain_scales_.end(), 0.0));
    return output.str();
  }

  void BeginAdaptiveIteration(int num_models) {
    if (!adaptive_enabled_) {
      return;
    }
    CHECK_NOTNULL(adaptive_sampler_.get());
    CHECK(!adaptive_iteration_active_);
    CHECK(!adaptive_sampler_->awaiting_update());
    CHECK(adaptive_previous_selected_unit_positions_.empty());
    CHECK_GT(num_models, 0);
    adaptive_iteration_active_ = true;
    adaptive_mask_changed_ = false;

    // Selection is a read-only proposal over committed controller history.
    // It occurs before the current GOSS draw and is committed only after
    // native LightGBM accepts the complete boosting iteration.
    const auto& selected_positions = adaptive_sampler_->SelectNextSubset();
    if (selected_positions != adaptive_selected_unit_positions_) {
      adaptive_mask_changed_ = true;
      adaptive_previous_selected_unit_positions_.swap(
          adaptive_selected_unit_positions_);
      adaptive_selected_unit_positions_.assign(
          selected_positions.begin(), selected_positions.end());
      ApplyAdaptiveUnitSelection();
    }
    CHECK(!adaptive_selected_unit_positions_.empty());
    CHECK_LE(
        adaptive_selected_unit_positions_.size(),
        adaptive_unit_inner_features_.size());

    adaptive_iteration_opportunity_sum_.assign(
        adaptive_selected_unit_positions_.size(), 0.0);
    adaptive_selected_unit_gains_scratch_.clear();
    adaptive_selected_unit_gains_scratch_.reserve(
        adaptive_selected_unit_positions_.size());
    if (!adaptive_gain_scales_.empty()) {
      CHECK_EQ(
          adaptive_gain_scales_.size(), static_cast<size_t>(num_models));
    }
    adaptive_iteration_next_gain_scales_.clear();
    adaptive_iteration_next_gain_scales_.reserve(num_models);
    adaptive_iteration_positive_unit_gain_counts_.clear();
    adaptive_iteration_positive_unit_gain_counts_.reserve(num_models);
    adaptive_models_expected_ = num_models;
    adaptive_models_seen_ = 0;
  }

  void PrepareAdaptiveTree() const {
    if (!adaptive_enabled_) {
      return;
    }
    CHECK(adaptive_iteration_active_);
    CHECK_LT(adaptive_models_seen_, adaptive_models_expected_);
    CHECK(!adaptive_selected_unit_positions_.empty());
  }

  void ResetAdaptiveRootGains(std::vector<double>* root_gains) const {
    if (!adaptive_enabled_) {
      return;
    }
    CHECK_NOTNULL(root_gains);
    CHECK_EQ(root_gains->size(), is_feature_used_.size());
    // Accumulation reads only members of the current selected units. Clear
    // exactly those writable slots; an omitted slot is cleared before it is
    // ever read again if its unit re-enters a later iteration.
    for (int inner_feature : adaptive_selected_inner_features_) {
      CHECK_GE(inner_feature, 0);
      CHECK_LT(inner_feature, static_cast<int>(root_gains->size()));
      (*root_gains)[inner_feature] = 0.0;
    }
  }

  bool AccumulateAdaptiveModel(
      const std::vector<double>& inner_feature_root_gains) {
    if (!adaptive_enabled_) {
      return false;
    }
    CHECK_NOTNULL(adaptive_sampler_.get());
    CHECK(adaptive_iteration_active_);
    CHECK_LT(adaptive_models_seen_, adaptive_models_expected_);
    CHECK_EQ(inner_feature_root_gains.size(), is_feature_used_.size());
    CHECK_EQ(
        adaptive_iteration_opportunity_sum_.size(),
        adaptive_selected_unit_positions_.size());
    const double current_scale =
        adaptive_gain_scales_.empty()
            ? 0.0
            : adaptive_gain_scales_[adaptive_models_seen_];

    adaptive_selected_unit_gains_scratch_.clear();
    for (size_t selected_index = 0;
         selected_index < adaptive_selected_unit_positions_.size();
         ++selected_index) {
      const int unit_position =
          adaptive_selected_unit_positions_[selected_index];
      CHECK_GE(unit_position, 0);
      CHECK_LT(
          static_cast<size_t>(unit_position),
          adaptive_unit_inner_features_.size());
      const int inner_feature = adaptive_unit_inner_features_[unit_position];
      const double feature_gain = inner_feature_root_gains[inner_feature];
      adaptive_selected_unit_gains_scratch_.push_back(feature_gain);
      const double opportunity =
          AdaptiveFeatureRootGain::NormalizeOpportunity(
              feature_gain, current_scale,
              adaptive_canonical_unit_ids_[unit_position]);
      CHECK(std::isfinite(opportunity));
      CHECK_GE(opportunity, 0.0);
      CHECK_LE(opportunity, 1.0);
      adaptive_iteration_opportunity_sum_[selected_index] += opportunity;
    }

    size_t positive_count = 0;
    const double observed_scale =
        AdaptiveFeatureRootGain::PositiveMedianInPlace(
            &adaptive_selected_unit_gains_scratch_, &positive_count);
    adaptive_iteration_next_gain_scales_.push_back(
        observed_scale > 0.0 ? observed_scale : current_scale);
    adaptive_iteration_positive_unit_gain_counts_.push_back(
        positive_count);

    ++adaptive_models_seen_;
    return adaptive_models_seen_ == adaptive_models_expected_;
  }

  void EndAdaptiveIteration() {
    if (!adaptive_enabled_) {
      return;
    }
    CHECK_NOTNULL(adaptive_sampler_.get());
    CHECK(adaptive_iteration_active_);
    CHECK(adaptive_sampler_->awaiting_update());
    CHECK_EQ(adaptive_models_seen_, adaptive_models_expected_);
    CHECK_EQ(
        adaptive_iteration_next_gain_scales_.size(),
        static_cast<size_t>(adaptive_models_expected_));
    CHECK_EQ(
        adaptive_iteration_positive_unit_gain_counts_.size(),
        static_cast<size_t>(adaptive_models_expected_));
    CHECK_EQ(
        adaptive_iteration_opportunity_sum_.size(),
        adaptive_selected_unit_positions_.size());

    // Binary/regression iterations need no averaging pass: multiplication by
    // exactly one would leave every statistic unchanged.
    if (adaptive_models_expected_ > 1) {
      const double inverse_models =
          1.0 / static_cast<double>(adaptive_models_expected_);
      for (double& value : adaptive_iteration_opportunity_sum_) {
        value *= inverse_models;
      }
    }

    const bool is_calibration =
        adaptive_sampler_->selection_round() == 1;
    if (is_calibration) {
      // Allocate/copy before the controller commit. A failure here leaves
      // committed controller and scale state untouched.
      adaptive_calibration_gain_scales_ =
          adaptive_iteration_next_gain_scales_;
      adaptive_calibration_positive_unit_gain_counts_ =
          adaptive_iteration_positive_unit_gain_counts_;
    }

    // This is the sole persistent controller mutation. It occurs only after
    // every class tree and the native boosting iteration have been accepted.
    adaptive_sampler_->Update(adaptive_iteration_opportunity_sum_);
    adaptive_gain_scales_.swap(adaptive_iteration_next_gain_scales_);

    if (adaptive_sampler_->detailed_receipt_available()) {
      Log::Info("%s", AdaptiveDetailedReceiptLine().c_str());
    }
    if (is_calibration && adaptive_sampler_->detailed_logging_enabled()) {
      Log::Info("%s", AdaptiveCalibrationReceiptLine().c_str());
    }
    adaptive_previous_selected_unit_positions_.clear();
    adaptive_mask_changed_ = false;
    adaptive_iteration_active_ = false;
    adaptive_models_expected_ = 0;
    adaptive_models_seen_ = 0;
    adaptive_iteration_opportunity_sum_.clear();
    adaptive_iteration_next_gain_scales_.clear();
    adaptive_iteration_positive_unit_gain_counts_.clear();
    adaptive_selected_unit_gains_scratch_.clear();
  }

  void AbortAdaptiveIteration() noexcept {
    if (!adaptive_enabled_) {
      return;
    }
    if (adaptive_sampler_ != nullptr) {
      adaptive_sampler_->AbortPendingSelection();
    }
    if (adaptive_mask_changed_) {
      adaptive_selected_unit_positions_.swap(
          adaptive_previous_selected_unit_positions_);
      ApplyAdaptiveUnitSelection();
    }
    adaptive_previous_selected_unit_positions_.clear();
    adaptive_mask_changed_ = false;
    adaptive_iteration_opportunity_sum_.clear();
    adaptive_iteration_next_gain_scales_.clear();
    adaptive_iteration_positive_unit_gain_counts_.clear();
    adaptive_selected_unit_gains_scratch_.clear();
    adaptive_iteration_active_ = false;
    adaptive_models_expected_ = 0;
    adaptive_models_seen_ = 0;
  }

  void SetIsFeatureUsedByTree(int fid, bool val) {
    is_feature_used_[fid] = val;
  }

 private:
  void ApplyAdaptiveUnitSelection() {
    CHECK(std::is_sorted(
        adaptive_selected_unit_positions_.begin(),
        adaptive_selected_unit_positions_.end()));
    CHECK(std::is_sorted(
        adaptive_previous_selected_unit_positions_.begin(),
        adaptive_previous_selected_unit_positions_.end()));

    if (adaptive_previous_selected_unit_positions_.empty()) {
      // Initialization has no installed selection to update.
      std::fill(is_feature_used_.begin(), is_feature_used_.end(), 0);
      for (int unit_position : adaptive_selected_unit_positions_) {
        SetAdaptiveUnitUsed(unit_position, true);
      }
    } else {
      // Apply only the symmetric difference. During rollback the two vectors
      // exchange roles, so the same exact operation restores the committed
      // mask without allocating in the noexcept path.
      size_t previous_index = 0;
      size_t current_index = 0;
      while (previous_index <
                 adaptive_previous_selected_unit_positions_.size() &&
             current_index < adaptive_selected_unit_positions_.size()) {
        const int previous =
            adaptive_previous_selected_unit_positions_[previous_index];
        const int current =
            adaptive_selected_unit_positions_[current_index];
        if (previous == current) {
          ++previous_index;
          ++current_index;
        } else if (previous < current) {
          SetAdaptiveUnitUsed(previous, false);
          ++previous_index;
        } else {
          SetAdaptiveUnitUsed(current, true);
          ++current_index;
        }
      }
      while (previous_index <
             adaptive_previous_selected_unit_positions_.size()) {
        SetAdaptiveUnitUsed(
            adaptive_previous_selected_unit_positions_[previous_index],
            false);
        ++previous_index;
      }
      while (current_index < adaptive_selected_unit_positions_.size()) {
        SetAdaptiveUnitUsed(
            adaptive_selected_unit_positions_[current_index], true);
        ++current_index;
      }
    }
    RebuildAdaptiveSelectedFeatures();
  }

  void SetAdaptiveUnitUsed(int unit_position, bool is_used) {
    CHECK_GE(unit_position, 0);
    CHECK_LT(
        static_cast<size_t>(unit_position),
        adaptive_unit_inner_features_.size());
    const int inner_feature = adaptive_unit_inner_features_[unit_position];
    CHECK_GE(inner_feature, 0);
    CHECK_LT(inner_feature, static_cast<int>(is_feature_used_.size()));
    CHECK_EQ(is_feature_used_[inner_feature], is_used ? 0 : 1);
    is_feature_used_[inner_feature] = is_used ? 1 : 0;
  }

  void RebuildAdaptiveSelectedFeatures() {
    adaptive_selected_inner_features_.clear();
    used_feature_indices_.clear();
    for (int unit_position : adaptive_selected_unit_positions_) {
      CHECK_GE(unit_position, 0);
      CHECK_LT(
          static_cast<size_t>(unit_position),
          adaptive_unit_inner_features_.size());
      const int inner_feature = adaptive_unit_inner_features_[unit_position];
      CHECK_EQ(is_feature_used_[inner_feature], 1);
      const int valid_position =
          adaptive_valid_position_by_inner_feature_[inner_feature];
      CHECK_GE(valid_position, 0);
      adaptive_selected_inner_features_.push_back(inner_feature);
      used_feature_indices_.push_back(valid_position);
    }
    std::sort(
        adaptive_selected_inner_features_.begin(),
        adaptive_selected_inner_features_.end());
    std::sort(used_feature_indices_.begin(), used_feature_indices_.end());
    CHECK(
        std::adjacent_find(
            adaptive_selected_inner_features_.begin(),
            adaptive_selected_inner_features_.end()) ==
        adaptive_selected_inner_features_.end());
    CHECK(
        std::adjacent_find(
            used_feature_indices_.begin(), used_feature_indices_.end()) ==
        used_feature_indices_.end());
  }

  void InitializeAdaptiveSampler() {
    CHECK_NOTNULL(train_data_);
    if (adaptive_sampler_ == nullptr) {
      Log::Fatal(
          "Adaptive selectable-unit sampler was not configured before "
          "training data initialization.");
    }

    adaptive_unit_inner_features_.clear();
    adaptive_valid_position_by_inner_feature_.assign(
        train_data_->num_features(), -1);
    adaptive_canonical_unit_ids_.clear();
    for (size_t valid_position = 0;
         valid_position < valid_feature_indices_.size(); ++valid_position) {
      const int inner_feature = train_data_->InnerFeatureIndex(
          valid_feature_indices_[valid_position]);
      CHECK_GE(inner_feature, 0);
      adaptive_valid_position_by_inner_feature_[inner_feature] =
          static_cast<int>(valid_position);
    }
    adaptive_unit_inner_features_.reserve(valid_feature_indices_.size());
    adaptive_canonical_unit_ids_.reserve(valid_feature_indices_.size());
    for (int canonical_feature : valid_feature_indices_) {
      const int inner_feature = train_data_->InnerFeatureIndex(canonical_feature);
      CHECK_GE(inner_feature, 0);
      CHECK_EQ(train_data_->RealFeatureIndex(inner_feature), canonical_feature);
      adaptive_unit_inner_features_.push_back(inner_feature);
      adaptive_canonical_unit_ids_.push_back(canonical_feature);
    }
    CHECK(!adaptive_canonical_unit_ids_.empty());
    CHECK_EQ(
        adaptive_unit_inner_features_.size(),
        adaptive_canonical_unit_ids_.size());
    CHECK_EQ(
        adaptive_valid_position_by_inner_feature_.size(),
        is_feature_used_.size());
    CHECK(
        std::adjacent_find(
            adaptive_canonical_unit_ids_.begin(),
            adaptive_canonical_unit_ids_.end()) ==
        adaptive_canonical_unit_ids_.end());

    adaptive_sampler_->Reset(adaptive_canonical_unit_ids_);
    adaptive_iteration_active_ = false;
    adaptive_mask_changed_ = false;
    adaptive_models_expected_ = 0;
    adaptive_models_seen_ = 0;
    adaptive_iteration_opportunity_sum_.clear();
    adaptive_iteration_opportunity_sum_.reserve(
        adaptive_unit_inner_features_.size());
    adaptive_iteration_next_gain_scales_.clear();
    adaptive_iteration_positive_unit_gain_counts_.clear();
    adaptive_selected_unit_gains_scratch_.clear();
    adaptive_selected_unit_gains_scratch_.reserve(
        adaptive_unit_inner_features_.size());
    adaptive_gain_scales_.clear();
    adaptive_calibration_gain_scales_.clear();
    adaptive_calibration_positive_unit_gain_counts_.clear();
    adaptive_previous_selected_unit_positions_.clear();
    // Reserve the whole fixed universe once so changed-selection rollback is
    // allocation-free inside AbortAdaptiveIteration() noexcept.
    used_feature_indices_.clear();
    used_feature_indices_.reserve(valid_feature_indices_.size());
    adaptive_selected_inner_features_.clear();
    adaptive_selected_inner_features_.reserve(
        valid_feature_indices_.size());
    need_reset_bytree_ = true;
    adaptive_selected_unit_positions_.resize(
        adaptive_unit_inner_features_.size());
    std::iota(
        adaptive_selected_unit_positions_.begin(),
        adaptive_selected_unit_positions_.end(), 0);
    ApplyAdaptiveUnitSelection();
    used_cnt_bytree_ = static_cast<int>(used_feature_indices_.size());
  }

  const Dataset* train_data_;
  double fraction_bytree_;
  double fraction_bynode_;
  bool need_reset_bytree_;
  int used_cnt_bytree_;
  int seed_;
  Random random_;
  std::vector<int8_t> is_feature_used_;
  std::vector<int> used_feature_indices_;
  std::vector<int> valid_feature_indices_;
  bool adaptive_enabled_;
  std::unique_ptr<AdaptiveFeatureMcb> adaptive_sampler_;
  std::vector<int> adaptive_unit_inner_features_;
  std::vector<int> adaptive_valid_position_by_inner_feature_;
  std::vector<int> adaptive_selected_inner_features_;
  std::vector<int> adaptive_canonical_unit_ids_;
  std::vector<int> adaptive_selected_unit_positions_;
  std::vector<int> adaptive_previous_selected_unit_positions_;
  bool adaptive_iteration_active_ = false;
  bool adaptive_mask_changed_ = false;
  int adaptive_models_expected_ = 0;
  int adaptive_models_seen_ = 0;
  std::vector<double> adaptive_iteration_opportunity_sum_;
  std::vector<double> adaptive_iteration_next_gain_scales_;
  std::vector<size_t> adaptive_iteration_positive_unit_gain_counts_;
  std::vector<double> adaptive_selected_unit_gains_scratch_;
  std::vector<double> adaptive_gain_scales_;
  std::vector<double> adaptive_calibration_gain_scales_;
  std::vector<size_t> adaptive_calibration_positive_unit_gain_counts_;
  /*! \brief interaction constraints index in original (raw data) features */
  std::vector<std::unordered_set<int>> interaction_constraints_;
};

}  // namespace LightGBM
#endif  // LIGHTGBM_SRC_TREELEARNER_COL_SAMPLER_HPP_
