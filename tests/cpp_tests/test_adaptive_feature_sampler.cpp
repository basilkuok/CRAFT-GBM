/*!
 * Copyright (c) 2026-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#include <gtest/gtest.h>

#include <LightGBM/c_api.h>
#include <LightGBM/config.h>
#include <LightGBM/dataset.h>
#include <LightGBM/tree.h>

#include "../../src/treelearner/col_sampler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef JAEGER_TEST_HOOKS
namespace {

class DatasetHandleOwner {
 public:
  ~DatasetHandleOwner() {
    if (handle != nullptr) {
      LGBM_DatasetFree(handle);
    }
  }

  DatasetHandle handle = nullptr;
};

class TemporaryBinaryFile {
 public:
  explicit TemporaryBinaryFile(std::string filename)
      : filename_(std::move(filename)) {}

  ~TemporaryBinaryFile() {
    std::remove(filename_.c_str());
  }

  const std::string& filename() const {
    return filename_;
  }

 private:
  std::string filename_;
};

class AdaptiveFeatureSamplerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    constexpr int kRows = 100;
    constexpr int kColumns = 5;
    std::vector<double> values(kRows * kColumns, 0.0);
    for (int row = 0; row < 30; ++row) {
      values[row * kColumns] = 1.0 + (row % 3);
    }
    for (int row = 30; row < 60; ++row) {
      values[row * kColumns + 1] = 1.0 + (row % 3);
    }
    // Column 2 joins the dense, conflict-free ordinary EFB group. Columns
    // 3 and 4 conflict with each other and form one native multi-value group.
    for (int row = 60; row < 65; ++row) {
      values[row * kColumns + 2] = 1.0 + (row % 2);
      values[row * kColumns + 3] = 1.0 + (row % 2);
      values[row * kColumns + 4] = 2.0 + (row % 2);
    }

    ASSERT_EQ(
        LGBM_DatasetCreateFromMat(
            values.data(), C_API_DTYPE_FLOAT64, kRows, kColumns, 1,
            "max_bin=8 min_data_in_bin=1 min_data_in_leaf=1 "
            "feature_pre_filter=false enable_bundle=true "
            "is_enable_sparse=true",
            nullptr, &handle_),
        0);
    dataset_ = static_cast<LightGBM::Dataset*>(handle_);
    ASSERT_NE(dataset_, nullptr);
    ASSERT_EQ(dataset_->num_features(), kColumns);

    std::map<int, std::vector<int>> members_by_group;
    for (int inner_feature = 0;
         inner_feature < dataset_->num_features(); ++inner_feature) {
      members_by_group[dataset_->Feature2Group(inner_feature)].push_back(
          inner_feature);
    }
    bool found_multi_member_ordinary_group = false;
    bool found_multi_member_multi_value_group = false;
    for (const auto& item : members_by_group) {
      if (item.second.size() <= 1) {
        continue;
      }
      if (dataset_->IsMultiGroup(item.first)) {
        found_multi_member_multi_value_group = true;
      } else {
        found_multi_member_ordinary_group = true;
      }
    }
    ASSERT_TRUE(found_multi_member_ordinary_group);
    ASSERT_TRUE(found_multi_member_multi_value_group);
  }

  void TearDown() override {
    if (handle_ != nullptr) {
      EXPECT_EQ(LGBM_DatasetFree(handle_), 0);
    }
  }

  static LightGBM::Config AdaptiveConfig() {
    LightGBM::Config config;
    config.num_iterations = 5;
    config.feature_fraction = 1.0;
    config.feature_fraction_bynode = 1.0;
    config.adaptive_feature_selection = true;
    config.adaptive_feature_confidence_delta = 0.05;
    config.adaptive_feature_log_period = 0;
    return config;
  }

  std::vector<double> InnerGains(
      const std::vector<double>& canonical_gains) const {
    EXPECT_EQ(
        canonical_gains.size(),
        static_cast<size_t>(dataset_->num_total_features()));
    std::vector<double> gains(dataset_->num_features(), 0.0);
    for (int canonical_feature = 0;
         canonical_feature < dataset_->num_total_features();
         ++canonical_feature) {
      const int inner_feature =
          dataset_->InnerFeatureIndex(canonical_feature);
      EXPECT_GE(inner_feature, 0);
      gains[inner_feature] = canonical_gains[canonical_feature];
    }
    return gains;
  }

  std::vector<double> CalibrationFeatureGains() const {
    return InnerGains({1.0, 9.0, 4.0, 2.0, 0.0});
  }

  std::vector<int> ExpectedInnerFeatureIndices() const {
    std::vector<int> result;
    for (int canonical_feature : dataset_->ValidFeatureIndices()) {
      const int inner_feature =
          dataset_->InnerFeatureIndex(canonical_feature);
      EXPECT_GE(inner_feature, 0);
      result.push_back(inner_feature);
    }
    return result;
  }

  void CommitCalibration(LightGBM::ColSampler* sampler) const {
    sampler->BeginAdaptiveIteration(1);
    sampler->PrepareAdaptiveTree();
    EXPECT_TRUE(
        sampler->AccumulateAdaptiveModel(CalibrationFeatureGains()));
    sampler->EndAdaptiveIteration();
  }

  DatasetHandle handle_ = nullptr;
  LightGBM::Dataset* dataset_ = nullptr;
};

TEST_F(
    AdaptiveFeatureSamplerTest,
    SelectedSubrowCopyUsesOrdinaryEfbClosureAndSelectedMultiMembers) {
  std::map<int, std::vector<int>> members_by_group;
  for (int feature = 0; feature < dataset_->num_features(); ++feature) {
    members_by_group[dataset_->Feature2Group(feature)].push_back(feature);
  }

  std::vector<int> ordinary_features;
  std::vector<int> multi_features;
  for (const auto& item : members_by_group) {
    if (item.second.size() > 1 && dataset_->IsMultiGroup(item.first)) {
      multi_features = item.second;
    } else if (item.second.size() > 1) {
      ordinary_features = item.second;
    }
  }
  ASSERT_GE(ordinary_features.size(), 2U);
  ASSERT_GE(multi_features.size(), 2U);

  const std::vector<LightGBM::data_size_t> first_rows({60, 61, 62, 63});
  const std::vector<LightGBM::data_size_t> second_rows({61, 62, 63, 64});
  const auto bin_at = [](const LightGBM::Dataset* data, int feature,
                         LightGBM::data_size_t row) {
    std::unique_ptr<LightGBM::BinIterator> iterator(
        data->FeatureIterator(feature));
    iterator->Reset(row);
    return iterator->Get(row);
  };

  LightGBM::Dataset subset(
      static_cast<LightGBM::data_size_t>(first_rows.size()));
  subset.CopyFeatureMapperFrom(dataset_);
  subset.CopySubrow(dataset_, first_rows.data(),
                    static_cast<LightGBM::data_size_t>(first_rows.size()),
                    false);

  std::vector<int8_t> mask(dataset_->num_features(), 0);
  mask[ordinary_features.front()] = 1;
  mask[multi_features.front()] = 1;
  subset.CopySubrow(dataset_, second_rows.data(),
                    static_cast<LightGBM::data_size_t>(second_rows.size()),
                    false, &mask);

  EXPECT_EQ(subset.num_features(), dataset_->num_features());
  EXPECT_EQ(subset.num_total_features(), dataset_->num_total_features());
  EXPECT_EQ(subset.num_feature_groups(), dataset_->num_feature_groups());
  for (int feature = 0; feature < dataset_->num_features(); ++feature) {
    EXPECT_EQ(subset.RealFeatureIndex(feature),
              dataset_->RealFeatureIndex(feature));
    EXPECT_EQ(subset.Feature2Group(feature), dataset_->Feature2Group(feature));
    EXPECT_EQ(subset.Feature2SubFeature(feature),
              dataset_->Feature2SubFeature(feature));
  }

  bool observed_ordinary_sibling_change = false;
  bool observed_unselected_multi_stale_value = false;
  for (size_t local_row = 0; local_row < first_rows.size(); ++local_row) {
    for (int feature : ordinary_features) {
      const uint32_t expected =
          bin_at(dataset_, feature, second_rows[local_row]);
      EXPECT_EQ(bin_at(&subset, feature,
                       static_cast<LightGBM::data_size_t>(local_row)),
                expected);
      if (feature != ordinary_features.front() &&
          expected != bin_at(dataset_, feature, first_rows[local_row])) {
        observed_ordinary_sibling_change = true;
      }
    }

    EXPECT_EQ(bin_at(&subset, multi_features.front(),
                     static_cast<LightGBM::data_size_t>(local_row)),
              bin_at(dataset_, multi_features.front(),
                     second_rows[local_row]));
    for (size_t member = 1; member < multi_features.size(); ++member) {
      const int feature = multi_features[member];
      const uint32_t old_value =
          bin_at(dataset_, feature, first_rows[local_row]);
      EXPECT_EQ(bin_at(&subset, feature,
                       static_cast<LightGBM::data_size_t>(local_row)),
                old_value);
      if (old_value != bin_at(dataset_, feature, second_rows[local_row])) {
        observed_unselected_multi_stale_value = true;
      }
    }
  }
  EXPECT_TRUE(observed_ordinary_sibling_change);
  EXPECT_TRUE(observed_unselected_multi_stale_value);

  // A null mask is exactly the native all-column CopySubrow path.
  LightGBM::Dataset native_subset(
      static_cast<LightGBM::data_size_t>(second_rows.size()));
  LightGBM::Dataset null_mask_subset(
      static_cast<LightGBM::data_size_t>(second_rows.size()));
  native_subset.CopyFeatureMapperFrom(dataset_);
  null_mask_subset.CopyFeatureMapperFrom(dataset_);
  native_subset.CopySubrow(
      dataset_, second_rows.data(),
      static_cast<LightGBM::data_size_t>(second_rows.size()), false);
  null_mask_subset.CopySubrow(
      dataset_, second_rows.data(),
      static_cast<LightGBM::data_size_t>(second_rows.size()), false, nullptr);
  for (int feature = 0; feature < dataset_->num_features(); ++feature) {
    for (size_t local_row = 0; local_row < second_rows.size(); ++local_row) {
      EXPECT_EQ(bin_at(&native_subset, feature,
                       static_cast<LightGBM::data_size_t>(local_row)),
                bin_at(dataset_, feature, second_rows[local_row]));
      EXPECT_EQ(bin_at(&native_subset, feature,
                       static_cast<LightGBM::data_size_t>(local_row)),
                bin_at(&null_mask_subset, feature,
                       static_cast<LightGBM::data_size_t>(local_row)));
    }
  }
}

TEST_F(
    AdaptiveFeatureSamplerTest,
    NativeHistogramStateHandlesChangedMasksAndRowSubsets) {
  const std::vector<int8_t> all_features(dataset_->num_features(), 1);
  std::vector<LightGBM::score_t> gradients(dataset_->num_data(), 1.0);
  std::vector<LightGBM::score_t> hessians(dataset_->num_data(), 1.0);
  std::unique_ptr<LightGBM::TrainingShareStates> state(
      dataset_->GetShareStates<false, 0>(
          gradients.data(), hessians.data(), all_features, false,
          true, false, 0));
  LightGBM::Dataset subset(4);
  LightGBM::Dataset reference_subset(4);
  subset.CopyFeatureMapperFrom(dataset_);
  reference_subset.CopyFeatureMapperFrom(dataset_);

  const std::vector<std::vector<LightGBM::data_size_t>> row_sets = {
      {0, 30, 60, 61}, {1, 31, 62, 63}, {2, 32, 63, 64},
      {0, 30, 61, 62}, {1, 31, 60, 64}};
  // Select one multi-value member, switch to its sibling, keep that mask
  // with different rows, select both members, then return to the first mask.
  const std::vector<std::vector<int>> selected_features = {
      {0, 3}, {1, 4}, {1, 4}, {2, 3, 4}, {0, 3}};
  for (size_t iteration = 0; iteration < row_sets.size(); ++iteration) {
    SCOPED_TRACE(iteration);
    const auto& rows = row_sets[iteration];
    std::vector<int8_t> mask(dataset_->num_features(), 0);
    for (int feature : selected_features[iteration]) {
      mask[dataset_->InnerFeatureIndex(feature)] = 1;
    }
    for (size_t row = 0; row < rows.size(); ++row) {
      gradients[row] = static_cast<LightGBM::score_t>(rows[row] + 1);
      hessians[row] = static_cast<LightGBM::score_t>(1 + rows[row] % 3);
    }
    subset.CopySubrow(dataset_, rows.data(), 4, false, &mask);
    reference_subset.CopySubrow(dataset_, rows.data(), 4, false);
    state->SetUseSubrow(true);
    state->SetSubrowCopied(false);
    state->bagging_use_indices = rows.data();
    state->bagging_indices_cnt = 4;
    subset.InitTrain(mask, state.get());

    std::unique_ptr<LightGBM::TrainingShareStates> reference_state(
        reference_subset.GetShareStates<false, 0>(
            gradients.data(), hessians.data(), all_features, false,
            true, false, 0));
    reference_state->SetUseSubrow(false);
    reference_state->bagging_use_indices = nullptr;
    reference_state->bagging_indices_cnt = 0;
    reference_subset.InitTrain(all_features, reference_state.get());
    ASSERT_EQ(state->feature_hist_offsets(),
              reference_state->feature_hist_offsets());
    std::vector<LightGBM::hist_t> actual(
        state->num_hist_total_bin() * 2, 0.0);
    std::vector<LightGBM::hist_t> expected(actual.size(), 0.0);
    std::vector<LightGBM::score_t> ordered_gradients(4);
    std::vector<LightGBM::score_t> ordered_hessians(4);
    subset.ConstructHistograms<false, 0>(
        mask, nullptr, 4, gradients.data(), hessians.data(),
        ordered_gradients.data(), ordered_hessians.data(), state.get(),
        actual.data());
    reference_subset.ConstructHistograms<false, 0>(
        all_features, nullptr, 4, gradients.data(), hessians.data(),
        ordered_gradients.data(), ordered_hessians.data(),
        reference_state.get(), expected.data());
    for (int feature : selected_features[iteration]) {
      const int inner = dataset_->InnerFeatureIndex(feature);
      const size_t offset = state->feature_hist_offsets()[inner] * 2;
      const int bins = dataset_->FeatureNumBin(inner) -
          (dataset_->FeatureBinMapper(inner)->GetMostFreqBin() == 0);
      for (int entry = 0; entry < bins * 2; ++entry) {
        EXPECT_DOUBLE_EQ(actual[offset + entry], expected[offset + entry])
            << "feature=" << feature << " histogram_entry=" << entry;
      }
    }
  }
}

TEST_F(
    AdaptiveFeatureSamplerTest,
    PrefilteredNonconsecutiveFeatureMappingSurvivesBinaryReload) {
  constexpr int kRows = 64;
  constexpr int kColumns = 4;
  std::vector<double> values(kRows * kColumns, 0.0);
  for (int row = 0; row < kRows; ++row) {
    values[row * kColumns] = static_cast<double>(row % 2);
    values[row * kColumns + 1] = 0.0;
    values[row * kColumns + 2] = static_cast<double>(row % 3);
    values[row * kColumns + 3] = 7.0;
  }

  const char* dataset_parameters =
      "max_bin=16 min_data_in_bin=1 min_data_in_leaf=1 "
      "feature_pre_filter=true enable_bundle=true is_enable_sparse=true";

  DatasetHandleOwner original;
  ASSERT_EQ(
      LGBM_DatasetCreateFromMat(
          values.data(), C_API_DTYPE_FLOAT64, kRows, kColumns, 1,
          dataset_parameters, nullptr, &original.handle),
      0);
  auto* original_dataset =
      static_cast<LightGBM::Dataset*>(original.handle);
  ASSERT_NE(original_dataset, nullptr);
  ASSERT_EQ(original_dataset->num_total_features(), kColumns);
  ASSERT_EQ(original_dataset->num_features(), 2);
  EXPECT_EQ(
      original_dataset->ValidFeatureIndices(),
      std::vector<int>({0, 2}));

  const auto unique_suffix =
      std::chrono::steady_clock::now().time_since_epoch().count();
  TemporaryBinaryFile binary_file(
      "jaeger_v7_feature_mapping_" + std::to_string(unique_suffix) + ".bin");
  ASSERT_EQ(
      LGBM_DatasetSaveBinary(
          original.handle, binary_file.filename().c_str()),
      0);

  DatasetHandleOwner reloaded;
  ASSERT_EQ(
      LGBM_DatasetCreateFromFile(
          binary_file.filename().c_str(), dataset_parameters, nullptr,
          &reloaded.handle),
      0);
  auto* reloaded_dataset =
      static_cast<LightGBM::Dataset*>(reloaded.handle);
  ASSERT_NE(reloaded_dataset, nullptr);
  ASSERT_EQ(reloaded_dataset->num_total_features(), kColumns);
  ASSERT_EQ(reloaded_dataset->num_features(), 2);
  EXPECT_EQ(
      reloaded_dataset->ValidFeatureIndices(),
      std::vector<int>({0, 2}));

  const std::vector<int> expected_canonical_features({0, 2});
  const std::vector<int> expected_inner_features({
      original_dataset->InnerFeatureIndex(0),
      original_dataset->InnerFeatureIndex(2)});
  ASSERT_GE(expected_inner_features[0], 0);
  ASSERT_GE(expected_inner_features[1], 0);
  ASSERT_NE(expected_inner_features[0], expected_inner_features[1]);
  EXPECT_EQ(
      std::vector<int>({
          reloaded_dataset->InnerFeatureIndex(0),
          reloaded_dataset->InnerFeatureIndex(2)}),
      expected_inner_features);

  for (LightGBM::Dataset* mapped_dataset :
       {original_dataset, reloaded_dataset}) {
    LightGBM::Config config = AdaptiveConfig();
    LightGBM::ColSampler sampler(&config);
    sampler.SetTrainingData(mapped_dataset);
    EXPECT_EQ(
        sampler.AdaptiveCanonicalUnitIdsForTesting(),
        expected_canonical_features);
    EXPECT_EQ(
        sampler.AdaptiveInnerFeaturesForTesting(),
        expected_inner_features);
    for (size_t position = 0;
         position < expected_canonical_features.size(); ++position) {
      EXPECT_EQ(
          mapped_dataset->RealFeatureIndex(expected_inner_features[position]),
          expected_canonical_features[position]);
    }
    EXPECT_EQ(mapped_dataset->InnerFeatureIndex(1), -1);
    EXPECT_EQ(mapped_dataset->InnerFeatureIndex(3), -1);
  }
}

TEST_F(
    AdaptiveFeatureSamplerTest,
    NativeEfbLayoutPreservesOneToOneOriginalFeatureSelectionAndRewards) {
  const LightGBM::Config config = AdaptiveConfig();
  LightGBM::ColSampler sampler(&config);
  sampler.SetTrainingData(dataset_);

  const auto expected_inner_features = ExpectedInnerFeatureIndices();
  const auto& actual_units =
      sampler.AdaptiveInnerFeaturesForTesting();
  EXPECT_EQ(actual_units, expected_inner_features);
  ASSERT_EQ(actual_units.size(), 5U);
  EXPECT_EQ(
      sampler.AdaptiveCanonicalUnitIdsForTesting(),
      dataset_->ValidFeatureIndices());
  for (size_t position = 0; position < actual_units.size();
       ++position) {
    EXPECT_EQ(
        dataset_->RealFeatureIndex(actual_units[position]),
        dataset_->ValidFeatureIndices()[position]);
  }
  const auto& calibration_controller =
      sampler.AdaptiveControllerForTesting();
  EXPECT_GT(calibration_controller.empirical_bernstein_rho(), 0.0);
  EXPECT_EQ(calibration_controller.calibration_reference_count(), 300);
  EXPECT_DOUBLE_EQ(calibration_controller.variance_reference(), 75.0);
  const double reference_root =
      calibration_controller.variance_reference() /
      calibration_controller.empirical_bernstein_rho();
  EXPECT_NEAR(
      reference_root - std::log1p(reference_root),
      2.0 * (calibration_controller.log_inverse_alpha() - std::log(2.0)), 2e-14);

  // The first three original features share one native EFB storage group.
  EXPECT_EQ(
      dataset_->Feature2Group(actual_units[0]),
      dataset_->Feature2Group(actual_units[1]));
  EXPECT_EQ(
      dataset_->Feature2Group(actual_units[1]),
      dataset_->Feature2Group(actual_units[2]));
  EXPECT_FALSE(
      dataset_->IsMultiGroup(
          dataset_->Feature2Group(actual_units[0])));

  CommitCalibration(&sampler);
  EXPECT_EQ(sampler.AdaptiveControllerForTesting().round(), 1);
  const std::string calibration = sampler.AdaptiveCalibrationReceiptLine();
  EXPECT_NE(
      calibration.find("calibration_gain_scales=[3]"),
      std::string::npos);
  EXPECT_NE(
      calibration.find(
          "calibration_positive_unit_gain_counts=[4]"),
      std::string::npos);

  sampler.BeginAdaptiveIteration(1);
  EXPECT_EQ(
      sampler.AdaptiveSelectedUnitPositionsForTesting(),
      std::vector<int>({0, 1, 2, 3, 4}));
  const auto& selected_mask = sampler.is_feature_used_bytree();
  EXPECT_EQ(
      std::count(selected_mask.begin(), selected_mask.end(), int8_t{1}),
      5);

  std::vector<double> reset_probe(dataset_->num_features(), 9.0);
  sampler.ResetAdaptiveRootGains(&reset_probe);
  EXPECT_EQ(
      reset_probe,
      std::vector<double>(dataset_->num_features(), 0.0));

  sampler.PrepareAdaptiveTree();
  EXPECT_TRUE(sampler.AccumulateAdaptiveModel(
      InnerGains({2.75, 11.0, 1.0, 5.5, 99.0})));
  EXPECT_EQ(sampler.AdaptiveControllerForTesting().round(), 1);
  sampler.EndAdaptiveIteration();

  const auto& controller = sampler.AdaptiveControllerForTesting();
  EXPECT_EQ(controller.round(), 2);
  const std::vector<double>& accumulated_gain =
      controller.accumulated_gain();
  const std::vector<double>& exposure_count =
      controller.exposure_count();
  ASSERT_EQ(accumulated_gain.size(), 5U);
  EXPECT_NEAR(accumulated_gain[0], 11.0 / 23.0, 1e-15);
  EXPECT_NEAR(accumulated_gain[1], 11.0 / 14.0, 1e-15);
  EXPECT_DOUBLE_EQ(accumulated_gain[2], 0.25);
  EXPECT_NEAR(accumulated_gain[3], 11.0 / 17.0, 1e-15);
  EXPECT_NEAR(accumulated_gain[4], 33.0 / 34.0, 1e-15);
  EXPECT_EQ(exposure_count, std::vector<double>(5, 1.0));
  for (size_t position = 0; position < accumulated_gain.size(); ++position) {
    const double residual = accumulated_gain[position] - 0.5;
    EXPECT_DOUBLE_EQ(controller.empirical_variance()[position], residual * residual);
  }
  EXPECT_NE(
      sampler.AdaptiveFinalReceiptLine().find(
          "current_gain_scales=[5.5]"),
      std::string::npos);
}

TEST_F(
    AdaptiveFeatureSamplerTest,
    PeriodicDetailedReceiptReportsFeatureMaskAndExactDirectEfbClosure) {
  LightGBM::Config config = AdaptiveConfig();
  config.adaptive_feature_log_period = 1;
  LightGBM::ColSampler sampler(&config);
  sampler.SetTrainingData(dataset_);

  CommitCalibration(&sampler);
  const std::string receipt = sampler.AdaptiveDetailedReceiptLine();
  EXPECT_NE(receipt.find("iteration=1"), std::string::npos);
  EXPECT_NE(
      receipt.find("diagnostic_state=pre_iteration_selection"),
      std::string::npos);
  EXPECT_NE(
      receipt.find("history_through_iteration=0"),
      std::string::npos);
  EXPECT_NE(
      receipt.find("canonical_unit_ids=[0,1,2,3,4]"),
      std::string::npos);
  EXPECT_NE(
      receipt.find("selected_canonical_unit_ids=[0,1,2,3,4]"),
      std::string::npos);
  EXPECT_NE(
      receipt.find("possible_best_canonical_unit_ids=[0,1,2,3,4]"),
      std::string::npos);
  EXPECT_NE(
      receipt.find("accumulated_gain_by_unit=[0,0,0,0,0]"),
      std::string::npos);
  EXPECT_NE(
      receipt.find("exposure_count_by_unit=[0,0,0,0,0]"),
      std::string::npos);
  EXPECT_NE(
      receipt.find("lower_confidence_bound_by_unit=[0,0,0,0,0]"),
      std::string::npos);
  EXPECT_NE(
      receipt.find("upper_confidence_bound_by_unit=[1,1,1,1,1]"),
      std::string::npos);
  EXPECT_NE(
      receipt.find("original_feature_mask_available=true"),
      std::string::npos);
  EXPECT_NE(
      receipt.find("original_feature_ids=[0,1,2,3,4]"),
      std::string::npos);
  EXPECT_NE(
      receipt.find(
          "selected_original_feature_ids=[0,1,2,3,4]"),
      std::string::npos);
  EXPECT_NE(
      receipt.find(
          "possible_best_original_feature_ids=[0,1,2,3,4]"),
      std::string::npos);

  std::vector<int> expected_groups;
  for (int inner_feature = 0;
       inner_feature < dataset_->num_features(); ++inner_feature) {
    expected_groups.push_back(dataset_->Feature2Group(inner_feature));
  }
  std::sort(expected_groups.begin(), expected_groups.end());
  expected_groups.erase(
      std::unique(expected_groups.begin(), expected_groups.end()),
      expected_groups.end());
  std::ostringstream expected_group_ids;
  expected_group_ids << "direct_efb_group_ids=[";
  for (size_t index = 0; index < expected_groups.size(); ++index) {
    if (index > 0) {
      expected_group_ids << ',';
    }
    expected_group_ids << expected_groups[index];
  }
  expected_group_ids << ']';
  EXPECT_NE(receipt.find(expected_group_ids.str()), std::string::npos);
  EXPECT_NE(
      receipt.find(
          "direct_efb_group_count=" +
          std::to_string(expected_groups.size())),
      std::string::npos);
  EXPECT_NE(
      receipt.find("direct_efb_group_original_feature_ids=["),
      std::string::npos);
}

TEST_F(
    AdaptiveFeatureSamplerTest,
    FixedBudgetUsesExactFeatureCountAndRestoresMaskOnAbort) {
  LightGBM::Config original_config = AdaptiveConfig();
  original_config.feature_fraction = 0.4;
  LightGBM::ColSampler original(&original_config);
  original.SetTrainingData(dataset_);
  CommitCalibration(&original);
  original.BeginAdaptiveIteration(1);
  original.PrepareAdaptiveTree();
  EXPECT_TRUE(
      original.AccumulateAdaptiveModel(CalibrationFeatureGains()));
  original.EndAdaptiveIteration();

  const std::vector<int8_t> full_original_mask =
      original.is_feature_used_bytree();
  original.BeginAdaptiveIteration(1);
  EXPECT_EQ(
      original.AdaptiveSelectedUnitPositionsForTesting(),
      std::vector<int>({0, 1}));
  EXPECT_EQ(
      original.AdaptiveUsedFeaturePositionsForTesting(),
      std::vector<int>({0, 1}));
  EXPECT_EQ(
      std::count(
          original.is_feature_used_bytree().begin(),
          original.is_feature_used_bytree().end(), int8_t{1}),
      2);
  std::vector<double> original_reset_probe(
      dataset_->num_features(), 9.0);
  original.ResetAdaptiveRootGains(&original_reset_probe);
  for (int inner_feature = 0;
       inner_feature < dataset_->num_features(); ++inner_feature) {
    EXPECT_DOUBLE_EQ(
        original_reset_probe[inner_feature],
        original.is_feature_used_bytree()[inner_feature] == 1 ? 0.0 : 9.0);
  }
  original.AbortAdaptiveIteration();
  EXPECT_EQ(original.is_feature_used_bytree(), full_original_mask);
  original.BeginAdaptiveIteration(1);
  EXPECT_EQ(
      original.AdaptiveSelectedUnitPositionsForTesting(),
      std::vector<int>({0, 1}));
  original.AbortAdaptiveIteration();
}

TEST_F(
    AdaptiveFeatureSamplerTest,
    TwoOutputsUseIndependentFeatureGainsAndAverageOneControllerUpdate) {
  const LightGBM::Config config = AdaptiveConfig();
  LightGBM::ColSampler sampler(&config);
  sampler.SetTrainingData(dataset_);

  sampler.BeginAdaptiveIteration(2);
  sampler.PrepareAdaptiveTree();
  EXPECT_FALSE(sampler.AccumulateAdaptiveModel(
      InnerGains({1.0, 9.0, 4.0, 2.0, 0.0})));
  EXPECT_EQ(sampler.AdaptiveControllerForTesting().round(), 0);
  sampler.PrepareAdaptiveTree();
  EXPECT_TRUE(sampler.AccumulateAdaptiveModel(
      InnerGains({2.0, 6.0, 3.0, 12.0, 4.0})));
  EXPECT_EQ(sampler.AdaptiveControllerForTesting().round(), 0);
  sampler.EndAdaptiveIteration();
  EXPECT_EQ(sampler.AdaptiveControllerForTesting().round(), 1);
  const std::string calibration = sampler.AdaptiveCalibrationReceiptLine();
  EXPECT_NE(
      calibration.find("calibration_gain_scales=[3,4]"),
      std::string::npos);
  EXPECT_NE(
      calibration.find(
          "calibration_positive_unit_gain_counts=[4,5]"),
      std::string::npos);

  sampler.BeginAdaptiveIteration(2);
  EXPECT_EQ(
      sampler.AdaptiveSelectedUnitPositionsForTesting(),
      std::vector<int>({0, 1, 2, 3, 4}));
  sampler.PrepareAdaptiveTree();
  EXPECT_FALSE(sampler.AccumulateAdaptiveModel(
      InnerGains({2.75, 11.0, 1.0, 5.5, 99.0})));
  EXPECT_EQ(sampler.AdaptiveControllerForTesting().round(), 1);
  sampler.PrepareAdaptiveTree();
  EXPECT_TRUE(sampler.AccumulateAdaptiveModel(
      InnerGains({1.0, 3.0, 2.0, 12.0, 99.0})));
  EXPECT_EQ(sampler.AdaptiveControllerForTesting().round(), 1);
  sampler.EndAdaptiveIteration();

  const auto& controller = sampler.AdaptiveControllerForTesting();
  EXPECT_EQ(controller.round(), 2);
  const std::vector<double>& accumulated_gain =
      controller.accumulated_gain();
  const std::vector<double>& exposure_count =
      controller.exposure_count();
  ASSERT_EQ(accumulated_gain.size(), 5U);
  EXPECT_NEAR(
      accumulated_gain[0], 0.5 * (11.0 / 23.0 + 1.0 / 5.0), 1e-15);
  EXPECT_NEAR(
      accumulated_gain[1], 0.5 * (11.0 / 14.0 + 3.0 / 7.0), 1e-15);
  EXPECT_NEAR(
      accumulated_gain[2], 0.5 * (1.0 / 4.0 + 1.0 / 3.0), 1e-15);
  EXPECT_NEAR(
      accumulated_gain[3], 0.5 * (11.0 / 17.0 + 3.0 / 4.0), 1e-15);
  EXPECT_NEAR(
      accumulated_gain[4], 0.5 * (33.0 / 34.0 + 99.0 / 103.0),
      1e-15);
  EXPECT_EQ(exposure_count, std::vector<double>(5, 1.0));
  // One residual of the averaged opportunity, not one residual per class.
  for (size_t position = 0; position < accumulated_gain.size(); ++position) {
    const double residual = accumulated_gain[position] - 0.5;
    EXPECT_DOUBLE_EQ(controller.empirical_variance()[position], residual * residual);
  }
  EXPECT_NE(
      sampler.AdaptiveFinalReceiptLine().find(
          "current_gain_scales=[5.5,3]"),
      std::string::npos);
}

TEST_F(
    AdaptiveFeatureSamplerTest,
    ChangedDynamicMaskAbortRestoresCommittedStateAndDeterministicRetry) {
  LightGBM::Config config = AdaptiveConfig();
  config.num_iterations = 300;
  config.adaptive_feature_confidence_delta = 0.99;
  LightGBM::ColSampler sampler(&config);
  sampler.SetTrainingData(dataset_);
  // Calibrate a predictable positive scale for the first original feature.
  sampler.BeginAdaptiveIteration(1);
  sampler.PrepareAdaptiveTree();
  EXPECT_TRUE(
      sampler.AccumulateAdaptiveModel(InnerGains({1.0, 0.0, 0.0, 0.0, 0.0})));
  sampler.EndAdaptiveIteration();

  const std::vector<int> unchanged_selected =
      sampler.AdaptiveSelectedUnitPositionsForTesting();
  const std::vector<int8_t> unchanged_mask =
      sampler.is_feature_used_bytree();
  const std::vector<int> unchanged_used_features =
      sampler.AdaptiveUsedFeaturePositionsForTesting();
  sampler.BeginAdaptiveIteration(1);
  EXPECT_EQ(
      sampler.AdaptiveSelectedUnitPositionsForTesting(),
      unchanged_selected);
  EXPECT_EQ(sampler.is_feature_used_bytree(), unchanged_mask);
  EXPECT_EQ(
      sampler.AdaptiveUsedFeaturePositionsForTesting(),
      unchanged_used_features);
  sampler.AbortAdaptiveIteration();
  EXPECT_EQ(
      sampler.AdaptiveSelectedUnitPositionsForTesting(),
      unchanged_selected);
  EXPECT_EQ(sampler.is_feature_used_bytree(), unchanged_mask);
  EXPECT_EQ(
      sampler.AdaptiveUsedFeaturePositionsForTesting(),
      unchanged_used_features);

  bool observed_changed_proposal = false;
  double high_gain = 2.0;
  while (sampler.AdaptiveControllerForTesting().round() <
             config.num_iterations - 1 &&
         !observed_changed_proposal) {
    const int committed_round =
        sampler.AdaptiveControllerForTesting().round();
    const uint64_t committed_exclusion_events =
        sampler.AdaptiveControllerForTesting().total_exclusion_events();
    const std::vector<int> committed_features =
        sampler.AdaptiveSelectedUnitPositionsForTesting();
    const std::vector<int> committed_used_features =
        sampler.AdaptiveUsedFeaturePositionsForTesting();
    const std::vector<int8_t> committed_mask =
        sampler.is_feature_used_bytree();
    const std::string committed_receipt =
        sampler.AdaptiveFinalReceiptLine();
    const auto committed_variance =
        sampler.AdaptiveControllerForTesting().empirical_variance();

    sampler.BeginAdaptiveIteration(1);
    const std::vector<int> attempted_features =
        sampler.AdaptiveSelectedUnitPositionsForTesting();
    const std::vector<int> attempted_used_features =
        sampler.AdaptiveUsedFeaturePositionsForTesting();
    const std::vector<int8_t> attempted_mask =
        sampler.is_feature_used_bytree();
    EXPECT_EQ(attempted_used_features, attempted_features);
    const std::vector<double> attempted_gains =
        InnerGains({high_gain, 0.0, 0.0, high_gain, 0.0});
    if (attempted_mask == committed_mask) {
      sampler.PrepareAdaptiveTree();
      EXPECT_TRUE(sampler.AccumulateAdaptiveModel(attempted_gains));
      sampler.EndAdaptiveIteration();
      high_gain *= 2.0;
      continue;
    }

    observed_changed_proposal = true;
    EXPECT_LT(attempted_features.size(), committed_features.size());
    const int first_inner = dataset_->InnerFeatureIndex(0);
    const int ordinary_sibling_inner = dataset_->InnerFeatureIndex(1);
    ASSERT_GE(first_inner, 0);
    ASSERT_GE(ordinary_sibling_inner, 0);
    ASSERT_EQ(
        dataset_->Feature2Group(first_inner),
        dataset_->Feature2Group(ordinary_sibling_inner));
    ASSERT_FALSE(
        dataset_->IsMultiGroup(dataset_->Feature2Group(first_inner)));
    EXPECT_EQ(attempted_mask[first_inner], 1);
    EXPECT_EQ(attempted_mask[ordinary_sibling_inner], 0);

    const int multi_selected_inner = dataset_->InnerFeatureIndex(3);
    const int multi_omitted_inner = dataset_->InnerFeatureIndex(4);
    ASSERT_GE(multi_selected_inner, 0);
    ASSERT_GE(multi_omitted_inner, 0);
    ASSERT_EQ(
        dataset_->Feature2Group(multi_selected_inner),
        dataset_->Feature2Group(multi_omitted_inner));
    ASSERT_TRUE(
        dataset_->IsMultiGroup(
            dataset_->Feature2Group(multi_selected_inner)));
    EXPECT_EQ(attempted_mask[multi_selected_inner], 1);
    EXPECT_EQ(attempted_mask[multi_omitted_inner], 0);

    std::vector<double> sparse_reset_probe(
        dataset_->num_features(), 9.0);
    sampler.ResetAdaptiveRootGains(&sparse_reset_probe);
    for (int inner_feature = 0;
         inner_feature < dataset_->num_features(); ++inner_feature) {
      EXPECT_DOUBLE_EQ(
          sparse_reset_probe[inner_feature],
          attempted_mask[inner_feature] == 1 ? 0.0 : 9.0);
    }

    sampler.PrepareAdaptiveTree();
    EXPECT_TRUE(sampler.AccumulateAdaptiveModel(attempted_gains));
    sampler.AbortAdaptiveIteration();

    EXPECT_EQ(
        sampler.AdaptiveControllerForTesting().round(), committed_round);
    EXPECT_FALSE(sampler.AdaptiveControllerForTesting().awaiting_update());
    EXPECT_EQ(
        sampler.AdaptiveControllerForTesting().total_exclusion_events(),
        committed_exclusion_events);
    EXPECT_EQ(
        sampler.AdaptiveSelectedUnitPositionsForTesting(), committed_features);
    EXPECT_EQ(
        sampler.AdaptiveUsedFeaturePositionsForTesting(),
        committed_used_features);
    EXPECT_EQ(sampler.is_feature_used_bytree(), committed_mask);
    EXPECT_EQ(sampler.AdaptiveFinalReceiptLine(), committed_receipt);
    EXPECT_EQ(sampler.AdaptiveControllerForTesting().empirical_variance(),
              committed_variance);

    std::vector<double> rollback_reset_probe(
        dataset_->num_features(), 9.0);
    sampler.ResetAdaptiveRootGains(&rollback_reset_probe);
    for (int inner_feature = 0;
         inner_feature < dataset_->num_features(); ++inner_feature) {
      EXPECT_DOUBLE_EQ(
          rollback_reset_probe[inner_feature],
          committed_mask[inner_feature] == 1 ? 0.0 : 9.0);
    }

    sampler.BeginAdaptiveIteration(1);
    EXPECT_EQ(
        sampler.AdaptiveSelectedUnitPositionsForTesting(), attempted_features);
    EXPECT_EQ(
        sampler.AdaptiveUsedFeaturePositionsForTesting(),
        attempted_used_features);
    EXPECT_EQ(sampler.is_feature_used_bytree(), attempted_mask);
    std::vector<double> retry_reset_probe(
        dataset_->num_features(), 9.0);
    sampler.ResetAdaptiveRootGains(&retry_reset_probe);
    for (int inner_feature = 0;
         inner_feature < dataset_->num_features(); ++inner_feature) {
      EXPECT_DOUBLE_EQ(
          retry_reset_probe[inner_feature],
          attempted_mask[inner_feature] == 1 ? 0.0 : 9.0);
    }
    sampler.PrepareAdaptiveTree();
    EXPECT_TRUE(sampler.AccumulateAdaptiveModel(attempted_gains));
    sampler.EndAdaptiveIteration();

    int retained = 0;
    for (int position : attempted_features) {
      retained += static_cast<int>(
          std::find(
              committed_features.begin(), committed_features.end(),
              position) != committed_features.end());
    }
    const int expected_exclusions =
        static_cast<int>(committed_features.size()) - retained;
    EXPECT_EQ(
        sampler.AdaptiveControllerForTesting().round(),
        committed_round + 1);
    EXPECT_EQ(
        sampler.AdaptiveControllerForTesting()
            .last_committed_newly_excluded_count(),
        expected_exclusions);
    EXPECT_EQ(
        sampler.AdaptiveControllerForTesting().total_exclusion_events(),
        committed_exclusion_events +
            static_cast<uint64_t>(expected_exclusions));
  }
  EXPECT_TRUE(observed_changed_proposal);
}

}  // namespace
#endif
