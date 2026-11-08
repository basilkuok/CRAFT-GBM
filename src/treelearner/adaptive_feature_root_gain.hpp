/*!
 * Copyright (c) 2026-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_ADAPTIVE_FEATURE_ROOT_GAIN_HPP_
#define LIGHTGBM_SRC_TREELEARNER_ADAPTIVE_FEATURE_ROOT_GAIN_HPP_

#include <LightGBM/utils/log.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace LightGBM {

// Bounded transform for JAEGER's feature root-split opportunity. Each feature
// uses its native root gain. S_{m,t} is the median strictly positive
// selected-feature gain from the prior committed iteration. If
// no positive gain is observed, the last positive scale is carried forward.
class AdaptiveFeatureRootGain {
 public:
  static bool IsInitialRootSearch(
      int num_leaves, int smaller_leaf, int larger_leaf) {
    return num_leaves == 1 && smaller_leaf == 0 && larger_leaf < 0;
  }

  static double NormalizeOpportunity(
      double gain, double scale, int canonical_unit) {
    if (!std::isfinite(gain) || gain < 0.0) {
      Log::Fatal(
          "Adaptive selectable-unit selection received invalid root gain "
          "%.17g for canonical unit %d.", gain, canonical_unit);
    }
    if (!std::isfinite(scale) || scale < 0.0) {
      Log::Fatal(
          "Adaptive selectable-unit selection received invalid lagged gain "
          "scale %.17g for canonical unit %d.", scale,
          canonical_unit);
    }
    if (gain == 0.0) {
      return 0.0;
    }
    if (scale == 0.0) {
      return 1.0;
    }
    const double result =
        gain <= scale
            ? (gain / scale) / (1.0 + gain / scale)
            : 1.0 / (1.0 + scale / gain);
    CHECK(std::isfinite(result));
    CHECK_GE(result, 0.0);
    CHECK_LE(result, 1.0);
    return result;
  }

  static double PositiveMedianInPlace(
      std::vector<double>* values,
      size_t* positive_count = nullptr) {
    CHECK_NOTNULL(values);
    size_t write = 0;
    for (double value : *values) {
      if (!std::isfinite(value) || value < 0.0) {
        Log::Fatal(
            "Adaptive selectable-unit selection received invalid root "
            "gain %.17g.", value);
      }
      if (value > 0.0) {
        (*values)[write++] = value;
      }
    }
    values->resize(write);
    if (positive_count != nullptr) {
      *positive_count = write;
    }
    if (write == 0) {
      return 0.0;
    }
    const size_t upper = write / 2;
    std::nth_element(
        values->begin(), values->begin() + upper, values->end());
    const double upper_value = (*values)[upper];
    const double lower_value =
        write % 2 == 1
            ? upper_value
            : *std::max_element(
                  values->begin(), values->begin() + upper);
    const double result =
        write % 2 == 1
            ? upper_value
            : lower_value + (upper_value - lower_value) * 0.5;
    CHECK(std::isfinite(result));
    CHECK_GT(result, 0.0);
    return result;
  }
};

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_TREELEARNER_ADAPTIVE_FEATURE_ROOT_GAIN_HPP_
