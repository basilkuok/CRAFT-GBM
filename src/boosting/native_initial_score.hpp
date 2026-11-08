/*!
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_BOOSTING_NATIVE_INITIAL_SCORE_HPP_
#define LIGHTGBM_SRC_BOOSTING_NATIVE_INITIAL_SCORE_HPP_

#include <LightGBM/config.h>
#include <LightGBM/dataset.h>
#include <LightGBM/network.h>
#include <LightGBM/objective_function.h>

#include <cmath>

namespace LightGBM {

// The objective owns the definition of its automatic initial raw score. Keep
// this separate from the initial-score policy: the no-split fallback also uses
// the automatic score when boost_from_average is disabled.
inline double ObtainAutomaticInitialScore(const ObjectiveFunction* objective, int class_id) {
  double init_score = 0.0;
  if (objective != nullptr) {
    init_score = objective->BoostFromScore(class_id);
  }
  if (Network::num_machines() > 1) {
    init_score = Network::GlobalSyncUpByMean(init_score);
  }
  return init_score;
}

inline bool ShouldBoostFromAverage(const Config& config, const Dataset* train_data) {
  return config.boost_from_average || (train_data != nullptr && train_data->num_features() == 0);
}

// Raw score for a fresh fit without Dataset init_score. This is shared by
// GBDT and the read-only Dataset query; score-cache updates stay in GBDT.
inline double ObtainNativeInitialScore(const Config& config, const Dataset* train_data,
                                       const ObjectiveFunction* objective, int class_id) {
  if (objective != nullptr && ShouldBoostFromAverage(config, train_data)) {
    const double init_score = ObtainAutomaticInitialScore(objective, class_id);
    if (std::fabs(init_score) > kEpsilon) {
      return init_score;
    }
  }
  return 0.0;
}

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_BOOSTING_NATIVE_INITIAL_SCORE_HPP_
