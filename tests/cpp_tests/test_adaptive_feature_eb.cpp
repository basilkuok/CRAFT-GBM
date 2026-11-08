/*!
 * Copyright (c) 2026 The LightGBM developers.
 * Licensed under the MIT License. See LICENSE in the project root.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "../../src/treelearner/adaptive_feature_eb.hpp"

namespace {

using LightGBM::GammaExponentialMixtureBoundary;

TEST(GammaExponentialMixtureTest, UpperRootsMatchIndependentMultiprecision) {
  // Independently generated with mpmath at 65 decimal digits, using the
  // original Proposition 9 formula (not the stable rearrangement in C++).
  // Inputs are the exact binary64 numbers written below. Each reference root
  // used 230 bisection iterations. See empirical_bernstein_reference.py.
  const double references[][4] = {
      {12.206072645530174, 2.8480311531867977, 0, 17.220121654989391},
      {12.206072645530174, 2.8480311531867977, 0.5, 17.937172439271507},
      {12.206072645530174, 2.8480311531867977, 1, 18.604672972537639},
      {12.206072645530174, 2.8480311531867977, 5, 22.902221609914047},
      {12.206072645530174, 2.8480311531867977, 25, 35.879778270684973},
      {12.206072645530174, 2.8480311531867977, 75, 54.872203069296972},
      {12.206072645530174, 2.8480311531867977, 10000, 572.2554148418368},
      {3.6888794541139363, 9.7949999999999999, 10, 14.060189878506035},
      {0.69314718055994529, 1, 0, 1.110604056747859},
      {50, 0.01, 0.10000000000000001, 52.78835608918741},
      {700, 0.050000000000000003, 100, 938.53818954105952},
      {12.206072645530174, 50, 31, 51.896609470660863},
  };
  for (const auto& row : references) {
    SCOPED_TRACE(::testing::Message() << "L=" << row[0]
        << " rho=" << row[1] << " V=" << row[2]);
    GammaExponentialMixtureBoundary boundary(row[0], row[1]);
    const double upper = boundary.UncachedBoundary(row[2]);
    EXPECT_GE(upper, row[3]);
    EXPECT_LE(upper - row[3], 2e-8 * (1.0 + row[3]));
    EXPECT_GT(boundary.LogMixture(upper, row[2]), row[0]);
    EXPECT_EQ(boundary.numerical_failure_count(), 0U);
  }
}

TEST(GammaExponentialMixtureTest, FixedCalibrationUsesDoubledOneSidedAlpha) {
  EXPECT_NEAR(GammaExponentialMixtureBoundary::CalibratedRho(
      std::log(100000.0), 300.0), 2.8480311531867977, 2e-14);
  EXPECT_NEAR(GammaExponentialMixtureBoundary::CalibratedRho(
      std::log(56100.0), 300.0), 2.984126795365557, 2e-14);
  const double log_feature_alpha = std::log(20.0);
  const double rho = GammaExponentialMixtureBoundary::CalibratedRho(
      log_feature_alpha, 300.0);
  const double c = 75.0 / rho;
  EXPECT_NEAR(c - std::log1p(c), 2.0 * std::log(20.0), 2e-14);
  // Independently checked at 70 decimal digits with mpmath. Forming the
  // one-sided log first and subtracting log(2) corrupts this tiny calibration a.
  const double almost_one = std::nextafter(1.0, 0.0);
  EXPECT_NEAR(GammaExponentialMixtureBoundary::CalibratedRho(
      -std::log(almost_one), 300.0), 3558984935.9094334, 1e-6);
}

TEST(GammaExponentialMixtureTest, VarianceGridAlwaysRoundsUpIncludingTinyValues) {
  const double infinity = std::numeric_limits<double>::infinity();
  const double inputs[] = {
      0.0, std::numeric_limits<double>::denorm_min(), 1e-25, 1e-10,
      std::nextafter(1.0 / 128.0, 0.0), 1.0 / 128.0,
      std::nextafter(1.0 / 128.0, infinity), 0.4, 0.5,
      std::nextafter(1.0, 0.0), 1.0, std::nextafter(1.0, infinity),
      3.0, 7.0, 31.0, 99.9, 10000.0, 1e12};
  double previous = 0.0;
  for (double variance : inputs) {
    const double rounded = GammaExponentialMixtureBoundary::RoundedVariance(variance);
    EXPECT_GE(rounded, variance);
    EXPECT_GE(rounded, previous);
    EXPECT_LE(rounded - variance, (1.0 + variance) / 128.0);
    previous = rounded;
  }
  EXPECT_EQ(GammaExponentialMixtureBoundary::RoundedVariance(0.0), 0.0);
  EXPECT_EQ(GammaExponentialMixtureBoundary::RoundedVariance(0.5), 0.5);
}

TEST(GammaExponentialMixtureTest, SharedCacheReusesNodesAndResetDiscardsParameters) {
  GammaExponentialMixtureBoundary boundary(std::log(200000.0), 2.8480311531867977);
  const double first = boundary.Boundary(0.4);
  EXPECT_DOUBLE_EQ(boundary.Boundary(0.401), first);
  EXPECT_DOUBLE_EQ(boundary.Boundary(0.405), first);
  EXPECT_DOUBLE_EQ(boundary.Boundary(0.40625), first);
  EXPECT_EQ(boundary.cached_boundary_count(), 1U);
  EXPECT_EQ(boundary.cache_hits(), 3U);
  EXPECT_GE(first, boundary.UncachedBoundary(0.4));
  const double next = boundary.Boundary(std::nextafter(
      0.40625, std::numeric_limits<double>::infinity()));
  EXPECT_GT(next, first);
  EXPECT_EQ(boundary.cached_boundary_count(), 2U);
  boundary.Reset(std::log(400000.0), 2.0);
  EXPECT_EQ(boundary.cached_boundary_count(), 0U);
  EXPECT_EQ(boundary.cache_hits(), 0U);
  EXPECT_NE(boundary.Boundary(0.4), first);
  EXPECT_EQ(boundary.numerical_failure_count(), 0U);
}

TEST(GammaExponentialMixtureTest, StableLargeShapesAndTinyErrorProbabilities) {
  const double shapes[] = {1e-6, 0.05, 2.8480311531867977, 32.0, 1e6};
  const double variances[] = {0.0, 1.0, 31.0, 1e4, 1e8};
  for (double rho : shapes) {
    GammaExponentialMixtureBoundary boundary(700.0, rho);
    EXPECT_NEAR(boundary.LogMixture(0.0, 0.0), 0.0, 1e-13);
    double previous = 0.0;
    for (double variance : variances) {
      SCOPED_TRACE(::testing::Message() << "rho=" << rho << " V=" << variance);
      const double upper = boundary.UncachedBoundary(variance);
      EXPECT_TRUE(std::isfinite(upper));
      EXPECT_GT(boundary.LogMixture(upper, variance), 700.0);
      EXPECT_GE(upper, previous);
      previous = upper;
    }
    EXPECT_EQ(boundary.numerical_failure_count(), 0U);
  }
}

TEST(GammaExponentialMixtureTest, ExtremeSpecialFunctionFailureCannotNarrowInterval) {
  // Some Boost versions exceed their iteration limit for shape near 1e12.
  // Such a failure must be counted and return infinity; versions which can
  // evaluate this shape must return an upper root of the requested threshold.
  for (double rho : {0.05, 1e12}) {
    GammaExponentialMixtureBoundary boundary(700.0, rho);
    const double upper = boundary.UncachedBoundary(1e12);
    if (std::isinf(upper)) {
      EXPECT_GT(boundary.numerical_failure_count(), 0U);
    } else {
      ASSERT_TRUE(std::isfinite(upper));
      EXPECT_GT(boundary.LogMixture(upper, 1e12), 700.0);
      EXPECT_EQ(boundary.numerical_failure_count(), 0U);
    }
  }
}

TEST(GammaExponentialMixtureTest, NumericalFailuresAreConservativeAndCounted) {
  GammaExponentialMixtureBoundary boundary(12.0, 3.0);
  EXPECT_TRUE(std::isinf(boundary.Boundary(1e13)));
  EXPECT_TRUE(std::isinf(boundary.Boundary(std::numeric_limits<double>::infinity())));
  EXPECT_EQ(boundary.numerical_failure_count(), 2U);
  boundary.Reset(12.0, 1e13);
  EXPECT_TRUE(std::isinf(boundary.Boundary(0.5)));
  EXPECT_EQ(boundary.numerical_failure_count(), 1U);
  boundary.Reset(12.0, 3.0);
  EXPECT_TRUE(std::isfinite(boundary.Boundary(0.5)));
  EXPECT_EQ(boundary.numerical_failure_count(), 0U);
}

TEST(GammaExponentialMixtureTest, CorruptInputsFailExplicitly) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  GammaExponentialMixtureBoundary uninitialized;
  EXPECT_THROW(uninitialized.Boundary(0.0), std::logic_error);
  EXPECT_THROW(uninitialized.Reset(nan, 1.0), std::invalid_argument);
  EXPECT_THROW(uninitialized.Reset(12.0, 0.0), std::invalid_argument);
  GammaExponentialMixtureBoundary boundary(12.0, 3.0);
  EXPECT_THROW(boundary.Boundary(-1.0), std::invalid_argument);
  EXPECT_THROW(boundary.Boundary(nan), std::invalid_argument);
  EXPECT_THROW(GammaExponentialMixtureBoundary::CalibratedRho(0.0, 300.0),
               std::invalid_argument);
  EXPECT_THROW(GammaExponentialMixtureBoundary::CalibratedRho(12.0, -1.0),
               std::invalid_argument);
}

}  // namespace
