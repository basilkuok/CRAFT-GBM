/*!
 * Copyright (c) 2026 The LightGBM developers.
 * Licensed under the MIT License. See LICENSE in the project root.
 *
 * Gamma-exponential mixture formula adapted from confseq's
 * src/confseq/uniform_boundaries.h, revision
 * 5ffe733ca2447a2e28c2c91f3b00086173f2ab2c:
 * https://github.com/gostevehoward/confseq
 *
 * Copyright (c) 2021 confseq developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef LIGHTGBM_TREELEARNER_ADAPTIVE_FEATURE_EB_HPP_
#define LIGHTGBM_TREELEARNER_ADAPTIVE_FEATURE_EB_HPP_

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/log1p.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace LightGBM {

// Howard et al. (2021), Theorem 4 and Proposition 9, with range c = 1.
// Parameters are frozen before the confidence history starts. Each call returns
// a numerator; the caller divides by its selected-observation count. This helper
// controls ONE tail and must receive log(2 * number_of_units / delta).
//
// Evaluation uses established Boost.Math special functions, long double, a
// cancellation-resistant formula, an inflated log threshold, the upper root
// bracket, and upward rounding of empirical variance. The numerical guard is
// checked against independent high-precision references; this is not a formal
// interval-arithmetic certification of Boost or the platform math library.
// Outside the tested numerical domain or on a numerical failure we return
// infinity, yielding the full [0, 1] interval, and increment a diagnostic count.
// This conservative failure policy does not impose a statistical time horizon.
// The mutable shared cache is used by the controller's serial mask computation.
class GammaExponentialMixtureBoundary {
 public:
  static constexpr int kGridBins = 128;

  GammaExponentialMixtureBoundary() = default;

  GammaExponentialMixtureBoundary(double log_inverse_alpha, double rho) {
    Reset(log_inverse_alpha, rho);
  }

  void Reset(double log_inverse_alpha, double rho) {
    if (!std::isfinite(log_inverse_alpha) || log_inverse_alpha <= 0.0 ||
        !std::isfinite(rho) || rho <= 0.0) {
      throw std::invalid_argument("Invalid gamma-exponential mixture parameters");
    }
    log_inverse_alpha_ = log_inverse_alpha;
    rho_ = rho;
    initialized_ = true;
    ClearCache();
    // This initialization may fail on unsupported special-function inputs.
    normalization_valid_ = false;
    if (rho >= 1e-12 && rho <= 1e12 && log_inverse_alpha <= 1e6) {
      try {
        const long double r = rho;
        log_gamma_probability_rho_ = LogGammaProbability(r, r);
        stirling_rho_ = StirlingRemainder(r);
        normalization_valid_ = std::isfinite(log_gamma_probability_rho_) &&
                               std::isfinite(stirling_rho_);
      } catch (const std::exception&) {
        normalization_valid_ = false;
      }
    }
  }

  // Proposition 3's exact NORMAL-mixture calibration, used as the fixed
  // approximation suggested in Section 3.5 for the gamma mixture. Pass
  // log(number_of_units / delta) directly, before adding log(2) for the boundary;
  // subtracting log(2) afterward loses precision when units=1 and delta is near 1.
  // reference_count / 4 is a proxy for
  // intrinsic time; it is NOT an upper bound on observed empirical variance.
  static double CalibratedRho(double log_feature_inverse_alpha,
                              double reference_count) {
    const long double a = log_feature_inverse_alpha;
    if (!std::isfinite(log_feature_inverse_alpha) || a <= 0.0L ||
        !std::isfinite(reference_count) || reference_count <= 0.0) {
      throw std::invalid_argument("Invalid gamma-exponential calibration inputs");
    }
    const long double target = 2.0L * a;
    long double lower = 0.0L;
    long double upper = std::max(1.0L, 2.0L * target);
    while (-boost::math::log1pmx(upper) < target) {
      upper *= 2.0L;
    }
    for (int iteration = 0; iteration < 160; ++iteration) {
      const long double middle = lower + (upper - lower) / 2.0L;
      if (middle == lower || middle == upper) {
        break;
      }
      if (-boost::math::log1pmx(middle) < target) {
        lower = middle;
      } else {
        upper = middle;
      }
    }
    return static_cast<double>(reference_count / (4.0L * upper));
  }

  double Boundary(double variance) const {
    RequireInitialized();
    CheckVariance(variance);
    if (!normalization_valid_ || !std::isfinite(variance) || variance > 1e12) {
      ++numerical_failure_count_;
      return Infinity();
    }
    const GridPoint grid = UpperGridPoint(variance);
    if (cache_.size() <= grid.index) {
      cache_.resize(grid.index + 1, std::numeric_limits<double>::quiet_NaN());
    }
    if (std::isnan(cache_[grid.index])) {
      cache_[grid.index] = UncachedBoundary(grid.variance);
      ++cached_boundary_count_;
    } else {
      ++cache_hits_;
    }
    return cache_[grid.index];
  }

  // Exposed for focused numerical tests and independent reference checks. The
  // production controller uses Boundary(), which also applies the shared grid.
  double UncachedBoundary(double variance) const {
    RequireInitialized();
    CheckVariance(variance);
    if (!normalization_valid_ || !std::isfinite(variance) || variance > 1e12) {
      ++numerical_failure_count_;
      return Infinity();
    }
    try {
      const long double v = variance;
      const long double threshold = log_inverse_alpha_ +
          1e-9L * (1.0L + log_inverse_alpha_);
      long double lower = 0.0L;
      long double upper = std::sqrt((v + rho_) *
          (2.0L * threshold + std::log1p(v / rho_) + 2.0L)) +
          2.0L * (threshold + 1.0L);
      bool bracketed = false;
      for (int iteration = 0; iteration < 128; ++iteration) {
        const long double value = LogMixtureImpl(upper, v);
        if (!std::isfinite(value)) {
          break;
        }
        if (value >= threshold) {
          bracketed = true;
          break;
        }
        upper *= 2.0L;
      }
      if (!bracketed) {
        ++numerical_failure_count_;
        return Infinity();
      }
      for (int iteration = 0; iteration < 100; ++iteration) {
        const long double middle = lower + (upper - lower) / 2.0L;
        if (upper - lower <= 1e-13L * (1.0L + upper)) {
          break;
        }
        const long double value = LogMixtureImpl(middle, v);
        if (!std::isfinite(value)) {
          ++numerical_failure_count_;
          return Infinity();
        }
        if (value < threshold) {
          lower = middle;
        } else {
          upper = middle;
        }
      }
      // Conversion to double can round down: move one representable value up.
      return std::nextafter(static_cast<double>(upper), Infinity());
    } catch (const std::exception&) {
      ++numerical_failure_count_;
      return Infinity();
    }
  }

  double LogMixture(double s, double variance) const {
    RequireInitialized();
    CheckVariance(variance);
    if (!normalization_valid_ || !std::isfinite(s) || s < 0.0 ||
        !std::isfinite(variance) || variance > 1e12) {
      throw std::invalid_argument("Unsupported gamma-exponential diagnostic input");
    }
    return static_cast<double>(LogMixtureImpl(s, variance));
  }

  static double RoundedVariance(double variance) {
    CheckVariance(variance);
    if (!std::isfinite(variance) || variance > 1e12) {
      return Infinity();
    }
    return UpperGridPoint(variance).variance;
  }

  void ClearCache() const {
    cache_.clear();
    cached_boundary_count_ = 0;
    cache_hits_ = 0;
    numerical_failure_count_ = 0;
  }

  double rho() const { return rho_; }
  double log_inverse_alpha() const { return log_inverse_alpha_; }
  int grid_bins() const { return kGridBins; }
  size_t cached_boundary_count() const { return cached_boundary_count_; }
  size_t cache_hits() const { return cache_hits_; }
  size_t numerical_failure_count() const { return numerical_failure_count_; }

 private:
  struct GridPoint {
    size_t index;
    double variance;
  };

  static double Infinity() { return std::numeric_limits<double>::infinity(); }

  static void CheckVariance(double variance) {
    if (std::isnan(variance) || variance < 0.0) {
      throw std::invalid_argument("Empirical variance must be nonnegative");
    }
  }

  void RequireInitialized() const {
    if (!initialized_) {
      throw std::logic_error("Gamma-exponential boundary has not been initialized");
    }
  }

  static GridPoint UpperGridPoint(double variance) {
    int exponent = 0;
    const long double x = 1.0L + variance;
    std::frexp(x, &exponent);
    --exponent;
    const long double step = std::ldexp(1.0L, exponent) / kGridBins;
    size_t bin = static_cast<size_t>(std::ceil(x / step)) - kGridBins;
    double rounded = static_cast<double>((kGridBins + bin) * step - 1.0L);
    // Important when positive variance is too small to change 1 + variance.
    if (rounded < variance) {
      ++bin;
      rounded = static_cast<double>((kGridBins + bin) * step - 1.0L);
    }
    return {static_cast<size_t>(exponent) * kGridBins + bin, rounded};
  }

  static long double LogGammaProbability(long double a, long double x) {
    const long double probability = boost::math::gamma_p(a, x);
    if (!(probability > 0.0L && probability <= 1.0L)) {
      throw std::runtime_error("Invalid regularized incomplete gamma result");
    }
    return std::log(probability);
  }

  // R(z) = log Gamma(z) - (z - 1/2) log(z) + z - log(sqrt(2*pi)).
  // The alternating Stirling expansion avoids catastrophic cancellation for
  // large shapes. At z >= 32, the first omitted term is below 2e-22.
  static long double StirlingRemainder(long double z) {
    if (z < 32.0L) {
      return boost::math::lgamma(z) - (z - 0.5L) * std::log(z) + z -
          0.918938533204672741780329736405617639861L;
    }
    const long double inverse = 1.0L / z;
    const long double square = inverse * inverse;
    return inverse * (1.0L / 12.0L + square * (-1.0L / 360.0L +
        square * (1.0L / 1260.0L + square * (-1.0L / 1680.0L +
        square * (1.0L / 1188.0L - square * 691.0L / 360360.0L)))));
  }

  long double LogMixtureImpl(long double s, long double variance) const {
    const long double shape = variance + rho_;
    return -shape * boost::math::log1pmx(s / shape) -
        0.5L * std::log1p(variance / rho_) +
        StirlingRemainder(shape) - stirling_rho_ +
        LogGammaProbability(shape, shape + s) - log_gamma_probability_rho_;
  }

  bool initialized_ = false;
  bool normalization_valid_ = false;
  double log_inverse_alpha_ = std::numeric_limits<double>::quiet_NaN();
  double rho_ = std::numeric_limits<double>::quiet_NaN();
  long double log_gamma_probability_rho_ = 0.0L;
  long double stirling_rho_ = 0.0L;
  mutable std::vector<double> cache_;
  mutable size_t cached_boundary_count_ = 0;
  mutable size_t cache_hits_ = 0;
  mutable size_t numerical_failure_count_ = 0;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_TREELEARNER_ADAPTIVE_FEATURE_EB_HPP_
