# CRAFT-GBM

CRAFT-GBM (Confidence-Ranked Adaptive Feature Tracking) is the native CPU
LightGBM extension accompanying **CRAFT-GBM: Selecting Best Features for
Gradient-Boosted Machine Learning**, by Basil Kuok, Yi Zhou, Yao Xuan,
Shuping Li, and Ke-Wei Huang.

This repository releases the model implementation and a small executable example.
It does not contain the full RQ1/RQ2 benchmark suite, competing methods, datasets,
or measured results; those belong to the separate supplementary code package.
The examples verify model operation, not the paper's aggregate performance claims.

## Build and run

The supported reproduction environment is Linux with Python 3.12, a C++17
compiler with OpenMP, CMake 3.28 or newer, and Boost.Math headers. GCC 13.3.0 and
Boost 1.82.0 were used for release verification. Eigen, fmt, fast_double_parser,
and nanoarrow source dependencies are already included; no Git submodules are
needed. A standard Linux development installation with `build-essential`,
`cmake`, and `libboost-dev` supplies the native prerequisites when its CMake
version is recent enough.

```bash
git clone https://github.com/basilkuok/CRAFT-GBM.git
cd CRAFT-GBM
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/build_native.py --jobs 4
export PYTHONPATH="$PWD/python-package${PYTHONPATH:+:$PYTHONPATH}"
python examples/train_craft.py --mode endogenous
python examples/train_craft.py --mode exogenous
```

If Boost is outside the compiler's normal search path, add
`--boost-root /path/to/boost` to the build command. The directory may contain
`include/boost` or `boost` directly. A compatible C++ runtime must be available
when loading the library; an older Conda `libstdc++` must not shadow the compiler's
runtime.

The helper builds the CLI and shared library, then places the shared library and
version file where the bundled Python bindings expect them. Import this checkout's
`lightgbm` package via `PYTHONPATH`; installing the ordinary PyPI `lightgbm` package
would select a different implementation. The `lightgbm` import name and existing
parameter/serialization names are intentionally retained for compatibility.

## Selection modes and stopping

Feature selection must be enabled with `adaptive_feature_selection=true`.
The inherited native default is `false`.

| Setting | CRAFT-GBM behavior |
|---|---|
| `feature_fraction=1.0` | Endogenous selection: full features through round 20, when reached; from round 21 retain the features whose upper bounds reach the largest lower bound. |
| `0 < feature_fraction < 1.0` | Exogenous selection: full features for rounds 1 and 2; subsequently select the native rounded feature count by confidence-interval midpoint, breaking ties by exposure and canonical feature ID. |
| `adaptive_feature_confidence_delta=0.05` | Default joint error allocation. |
| `adaptive_feature_calibration_reference_count=300` | Default fixed mixture calibration reference; the paper uses 300 in RQ1 and 1000 in RQ2. This is not a training horizon or stopping bound. |

Round 1 initializes the lagged gain scales; confidence observations begin at round
2. Features omitted from a round receive no observations. Native EFB storage is
preserved, and selection precedes the current GOSS draw.

The selector itself does **not** install the paper's stopping rule.
`examples/train_craft.py` supplies the RQ1 controller: evaluate complete-training
loss from the native initial predictor, use a fixed improvement threshold of
`0.001 * initial_loss` and patience 10, initialize the anchor after round 2,
and return the completed model at the patience stop. Endogenous training receives
no external round cap. The 10,002-round bound is an integrity check, not a tuning
parameter. Exogenous examples instead use an explicit fixed round count.

The inherited `train_with_validation_stopping` API has different semantics: it
uses validation loss and returns its best prefix. It is retained for compatibility
and must not be substituted for the paper's RQ1 controller. Test or validation
labels do not enter the example's endogenous stopping decisions.

## Supported scope

The selector supports fresh, continuous, single-machine CPU GBDT fits with native
objectives, the serial tree learner, `force_col_wise=true`, native EFB enabled,
`feature_fraction_bynode=1.0`, and native GOSS or disabled bagging. The example enables adaptive selection and forced column-wise training; the
remaining supported options use native defaults. Unsupported selector configurations include GPU and
distributed training, resumed training, custom objectives, native early stopping,
linear trees, quantized gradients, extra trees, feature constraints, and forced
splits. R, Java, and GPU builds are not release targets.

The statistical target is the conditional transformed root gain averaged over
each feature's observed rounds. It is not an instantaneous gain ranking or a
guarantee about downstream test accuracy. Full experimental reproduction also
requires the paper's data, training, timing, and evaluation protocols.

## Targeted checks

After building and setting `PYTHONPATH` as above:

```bash
python -m unittest discover -s tests -p test_craft.py -v
```

Optional native tests use the supplied development test sources. Google Test
1.14.0 is fetched if a system installation is unavailable:

```bash
python scripts/build_native.py --jobs 4 --with-cpp-tests
./testlightgbm --gtest_filter='GammaExponentialMixtureTest.*:AdaptiveFeature*.*'
```

Only these CRAFT-specific native suites are part of the release verification;
upstream tests requiring datasets outside this repository are not included in
that claim.

## Source identity and version

The release preserves the 803 files of the supplied model-source snapshot without
algorithm changes. `SOURCE_MANIFEST.json` records their relative paths, sizes, and
SHA-256 digests. Standalone documentation, the build helper, example, and tests
are release additions. `RELEASE_VERIFICATION.json` records the checks performed.

The CRAFT-GBM release version is **v1.0.0**. `VERSION.txt` and
`lightgbm.__version__` retain the upstream LightGBM base version **4.7.0**; they are
not the CRAFT-GBM release identifier. Use the release tag when citing or reproducing
this source snapshot. See `CITATION.cff` for the software citation metadata.

## License and attribution

The root `LICENSE` retains the LightGBM MIT license. Third-party code retains its
own notices: Eigen (`external_libs/eigen/COPYING.*`), fmt
(`external_libs/fmt/LICENSE`), fast_double_parser
(`external_libs/fast_double_parser/LICENSE*`), and nanoarrow
(`external_libs/nanoarrow/LICENSE.txt` and `NOTICE.txt`). The empirical-Bernstein
implementation also retains the confseq MIT notice and source revision in
`src/treelearner/adaptive_feature_eb.hpp`.
