# CRAFT-GBM

CRAFT-GBM (Confidence-Ranked Adaptive Feature Tracking) is the native CPU
LightGBM extension accompanying the paper
[**CRAFT-GBM: Selecting Best Features for Gradient-Boosted Machine Learning**](https://arxiv.org/abs/2611.11914),
by Basil Kuok, Yi Zhou, Yao Xuan, Shuping Li, and Ke-Wei Huang.

This repository releases the CRAFT-GBM model source code for direct use by the
research community and the public. Feedback and pull requests are welcome.

## Build and run

The supported reproduction environment is Linux with Python 3.12, a C++17
compiler with OpenMP, CMake 3.28 or newer, and Boost.Math headers. GCC 13.3.0 and
Boost 1.82.0 were used for release verification. Eigen, fmt, fast_double_parser,
and nanoarrow source dependencies are already included; no Git submodules are
needed. A standard Linux development installation with `build-essential`,
`cmake`, and `libboost-dev` supplies the native prerequisites when its CMake
version is recent enough.

Python must include `venv` and `ensurepip`. If environment creation reports that
`ensurepip` is unavailable on Debian/Ubuntu, install the matching
`python3.12-venv` system package before continuing.

```bash
git clone https://github.com/basilkuok/CRAFT-GBM.git
cd CRAFT-GBM
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_install.py --preflight
python scripts/build_native.py --jobs 4
python -m pip install --no-deps -e .
python scripts/check_install.py
python examples/train_craft.py --mode endogenous
python examples/train_craft.py --mode exogenous
```

If Boost is outside the compiler's normal search path, add
`--boost-root /path/to/boost` to the build command. The directory may contain
`include/boost` or `boost` directly. A compatible C++ runtime must be available
when loading the library; an older Conda `libstdc++` must not shadow the compiler's
runtime.
An error such as `GLIBCXX_3.4.32 not found` identifies this runtime mismatch.
Use a Python installation that loads the compiler's compatible C++ runtime, or
update the runtime in your Conda environment to match the compiler. The setup
tools do not change system libraries or global environment settings.

The build helper compiles the CLI and shared library and places the library and
version file where the bundled Python bindings expect them. The editable install
then makes this checkout importable from other working directories without
setting `PYTHONPATH`. Build before installing, keep the checkout in place, and
rerun the native build after changing native sources. This installation path is
for the local source checkout; it does not supply a prebuilt binary wheel.

Use a dedicated environment: ordinary PyPI `lightgbm` and CRAFT-GBM share the
`lightgbm` import name and should not be installed together. Installing ordinary
`lightgbm` does not install CRAFT-GBM. If the diagnostic reports a conflicting
package, create a fresh environment and follow the commands above. Existing
parameter and serialization names are retained for compatibility.

The installation diagnostic reports the interpreter, Python package, native
library, and CRAFT-specific capabilities. For notebooks, select the kernel using
this environment's Python; restart an existing kernel after installation. If a
library fails to load, check the reported path and the C++ runtime note above.
Linux with Python 3.12 is the verified environment; this release does not claim
Windows, macOS, or other Python-version compatibility.

## Train on your own data

Import the public helper from `lightgbm.craft`. It uses a LightGBM parameter
dictionary and `lgb.Dataset`, enables adaptive feature selection and column-wise
training, and returns a `CraftResult`. For binary classification, with your
training arrays `X_train`, `y_train` and prediction array `X_test`:

```python
import lightgbm as lgb
from lightgbm.craft import train_craft

# Endogenous: the training history determines the feature count and stopping.
result = train_craft(
    {"objective": "binary", "feature_fraction": 1.0},
    lgb.Dataset(X_train, label=y_train),
)
probabilities = result.booster.predict(X_test)
result.booster.save_model("craft_endogenous.txt")
print(result.stop_reason, result.booster.current_iteration())
```

For an explicit feature fraction and fixed training horizon:

```python
import lightgbm as lgb
from lightgbm.craft import train_craft

result = train_craft(
    {"objective": "binary", "feature_fraction": 0.5},
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=100,
)
probabilities = result.booster.predict(X_test)
result.booster.save_model("craft_exogenous.txt")
```

`result.booster` is the trained LightGBM booster. `result.training_losses`
contains the native initial training loss followed by one loss per completed
round. `result.stop_reason` is `training_loss_patience` or `fixed_horizon`.
Reload a saved model for prediction with `lgb.Booster(model_file="craft_endogenous.txt")`.
Use `num_threads` in the parameter dictionary to choose the CPU allocation; the
helper defaults to two threads.

The helper supports these canonical objective/metric pairs. Omit `metric` to
select the corresponding metric automatically; for multiclass classification,
also set `num_class`.

| `objective` | Training-loss metric |
|---|---|
| `binary` | `binary_logloss` |
| `multiclass` | `multi_logloss` |
| `regression` | `rmse` |

Use the canonical `feature_fraction` parameter and the `num_boost_round` function
argument rather than their aliases. Endogenous training rejects an external round
limit; exogenous training requires a positive integer `num_boost_round`. The
helper's supported objectives are narrower than the native selector's scope.

## Selection modes and stopping

Feature selection must be enabled with `adaptive_feature_selection=true`.
The public helper enables it automatically; the inherited native default is
`false`. With selection enabled, `feature_fraction=1.0` means endogenous
selection, not full-feature LightGBM throughout training.

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
`lightgbm.craft.train_craft` supplies the RQ1 controller: evaluate complete-training
loss from the native initial predictor, use a fixed improvement threshold of
`0.001 * initial_loss` and patience 10, initialize the anchor after round 2,
and return the completed model at the patience stop. Endogenous training receives
no external round cap. The 10,002-round bound is an integrity check, not a tuning
parameter. Exogenous examples instead use an explicit fixed round count.

The supplied demonstration explicitly sets the calibration reference count to
300 for endogenous mode and 1000 for exogenous mode, following the respective
paper experiments. Choosing exogenous mode alone does not change the native
default of 300; the own-data examples above leave that default unchanged.

The inherited `train_with_validation_stopping` API has different semantics: it
uses validation loss and returns its best prefix. It is retained for compatibility
and must not be substituted for the paper's RQ1 controller. Test or validation
labels do not enter the helper's endogenous stopping decisions.

## Supported scope

The selector supports fresh, continuous, single-machine CPU GBDT fits with native
objectives, the serial tree learner, `force_col_wise=true`, native EFB enabled,
`feature_fraction_bynode=1.0`, and native GOSS or disabled bagging. The helper
enables adaptive selection and forced column-wise training; the remaining
supported options use native defaults. Unsupported selector configurations include GPU and
distributed training, resumed training, custom objectives, native early stopping,
linear trees, quantized gradients, extra trees, feature constraints, and forced
splits. R, Java, and GPU builds are not release targets.

The statistical target is the conditional transformed root gain averaged over
each feature's observed rounds. It is not an instantaneous gain ranking or a
guarantee about downstream test accuracy. Full experimental reproduction also
requires the paper's data, training, timing, and evaluation protocols.

## Targeted checks

After completing the build and editable installation above:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
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

The repository preserves the 803 files of the supplied model-source snapshot without
algorithm changes. `SOURCE_MANIFEST.json` records their relative paths, sizes, and
SHA-256 digests. Standalone documentation, the build helper, example, and tests
are release additions. `RELEASE_VERIFICATION.json` records the checks performed
for the published v1.0.0 release.
`USABILITY_VERIFICATION.json` records the installation checks and exact
before/after helper comparisons for this usability update.

The published CRAFT-GBM release is **v1.0.0**; the editable distribution for this
usability update is **craft-gbm 1.1.0.dev0**, a development version.
`VERSION.txt` and
`lightgbm.__version__` retain the upstream LightGBM base version **4.7.0**; they are
not the CRAFT-GBM release identifier. Use the release tag when citing or reproducing
this source snapshot. See `CITATION.cff` for the published software citation and
the accompanying arXiv paper. Record the Git commit when using a development
checkout.

## License and attribution

The root `LICENSE` retains the LightGBM MIT license. Third-party code retains its
own notices: Eigen (`external_libs/eigen/COPYING.*`), fmt
(`external_libs/fmt/LICENSE`), fast_double_parser
(`external_libs/fast_double_parser/LICENSE*`), and nanoarrow
(`external_libs/nanoarrow/LICENSE.txt` and `NOTICE.txt`). The empirical-Bernstein
implementation also retains the confseq MIT notice and source revision in
`src/treelearner/adaptive_feature_eb.hpp`.
