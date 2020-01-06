# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Added
- Add support for shared steps ([PR #19](https://github.com/alegonz/baikal/pull/19)). Now steps can be called several times on different inputs.
    - **This is a backwards-incompatible change.** The outputs of the steps now follow the following format: `step_name/port/output_number`.
      (Previously it was `step_name/output_number`)
- Add option to include targets in `plot_model` ([PR #20](https://github.com/alegonz/baikal/pull/20)).

### Changed
- Move `compute_func` (previously `function`) and `trainable` args to `Step.__call__` ([PR #18](https://github.com/alegonz/baikal/pull/18)).
    - **This is a backwards-incompatible change.**

### Fixed
- Specify `scikit-learn` instead of `sklearn` in package dependencies
- Add clarification in that steps must be named in `build_fn` when using `SKLearnWrapper`
- Fix bug where the compute function was not being transferred when replacing a step in `Model.set_params`.
- Fix some bugs in `plot_model` ([PR #20](https://github.com/alegonz/baikal/pull/20)).

## [0.2.0] - 2019-11-16
### Added
- This CHANGELOG file.
- Introduced new targets API ([PR #1](https://github.com/alegonz/baikal/pull/1)).
    - Steps now take an optional `targets` argument at call time to specify inputs for target data at fit time.
    - Correspondingly, `Model` also takes an additional argument for these targets.
    - The `extra_targets` argument in `Model.fit` was removed.
- Step enhancements
    - `make_step` factory function to ease definition of steps.
    - Added support for function arguments to Lambda step ([(PR #8)](https://github.com/alegonz/baikal/pull/8)).
    - Added new Split step ([(PR #9)](https://github.com/alegonz/baikal/pull/9)).

### Fixed
- Some refactoring and minor fixes.
- Bug fixes ([(PR #6)](https://github.com/alegonz/baikal/pull/6), [(PR #7)](https://github.com/alegonz/baikal/pull/7))

## [0.1.0] - 2019-06-01
### Added
- Everything. This is the first (pre-release) version.

[Unreleased]: https://github.com/alegonz/baikal/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/alegonz/baikal/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/alegonz/baikal/releases/tag/v0.1.0

<!---
Release diff tags are written as in the example below:
[0.2.0]: https://github.com/alegonz/baikal/compare/v0.1.0...v0.2.0
-->
