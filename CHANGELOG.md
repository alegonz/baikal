# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.3.1] - 2020-04-26
### Fixed
- Fix bug where `get_params` would break when the base class did not implement 
  an `__init__` method ([PR #32]).

## [0.3.0] - 2020-02-23
### Added
- Add support for shared steps ([PR #19](https://github.com/alegonz/baikal/pull/19)). 
  Now steps can be called several times on different inputs.
    - **This is a backwards-incompatible change.** The outputs of the steps now follow 
      the following format: `step_name:port/output_number`.
      (Previously it was `step_name/output_number`)
- Add option to include targets in `plot_model` ([PR #20](https://github.com/alegonz/baikal/pull/20)).
- Add new `fit_compute_func` argument to `Step.__call__` that allows to specify custom 
  behavior at fit time ([PR #22](https://github.com/alegonz/baikal/pull/22)).
- Add documentation built with Sphinx and hosted on [baikal.readthedocs.io](https://baikal.readthedocs.io/en/latest) 
  ([PR #29](https://github.com/alegonz/baikal/pull/29)).

### Changed
- Move `compute_func` (previously `function`) and `trainable` args to `Step.__call__` 
  ([PR #18](https://github.com/alegonz/baikal/pull/18)).
    - Also, the default value is changed from `None` to `"auto"`.
    - **This is a backwards-incompatible change.**
- Raise `RuntimeError` chained with the original exception in `Model.fit` and `Model.predict`. 

### Fixed
- Add clarification in that steps must be named in `build_fn` when using `SKLearnWrapper`
- Fix bug where the compute function was not being transferred when replacing a step in `Model.set_params`.
- Fix an API inconsistency regarding the handling of the arguments of fit/compute for 
  steps with multiple inputs and targets ([PR #21](https://github.com/alegonz/baikal/pull/21)).
- Fix several bugs in `plot_model` (it was largely broken) 
  ([PR #20](https://github.com/alegonz/baikal/pull/20), [PR #24](https://github.com/alegonz/baikal/pull/24)).

## [0.2.0] - 2019-11-16
### Added
- This CHANGELOG file.
- Introduced new targets API ([PR #1](https://github.com/alegonz/baikal/pull/1)).
    - Steps now take an optional `targets` argument at call time to specify inputs for 
      target data at fit time.
    - Correspondingly, `Model` also takes an additional argument for these targets.
    - The `extra_targets` argument in `Model.fit` was removed.
- Step enhancements
    - `make_step` factory function to ease definition of steps.
    - Added support for function arguments to Lambda step ([PR #8](https://github.com/alegonz/baikal/pull/8)).
    - Added new Split step ([PR #9](https://github.com/alegonz/baikal/pull/9)).

### Fixed
- Some refactoring and minor fixes.
- Bug fixes ([PR #6](https://github.com/alegonz/baikal/pull/6), [PR #7](https://github.com/alegonz/baikal/pull/7))

## [0.1.0] - 2019-06-01
### Added
- Everything. This is the first (pre-release) version.

[Unreleased]: https://github.com/alegonz/baikal/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/alegonz/baikal/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/alegonz/baikal/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/alegonz/baikal/releases/tag/v0.1.0

<!---
Release diff tags are written as in the example below:
[0.2.0]: https://github.com/alegonz/baikal/compare/v0.1.0...v0.2.0
-->
