# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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

### Fixes
- Some refactoring and minor fixes.
- Bug fixes ([(PR #6)](https://github.com/alegonz/baikal/pull/6), [(PR #7)](https://github.com/alegonz/baikal/pull/7))

## [0.1.0] - 2019-06-01
### Added
- Everything. This is the first (pre-release) version.

[0.2.0]: https://github.com/alegonz/baikal/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/alegonz/baikal/releases/tag/v0.1.0

<!---
Release diff tags are written as in the example below:
[0.2.0]: https://github.com/alegonz/baikal/compare/v0.1.0...v0.2.0
-->
