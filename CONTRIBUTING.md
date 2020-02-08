# Contributing guidelines

Bug reports and fixes are always welcome!

Contributions to extend/refactor/improve/document the API are also welcome! **baikal** 
is currently a one-man operation, and it could benefit from more minds and hands working 
on it :)

If you would like to contribute to the project (thank you!), please follow the guidelines 
below.
  
## Bug reports
1. Check if the bug happens in master. If the bug persists, then
2. Check the issues page to see if the issue has been reported, solved or closed before. 
   Make sure to remove the `is:open` qualifier so that closed issues are also visible.
   If the bug is indeed new, then
3. Open a new issue and provide a brief explanation of the bug describing the expected and
   the actual behavior, and add a code sample to reproduce it. Please refer to the template
   provided when clicking the "New issue" button.
4. If possible, try to fix it and submit a PR yourself :)
  
## Feature requests
1. Check in the issues page if a similar idea has already been proposed. If it hasn't, then
2. Open an issue describing the feature and why it would be useful and important to have.
   The feature must be accompanied with a code snippet showing how the feature would be
   used. Please refer to the template provided when clicking the "New issue" button.
3. Make a case for your proposal and address any questions/comments/suggestions.
4. If the feature is accepted, you may go ahead and submit a PR.

**baikal** goal is to make building complex machine learning pipelines easier, so a good 
API feature has (ideally all) the following traits:

* makes a task easier,
* is of general use,
* is intuitive,
* is hard to use incorrectly,
* makes code more readable.

## Submitting a pull request

* **Scope**: A PR must address one issue (unless the same solution fixes two or more 
  issues of course) and should be decoupled from any other proposed changes as much as 
  possible. If the PR involves several changes, it might be more appropriate to split it
  into several PRs, as several PRs are easier to review/understand/backport/revert than 
  one huge PR. Please add a reference to the related issue in the description (e.g. 
  `Fixes #123`, `Implements #456`), this will close the issue automatically when the PR
  is merged.
* **Tests**: Existing tests **must** pass and no line should be left uncovered. If the 
  PR fixes a bug, it should also add a test covering the case where the bug happens. If 
  the PR introduces a new feature, it should add the appropriate tests confirming the 
  correct functioning of the feature. Remember that the reported coverage is only line and 
  branch coverage. If possible, go the extra mile and devise tests that cover more complex
  yet important interactions of multiple conditions. For a new API feature, usually the 
  feature use cases can also serve as the test cases, so you might be able to shoot two 
  birds with one stone!
* **Code format**: This project adopts the black code format. Make sure to setup the 
  pre-commit hook before committing any changes.
* **Commits**: Commits, like PRs, should be granular and decoupled from each other. 
  Ideally, the PR's commit history tells a story: the reviewer should be able to easily 
  grasp what changes were made when glancing at the commit history. Please add descriptive 
  commit messages and avoid cryptic messages like `Some refactoring` or `More fixes`. When
  writing a commit message, usually the **why** is more important than the **what** (one 
  can check the diff for that), so try to explain the reasons for that change. Remember: 
  the audience of a commit message is another developer in the future (including your future 
  self) that might need to understand the reasons why and the context where the changes 
  happened.
* **Documentation**: Any changes must be accompanied by the appropriate documentation, if 
  applicable. This might include adding or revising the docstrings, updating the user 
  guide, or adding an example.
* **Changelog**: Please update the [Changelog](CHANGELOG.md) appropriately.
* **License**: by submitting a pull request to the project, you’re offering your changes 
  under this project’s [license](LICENSE). 

## Setting up the development environment
1. Clone the project.
2. From the project root folder run: `make setup_dev`.
    - This will create a virtualenv and install the package in development mode.
    - It will also install a pre-commit hook for the black code formatter.
    - You need Python 3.5 or above.
3. To run the tests use: `make test`, or `make test-cov` to include coverage.
    - The tests include a test for the plot utility, so you need to install graphviz.
