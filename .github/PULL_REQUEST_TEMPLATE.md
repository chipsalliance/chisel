### Contributor Checklist

- [ ] Did you add Scaladoc to every public function/method?
- [ ] Did you add at least one test demonstrating the PR?
- [ ] Did you delete any extraneous printlns/debugging code?
- [ ] Did you specify the type of improvement?
- [ ] Did you add appropriate documentation in `docs/src`?
- [ ] Did you request a desired merge strategy?
- [ ] Did you add text to be included in the Release Notes for this change?

<!--
If you PR has any impact on the user API or affects backend code generation,
please describe the change in the "Release Notes" section below.
-->

#### Type of Improvement

<!-- Choose one or more from the following (delete those that do not apply): -->
- Feature (or new API)
- API modification
- API deprecation
- Backend code generation
- Performance improvement
- Bugfix
- Documentation or website-related
- Dependency update
- Internal or build-related (includes code refactoring/cleanup)


#### Desired Merge Strategy

<!-- If approved, how should this PR be merged? Delete those that do not apply -->
- Squash: The PR will be squashed and merged (choose this if you have no preference).
- Rebase: You will rebase the PR onto master and it will be merged with a merge commit.

#### Release Notes
<!--
The title of your PR will be included in the release notes in addition to any text in this section.
Please be sure to elaborate on any API changes or deprecations and any impact on backend code generation.
-->

### Reviewer Checklist (only modified by reviewer)
- [ ] Did you add the appropriate labels? (Select the most appropriate one based on the "Type of Improvement")
- [ ] Did you mark the proper milestone (Bug fix: `3.5.x`, `3.6.x`, or `5.x` depending on impact, API modification or big change: `6.0`)?
- [ ] Did you review?
- [ ] Did you check whether all relevant Contributor checkboxes have been checked?
- [ ] Did you do one of the following when ready to merge:
  - [ ] Squash: You/ the contributor `Enable auto-merge (squash)`, clean up the commit message, and label with `Please Merge`.
  - [ ] Merge: Ensure that contributor has cleaned up their commit history, then merge with `Create a merge commit`.
