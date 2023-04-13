## GUIDE TO CONTRIBUTING

1. If you need help on making a pull request, follow this [guide](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

2. To understand how to compile and test chisel3 from the source code, follow the instructions in [SETUP.md](https://github.com/chipsalliance/chisel3/blob/master/SETUP.md).

3. In order to contribute to chisel3, you'll need to sign the CLA agreement. You will be asked to sign it upon your first pull request.

<!-- This ones helped me a lot -->

4. To introduce yourself and get help, you can join the [gitter](https://gitter.im/freechipsproject/chisel3) forum. If you have any questions or concerns, you can get help there.

5. You can peruse the [good-first-issues](https://github.com/chipsalliance/chisel3/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for easy tasks to start with. Another easy thing to start with is doing your own pass of the [website](https://www.chisel-lang.org/chisel3/docs/introduction.html) looking for typos, pages missing their titles, etc. The sources for the website are [here](https://github.com/chipsalliance/chisel3/tree/master/docs).

6. Please make your PRs against the `master` branch. The project admins, when reviewing your PR, will decide which stable version (if any) your change should be backported to. The backports will be opened automatically on your behalf once your `master` PR is merged.

7. The PR template will require you to select "Type of Improvement." A reviewer or someone with write access will add the appropriate label to your PR based on this type of improvement which will include your PR in the correct category in the release notes.

8. If your backport PR(s) get labeled with `bp-conflict`, it means they cannot be automatically be merged. You can help get them merged by openening a PR against the already-existing backport branch (will be named something like `mergify/bp/3.5.x/pr-2512`) with the necessary cleanup changes. The admins will merge your cleanup PR and remove the `bp-conflict` label if appropriate.

 
### Frequently Asked Questions

#### I'm failing the scalafmt check. How do I make sure my code is formatted?

From the Chisel3 root directory, run:

```
sbt scalafmtAll
```

You may need to specify the version, at time of writing this is:

```
sbt ++2.12.15 scalafmtAll
```

#### How do I update PRs from before Scalafmt was applied?

Just before the release of Chisel v3.5.0, we started using [Scalafmt](https://scalameta.org/scalafmt/) in this repository.
Scalafmt is a code formatter; thus when it was applied in https://github.com/chipsalliance/chisel3/pull/2246 it caused a nearly 10,000 line diff.
This introduces merge conflicts with almost every single branch from before that change.
In some cases, it can be simple enough to just resolve any conflicts with a standard merge or rebase, but for some branches, this is too complicated.
For such complex cases, we recommend the following procedure.

##### Merge-based flow

To make the following commands copy-pastable, we recommend exporting your branch name as an environment variable:

```bash
export MY_BRANCH=<name of your branch>
```

First, you should back up a copy of your branch in case anything goes wrong.

```bash
git branch $MY_BRANCH-backup
```

We also need to make sure that you have the upstream Chisel repo as a remote.

```bash
git remote add upstream https://github.com/chipsalliance/chisel3.git
git fetch upstream
```

You can then run the following commands:

```bash
# Make sure your branch is checked out
git checkout $MY_BRANCH

# Merge with the commit just before Scalafmt was applied
# It is possible you will have to resolve conflicts at this point
git merge upstream/just-before-scalafmt
# just-before-scalafmt is a branch that points to a commit, in case the branch gets deleted, you can instead run:
# git branch just-before-scalafmt dd36f97a82746cec0b25b94651581fe799e24579
# git merge just-before-scalafmt

# Format everything using Scalafmt
sbt scalafmtAll

# Commit the reformatted code
git commit -a -m 'Apply Scalafmt'

# Now we need to merge our changes with just after Scalafmt as applied
# Any conflicts are from our changes so we tell git to resolve conflicts by picking our changes
git merge upstream/just-after-scalafmt -X ours
# just-before-scalafmt is a branch that points to a commit, in case the branch gets deleted, you can instead run:
# git branch just-after-scalafmt 3131c0daad41dea78bede4517669e376c41a325a
# git merge just-after-scalafmt -X ours

# Most conflicts from code formatting should be resolved, now you can merge master
# There may still be some silly conflicts, but most should be avoided
git merge upstream/master

# Don't forget to push your changes!
git push origin $MY_BRANCH
```

