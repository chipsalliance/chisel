## GUIDE TO CONTRIBUTING

1. If you need help on making a pull request, follow this [guide](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

2. To understand how to compile and test chisel3 from the source code, install the [required dependencies](https://www.chisel-lang.org/docs/installation).

3. In order to contribute to chisel3, you'll need to sign the CLA agreement. You will be asked to sign it upon your first pull request.

<!-- This ones helped me a lot -->

4. To introduce yourself and get help, you can join the [gitter](https://gitter.im/freechipsproject/chisel3) forum. If you have any questions or concerns, you can get help there.

5. You can peruse the [good-first-issues](https://github.com/chipsalliance/chisel3/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for easy tasks to start with. Another easy thing to start with is doing your own pass of the [website](https://www.chisel-lang.org/chisel3/docs/introduction.html) looking for typos, pages missing their titles, etc. The sources for the website are [here](https://github.com/chipsalliance/chisel3/tree/master/docs).

6. Please make your PRs against the `main` branch. The project admins, when reviewing your PR, will decide which stable version (if any) your change should be backported to. They will apply the appropriate `milestone` marker which controls which branches the backport will be opened to. The backports will be opened automatically on your behalf once your `main` PR is merged.

7. The PR template will require you to select "Type of Improvement." A reviewer or someone with write access will add the appropriate label to your PR based on this type of improvement which will include your PR in the correct category in the release notes.

8. If your backport PR(s) get labeled with `bp-conflict`, it means they cannot be automatically be merged. You can help get them merged by openening a PR against the already-existing backport branch (will be named something like `mergify/bp/3.5.x/pr-2512`) with the necessary cleanup changes. The admins will merge your cleanup PR and remove the `bp-conflict` label if appropriate.


### Frequently Asked Questions

#### I'm failing the formatting check. How do I make sure my code is formatted?

From the Chisel root directory, run:

```sh
# Reformat normal source files
./mill __.reformat

# Reformat mill build files
./mill --meta-level 1 mill.scalalib.scalafmt.ScalafmtModule/reformatAll sources
```
