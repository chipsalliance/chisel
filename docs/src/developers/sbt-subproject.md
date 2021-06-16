---
layout: docs
title:  "Developers"
section: "chisel3"
---

# Chisel as an sbt subproject

In order to use the constructs defined in the Chisel3 library, those definitions must be made available to the Scala
compiler at the time a project dependent on them is compiled.
For sbt-based builds there are fundamentally two ways to do this:
* provide a library dependency on the published Chisel3 jars via sbt's `libraryDependencies` setting,
* clone the Chisel3 git repository and include the source code as a subproject of a dependent project.

The former of these two approaches is used by the chisel-tutorial project.
It is the simplest approach and assumes you do not require tight control over Chisel3 source code and are content with the
published release versions of Chisel3.

The latter approach should be used by Chisel3 projects that require finer control over Chisel3 source code.

It's hard to predict in advance the future requirements of a project, and it would be advantageous to be able to
switch between the two approaches relatively easily.
In order to accomplish this, we provide the `sbt-chisel-dep` plugin that allows the developer to concisely specify
Chisel3 subproject dependencies and switch between subproject and library dependency support based on the presence of
a directory (or symbolic link) in the root of the dependent project.

The chisel-template project uses this plugin to support switching between either dependency (subproject or library).
By default, the chisel-template project does not contain a chisel3 subproject directory, and hence, uses a library dependency
on chisel3 (and related Chisel3 projects).
However, if you clone the chisel3 GitHub project from the root directory of the chisel-template project, creating a chisel3
subdirectory, the `sbt-chisel-dep` plugin will take note of the chisel3 project subdirectory,
and provide an sbt subproject dependency in place of the library dependency.

Checkout the [README for the `sbt-chisel-dep`](https://github.com/ucb-bar/sbt-chisel-dep) project for instructions on its usage.

Example versions of the build.sbt and specification of the sbt-chisel-dep plugin are available from the [skeleton branch of the chisel-template repository](https://github.com/ucb-bar/chisel-template/tree/skeleton).
