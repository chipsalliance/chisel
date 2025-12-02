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

---

## Building and Testing

### Dependencies

Chisel uses the [Mill](https://mill-build.org/) build tool.
You can install it as described on the Chisel [installation instructions](https://www.chisel-lang.org/docs/installation), or just use the bootstrap script in this repository: `./mill`.
Developers should read the Mill documentation to understand the basic commands and use.

The main dependencies for development are the JDK and git.
Any JDK 11 or newer will work for most development, but note that developing the CIRCT Panama bindings requires Java 21.
[Coursier](https://get-coursier.io)'s command-line is useful for hot swapping JDKs.
For example, the following swap the JDK in your shell to the latest release of GraalVM Java 21:

```sh
eval $(cs java --jvm graalvm-java21 --env)
```

While the CIRCT Panama bindings require Java 21, publishing the Chisel plugin for versions < 2.13.11 requires Java 11.
To switch the JDK in your shell to the latest patch release of Temurin Java 11:

```sh
eval $(cs java --jvm temurin:11 --env)
```

LLVM lit can be installed with pip3 (you may need to update your `PATH` environment variable to include the install directory)
```sh
pip3 install lit
```

LLVM FileCheck can be installed by compiling LLVM or CIRCT from source, or from Jack's pre-built binaries at https://github.com/jackkoenig/filecheck.

### Useful commands

Mill's `resolve` command plus the wildcard `_` are useful for discovering available projects, tasks, and commands.

```sh
# See all projects
./mill resolve _

# See all cross-compile versions for the 'chisel' build unit
./mill resolve chisel._

# See all available tasks and commands for all 'chisel' build unit
./mill resolve chisel.__
```

You can compile everything with (note this includes the CIRCT Panama bindings so requires Java 21):
```sh
./mill compileAll
```

Note that this is a custom command.
Typical Mill docs will suggest `./mill __.compile` to compile everything, but this does not currently work due to the
work-in-progress addition of Scala 3 support.

Most testing can be done on just the Chisel build unit:
```sh
./mill chisel[].test
```

The `[]` exists because we are cross-compiling between Scala 2.13 and Scala 3.
You can pick a specific version, e.g. `./mill chisel[2.13.18]`.
Using `[]` will pick the first version in the list of supported versions which one can think about as the "default" version.

You can test everything with:
```sh
./mill __.test
```

Note the cross-version will likely change in the future, use `./mill resolve chisel._` to see latest version.

Chisel uses ScalaTest so you can run individual tests using standard ScalaTest commands and arguments, e.g.
```sh
./mill chisel[].test.testOnly chiselTests.VecLiteralSpec -- -z "lits must fit in vec element width"
```

### lit + FileCheck Tests

Some of our tests use LLVM's [lit](https://llvm.org/docs/CommandGuide/lit.html) and [FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html).

These tests are also run with mill:
```sh
./mill lit.cross[].run
```

If a test fails, you can use `--filter` to select for the failing test and use `-v` to show the debug information from `FileCheck`:
```sh
./mill lit.cross[].run --filter Module -v
```

### Formatting

Chisel enforces formatting using Scalafmt.
To format the code, run:

```sh
# Reformat normal source files
./mill __.reformat

# Reformat mill build files
./mill --meta-level 1 mill.scalalib.scalafmt.ScalafmtModule/reformatAll sources
```

---

## Internal Documentation

This section describes the internal Chisel components for contributors.

### CIRCT Converter

There is a highly experimental component CIRCT Converter (a.k.a. Panama Converter). It is powered by [Java's Panama framework](https://openjdk.org/projects/panama/) to interact directly with CIRCT by calling the C-APIs of MLIR and CIRCT directly from Scala, enabling seamless emitting Chisel IR to CIRCT FIRRTL Dialect IR (no serialization and deserialization for FIRRTL), flexible executing Passes with PassManager, lowering to / exporting SystemVerilog, accesing OM data, and more.

#### Directory `circtpanamabinding`

Here defines the needed CIRCT items that will be processed by the Panama framework's [jextract](https://github.com/openjdk/jextract) tool for codegen FFI Java code to use in Scala.

When you need to use new APIs, add their names to the corresponding files according to the category, then recompile, and the [mill build system](https://github.com/chipsalliance/chisel/blob/master/common.sc) will automatically process these files, invoking `jextract` to generate Java FFI code for you.

The Panama framework requires exact Java 21, and requires extra `javac` options to link libraries `--enable-native-access=ALL-UNNAMED --enable-preview -Djava.library.path=<lib-path>`. See examples from lit tests, e.g. [SmokeTest.sc](https://github.com/chipsalliance/chisel/blob/main/lit/tests/SmokeTest.sc).

#### Directory `panamalib`

It provides a type-safe wrapper for the FFI code generated by jextract.

#### File `PanamaCIRCTConverter.scala`

Here is the implementation of how each Chisel IR will be emitted to the CIRCT.

It needs to be highly synchronized with CIRCT upstream. When updating it, you can refer to [CIRCT's FIRRTL Dialect documentation](https://circt.llvm.org/docs/Dialects/FIRRTL/). Some dialect-specific types, conversions may require a specialized C-API function to return from CIRCT, in which case you can open a PR in CIRCT upstream to add it and use it here.

#### File `PanamaCIRCTPassManager.scala`

It provides a PassManager, which is used in place of the firtool cli.

#### File `PanamaCIRCTOM.scala`

After using PassManager to lowering the FIRRTL Dialect to the HW Dialect, you will be able to access the OM data in it.

The data types of the OM are defined here, as well as the way to access them. This is an example of printing out all the `Top_Class` fields.

```scala
val pm = converter.passManager()
assert(pm.populateFinalizeIR())
assert(pm.run())

val om = converter.om()
val evaluator = om.evaluator()

val top = evaluator.instantiate("Top_Class", Seq(om.newBasePathEmpty)).get
top.foreachField((name, value) => println(s".$name => { ${value.toString} }"))
```
