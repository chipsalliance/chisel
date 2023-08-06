# Getting Started with the CIRCT Project

## Overview

Welcome to the CIRCT project!

"CIRCT" stands for "Circuit IR Compilers and Tools".  The CIRCT project is an 
(experimental!) effort looking to apply MLIR and the LLVM development 
methodology to the domain of hardware design tools.

Take a look at the following diagram, which gives a brief overview of the 
current [dialects and how they interact](/includes/img/dialects.svg):

<p align="center"><img src="/includes/img/dialectlegend.svg"/></p>
<p align="center"><img src="/includes/img/dialects.svg"/></p>

## Setting this up

These commands can be used to setup CIRCT project:

1) **Install Dependencies** of LLVM/MLIR according to [the
  instructions](https://mlir.llvm.org/getting_started/), including cmake and 
ninja.

*Note:* CIRCT is known to build with at least GCC 9.4 and Clang 13.0.1, but
older versions may not be supported. It is recommended to use the same C++
toolchain to compile both LLVM and CIRCT to avoid potential issues.

If you plan to use the Python bindings, you should start by reading [the
 instructions](https://mlir.llvm.org/docs/Bindings/Python/#building) for building
the MLIR Python bindings, which describe extra dependencies, CMake variables,
and helpful Python development practices. Note the extra CMake variables, which
you will need to specify in step 3) below.

2) **Check out LLVM and CIRCT repos.**  CIRCT contains LLVM as a git
submodule.  The LLVM repo here includes staged changes to MLIR which
may be necessary to support CIRCT.  It also represents the version of
LLVM that has been tested.  MLIR is still changing relatively rapidly,
so feel free to use the current version of LLVM, but APIs may have
changed.

```
$ git clone git@github.com:circt/circt.git
$ cd circt
$ git submodule init
$ git submodule update
```

*Note:* The repository is set up so that `git submodule update` performs a 
shallow clone, meaning it downloads just enough of the LLVM repository to check 
out the currently specified commit. If you wish to work with the full history of
the LLVM repository, you can manually "unshallow" the submodule:

```
$ cd llvm
$ git fetch --unshallow
```

3) **Build and test LLVM/MLIR:**

```
$ cd circt
$ mkdir llvm/build
$ cd llvm/build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="X86;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG
$ ninja
$ ninja check-mlir
```

4) **Build and test CIRCT:**

```
$ cd circt
$ mkdir build
$ cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG
$ ninja
$ ninja check-circt
$ ninja check-circt-integration # Run the integration tests.
```

The `-DCMAKE_BUILD_TYPE=DEBUG` flag enables debug information, which makes the
whole tree compile slower, but allows you to step through code into the LLVM
and MLIR frameworks.

To get something that runs fast, use `-DCMAKE_BUILD_TYPE=Release` or
`-DCMAKE_BUILD_TYPE=RelWithDebInfo` if you want to go fast and optionally if
you want debug info to go with it.  `Release` mode makes a very large difference
in performance.

If you plan to use the Python bindings, you should also specify
`-DCIRCT_BINDINGS_PYTHON_ENABLED=ON`.

5) **Optionally configure your environment**:

It is useful to add the `.../circt/build/bin` and `.../circt/llvm/build/bin` 
directories to the end of your PATH, allowing you to use the tools like `circt-opt` 
in a natural way on the command line.  Similarly, you need to be in the build 
directory to invoke ninja, which is super annoying.  You might find a bash/zsh 
alias like this to be useful:

```bash
build() {
  (cd $HOME/Projects/circt/build/; ninja $1 $2 $3)
}
```

This allows you to invoke `build check-circt` from any directory and have it do
the right thing.

6) **Run the Verilator tests:** (optional)

[Verilator](https://github.com/verilator/verilator) can be used to check
SystemVerilog code. To run the tests, build or install a **recent** version
of Verilator (at least v4.034, ideally v4.110 or later to avoid a known bug).
(Some Linux distributions have *ancient* versions.) If Verilator is in your
PATH, `build check-circt` should run the tests which require Verilator.

We provide a script `utils/get-verilator.sh` to automate the download and
compilation of Verilator into a known location. The testing script will check
this location first. This script assumes that all the Verilator package
dependencies are installed on your system. They are:

- make
- autoconf
- g++
- flex
- bison
- libfl2     # Ubuntu only (ignore if gives error)
- libfl-dev  # Ubuntu only (ignore if gives error)

7) **Install Cap'nProto** (optional, affects ESI dialect only)

Some of the ESI dialect code requires [libcapnp](https://capnproto.org/), 0.9.1 or newer.
(Specifically, the [cosimulation](Dialects/ESI/cosim.md) component.) Most of
the ESI cosim integration tests also require the python bindings: pycapnp.
The `utils/get-capnp.sh` script downloads, compiles, and installs a known
good version to a directory within the circt source code. It optionally
installs pycapnp via 'pip3'. The capnp compile requires libtool.
Alternatively, you can use a docker image we provide via
`utils/run-docker.sh`.

8) **Install OR-Tools** (optional, enables additional schedulers)

[OR-Tools](https://developers.google.com/optimization/) is an open source
software suite for (mathematical) optimization. It provides a uniform interface
to several open-source and commercial solvers, e.g. for linear programs and
satisfiability problems. Here, it is optionally used in the static scheduling
infrastructure. Binary distributions often do not include the required CMake
build info. The `utils/get-or-tools.sh` script downloads, compiles, and
installs a known good version to a directory within the CIRCT source code,
where it is then picked up automatically by the build.

## Windows: notes on setting up with Ninja

Building on Windows using MSVC + Ninja + Python support is straight forward,
though full of landmines. Here are some notes:

- Ninja and cmake must be run in a VS Developer Command shell. If you use
Powershell and don't want to start the VS GUI, you can run:
```powershell
> $vsPath = &(Join-Path ${env:ProgramFiles(x86)} "\Microsoft Visual Studio\Installer\vswhere.exe") -property installationpath
> Import-Module (Get-ChildItem $vsPath -Recurse -File -Filter Microsoft.VisualStudio.DevShell.dll).FullName
> Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation
```

- VSCode's cmake configure does not operate properly with Python support. The
symptom is that the build will complete, but importing `circt` or `mlir` crashes
Python. Doing everything from the command line is the only way CIRCT compiles
have been made to work.

Cheat sheet for powershell:

```powershell
# Install cmake, ninja, and Visual Studio
> python -m pip install psutil pyyaml numpy pybind11

> $vsPath = &(Join-Path ${env:ProgramFiles(x86)} "\Microsoft Visual Studio\Installer\vswhere.exe") -property installationpath
> Import-Module (Get-ChildItem $vsPath -Recurse -File -Filter Microsoft.VisualStudio.DevShell.dll).FullName
> Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation

> cd <circt clone>
> cmake -B<build_dir> llvm/llvm `
    -GNinja `
    -DLLVM_ENABLE_PROJECTS=mlir `
    -DCMAKE_BUILD_TYPE=Debug `
    -DLLVM_TARGETS_TO_BUILD=X86 `
    -DLLVM_ENABLE_ASSERTIONS=ON `
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON `
    -DLLVM_EXTERNAL_PROJECTS=circt `
    -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR="$(PWD)" `
    -DCIRCT_BINDINGS_PYTHON_ENABLED=ON `
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
> ninja -C<build_dir> check-circt
```

## Submitting changes to CIRCT

The project is small so there are few formal process yet.  We generally follow
the LLVM and MLIR community practices, but we currently use pull requests and
GitHub issues.  Here are some high-level guidelines:

 * Please use clang-format in the LLVM style.  There are good plugins
   for common editors like VSCode, Atom, etc, or you can run it
   manually.  This makes code easier to read and understand.

 * Beyond mechanical formatting issues, please follow the [LLVM Coding
   Standards](https://llvm.org/docs/CodingStandards.html).

 * Please practice "[incremental development](https://llvm.org/docs/DeveloperPolicy.html#incremental-development)", 
   preferring to send a small series of incremental patches rather than large 
   patches.  There are other policies in the LLVM Developer Policy document that
   are worth skimming.

 * Please use "Squash and Merge" in PRs when they are approved - we don't
   need the intra-change history in the repository history.

 * Please create a PR to get a code review.  For reviewers, it is good to look
   at the primary author of the code you are touching to make sure they are at
   least CC'd on the PR.

## Submitting changes to LLVM / MLIR

This project depends on MLIR and LLVM, and it is occasionally useful to improve 
them.
To get set up for this:

1) Follow the "[How to Contribute](https://mlir.llvm.org/getting_started/Contributing/)" 
instructions, and install the right tools, e.g. 'arcanist' and `clang-format`.
2) Get an LLVM Phabricator account
3) [Ask for LLVM commit access](https://llvm.org/docs/DeveloperPolicy.html#obtaining-commit-access), 
the barrier is low.

### Submitting a patch

The patch flow goes like this:

1) Check out the LLVM mono repo (as described above).
2) Make changes to your codebase.
3) Stage your changes with `git add`.
4) Run clang-format to tidy up the details of the patch with `git clang-format origin/main`
5) Create a [patch in Phabricator](https://llvm.org/docs/Phabricator.html) with `arc diff`.
6) Iterate on review, changing your code and sending another patch with `arc diff`.

When your review converges and your patch is approved, do the following:

1) commit your changes with `git commit`
2) rebase your changes with `git pull --rebase`
3) retest your patch with `ninja check-mlir`  (or whatever other target makes sense)
4) push your changes with `git push`
