# Compiling CIRCT and PyCDE

PyCDE is compiled as a component of CIRCT which pulls in the LLVM/MLIR project
as a submodule. As such, compiling it is complex and takes some time. If you're
not a CIRCT or PyCDE developer, [use pip](https://pypi.org/project/pycde/).

## Cloning the repo

If you havent already, you need to clone the CIRCT repo. Unless you already
have contributor permissions to the LLVM project, the easiest way to develop
(with the ability to create and push branches) is to fork the repo in your
GitHub account. You can then clone your fork. The clone command should look
like this:

```bash
git clone git@github.com:<your_github_username>/circt.git <optional_repo_name>
```

If you don't envision needing that ability, you can clone the main repo
following the directions in step 2 of the [GettingStarted](GettingStarted.md) page.

After cloning, navigate to your repo root (circt is the default) and use the
following to pull down LLVM:

```bash
git submodule update --init
```

## PyCDE Installation

### Installing and Building with Wheels

If you are using a fork, you'll need the git tags since the package versioning step requires them:

```bash
git remote add upstream git@github.com:llvm/circt.git
git fetch upstream --tags
```

The simplest way to compile PyCDE for local use is to install it with the `pip
install` command:

```
$ cd circt
$ pip install frontends/PyCDE --use-feature=in-tree-build
```

If you just want to build the wheel, use the `pip wheel` command:

```
$ cd circt
$ pip wheel frontends/PyCDE --use-feature=in-tree-build
```

This will create a `pycde-<version>-<python version>-<platform>.whl` file in the root of the repo.

### Manual Compilation

Follow these steps to setup your repository for installing PyCDE via CMake.
Ensure that your repo has the proper Python requirements by running the
following from your CIRCT repo root:

```bash
python -m pip install -r frontends/PyCDE/python/requirements.txt
```

Although not scrictly needed for PyCDE develoment, scripts for some tools you
might want to install are located in utils/
(Cap'n Proto, Verilator, OR-Tools):

```bash
utils/get-capnp.sh
utils/get-verilator.sh
utils/get-or-tools
```

Install PyCDE with CMake. PyCDE requires cmake version >= 3.21:

```bash
mkdir build
cd build
cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_EXTERNAL_PROJECTS=circt \
    -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=.. \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DCIRCT_BINDINGS_PYTHON_ENABLED=ON \
    -DCIRCT_ENABLE_FRONTENDS=PyCDE
    -G Ninja ../llvm/llvm
```

Alternatively, you can pass the source and build paths to the CMake command and
build in the specified folder:

```bash
cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_EXTERNAL_PROJECTS=circt \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DCIRCT_BINDINGS_PYTHON_ENABLED=ON \
    -DCIRCT_ENABLE_FRONTENDS=PyCDE \
    -G Ninja \
    -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=<your_circt_repo_root_path> \
    -B<path_to_desired_build_dir> \
    <your_circt_repo_root_path>/llvm/llvm
```

Afterwards, use the following commands to ensure that CIRCT and PyCDE are built
and the tests pass:

```bash
ninja -C <path_to_your_circt_build> check-circt
ninja -C <path_to_your_circt_build> check-pycde
ninja -C <path_to_your_circt_build> check-pycde-integration
```

If you want to use PyCDE after compiling it, you must add the core CIRCT
bindings and PyCDE to your PYTHONPATH. This is quite helpful if you are hacking
on PyCDE as you don't have to run an install after each Python file change since
Ninja uses links in the build dir rather than copying the files as it does
during install.

```bash
export PYTHONPATH="<full_path_to_your_circt_build>/tools/circt/python_packages/pycde"
```

If you are installing PyCDE through `ninja install`, the libraries and Python modules will be installed into the correct location automatically.
