# Using the Python Bindings

If you are mainly interested in using CIRCT from Python scripts, you need to compile both LLVM/MLIR and CIRCT with Python bindings enabled. Furthermore, you must use a unified build, where LLVM/MLIR and CIRCT are compiled together in one step. 

CIRCT also includes an experimental, opinionated frontend for CIRCT's Python bindings, called [PyCDE](PyCDE).

## Installing and Building with Wheels

CIRCT provides a `setup.py` script that take care of configuring and building LLVM/MLIR, CIRCT, and CIRCT's Python bindings. You can install the CIRCT Python bindings with the `pip install` command:

```
$ cd circt
$ pip install lib/Bindings/Python --use-feature=in-tree-build
```

If you just want to build the wheel, use the `pip wheel` command:

```
$ cd circt
$ pip wheel lib/Bindings/Python --use-feature=in-tree-build
```

This will create a `circt_core-<version>-<python version>-<platform>.whl` file in the root of the repo.

There are some environment variables you can set to control the script. These should be prefixed to the above command(s), or `export`ed in your shell.

To specify an existing CMake build directory, you can set `CIRCT_CMAKE_BUILD_DIR`:

```
export CIRCT_CMAKE_BUILD_DIR=/path/to/your/build/dir
```

To specify an alternate LLVM directory, you can set `CIRCT_LLVM_DIR`:

```
export CIRCT_LLVM_DIR=/path/to/your/llvm
```

Finally, you can set other environment variables to control CMake. By default, the script uses the same settings as [Manual Compilation](#manual-compilation) below. It is recommended to use Ninja and CCache, which can be accomplished with:

```
export CMAKE_GENERATOR=Ninja CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache
```

All other [CMake environment variables](https://cmake.org/cmake/help/latest/manual/cmake-env-variables.7.html) can also be used.

## Manual Compilation

To manually compile LLVM/MLIR, CIRCT, and CIRCT's Python bindings, you can use a single CMake invocation like this:

```
$ cd circt
$ mkdir build
$ cd build
$ cmake -G Ninja ../llvm/llvm \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_EXTERNAL_PROJECTS=circt \
    -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=.. \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DCIRCT_BINDINGS_PYTHON_ENABLED=ON
```

Afterwards, use `ninja check-circt-integration` to ensure that the bindings work. (This will now additionally spin up a couple of Python scripts to test that they are accessible.)

### Without Installation

If you want to try the bindings fresh from the compiler without installation, you need to ensure Python can find the generated modules:

```
export PYTHONPATH="$PWD/llvm/build/tools/circt/python_packages/circt_core"
```

### With Installation

If you are installing CIRCT through `ninja install` anyway, the libraries and Python modules will be installed into the correct location automatically.

## Trying things out

Now you are able to use the CIRCT dialects and infrastructure from a Python interpreter and script:

```python
# silicon.py
import circt
from circt.ir import Context, InsertionPoint, IntegerType, Location, Module
from circt.dialects import hw, comb

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  i42 = IntegerType.get_signless(42)
  m = Module.create()
  with InsertionPoint(m.body):

    def magic(module):
      xor = comb.XorOp.create(module.a, module.b)
      return {"c": xor}

    hw.HWModuleOp(name="magic",
                  input_ports=[("a", i42), ("b", i42)],
                  output_ports=[("c", i42)],
                  body_builder=magic)
  print(m)
```

Running this script through `python3 silicon.py` should print the following MLIR:

```mlir
module {
  hw.module @magic(%a: i42, %b: i42) -> (c: i42) {
    %0 = comb.xor %a, %b : i42
    hw.output %0 : i42
  }
}
```
