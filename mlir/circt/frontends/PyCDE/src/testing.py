from .system import System
from .module import Module

import builtins
import inspect
from pathlib import Path
import subprocess
import re
import os


def unittestmodule(generate=True,
                   print=True,
                   run_passes=False,
                   print_after_passes=False,
                   emit_outputs=False,
                   debug=False,
                   **kwargs):
  """
  Like @module, but additionally performs system instantiation, generation,
  and printing to reduce boilerplate in tests.
  In case of wrapping a function, @testmodule accepts kwargs which are passed
  to the function as arguments.
  """

  def testmodule_inner(func_or_class):
    # Apply any provided kwargs if this was a function.
    if inspect.isfunction(func_or_class):
      mod = func_or_class(**kwargs)
    elif inspect.isclass(func_or_class) and issubclass(func_or_class, Module):
      mod = func_or_class
    else:
      raise AssertionError("unittest() must decorate a function or"
                           " a Module subclass")

    # Add the module to global scope in case it's referenced within the
    # module generator functions
    setattr(builtins, mod.__name__, mod)

    sys = System([mod], output_directory=f"out_{func_or_class.__name__}")
    if generate:
      sys.generate()
      if print:
        sys.print()
      if run_passes:
        sys.run_passes(debug)
      if print_after_passes:
        sys.print()
      if emit_outputs:
        sys.emit_outputs()

    return mod

  return testmodule_inner


def cocotest(func):
  # Set a flag on the function to indicate that it's a testbench.
  setattr(func, "_testbench", True)
  return func


_EXTRA_FLAG = "_extra_files"


def cocoextra(func):
  # cocotb extra files function. Will include all returned filepaths in the
  # cocotb run.
  setattr(func, _EXTRA_FLAG, True)
  return func


def _gen_cocotb_testfile(tests):
  """
  Converts testbench functions to cocotb-compatible versions..
  To do this cleanly, we need to detect the indent of the function,
  and remove it from the function implementation.
  """
  template = "import cocotb\n\n"

  for test in tests:
    src = inspect.getsourcelines(test)[0]
    indent = len(src[0]) - len(src[0].lstrip())
    src = [line[indent:] for line in src]
    # Remove the '@cocotest' decorator
    src = src[1:]
    # If the function was not async, make it so.
    if not src[0].startswith("async"):
      src[0] = "async " + src[0]

    # Append to the template as a cocotb test.
    template += "@cocotb.test()\n"
    template += "".join(src)
    template += "\n\n"

  return template


class _IVerilogHandler:
  """ Class for handling icarus-verilog specific commands and patching."""

  def __init__(self):
    # Ensure that iverilog is available in path and it is at least iverilog v11
    try:
      out = subprocess.check_output(["iverilog", "-V"])
    except subprocess.CalledProcessError:
      raise Exception("iverilog not found in path")

    # find the 'Icarus Verilog version #' string and extract the version number
    # using a regex
    ver_re = r"Icarus Verilog version (\d+\.\d+)"
    ver_match = re.search(ver_re, out.decode("utf-8"))
    if ver_match is None:
      raise Exception("Could not find Icarus Verilog version")
    ver = ver_match.group(1)
    if float(ver) < 11:
      raise Exception(f"Icarus Verilog version must be >= 11, got {ver}")

  def extra_compile_args(self, pycde_system: System):
    # If no timescale is defined in the source code, icarus assumes a
    # timescale of '1'. This prevents cocotb from creating small timescale clocks.
    # Since a timescale is not emitted by default from export-verilog, make our
    # lives easier and create a minimum timescale through the command-line.
    cmd_file = os.path.join(pycde_system._output_directory, "cmds.f")
    with open(cmd_file, "w+") as f:
      f.write("+timescale+1ns/1ps")

    return [f"-f{cmd_file}"]


def cocotestbench(pycde_mod, simulator='icarus', **kwargs):
  """
  Decorator class for defining a class as a PyCDE testbench.
  'pycde_mod' is the PyCDE module under test.
  Within the decorated class, functions with the '@cocotest' decorator
  will be converted to a cocotb-compatible testbench.
  kwargs will be forwarded to the cocotb-test 'run' function
  """

  # Ensure that system has 'make' available:
  try:
    subprocess.check_output(["make", "-v"])
  except subprocess.CalledProcessError:
    raise Exception(
        "'make' is not available, and is required to run cocotb tests.")

  # Some iverilog-specific checking. This is the main simulator currently used
  # for integration testing, and we require a minimum version for it to be
  # compatible with CIRCT output.
  simhandler = None
  if simulator == "icarus":
    simhandler = _IVerilogHandler()
  # else, let cocotb handle simulator verification.

  def testbenchmodule_inner(tb_class):
    sys = System([pycde_mod])
    sys.generate()
    sys.emit_outputs()
    testmodule = "test_" + pycde_mod.__name__

    # Find include files in the testbench.
    extra_files_funcs = [
        getattr(tb_class, a)
        for a in dir(tb_class)
        if getattr(getattr(tb_class, a), _EXTRA_FLAG, False)
    ]
    test_files = sys.mod_files.union(
        set(sum([f() for f in extra_files_funcs], [])))

    # Find functions with the testbench flag set.
    testbench_funcs = [
        getattr(tb_class, a)
        for a in dir(tb_class)
        if getattr(getattr(tb_class, a), "_testbench", False)
    ]

    # Generate the cocotb test file.
    testfile_path = Path(sys._output_directory, f"{testmodule}.py")
    with open(testfile_path, "w") as f:
      f.write(_gen_cocotb_testfile(testbench_funcs))

    # Simulator-specific extra compile args.
    if simhandler:
      compile_args = kwargs.get("compile_args", [])
      compile_args += simhandler.extra_compile_args(sys)
      kwargs["compile_args"] = compile_args

    from cocotb_test.simulator import run
    run(simulator=simulator,
        module=testmodule,
        toplevel=pycde_mod.__name__,
        toplevel_lang="verilog",
        verilog_sources=list(test_files),
        work_dir=sys._output_directory,
        **kwargs)

    return pycde_mod

  return testbenchmodule_inner
