# -*- Python -*-

import os
import platform
import re
import shutil
import subprocess
import tempfile
import warnings

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'PyCDE Integration'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.py']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.circt_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))
config.substitutions.append(('%shlibdir', config.circt_shlib_dir))
config.substitutions.append(('%INC%', config.circt_include_dir))
config.substitutions.append(('%PYTHON%', f'"{config.python_executable}"'))
config.substitutions.append(('%CIRCT_SOURCE%', config.circt_src_root))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# Set the timeout, if requested.
if config.timeout is not None and config.timeout != "":
  lit_config.maxIndividualTestTime = int(config.timeout)

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    'Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt', 'lit.cfg.py',
    'lit.local.cfg.py', 'support'
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.circt_obj_root,
                                     'frontends/pycde/integration_test')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)
# Substitute '%l' with the path to the build lib dir.

# Tweak the PYTHONPATH to include the binary dir.
if config.bindings_python_enabled:
  llvm_config.with_environment(
      'PYTHONPATH', [os.path.join(config.circt_python_packages_dir, 'pycde')],
      append_path=True)

tool_dirs = [
    config.circt_tools_dir, config.mlir_tools_dir, config.llvm_tools_dir
]
tools = ['circt-rtl-sim.py', 'esi-cosim-runner.py']

# Enable Icarus Verilog as a fallback if no other ieee-sim was detected.
if config.iverilog_path != "":
  tool_dirs.append(os.path.dirname(config.iverilog_path))
  tools.append('iverilog')
  tools.append('vvp')
  config.available_features.add('iverilog')
  config.available_features.add('ieee-sim')
  config.available_features.add('rtl-sim')
  config.substitutions.append(('%iverilog', config.iverilog_path))
  config.substitutions.append(('%ieee-sim', config.iverilog_path))

# Enable Verilator if it has been detected.
if config.verilator_path != "":
  tool_dirs.append(os.path.dirname(config.verilator_path))
  tools.append('verilator')
  config.available_features.add('verilator')
  config.available_features.add('rtl-sim')
  llvm_config.with_environment('VERILATOR_PATH', config.verilator_path)

# Enable Quartus if it has been detected.
if config.quartus_path != "":
  tool_dirs.append(os.path.dirname(config.quartus_path))
  tools.append('quartus')
  config.available_features.add('quartus')

# Enable Vivado if it has been detected.
if config.vivado_path != "":
  tool_dirs.append(config.vivado_path)
  tools.append('xvlog')
  tools.append('xelab')
  tools.append('xsim')
  config.available_features.add('ieee-sim')
  config.available_features.add('vivado')
  config.substitutions.append(
      ('%ieee-sim', os.path.join(config.vivado_path, "xsim")))
  config.substitutions.append(('%xsim%', os.path.join(config.vivado_path,
                                                      "xsim")))

# Enable Questa if it has been detected.
if config.questa_path != "":
  config.available_features.add('questa')
  config.available_features.add('ieee-sim')
  config.available_features.add('rtl-sim')
  if 'LM_LICENSE_FILE' in os.environ:
    llvm_config.with_environment('LM_LICENSE_FILE',
                                 os.environ['LM_LICENSE_FILE'])

  tool_dirs.append(config.questa_path)
  tools.append('vlog')
  tools.append('vsim')

  config.substitutions.append(
      ('%questa', os.path.join(config.questa_path, "vsim")))
  config.substitutions.append(
      ('%ieee-sim', os.path.join(config.questa_path, "vsim")))

ieee_sims = list(filter(lambda x: x[0] == '%ieee-sim', config.substitutions))
if len(ieee_sims) > 1:
  warnings.warn(
      f"You have multiple ieee-sim simulators configured, choosing: {ieee_sims[-1][1]}"
  )

# If the ieee-sim was selected to be iverilog in case no other simulators are
# available, define a feature flag to allow tests which cannot be simulated
# with iverilog to be disabled.
if ieee_sims and ieee_sims[-1][1] == config.iverilog_path:
  config.available_features.add('ieee-sim-iverilog')

# Enable ESI cosim tests if they have been built.
if config.esi_cosim_path != "":
  config.available_features.add('esi-cosim')
  config.substitutions.append(
      ('%ESIINC%', f'{config.circt_include_dir}/circt/Dialect/ESI/'))
  config.substitutions.append(('%ESICOSIM%', f'{config.esi_cosim_path}'))

# Enable ESI's Capnp tests if they're supported.
if config.esi_capnp != "":
  config.available_features.add('capnp')
  config.substitutions.append(('%CAPNP_CMAKE_DIR%', config.capnp_path))

llvm_config.add_tool_substitutions(tools, tool_dirs)
