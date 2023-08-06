#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build/install the circt-core python package.
# Note that this includes a relatively large build of LLVM (~2400 C++ files)
# and can take a considerable amount of time, especially with defaults.
# To install:
#   pip install . --use-feature=in-tree-build
# To build a wheel:
#   pip wheel . --use-feature=in-tree-build
#
# It is recommended to build with Ninja and ccache. To do so, set environment
# variables by prefixing to above invocations:
#   CMAKE_GENERATOR=Ninja \
#   CMAKE_C_COMPILER_LAUNCHER=ccache \
#   CMAKE_CXX_COMPILER_LAUNCHER=ccache
#
# On CIs, it is often advantageous to re-use/control the CMake build directory.
# This can be set with the CIRCT_CMAKE_BUILD_DIR env var.
#
# By default, this will use the llvm-project submodule included with CIRCT.
# This can be overridden with the CIRCT_LLVM_DIR env var.

import os
import platform
import shutil
import subprocess
import sys
import sysconfig

from distutils.command.build import build as _build
from setuptools import find_namespace_packages, setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


# Build phase discovery is unreliable. Just tell it what phases to run.
class CustomBuild(_build):

  def run(self):
    self.run_command("build_py")
    self.run_command("build_ext")
    self.run_command("build_scripts")


class CMakeExtension(Extension):

  def __init__(self, name, sourcedir=""):
    Extension.__init__(self, name, sources=[])
    self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_py):

  def run(self):
    target_dir = self.build_lib
    circt_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    cmake_build_dir = os.getenv("CIRCT_CMAKE_BUILD_DIR")
    if not cmake_build_dir:
      cmake_build_dir = os.path.join(circt_dir, "build")
    cmake_install_dir = os.path.join(cmake_build_dir, "..", "install")
    llvm_dir = os.getenv("CIRCT_LLVM_DIR")
    if not llvm_dir:
      llvm_dir = os.path.join(circt_dir, "llvm", "llvm")
    cmake_args = [
        "-DCMAKE_BUILD_TYPE=Release",  # not used on MSVC, but no harm
        "-DCMAKE_INSTALL_PREFIX={}".format(os.path.abspath(cmake_install_dir)),
        "-DCMAKE_OSX_DEPLOYMENT_TARGET=10.14",  # on OSX, min target for C++17
        "-DPython3_EXECUTABLE={}".format(sys.executable.replace("\\", "/")),
        "-DLLVM_ENABLE_PROJECTS=mlir",
        "-DLLVM_EXTERNAL_PROJECTS=circt",
        "-DLLVM_EXTERNAL_CIRCT_SOURCE_DIR={}".format(circt_dir),
        "-DLLVM_TARGETS_TO_BUILD=host",
        "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
        "-DCIRCT_BINDINGS_PYTHON_ENABLED=ON",
        "-DCIRCT_RELEASE_TAG_ENABLED=ON",
        "-DCIRCT_RELEASE_TAG=firtool",
    ]

    # HACK: CMake fails to auto-detect static linked Python installations, which
    # happens to be what exists on manylinux. We detect this and give it a dummy
    # library file to reference (which is checks exists but never gets
    # used).
    if platform.system() == "Linux":
      python_libdir = sysconfig.get_config_var('LIBDIR')
      python_library = sysconfig.get_config_var('LIBRARY')
      if python_libdir and not os.path.isabs(python_library):
        python_library = os.path.join(python_libdir, python_library)
      if python_library and not os.path.exists(python_library):
        print("Detected static linked python. Faking a library for cmake.")
        fake_libdir = os.path.join(cmake_build_dir, "fake_python", "lib")
        os.makedirs(fake_libdir, exist_ok=True)
        fake_library = os.path.join(fake_libdir,
                                    sysconfig.get_config_var('LIBRARY'))
        subprocess.check_call(["ar", "q", fake_library])
        cmake_args.append("-DPython3_LIBRARY:PATH={}".format(fake_library))

    os.makedirs(cmake_build_dir, exist_ok=True)
    if os.path.exists(cmake_install_dir):
      shutil.rmtree(cmake_install_dir)
    cmake_cache_file = os.path.join(cmake_build_dir, "CMakeCache.txt")
    if os.path.exists(cmake_cache_file):
      os.remove(cmake_cache_file)
    subprocess.check_call(["cmake", llvm_dir] + cmake_args, cwd=cmake_build_dir)
    subprocess.check_call(
        ["cmake", "--build", ".", "--target", "install-CIRCTPythonModules"],
        cwd=cmake_build_dir)
    if os.path.exists(target_dir):
      os.remove(target_dir)
    shutil.copytree(os.path.join(cmake_install_dir, "python_packages",
                                 "circt_core"),
                    target_dir,
                    symlinks=False)


class NoopBuildExtension(build_ext):

  def build_extension(self, ext):
    pass


setup(
    name="circt",
    version="0.0.1",
    author="Mike Urbach",
    author_email="mikeurbach@gmail.com",
    description="CIRCT Python Bindings",
    long_description="",
    include_package_data=True,
    ext_modules=[
        CMakeExtension("circt._mlir_libs._mlir"),
        CMakeExtension("circt._mlir_libs._circt"),
    ],
    cmdclass={
        "build": CustomBuild,
        "built_ext": NoopBuildExtension,
        "build_py": CMakeBuild,
    },
    zip_safe=False,
    packages=find_namespace_packages(include=[
        "circt",
        "circt.*",
    ]),
)
