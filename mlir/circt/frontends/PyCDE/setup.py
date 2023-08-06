# Build/install the pycde python package. Blatantly copied from npcomp.
# Note that this includes a relatively large build of LLVM (~2400 C++ files)
# and can take a considerable amount of time, especially with defaults.
# To install:
#   pip install . --use-feature=in-tree-build
# To build a wheel:
#   pip wheel . --use-feature=in-tree-build
#
# It is recommended to build with Ninja and ccache. To do so, set environment
# variables by prefixing to above invocations:
#   CMAKE_GENERATOR=Ninja CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache
#
# On CIs, it is often advantageous to re-use/control the CMake build directory.
# This can be set with the PYCDE_CMAKE_BUILD_DIR env var.
import os
import shutil
import subprocess
import sys

from distutils.command.build import build as _build
from setuptools import find_namespace_packages, setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

_thisdir = os.path.abspath(os.path.dirname(__file__))


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
    cmake_build_dir = os.getenv("PYCDE_CMAKE_BUILD_DIR")
    if not cmake_build_dir:
      cmake_build_dir = os.path.join(target_dir, "..", "cmake_build")
    cmake_install_dir = os.path.join(target_dir, "..", "cmake_install")
    circt_dir = os.path.abspath(
        os.environ.get("CIRCT_DIRECTORY", os.path.join(_thisdir, "..", "..")))
    src_dir = os.path.abspath(os.path.join(circt_dir, "llvm", "llvm"))
    cfg = "Release"
    cmake_args = [
        "-DCMAKE_INSTALL_PREFIX={}".format(os.path.abspath(cmake_install_dir)),
        "-DPython3_EXECUTABLE={}".format(sys.executable.replace("\\", "/")),
        "-DCMAKE_BUILD_TYPE={}".format(cfg),  # not used on MSVC, but no harm
        "-DLLVM_ENABLE_ASSERTIONS=ON",
        "-DLLVM_ENABLE_PROJECTS=mlir",
        "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
        "-DLLVM_TARGETS_TO_BUILD=host",
        "-DCIRCT_BINDINGS_PYTHON_ENABLED=ON",
        "-DCIRCT_ENABLE_FRONTENDS=PyCDE",
        "-DLLVM_EXTERNAL_PROJECTS=circt",
        "-DLLVM_EXTERNAL_CIRCT_SOURCE_DIR={}".format(circt_dir),
    ]
    if "CIRCT_EXTRA_CMAKE_ARGS" in os.environ:
      cmake_args += os.environ["CIRCT_EXTRA_CMAKE_ARGS"].split(" ")
    build_args = []
    build_parallelism = os.getenv("CMAKE_PARALLELISM")
    if build_parallelism:
      build_args.append(f"--parallel {build_parallelism}")
    else:
      build_args.append("--parallel")
    os.makedirs(cmake_build_dir, exist_ok=True)
    if os.path.exists(cmake_install_dir):
      shutil.rmtree(cmake_install_dir)
    cmake_cache_file = os.path.join(cmake_build_dir, "CMakeCache.txt")
    if os.path.exists(cmake_cache_file):
      os.remove(cmake_cache_file)
    subprocess.check_call(["cmake", src_dir] + cmake_args, cwd=cmake_build_dir)
    subprocess.check_call(["cmake", "--build", ".", "--target", "check-pycde"] +
                          build_args,
                          cwd=cmake_build_dir)
    install_cmd = ["cmake", "--build", ".", "--target", "install-PyCDE"]
    subprocess.check_call(install_cmd + build_args, cwd=cmake_build_dir)
    shutil.copytree(os.path.join(cmake_install_dir, "python_packages"),
                    target_dir,
                    symlinks=False,
                    dirs_exist_ok=True)


class NoopBuildExtension(build_ext):

  def build_extension(self, ext):
    pass


setup(name="pycde",
      version="0.0.1",
      author="John Demme",
      author_email="John.Demme@microsoft.com",
      description="Python CIRCT Design Entry",
      long_description="",
      include_package_data=True,
      ext_modules=[
          CMakeExtension("pycde.circt._mlir_libs._mlir"),
          CMakeExtension("pycde.circt._mlir_libs._circt"),
      ],
      install_requires=["numpy", "jinja2"],
      cmdclass={
          "build": CustomBuild,
          "built_ext": NoopBuildExtension,
          "build_py": CMakeBuild,
      },
      zip_safe=False,
      packages=find_namespace_packages(include=[
          "pycde",
          "pycde.*",
      ]))
