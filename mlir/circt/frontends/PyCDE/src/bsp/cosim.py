#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..common import Clock, Input
from ..module import Module, generator
from ..system import System
from ..types import types

from ..circt import ir
from ..circt.dialects import esi as raw_esi

import shutil
from pathlib import Path

__root_dir__ = Path(__file__).parent.parent


def CosimBSP(user_module):
  """Wrap and return a cosimulation 'board support package' containing
  'user_module'"""

  class top(Module):
    clk = Clock()
    rst = Input(types.int(1))

    @generator
    def build(ports):
      user_module(clk=ports.clk, rst=ports.rst)
      raw_esi.ServiceInstanceOp(result=[],
                                service_symbol=None,
                                impl_type=ir.StringAttr.get("cosim"),
                                inputs=[ports.clk.value, ports.rst.value])

      System.current().add_packaging_step(top.package)

    @staticmethod
    def package(sys: System):
      """Run the packaging to create a cosim package."""

      # When pycde is installed through a proper install, all of the collateral
      # files are under a dir called "collateral".
      collateral_dir = __root_dir__ / "collateral"
      if collateral_dir.exists():
        bin_dir = collateral_dir
        lib_dir = collateral_dir
        esi_inc_dir = collateral_dir
      else:
        # Build we also want to allow pycde to work in-tree for developers. The
        # necessary files are screwn around the build tree.
        build_dir = __root_dir__.parents[4]
        bin_dir = build_dir / "bin"
        lib_dir = build_dir / "lib"
        circt_inc_dir = build_dir / "tools" / "circt" / "include" / "circt"
        esi_inc_dir = circt_inc_dir / "Dialect" / "ESI"

      hw_src = sys.hw_output_dir
      for f in lib_dir.glob("*.so"):
        shutil.copy(f, hw_src)
      for f in lib_dir.glob("*.dll"):
        shutil.copy(f, hw_src)

      if not collateral_dir.exists():
        shutil.copy(bin_dir / "driver.cpp", hw_src)
        shutil.copy(bin_dir / "driver.sv", hw_src)
        shutil.copy(esi_inc_dir / "ESIPrimitives.sv", hw_src)
        shutil.copy(esi_inc_dir / "cosim" / "Cosim_DpiPkg.sv", hw_src)
        shutil.copy(esi_inc_dir / "cosim" / "Cosim_Endpoint.sv", hw_src)

      shutil.copy(__root_dir__ / "Makefile.cosim", sys.output_directory)
      shutil.copy(sys.hw_output_dir / "schema.capnp", sys.runtime_output_dir)

      # Copy everything from the 'runtime' directory
      shutil.copytree(esi_inc_dir, sys.runtime_output_dir, dirs_exist_ok=True)

  return top
