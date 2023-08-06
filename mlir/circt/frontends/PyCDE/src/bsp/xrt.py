#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..common import Clock, Input, Output
from ..constructs import ControlReg, Wire
from ..module import Module, generator
from ..system import System
from ..types import bit, types, Bits

import glob
from io import FileIO
import math
import pathlib
import shutil

__dir__ = pathlib.Path(__file__).parent

# Parameters for AXI4-Lite interface
axil_addr_width = 32
axil_data_width = 32
axil_data_width_bytes = int(axil_data_width / 8)

# Constants for MMIO registers
MagicNumberLo = 0xE5100E51  # ESI__ESI
MagicNumberHi = 0x207D98E5  # Random
VersionNumber = 0  # Version 0: format subject to change


# Signals from master
def axil_in_type(addr_width, data_width):
  return types.struct({
      "awvalid": types.i1,
      "awaddr": types.int(addr_width),
      "wvalid": types.i1,
      "wdata": types.int(data_width),
      "wstrb": types.int(data_width // 8),
      "arvalid": types.i1,
      "araddr": types.int(addr_width),
      "rready": types.i1,
      "bready": types.i1
  })


# Signals to master
def axil_out_type(data_width):
  return types.struct({
      "awready": types.i1,
      "wready": types.i1,
      "arready": types.i1,
      "rvalid": types.i1,
      "rdata": types.int(data_width),
      "rresp": types.i2,
      "bvalid": types.i1,
      "bresp": types.i2
  })


def output_tcl(os: FileIO):
  """Output Vitis tcl describing the registers."""

  from jinja2 import Environment, FileSystemLoader, StrictUndefined

  env = Environment(loader=FileSystemLoader(str(__dir__)),
                    undefined=StrictUndefined)
  template = env.get_template("xrt_package.tcl.j2")
  os.write(template.render(system_name=System.current().name))


def XrtBSP(user_module):
  """Use the Xilinx RunTime (XRT) shell to implement ESI services and build an
  image or emulation package.
  How to use this BSP:
  - Wrap your top PyCDE module in `XrtBSP`.
  - Run your script. This BSP will write a 'build package' to the output dir.
  This package contains a Makefile.xrt which (given a proper Vitis dev
  environment) will compile a hw image or hw_emu image. It is a free-standing
  build package -- you do not need PyCDE installed on the same machine as you
  want to do the image build.
  - To build the `hw` image, run 'make -f Makefile.xrt TARGET=hw'. If you want
  an image which runs on an Azure NP-series instance, run the 'azure' target
  (requires an Azure subscription set up with as per
  https://learn.microsoft.com/en-us/azure/virtual-machines/field-programmable-gate-arrays-attestation).
  This target requires a few environment variables to be set (which the Makefile
  will tell you about).
  - To build a hw emulation image, run with TARGET=hw_emu.
  - The makefile also builds a Python plugin. To specify the python version to
  build against (if different from the version ran by 'python3' in your
  environment), set the PYTHON variable (e.g. 'PYTHON=python3.9').
  """

  class XrtService(Module):
    clk = Clock(types.i1)
    rst = Input(types.i1)

    axil_in = Input(axil_in_type(axil_addr_width, axil_data_width))
    axil_out = Output(axil_out_type(axil_data_width))

    @generator
    def generate(self):
      clk = self.clk
      rst = self.rst

      sys: System = System.current()
      output_tcl((sys.hw_output_dir / "xrt_package.tcl").open("w"))

      ######
      # Write side.

      # So that we don't wedge the AXI-lite for writes, just ack all of them.
      write_happened = Wire(bit)
      latched_aw = ControlReg(self.clk, self.rst, [self.axil_in.awvalid],
                              [write_happened])
      latched_w = ControlReg(self.clk, self.rst, [self.axil_in.wvalid],
                             [write_happened])
      write_happened.assign(latched_aw & latched_w)

      ######
      # Read side.

      # Track the non-zero registers in the read address space.
      rd_addr_data = {
          16: Bits(32)(MagicNumberLo),
          20: Bits(32)(MagicNumberHi),
          24: Bits(32)(VersionNumber),
      }

      # Create an array out of the sparse value map 'rd_addr_data' and zero
      # constants. Then create a potentially giant mux. There's probably a much
      # better way to do this. I suspect this is a common high-level construct
      # which should be automatically optimized, but I'm not sure how common
      # this actually is.

      max_addr_log2 = int(
          math.ceil(math.log2(max([a for a in rd_addr_data.keys()]) + 1)))

      # Convert the sparse dict into a zero filled array.
      zero = types.int(axil_data_width)(0)
      rd_space = [zero] * int(math.pow(2, max_addr_log2))
      for (addr, val) in rd_addr_data.items():
        rd_space[addr] = val

      # Create the address index signal and do the muxing.
      addr_slice = self.axil_in.araddr.slice(
          types.int(axil_addr_width)(0), max_addr_log2)
      rd_addr = addr_slice.reg(clk, rst)
      rvalid = self.axil_in.arvalid.reg(clk, rst, cycles=2)
      rdata = types.array(types.int(axil_data_width),
                          len(rd_space))(rd_space)[rd_addr].reg(clk)

      # Assign the module outputs.
      self.axil_out = axil_out_type(axil_data_width)({
          "awready": 1,
          "wready": 1,
          "arready": 1,
          "rvalid": rvalid,
          "rdata": rdata,
          "rresp": 0,
          "bvalid": write_happened,
          "bresp": 0
      })

  class top(Module):
    ap_clk = Clock()
    ap_resetn = Input(types.i1)

    # AXI4-Lite slave interface
    s_axi_control_AWVALID = Input(types.i1)
    s_axi_control_AWREADY = Output(types.i1)
    s_axi_control_AWADDR = Input(types.int(axil_addr_width))
    s_axi_control_WVALID = Input(types.i1)
    s_axi_control_WREADY = Output(types.i1)
    s_axi_control_WDATA = Input(types.int(axil_data_width))
    s_axi_control_WSTRB = Input(types.int(axil_data_width // 8))
    s_axi_control_ARVALID = Input(types.i1)
    s_axi_control_ARREADY = Output(types.i1)
    s_axi_control_ARADDR = Input(types.int(axil_addr_width))
    s_axi_control_RVALID = Output(types.i1)
    s_axi_control_RREADY = Input(types.i1)
    s_axi_control_RDATA = Output(types.int(axil_data_width))
    s_axi_control_RRESP = Output(types.i2)
    s_axi_control_BVALID = Output(types.i1)
    s_axi_control_BREADY = Input(types.i1)
    s_axi_control_BRESP = Output(types.i2)

    @generator
    def construct(ports):

      axil_in_sig = axil_in_type(axil_addr_width, axil_data_width)({
          "awvalid": ports.s_axi_control_AWVALID,
          "awaddr": ports.s_axi_control_AWADDR,
          "wvalid": ports.s_axi_control_WVALID,
          "wdata": ports.s_axi_control_WDATA,
          "wstrb": ports.s_axi_control_WSTRB,
          "arvalid": ports.s_axi_control_ARVALID,
          "araddr": ports.s_axi_control_ARADDR,
          "rready": ports.s_axi_control_RREADY,
          "bready": ports.s_axi_control_BREADY,
      })

      rst = ~ports.ap_resetn

      xrt = XrtService(clk=ports.ap_clk, rst=rst, axil_in=axil_in_sig)

      axil_out = xrt.axil_out

      # AXI-Lite control
      ports.s_axi_control_AWREADY = axil_out['awready']
      ports.s_axi_control_WREADY = axil_out['wready']
      ports.s_axi_control_ARREADY = axil_out['arready']
      ports.s_axi_control_RVALID = axil_out['rvalid']
      ports.s_axi_control_RDATA = axil_out['rdata']
      ports.s_axi_control_RRESP = axil_out['rresp']
      ports.s_axi_control_BVALID = axil_out['bvalid']
      ports.s_axi_control_BRESP = axil_out['bresp']

      # Splice in the user's code
      # NOTE: the clock is `ports.ap_clk`
      #       and reset is `ports.ap_resetn` which is active low
      user_module(clk=ports.ap_clk, rst=rst)

      # Copy additional sources
      sys: System = System.current()
      sys.add_packaging_step(top.package)

    @staticmethod
    def package(sys: System):
      """Assemble a 'build' package which includes all the necessary build
      collateral (about which we are aware), build/debug scripts, and the
      generated runtime."""

      from jinja2 import Environment, FileSystemLoader, StrictUndefined

      sv_sources = glob.glob(str(__dir__ / '*.sv'))
      tcl_sources = glob.glob(str(__dir__ / '*.tcl'))
      for source in sv_sources + tcl_sources:
        shutil.copy(source, sys.hw_output_dir)

      env = Environment(loader=FileSystemLoader(str(__dir__)),
                        undefined=StrictUndefined)
      makefile_template = env.get_template("Makefile.xrt.j2")
      dst_makefile = sys.output_directory / "Makefile.xrt"
      dst_makefile.open("w").write(
          makefile_template.render(system_name=sys.name))

      shutil.copy(__dir__ / "xrt.ini", sys.output_directory / "xrt.ini")
      shutil.copy(__dir__ / "xsim.tcl", sys.output_directory / "xsim.tcl")

      runtime_dir = sys.runtime_output_dir / sys.name
      shutil.copy(__dir__ / "xrt_api.py", runtime_dir / "xrt.py")
      shutil.copy(__dir__ / "EsiXrtPython.cpp",
                  runtime_dir / "EsiXrtPython.cpp")

  return top
