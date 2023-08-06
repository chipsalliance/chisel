# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1
# RUN: ls %t/hw/top.sv
# RUN: ls %t/hw/Main.sv
# RUN: ls %t/hw/services.json
# RUN: ls %t/hw/ESILoopback.tcl
# RUN: ls %t/hw/filelist.f
# RUN: ls %t/hw/xsim.tcl
# RUN: ls %t/hw/xrt_package.tcl
# RUN: ls %t/runtime/ESILoopback/common.py
# RUN: ls %t/runtime/ESILoopback/__init__.py
# RUN: ls %t/runtime/ESILoopback/xrt.py
# RUN: ls %t/Makefile.xrt
# RUN: ls %t/xrt.ini
# RUN: ls %t/xsim.tcl
# RUN: ls %t/runtime/ESILoopback/EsiXrtPython.cpp

# RUN: FileCheck %s --input-file %t/hw/top.sv --check-prefix=TOP

import pycde
from pycde import Clock, Input, Module, generator, types
from pycde.bsp import XrtBSP

import sys


class Main(Module):
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    pass


gendir = sys.argv[1]
s = pycde.System(XrtBSP(Main),
                 name="ESILoopback",
                 output_directory=gendir,
                 sw_api_langs=["python"])
s.run_passes(debug=True)
s.compile()
s.package()

# TOP-LABEL: module top
# TOP:         #(parameter __INST_HIER = "INSTANTIATE_WITH_INSTANCE_PATH") (
# TOP:         input         ap_clk,
# TOP:                       ap_resetn,
# TOP:                       s_axi_control_AWVALID,
# TOP:         input  [31:0] s_axi_control_AWADDR,
# TOP:         input         s_axi_control_WVALID,
# TOP:         input  [31:0] s_axi_control_WDATA,
# TOP:         input  [3:0]  s_axi_control_WSTRB,
# TOP:         input         s_axi_control_ARVALID,
# TOP:         input  [31:0] s_axi_control_ARADDR,
# TOP:         input         s_axi_control_RREADY,
# TOP:                       s_axi_control_BREADY,
# TOP:         output        s_axi_control_AWREADY,
# TOP:                       s_axi_control_WREADY,
# TOP:                       s_axi_control_ARREADY,
# TOP:                       s_axi_control_RVALID,
# TOP:         output [31:0] s_axi_control_RDATA,
# TOP:         output [1:0]  s_axi_control_RRESP,
# TOP:         output        s_axi_control_BVALID,
# TOP:         output [1:0]  s_axi_control_BRESP

# TOP:         XrtService #(
# TOP:         ) XrtService (
# TOP:           .clk      (ap_clk),
# TOP:           .rst      (~ap_resetn),
# TOP:           .axil_in  (_GEN),
# TOP:           .axil_out (_XrtService_axil_out)
# TOP:         );

# TOP:         Main #(
# TOP:         ) Main (
# TOP:           .clk (ap_clk),
# TOP:           .rst (~ap_resetn)
# TOP:         );

# TOP:         assign s_axi_control_AWREADY = _XrtService_axil_out.awready;
# TOP:         assign s_axi_control_WREADY = _XrtService_axil_out.wready;
# TOP:         assign s_axi_control_ARREADY = _XrtService_axil_out.arready;
# TOP:         assign s_axi_control_RVALID = _XrtService_axil_out.rvalid;
# TOP:         assign s_axi_control_RDATA = _XrtService_axil_out.rdata;
# TOP:         assign s_axi_control_RRESP = _XrtService_axil_out.rresp;
# TOP:         assign s_axi_control_BVALID = _XrtService_axil_out.bvalid;
# TOP:         assign s_axi_control_BRESP = _XrtService_axil_out.bresp;

# TOP:       endmodule
