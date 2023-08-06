// REQUIRES: verilator
// RUN: circt-opt %s -export-verilog -verify-diagnostics -o %t2.mlir > %t1.sv
// RUN: verilator --lint-only --top-module top %t1.sv
// RUN: circt-rtl-sim.py %t1.sv --cycles 2 2>&1 | FileCheck %s

module {
  sv.interface @data_vr {
    sv.interface.signal @data : i32
    sv.interface.signal @valid : i1
    sv.interface.signal @ready : i1
    sv.interface.modport @data_in (input @data, input @valid, output @ready)
    sv.interface.modport @data_out (output @data, output @valid, input @ready)
  }

  // TODO: This ugly bit is because we don't yet have ExportVerilog support
  // for modports as module port declarations.
  hw.module.extern @Rcvr (%m: !sv.modport<@data_vr::@data_in>)
  sv.verbatim "module Rcvr (data_vr.data_in m);\nendmodule"

  hw.module @top (%clk: i1, %rst: i1) {
    %iface = sv.interface.instance : !sv.interface<@data_vr>

    %ifaceInPort = sv.modport.get %iface @data_in :
      !sv.interface<@data_vr> -> !sv.modport<@data_vr::@data_in>

    hw.instance "rcvr1" @Rcvr(m: %ifaceInPort: !sv.modport<@data_vr::@data_in>) -> ()

    hw.instance "rcvr2" @Rcvr(m: %ifaceInPort: !sv.modport<@data_vr::@data_in>) -> ()

    %c1 = hw.constant 1 : i1
    sv.interface.signal.assign %iface(@data_vr::@valid) = %c1 : i1

    sv.always posedge %clk {
      %validValue = sv.interface.signal.read %iface(@data_vr::@valid) : i1
      %fd = hw.constant 0x80000002 : i32
      sv.fwrite %fd, "valid: %d\n" (%validValue) : i1
    }
    // CHECK: valid: 1
    // CHECK: valid: 1
    // CHECK: valid: 1
    // CHECK: valid: 1
    // CHECK: valid: 1
    // CHECK: valid: 1
  }
}
