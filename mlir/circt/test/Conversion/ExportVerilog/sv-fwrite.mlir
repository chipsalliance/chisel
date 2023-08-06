// RUN: circt-opt %s -export-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

hw.module @top(%clock : i1, %reset: i1) -> () {
  sv.alwaysff(posedge %clock) {
    %0 = hw.constant 0x80000001 : i32
    %1 = hw.constant 0x80000002 : i32

    // CHECK: $fwrite(32'h80000001, "stdout");
    sv.fwrite %0, "stdout"

    // CHECK: $fwrite(32'h80000002, "stderr once");
    sv.fwrite %1, "stderr once"
    // CHECK: $fwrite(32'h80000002, "stderr twice");
    sv.fwrite %1, "stderr twice"

    // CHECK: $fwrite(32'h80000002, "direct fd");
    %2 = hw.constant 0x80000002 : i32
    sv.fwrite %2, "direct fd"
  }
}
