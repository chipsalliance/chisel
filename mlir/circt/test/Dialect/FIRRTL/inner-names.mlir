// RUN: circt-opt --verify-diagnostics %s | FileCheck %s

firrtl.circuit "Foo" {
  firrtl.extmodule @Bar()

  // CHECK-LABEL: firrtl.module @Foo
  // CHECK-SAME: in %value: !firrtl.uint<42> sym @symValue
  // CHECK-SAME: in %clock: !firrtl.clock sym @symClock
  // CHECK-SAME: in %reset: !firrtl.asyncreset sym @symReset
  firrtl.module @Foo(
    in %value: !firrtl.uint<42> sym @symValue,
    in %clock: !firrtl.clock sym @symClock,
    in %reset: !firrtl.asyncreset sym @symReset
  ) {
    // CHECK: firrtl.instance instName sym @instSym @Bar()
    firrtl.instance instName sym @instSym @Bar()
    // CHECK: %nodeName = firrtl.node sym @nodeSym %value : !firrtl.uint<42>
    %nodeName = firrtl.node sym @nodeSym %value : !firrtl.uint<42>
    // CHECK: %wireName = firrtl.wire sym @wireSym : !firrtl.uint<42>
    %wireName = firrtl.wire sym @wireSym : !firrtl.uint<42>
    // CHECK: %regName = firrtl.reg sym @regSym %clock : !firrtl.clock, !firrtl.uint<42>
    %regName = firrtl.reg sym @regSym %clock : !firrtl.clock, !firrtl.uint<42>
    // CHECK: %regResetName = firrtl.regreset sym @regResetSym %clock, %reset, %value : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<42>, !firrtl.uint<42>
    %regResetName = firrtl.regreset sym @regResetSym %clock, %reset, %value : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<42>, !firrtl.uint<42>
    // CHECK: %memName_port = firrtl.mem sym @memSym Undefined {depth = 8 : i64, name = "memName", portNames = ["port"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<42>>
    %memName_port = firrtl.mem sym @memSym Undefined {depth = 8 : i64, name = "memName", portNames = ["port"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<42>>
  }
}
