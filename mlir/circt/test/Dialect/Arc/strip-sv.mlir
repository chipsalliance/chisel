// RUN: circt-opt %s --arc-strip-sv --verify-diagnostics | FileCheck %s

// CHECK-NOT: sv.verbatim
// CHECK-NOT: sv.ifdef
sv.verbatim "// Standard header to adapt well known macros to our needs." {symbols = []}
sv.ifdef  "RANDOMIZE_REG_INIT" {
  sv.verbatim "`define RANDOMIZE" {symbols = []}
}

// CHECK-LABEL: hw.module @Foo(
hw.module @Foo(%clock: i1, %a: i4) -> (z: i4) {
  // CHECK-NEXT: [[REG:%.+]] = seq.compreg %a, %clock
  %0 = seq.firreg %a clock %clock : i4
  %1 = sv.wire : !hw.inout<i4>
  sv.assign %1, %0 : i4
  %2 = sv.read_inout %1 : !hw.inout<i4>
  // CHECK-NEXT: hw.output [[REG]]
  hw.output %2 : i4
}
// CHECK-NEXT: }

// CHECK-LABEL: hw.module.extern @PeripheryBus
hw.module.extern @PeripheryBus() -> (clock: i1, reset: i1)
// CHECK: hw.module @Top
hw.module @Top() {
  %c0_i7 = hw.constant 0 : i7
  // CHECK: %subsystem_pbus.clock, %subsystem_pbus.reset = hw.instance "subsystem_pbus" @PeripheryBus() -> (clock: i1, reset: i1)
  %subsystem_pbus.clock, %subsystem_pbus.reset = hw.instance "subsystem_pbus" @PeripheryBus() -> (clock: i1, reset: i1)
  // CHECK: [[RST:%.+]] = comb.mux %subsystem_pbus.reset, %c0_i7, %int_rtc_tick_value : i7
  // CHECK: %int_rtc_tick_value = seq.compreg [[RST]], %subsystem_pbus.clock : i7
  %int_rtc_tick_value = seq.firreg %int_rtc_tick_value clock %subsystem_pbus.clock reset sync %subsystem_pbus.reset, %c0_i7 : i7
}

// CHECK-NOT: sv.macro.decl
sv.macro.decl @RANDOM
