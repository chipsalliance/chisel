// RUN: circt-opt --firrtl-inject-dut-hier --firrtl-extract-instances --verify-diagnostics %s | FileCheck %s

// Tests extracted from:
// - test/scala/firrtl/ExtractClockGates.scala

//===----------------------------------------------------------------------===//
// ExtractClockGates Multigrouping
//===----------------------------------------------------------------------===//

// CHECK: firrtl.circuit "ExtractClockGatesMultigrouping"
firrtl.circuit "ExtractClockGatesMultigrouping" attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation", filename = "ClockGates.txt", group = "ClockGatesGroup"}, {class = "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation", name = "InjectedSubmodule"}]} {
  firrtl.extmodule private @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}
  // CHECK-LABEL: firrtl.module private @SomeModule
  firrtl.module private @SomeModule(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>) {
    // CHECK-NOT: firrtl.instance gate @EICG_wrapper
    %gate_in, %gate_en, %gate_out = firrtl.instance gate @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    firrtl.connect %gate_in, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %gate_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module private @InjectedSubmodule
  // CHECK: firrtl.instance inst0 sym [[INST0_SYM:@.+]] @SomeModule
  // CHECK: firrtl.instance inst1 sym [[INST1_SYM:@.+]] @SomeModule

  // CHECK-LABEL: firrtl.module private @ClockGatesGroup
  // CHECK: firrtl.instance gate @EICG_wrapper
  // CHECK: firrtl.instance gate @EICG_wrapper

  // CHECK-LABEL: firrtl.module private @DUTModule
  firrtl.module private @DUTModule(in %clock: !firrtl.clock, in %foo_en: !firrtl.uint<1>, in %bar_en: !firrtl.uint<1>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // CHECK-NOT: firrtl.instance gate @EICG_wrapper
    // CHECK-NOT: firrtl.instance gate @EICG_wrapper
    %inst0_clock, %inst0_en = firrtl.instance inst0 @SomeModule(in clock: !firrtl.clock, in en: !firrtl.uint<1>)
    %inst1_clock, %inst1_en = firrtl.instance inst1 @SomeModule(in clock: !firrtl.clock, in en: !firrtl.uint<1>)
    // CHECK: firrtl.instance ClockGatesGroup sym [[CLKGRP_SYM:@.+]] @ClockGatesGroup
    // CHECK: firrtl.instance InjectedSubmodule sym [[INJMOD_SYM:@.+]] @InjectedSubmodule
    firrtl.connect %inst0_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %inst1_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %inst0_en, %foo_en : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %inst1_en, %bar_en : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module @ExtractClockGatesMultigrouping
  firrtl.module @ExtractClockGatesMultigrouping(in %clock: !firrtl.clock, in %foo_en: !firrtl.uint<1>, in %bar_en: !firrtl.uint<1>) {
    %dut_clock, %dut_foo_en, %dut_bar_en = firrtl.instance dut  @DUTModule(in clock: !firrtl.clock, in foo_en: !firrtl.uint<1>, in bar_en: !firrtl.uint<1>)
    firrtl.connect %dut_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %dut_bar_en, %bar_en : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %dut_foo_en, %foo_en : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK: sv.verbatim "
  // CHECK-SAME{LITERAL}: clock_gate_1 -> {{0}}.{{1}}.{{2}}\0A
  // CHECK-SAME{LITERAL}: clock_gate_0 -> {{0}}.{{1}}.{{3}}\0A
  // CHECK-SAME: output_file = #hw.output_file<"ClockGates.txt", excludeFromFileList>
  // CHECK-SAME: symbols = [
  // CHECK-SAME: @DUTModule
  // CHECK-SAME: #hw.innerNameRef<@DUTModule::[[INJMOD_SYM]]>
  // CHECK-SAME: #hw.innerNameRef<@InjectedSubmodule::[[INST0_SYM]]>
  // CHECK-SAME: #hw.innerNameRef<@InjectedSubmodule::[[INST1_SYM]]>
  // CHECK-SAME: ]
}
