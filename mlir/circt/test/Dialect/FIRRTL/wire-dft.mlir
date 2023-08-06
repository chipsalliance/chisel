// RUN: circt-opt -firrtl-dft %s --verify-diagnostics | FileCheck %s

// The clock gate corresponds to this FIRRTL external module:
// ```firrtl
// extmodule EICG_wrapper :
//   input in : Clock
//   input test_en : UInt<1>
//   input en : UInt<1>
//   (input dft_clk_div_bypass : UInt<1>)?
//   output out : Clock
//   defname = EICG_wrapper
// ```

// Should not error when there is no enable.
// CHECK-LABEL: firrtl.circuit "NoDuts"
firrtl.circuit "NoDuts" {
  firrtl.module @NoDuts() {}
}

// Should be fine when there are no clock gates.
firrtl.circuit "NoClockGates" {
  firrtl.module @A() { }
  firrtl.module @NoClockGates() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // CHECK: firrtl.instance a @A()
    firrtl.instance a @A()
    %test_en = firrtl.wire {annotations = [{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]} : !firrtl.uint<1>
  }
}

// Simple example with the enable signal in the top level DUT module.
// CHECK-LABEL: firrtl.circuit "Simple"
firrtl.circuit "Simple" {
  firrtl.extmodule @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}
  
  // CHECK: firrtl.module @A(in %test_en: !firrtl.uint<1>)
  firrtl.module @A() {
    %in, %test_en, %en, %out = firrtl.instance eicg @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    // CHECK: firrtl.strictconnect %eicg_test_en, %test_en : !firrtl.uint<1>
  }

  firrtl.module @Simple() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // CHECK: %a_test_en = firrtl.instance a  @A(in test_en: !firrtl.uint<1>)
    firrtl.instance a @A()
    %test_en = firrtl.wire {annotations = [{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]} : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %a_test_en, %test_en
  }
}

// Complex example. The enable signal should flow using output ports up to the
// LCA, and downward to the leafs using input ports.  
// CHECK-LABEL: firrtl.circuit "TestHarness"
firrtl.circuit "TestHarness" {

  firrtl.extmodule @EICG_wrapper1(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}
  firrtl.extmodule @EICG_wrapper2(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}
  firrtl.extmodule @EICG_wrapper3(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, in dft_clk_div_bypass: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper_with_bypass"}

  // CHECK: firrtl.module @C(in %test_en: !firrtl.uint<1>, in %dft_clk_div_bypass: !firrtl.uint<1>)
  firrtl.module @C() {
    %eicg_in, %eicg_test_en, %eicg_en, %eicg_out = firrtl.instance eicg @EICG_wrapper1(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)

    %eicg3_in, %eicg3_test_en, %eicg3_en, %eicg3_dft_clk_div_bypass, %eicg3_out = firrtl.instance eicg3 @EICG_wrapper3(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, in dft_clk_div_bypass: !firrtl.uint<1>, out out: !firrtl.clock)

    // CHECK: firrtl.strictconnect %eicg_test_en, %test_en
    // CHECK: firrtl.strictconnect %eicg3_test_en, %test_en
    // CHECK: firrtl.strictconnect %eicg3_dft_clk_div_bypass, %dft_clk_div_bypass
  }

  // CHECK: firrtl.module @B(in %test_en: !firrtl.uint<1>)
  firrtl.module @B() {
    %eicg_in, %eicg_test_en, %eicg_en, %eicg_out = firrtl.instance eicg @EICG_wrapper1(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)

    // CHECK: firrtl.strictconnect %eicg_test_en, %test_en
  }

  // CHECK: firrtl.module @A(in %test_en: !firrtl.uint<1>, in %dft_clk_div_bypass: !firrtl.uint<1>)
  firrtl.module @A() {
    %eicg_in, %eicg_test_en, %eicg_en, %eicg_out = firrtl.instance eicg @EICG_wrapper2(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    // CHECK: %b_test_en = firrtl.instance b  @B(in test_en: !firrtl.uint<1>)
    firrtl.instance b @B()
    // CHECK: %c_test_en, %c_dft_clk_div_bypass = firrtl.instance c @C(in test_en: !firrtl.uint<1>, in dft_clk_div_bypass: !firrtl.uint<1>)
    firrtl.instance c @C()

    // CHECK: firrtl.strictconnect %eicg_test_en, %test_en : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %b_test_en, %test_en : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %c_test_en, %test_en : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %c_dft_clk_div_bypass, %dft_clk_div_bypass : !firrtl.uint<1>
  }

  // CHECK: firrtl.module @TestEn0(out %test_en: !firrtl.uint<1>)
  firrtl.module @TestEn0() {
    // A bundle type should be work for the enable signal using annotations with fieldIDs.
    %test_en = firrtl.wire {annotations = [{circt.fieldID = 3 : i32, class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]} : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
    // CHECK: %0 = firrtl.subindex %test_en_0[0] : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
    // CHECK: %1 = firrtl.subfield %0[qux] : !firrtl.bundle<baz: uint<1>, qux: uint<1>>
    // CHECK: firrtl.strictconnect %test_en, %1 : !firrtl.uint<1>
    firrtl.instance b @B()

    // CHECK: firrtl.strictconnect %b_test_en, %1 : !firrtl.uint<1>
  }

  // CHECK: firrtl.module @TestEn1(out %test_en: !firrtl.uint<1>, out %dft_clk_div_bypass: !firrtl.uint<1>)
  firrtl.module @TestEn1() {
    // CHECK: %test_en0_test_en = firrtl.instance test_en0  @TestEn0(out test_en: !firrtl.uint<1>)
    firrtl.instance test_en0 @TestEn0()

    firrtl.instance c @C()
    // CHECK: %c_test_en, %c_dft_clk_div_bypass = firrtl.instance c @C(in test_en: !firrtl.uint<1>, in dft_clk_div_bypass: !firrtl.uint<1>)

    // A bundle type should be work for the bypass signal using annotations with fieldIDs.
    %dft_clk_div_bypass = firrtl.wire {annotations = [{circt.fieldID = 3 : i32, class = "sifive.enterprise.firrtl.DFTClockDividerBypassAnnotation"}]} : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
    // CHECK: %0 = firrtl.subindex %dft_clk_div_bypass_0[0] : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
    // CHECK: %1 = firrtl.subfield %0[qux] : !firrtl.bundle<baz: uint<1>, qux: uint<1>>


    // CHECK: firrtl.strictconnect %test_en, %test_en0_test_en : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %c_test_en, %test_en0_test_en : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %dft_clk_div_bypass, %1 : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %c_dft_clk_div_bypass, %1 : !firrtl.uint<1>
  }

  // CHECK: firrtl.module @DUT()
  firrtl.module @DUT() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // CHECK: %a_test_en, %a_dft_clk_div_bypass = firrtl.instance a  @A(in test_en: !firrtl.uint<1>, in dft_clk_div_bypass: !firrtl.uint<1>)
    firrtl.instance a @A()
    // CHECK: %b_test_en = firrtl.instance b  @B(in test_en: !firrtl.uint<1>)
    firrtl.instance b @B()
    // CHECK: %test_en1_test_en, %test_en1_dft_clk_div_bypass = firrtl.instance test_en1  @TestEn1(out test_en: !firrtl.uint<1>, out dft_clk_div_bypass: !firrtl.uint<1>)
    firrtl.instance test_en1 @TestEn1()

    // CHECK: firrtl.strictconnect %a_test_en, %test_en1_test_en : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %b_test_en, %test_en1_test_en : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %a_dft_clk_div_bypass, %test_en1_dft_clk_div_bypass : !firrtl.uint<1>
  }

  // CHECK: firrtl.module @TestHarness()
  firrtl.module @TestHarness() {
    // CHECK: firrtl.instance dut  @DUT()
    firrtl.instance dut @DUT()

    // The clock gate outside of the DUT should not be wired.
    %eicg_in, %eicg_test_en, %eicg_en, %eicg_out = firrtl.instance eicg @EICG_wrapper2(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    // CHECK-NOT: firrtl.connect
  }
}

// Test enable signal as input to top module, and outside of DUT (issue #3784).
// CHECK-LABEL: firrtl.circuit "EnableOutsideDUT"
firrtl.circuit "EnableOutsideDUT" {
  firrtl.extmodule @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}

  // CHECK: firrtl.module @A(in %test_en: !firrtl.uint<1>)
  firrtl.module @A() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %in, %test_en, %en, %out = firrtl.instance eicg @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    // CHECK: firrtl.strictconnect %eicg_test_en, %test_en : !firrtl.uint<1>
  }

  // Regardless of enable signal origin, leave clocks outside DUT alone.
  firrtl.module @OutsideDUT() {
    %in, %test_en, %en, %out = firrtl.instance eicg @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    // CHECK-NOT: firrtl.connect
  }

  firrtl.module @EnableOutsideDUT(in %port_en: !firrtl.uint<1>) attributes {
    portAnnotations = [[{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]]
  } {
    // CHECK: firrtl.instance o @OutsideDUT
    firrtl.instance o @OutsideDUT()

    // CHECK: %a_test_en = firrtl.instance a  @A(in test_en: !firrtl.uint<1>)
    firrtl.instance a @A()
    // CHECK: firrtl.strictconnect %a_test_en, %port_en
  }
}

// Test enable signal outside DUT but not top.
// CHECK-LABEL: firrtl.circuit "EnableOutsideDUT2"
firrtl.circuit "EnableOutsideDUT2" {
  firrtl.extmodule @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}

  // CHECK: firrtl.module @A(in %test_en: !firrtl.uint<1>)
  firrtl.module @A() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %in, %test_en, %en, %out = firrtl.instance eicg @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    // CHECK: firrtl.strictconnect %eicg_test_en, %test_en : !firrtl.uint<1>
  }

  // Regardless of enable signal origin, leave clocks outside DUT alone.
  // CHECK: @OutsideDUT()
  firrtl.module @OutsideDUT() {
    %in, %test_en, %en, %out = firrtl.instance eicg @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    // CHECK-NOT: firrtl.strictconnect
  }

  // CHECK-LABEL: @enableSignal
  firrtl.module @enableSignal() {
    %test_en = firrtl.wire {annotations = [{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]} : !firrtl.uint<1>
  }

  // CHECK-LABEL: @EnableOutsideDUT2
  firrtl.module @EnableOutsideDUT2() {
    // CHECK: %[[en:.+]] = firrtl.instance en @enableSignal
    firrtl.instance en @enableSignal()
    // CHECK: firrtl.instance o @OutsideDUT
    firrtl.instance o @OutsideDUT()

    // CHECK: %[[a_en:.+]] = firrtl.instance a  @A(in test_en: !firrtl.uint<1>)
    firrtl.instance a @A()
    // CHECK: firrtl.strictconnect %[[a_en]], %[[en]]
  }
}

// Test ignore bypass with wrong port name or at different direction/type.
firrtl.circuit "BypassWrong" {
  // Warning if port is compatible but name doesn't match, conservatively skipping.
  // expected-warning @below {{compatible port in bypass position has wrong name, skipping}}
  firrtl.extmodule @EICG_wrapper_wrongName(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, in wrong_name: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper_name"}
  firrtl.extmodule @EICG_wrapper_wrongType(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, in dft_clk_div_bypass: !firrtl.uint<3>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper_type"}
  firrtl.extmodule @EICG_wrapper_wrongDirection(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out dft_clk_div_bypass: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper_dir"}

  firrtl.extmodule @EICG_wrapper_right(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, in dft_clk_div_bypass: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper_right"}

  // No bypass wiring.
  // CHECK-LABEL: @Gates(in %test_en: !firrtl.uint<1>)
  firrtl.module private @Gates() {
    %name_in, %name_test_en, %name_en, %name_wrong_name, %name_out = firrtl.instance name @EICG_wrapper_wrongName(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, in wrong_name: !firrtl.uint<1>, out out: !firrtl.clock)
    %type_in, %type_test_en, %type_en, %type_dft_clk_div_bypass, %type_out = firrtl.instance type @EICG_wrapper_wrongType(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, in dft_clk_div_bypass: !firrtl.uint<3>, out out: !firrtl.clock)

    %dir_in, %dir_test_en, %dir_en, %dir_dft_clk_div_bypass, %dir_out = firrtl.instance dir @EICG_wrapper_wrongDirection(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out dft_clk_div_bypass: !firrtl.uint<1>, out out: !firrtl.clock)
  }

  firrtl.module @BypassWrong() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %test_en = firrtl.wire {annotations = [{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]} : !firrtl.uint<1>
    %dft_clk_div_bypass = firrtl.wire {annotations = [{class = "sifive.enterprise.firrtl.DFTClockDividerBypassAnnotation"}]} : !firrtl.uint<1>

    firrtl.instance g @Gates()
  }
}

