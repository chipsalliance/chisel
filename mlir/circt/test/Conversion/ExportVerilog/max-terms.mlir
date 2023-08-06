// RUN: circt-opt -test-apply-lowering-options='options=maximumNumberOfTermsPerExpression=4,disallowLocalVariables' --export-verilog %s | FileCheck %s

// CHECK-LABEL: module large_use_in_procedural
hw.module @large_use_in_procedural(%clock: i1, %a: i1) {
  // CHECK: wire [[GEN_1:long_concat]] = a + a + a + a + a;

  // CHECK: always
  sv.always {
    sv.ifdef.procedural "FOO" {
      // This expression should be hoisted and spilled.
      // If there is a namehint, we should use the name.
      %1 = comb.add %a, %a, %a, %a, %a {sv.namehint = "long_concat"}: i1
      // CHECK: if ([[GEN_1]])
      sv.if %1 {
        sv.exit
      }
      %2 = comb.add %a, %a, %a, %a : i1
      // CHECK: if (a + a + a + a)
      sv.if %2 {
        sv.exit
      }
    }
  }

  // CHECK: reg [[REG:.+]];
  %reg = sv.reg : !hw.inout<i1>

  // CHECK: wire [[GEN_0:.+]] = reg_0 + reg_0 + reg_0 + reg_0 + reg_0;
  sv.alwaysff(posedge %clock) {
    // CHECK: always
    // CHECK: [[REG]] = a;
    sv.bpassign %reg, %a : i1
    %0 = sv.read_inout %reg : !hw.inout<i1>
    %1 = comb.add %0, %0, %0, %0, %0 : i1
    // CHECK: if ([[GEN_0]])
    sv.if %1 {
      sv.exit
    }
  }
}

// CHECK-LABEL: module large_use_in_procedural_successive
hw.module @large_use_in_procedural_successive(%clock: i1, %a: i1) {
  sv.always posedge %clock {
    %0 = comb.and %a, %a, %a, %a, %a : i1
    %1 = comb.and %a, %a, %a, %a, %a : i1
    // CHECK:      wire {{.*}} = a & a & a & a & a;
    // CHECK-NEXT: wire {{.*}} = a & a & a & a & a;
    sv.if %0 {
      sv.exit
    }
    sv.if %1 {
      sv.exit
    }
  }
}

// CHECK-LABEL: module dont_spill_to_procedural_regions
hw.module @dont_spill_to_procedural_regions(%z: i10) -> () {
  %r1 = sv.reg : !hw.inout<i1>
  %r2 = sv.reg : !hw.inout<i10>
  // CHECK: wire [9:0] _GEN = r2 + r2 + r2 + r2 + r2;
  // CHECK: initial begin
  // CHECK-NEXT:   `ifdef BAR
  // CHECK-NEXT:      r1 <= _GEN == z;
  // CHECK-NEXT:   `endif
  // CHECK-NEXT: end // initial
  sv.initial {
    %x = sv.read_inout %r2: !hw.inout<i10>
    sv.ifdef.procedural "BAR" {
      %2 = comb.add %x, %x, %x, %x, %x : i10
      %3 = comb.icmp eq %2, %z: i10
      sv.passign %r1, %3: i1
    }
  }
  hw.output
}
