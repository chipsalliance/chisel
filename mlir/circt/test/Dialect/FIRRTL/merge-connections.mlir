// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(merge-connections)))' %s | FileCheck %s --check-prefixes=CHECK,COMMON
// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(merge-connections{aggressive-merging=true})))' %s | FileCheck %s --check-prefixes=AGGRESSIVE,COMMON

firrtl.circuit "Test"   {
  // circuit Test :
  //   module Test :
  //     input a : {c: {clock: Clock, valid:UInt<1>}[2]}
  //     output b : {c: {clock: Clock, valid:UInt<1>}[2]}
  //     b <= a
  // COMMON-LABEL: firrtl.module @Test(
  // COMMON-NEXT:    firrtl.strictconnect %b, %a
  // COMMON-NEXT:  }
  firrtl.module @Test(in %a: !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>, out %b: !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>) {
     %0 = firrtl.subindex %a[0] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
     %1 = firrtl.subindex %b[0] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
     %2 = firrtl.subfield %0[clock] : !firrtl.bundle<clock: clock, valid: uint<1>>
     %3 = firrtl.subfield %1[clock] : !firrtl.bundle<clock: clock, valid: uint<1>>
     firrtl.strictconnect %3, %2 : !firrtl.clock
     %4 = firrtl.subfield %0[valid] : !firrtl.bundle<clock: clock, valid: uint<1>>
     %5 = firrtl.subfield %1[valid] : !firrtl.bundle<clock: clock, valid: uint<1>>
     firrtl.strictconnect %5, %4 : !firrtl.uint<1>
     %6 = firrtl.subindex %a[1] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
     %7 = firrtl.subindex %b[1] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
     %8 = firrtl.subfield %6[clock] : !firrtl.bundle<clock: clock, valid: uint<1>>
     %9 = firrtl.subfield %7[clock] : !firrtl.bundle<clock: clock, valid: uint<1>>
     firrtl.strictconnect %9, %8 : !firrtl.clock
     %10 = firrtl.subfield %6[valid] : !firrtl.bundle<clock: clock, valid: uint<1>>
     %11 = firrtl.subfield %7[valid] : !firrtl.bundle<clock: clock, valid: uint<1>>
     firrtl.strictconnect %11, %10 : !firrtl.uint<1>
  }

  // circuit Bar :
  //   module Bar :
  //     output a : {b: UInt<1>, c:UInt<1>}
  //     a.b <= UInt<1>(0)
  //     a.c <= UInt<1>(1)
  // COMMON-LABEL: firrtl.module @Constant(
  // COMMON-NEXT:    %0 = firrtl.aggregateconstant [0 : ui1, 1 : ui1]
  // COMMON-NEXT:    firrtl.strictconnect %a, %0
  // COMMON-NEXT:  }
  firrtl.module @Constant(out %a: !firrtl.bundle<b: uint<1>, c: uint<1>>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %0 = firrtl.subfield %a[b] : !firrtl.bundle<b: uint<1>, c: uint<1>>
    %1 = firrtl.subfield %a[c] : !firrtl.bundle<b: uint<1>, c: uint<1>>
    firrtl.strictconnect %0, %c0_ui1 : !firrtl.uint<1>
    firrtl.strictconnect %1, %c1_ui1 : !firrtl.uint<1>
  }

  // AGGRESSIVE-LABEL:  firrtl.module @ConcatToVector(
  // AGGRESSIVE-NEXT:     %0 = firrtl.vectorcreate %s1, %s2 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.vector<uint<1>, 2>
  // AGGRESSIVE-NEXT:     firrtl.strictconnect %sink, %0
  // AGGRESSIVE-NEXT:   }
  // CHECK-LABEL:       firrtl.module @ConcatToVector(
  // CHECK-NEXT:          %0 = firrtl.subindex %sink[1]
  // CHECK-NEXT:          %1 = firrtl.subindex %sink[0]
  // CHECK-NEXT:          firrtl.strictconnect %1, %s1
  // CHECK-NEXT:          firrtl.strictconnect %0, %s2
  // CHECK-NEXT:        }

  firrtl.module @ConcatToVector(in %s1: !firrtl.uint<1>, in %s2: !firrtl.uint<1>, out %sink: !firrtl.vector<uint<1>, 2>) {
    %0 = firrtl.subindex %sink[1] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %sink[0] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %1, %s1 : !firrtl.uint<1>
    firrtl.strictconnect %0, %s2 : !firrtl.uint<1>
  }

  // Check that we don't use %s1 as a source value.
  // AGGRESSIVE-LABEL:   firrtl.module @FailedToUseAggregate(
  // AGGRESSIVE-NEXT:    %0 = firrtl.subindex %s1[0]
  // AGGRESSIVE-NEXT:    %1 = firrtl.vectorcreate %0, %s2
  // AGGRESSIVE-NEXT:    firrtl.strictconnect %sink, %1
  // AGGRESSIVE-NEXT:   }
  // CHECK-LABEL:       firrtl.module @FailedToUseAggregate(
  // CHECK-NEXT:         %0 = firrtl.subindex %sink[1]
  // CHECK-NEXT:         %1 = firrtl.subindex %s1[0]
  // CHECK-NEXT:         %2 = firrtl.subindex %sink[0]
  // CHECK-NEXT:         firrtl.strictconnect %2, %1
  // CHECK-NEXT:         firrtl.strictconnect %0, %s2
  // CHECK-NEXT:        }
  firrtl.module @FailedToUseAggregate(in %s1: !firrtl.vector<uint<1>, 2>, in %s2: !firrtl.uint<1>, out %sink: !firrtl.vector<uint<1>, 2>) {
    %0 = firrtl.subindex %sink[1] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %s1[0] : !firrtl.vector<uint<1>, 2>
    %2 = firrtl.subindex %sink[0] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %2, %1 : !firrtl.uint<1>
    firrtl.strictconnect %0, %s2 : !firrtl.uint<1>
  }


  // Check that we don't use strictconnect when merging connections into non-passive type.
  // COMMON-LABEL:   firrtl.module private @DUT
  // COMMON-NEXT:     %p = firrtl.wire
  // COMMON-NEXT:     %0 = firrtl.subfield
  // CHECK-NEXT:      firrtl.strictconnect %0, %x_a
  // COMMON-NEXT:     %1 = firrtl.subfield
  // COMMON-NEXT:     firrtl.strictconnect %x_b, %1
  // COMMON-NEXT:     firrtl.strictconnect %y_a, %0
  // CHECK-NEXT:      firrtl.strictconnect %1, %y_b
  // AGGRESSIVE-NEXT: %[[AGG:.+]] = firrtl.bundlecreate
  // AGGRESSIVE-NEXT: firrtl.connect %p, %[[AGG]]
  // COMMON-NEXT:   }
  firrtl.module private @DUT(in %x_a: !firrtl.uint<2>,
                             out %x_b: !firrtl.uint<2>,
                             out %y_a: !firrtl.uint<2>,
                             in %y_b: !firrtl.uint<2>) {
    %p = firrtl.wire : !firrtl.bundle<a: uint<2>, b flip: uint<2>>
    %0 = firrtl.subfield %p[a] : !firrtl.bundle<a: uint<2>, b flip: uint<2>>
    firrtl.strictconnect %0, %x_a : !firrtl.uint<2>
    %1 = firrtl.subfield %p[b] : !firrtl.bundle<a: uint<2>, b flip: uint<2>>
    firrtl.strictconnect %x_b, %1 : !firrtl.uint<2>
    firrtl.strictconnect %y_a, %0 : !firrtl.uint<2>
    firrtl.strictconnect %1, %y_b : !firrtl.uint<2>
  }
}

