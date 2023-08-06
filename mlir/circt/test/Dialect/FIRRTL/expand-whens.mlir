// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-expand-whens)))' %s | FileCheck %s
firrtl.circuit "ExpandWhens" {
firrtl.module @ExpandWhens () {}

// Test that last connect semantics are resolved for connects.
firrtl.module @shadow_connects(out %out : !firrtl.uint<1>) {
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  firrtl.connect %out, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %out, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
}
// CHECK-LABEL: firrtl.module @shadow_connects(out %out: !firrtl.uint<1>) {
// CHECK-NEXT:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
// CHECK-NEXT:   %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
// CHECK-NEXT:   firrtl.connect %out, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT: }


// Test that last connect semantics are resolved in a WhenOp
firrtl.module @shadow_when(in %p : !firrtl.uint<1>) {
  %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
  %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
  firrtl.when %p : !firrtl.uint<1> {
    %w = firrtl.wire : !firrtl.uint<2>
    firrtl.connect %w, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
    firrtl.connect %w, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
}
// CHECK-LABEL: firrtl.module @shadow_when(in %p: !firrtl.uint<1>) {
// CHECK-NEXT:   %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
// CHECK-NEXT:   %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
// CHECK-NEXT:   %w = firrtl.wire  : !firrtl.uint<2>
// CHECK-NEXT:   firrtl.connect %w, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT: }


// Test all simulation constructs
firrtl.module @simulation(in %clock : !firrtl.clock, in %p : !firrtl.uint<1>, in %enable : !firrtl.uint<1>, in %reset : !firrtl.uint<1>) {
  firrtl.when %p : !firrtl.uint<1> {
    firrtl.printf %clock, %enable, "CIRCT Rocks!" : !firrtl.clock, !firrtl.uint<1>
    firrtl.stop %clock, %enable, 0 : !firrtl.clock, !firrtl.uint<1>
    firrtl.assert %clock, %p, %enable, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.assume %clock, %p, %enable, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.cover %clock, %p, %enable, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
  } else {
    firrtl.printf %clock, %reset, "CIRCT Rocks!" : !firrtl.clock, !firrtl.uint<1>
    firrtl.stop %clock, %enable, 1 : !firrtl.clock, !firrtl.uint<1>
    firrtl.assert %clock, %p, %enable, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.assume %clock, %p, %enable, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.cover %clock, %p, %enable, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
  }
}
// CHECK-LABEL: firrtl.module @simulation(in %clock: !firrtl.clock, in %p: !firrtl.uint<1>, in %enable: !firrtl.uint<1>, in %reset: !firrtl.uint<1>) {
// CHECK-NEXT:   %0 = firrtl.and %p, %enable : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   firrtl.printf %clock, %0, "CIRCT Rocks!" : !firrtl.clock, !firrtl.uint<1>
// CHECK-NEXT:   %1 = firrtl.and %p, %enable : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   firrtl.stop %clock, %1, 0 : !firrtl.clock, !firrtl.uint<1>
// CHECK-NEXT:   %2 = firrtl.and %p, %enable : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   firrtl.assert %clock, %p, %2, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false}
// CHECK-NEXT:   %3 = firrtl.and %p, %enable : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   firrtl.assume %clock, %p, %3, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false}
// CHECK-NEXT:   %4 = firrtl.and %p, %enable : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   firrtl.cover %clock, %p, %4, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT:   %5 = firrtl.not %p : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   %6 = firrtl.and %5, %reset : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   firrtl.printf %clock, %6, "CIRCT Rocks!" : !firrtl.clock, !firrtl.uint<1>
// CHECK-NEXT:   %7 = firrtl.and %5, %enable : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   firrtl.stop %clock, %7, 1 : !firrtl.clock, !firrtl.uint<1>
// CHECK-NEXT:   %8 = firrtl.and %5, %enable : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   firrtl.assert %clock, %p, %8, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false}
// CHECK-NEXT:   %9 = firrtl.and %5, %enable : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   firrtl.assume %clock, %p, %9, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false}
// CHECK-NEXT:   %10 = firrtl.and %5, %enable : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   firrtl.cover %clock, %p, %10, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false}
// CHECK-NEXT: }


// Test nested when operations work correctly.
firrtl.module @nested_whens(in %clock : !firrtl.clock, in %p0 : !firrtl.uint<1>, in %p1 : !firrtl.uint<1>, in %enable : !firrtl.uint<1>, in %reset : !firrtl.uint<1>) {
  firrtl.when %p0 : !firrtl.uint<1> {
    firrtl.when %p1 : !firrtl.uint<1> {
      firrtl.printf %clock, %enable, "CIRCT Rocks!" : !firrtl.clock, !firrtl.uint<1>
    }
  }
}
// CHECK-LABEL: firrtl.module @nested_whens(in %clock: !firrtl.clock, in %p0: !firrtl.uint<1>, in %p1: !firrtl.uint<1>, in %enable: !firrtl.uint<1>, in %reset: !firrtl.uint<1>) {
// CHECK-NEXT:   %0 = firrtl.and %p0, %p1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   %1 = firrtl.and %0, %enable : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   firrtl.printf %clock, %1, "CIRCT Rocks!" : !firrtl.clock, !firrtl.uint<1>
// CHECK-NEXT: }


// Test that a parameter set in both sides of the connect is resolved. The value
// used is local to each region.
firrtl.module @set_in_both(in %clock : !firrtl.clock, in %p : !firrtl.uint<1>, out %out : !firrtl.uint<2>) {
  firrtl.when %p : !firrtl.uint<1> {
    %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
    firrtl.connect %out, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  } else {
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    firrtl.connect %out, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
}
// CHECK-LABEL: firrtl.module @set_in_both(in %clock: !firrtl.clock, in %p: !firrtl.uint<1>, out %out: !firrtl.uint<2>) {
// CHECK-NEXT:   %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
// CHECK-NEXT:   %0 = firrtl.not %p : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
// CHECK-NEXT:   %1 = firrtl.mux(%p, %c0_ui2, %c1_ui2) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
// CHECK-NEXT:   firrtl.connect %out, %1 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT: }


// Test that a parameter set before a WhenOp, and then in both sides of the
// WhenOp is resolved.
firrtl.module @set_before_and_in_both(in %clock : !firrtl.clock, in %p : !firrtl.uint<1>, out %out : !firrtl.uint<2>) {
  %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
  %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
  %c2_ui2 = firrtl.constant 2 : !firrtl.uint<2>
  firrtl.connect %out, %c2_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  firrtl.when %p : !firrtl.uint<1> {
    firrtl.connect %out, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  } else {
     firrtl.connect %out, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
}
// CHECK-LABEL: firrtl.module @set_before_and_in_both(in %clock: !firrtl.clock, in %p: !firrtl.uint<1>, out %out: !firrtl.uint<2>) {
// CHECK-NEXT:   %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
// CHECK-NEXT:   %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
// CHECK-NEXT:   %c2_ui2 = firrtl.constant 2 : !firrtl.uint<2>
// CHECK-NEXT:   %0 = firrtl.not %p : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   %1 = firrtl.mux(%p, %c0_ui2, %c1_ui2) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
// CHECK-NEXT:   firrtl.connect %out, %1 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT: }


// Test that a parameter set in a WhenOp is not the last connect.
firrtl.module @set_after(in %clock : !firrtl.clock, in %p : !firrtl.uint<1>, out %out : !firrtl.uint<2>) {
  %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
  %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
  %c2_ui2 = firrtl.constant 2 : !firrtl.uint<2>
  firrtl.when %p : !firrtl.uint<1> {
    firrtl.connect %out, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  } else {
    firrtl.connect %out, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  firrtl.connect %out, %c2_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
}
// CHECK-LABEL: firrtl.module @set_after(in %clock: !firrtl.clock, in %p: !firrtl.uint<1>, out %out: !firrtl.uint<2>) {
// CHECK-NEXT:   %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
// CHECK-NEXT:   %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
// CHECK-NEXT:   %c2_ui2 = firrtl.constant 2 : !firrtl.uint<2>
// CHECK-NEXT:   %0 = firrtl.not %p : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   %1 = firrtl.mux(%p, %c0_ui2, %c1_ui2) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
// CHECK-NEXT:   firrtl.connect %out, %c2_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT: }


// Test that wire written to in only the thenblock is resolved.
firrtl.module @set_in_then0(in %clock : !firrtl.clock, in %p : !firrtl.uint<1>, out %out : !firrtl.uint<2>) {
  %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
  %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
  firrtl.connect %out, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  firrtl.when %p : !firrtl.uint<1> {
    firrtl.connect %out, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
}
// CHECK-LABEL: firrtl.module @set_in_then0(in %clock: !firrtl.clock, in %p: !firrtl.uint<1>, out %out: !firrtl.uint<2>) {
// CHECK-NEXT:   %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
// CHECK-NEXT:   %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
// CHECK-NEXT:   %0 = firrtl.mux(%p, %c1_ui2, %c0_ui2) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
// CHECK-NEXT:   firrtl.connect %out, %0 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT: }


// Test that wire written to in only the then block is resolved.
firrtl.module @set_in_then1(in %clock : !firrtl.clock, in %p : !firrtl.uint<1>, out %out : !firrtl.uint<2>) {
  %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
  %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
  firrtl.when %p : !firrtl.uint<1> {
    firrtl.connect %out, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  firrtl.connect %out, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
}
// CHECK-LABEL: firrtl.module @set_in_then1(in %clock: !firrtl.clock, in %p: !firrtl.uint<1>, out %out: !firrtl.uint<2>) {
// CHECK-NEXT:   %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
// CHECK-NEXT:   %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
// CHECK-NEXT:   firrtl.connect %out, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT: }


// Test that wire written to in only the else is resolved.
firrtl.module @set_in_else0(in %p : !firrtl.uint<1>, out %out : !firrtl.uint<2>) {
  %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
  %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
  firrtl.connect %out, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  firrtl.when %p : !firrtl.uint<1> {
  } else {
    firrtl.connect %out, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
}
// CHECK-LABEL: firrtl.module @set_in_else0(in %p: !firrtl.uint<1>, out %out: !firrtl.uint<2>) {
// CHECK-NEXT:   %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
// CHECK-NEXT:   %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
// CHECK-NEXT:   %0 = firrtl.not %p : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   %1 = firrtl.mux(%p, %c0_ui2, %c1_ui2) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
// CHECK-NEXT:   firrtl.connect %out, %1 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT: }


// Test that when there is implicit extension, the mux infers the correct type.
firrtl.module @check_mux_return_type(in %p : !firrtl.uint<1>, out %out : !firrtl.uint<2>) {
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
  firrtl.connect %out, %c0_ui1 : !firrtl.uint<2>, !firrtl.uint<1>
  firrtl.when %p : !firrtl.uint<1> {
  } else {
    firrtl.connect %out, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  // CHECK: firrtl.mux(%p, %c0_ui1, %c1_ui2) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<2>) -> !firrtl.uint<2>
}

// Test that wire written to in only the else block is resolved.
firrtl.module @set_in_else1(in %clock : !firrtl.clock, in %p : !firrtl.uint<1>, out %out : !firrtl.uint<2>) {
  %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
  %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
  firrtl.when %p : !firrtl.uint<1> {
  } else {
    firrtl.connect %out, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  firrtl.connect %out, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
}
// CHECK-LABEL: firrtl.module @set_in_else1(in %clock: !firrtl.clock, in %p: !firrtl.uint<1>, out %out: !firrtl.uint<2>) {
// CHECK-NEXT:   %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
// CHECK-NEXT:   %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
// CHECK-NEXT:   %0 = firrtl.not %p : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   firrtl.connect %out, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT: }

// Check that nested WhenOps work.
firrtl.module @nested(in %clock : !firrtl.clock, in %p0 : !firrtl.uint<1>, in %p1 : !firrtl.uint<1>, out %out : !firrtl.uint<2>) {
  %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
  %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
  %c2_ui2 = firrtl.constant 2 : !firrtl.uint<2>

  firrtl.connect %out, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  firrtl.when %p0 : !firrtl.uint<1> {
    firrtl.when %p1 : !firrtl.uint<1> {
      firrtl.connect %out, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
    }
  }
}
// CHECK-LABEL: firrtl.module @nested(in %clock: !firrtl.clock, in %p0: !firrtl.uint<1>, in %p1: !firrtl.uint<1>, out %out: !firrtl.uint<2>) {
// CHECK-NEXT:   %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
// CHECK-NEXT:   %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
// CHECK-NEXT:   %c2_ui2 = firrtl.constant 2 : !firrtl.uint<2>
// CHECK-NEXT:   %0 = firrtl.and %p0, %p1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   %1 = firrtl.mux(%p1, %c1_ui2, %c0_ui2) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
// CHECK-NEXT:   %2 = firrtl.mux(%p0, %1, %c0_ui2) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
// CHECK-NEXT:   firrtl.connect %out, %2 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT: }


// Check that nested WhenOps work.
firrtl.module @nested2(in %clock : !firrtl.clock, in %p0 : !firrtl.uint<1>, in %p1 : !firrtl.uint<1>, out %out : !firrtl.uint<2>) {
  %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
  %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
  %c2_ui2 = firrtl.constant 2 : !firrtl.uint<2>
  %c3_ui2 = firrtl.constant 3 : !firrtl.uint<2>

  firrtl.when %p0 : !firrtl.uint<1> {
    firrtl.when %p1 : !firrtl.uint<1> {
      firrtl.connect %out, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
    } else {
      firrtl.connect %out, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
    }
  } else {
    firrtl.when %p1 : !firrtl.uint<1> {
      firrtl.connect %out, %c2_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
    } else {
      firrtl.connect %out, %c3_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
    }
  }
}
//CHECK-LABEL: firrtl.module @nested2(in %clock: !firrtl.clock, in %p0: !firrtl.uint<1>, in %p1: !firrtl.uint<1>, out %out: !firrtl.uint<2>) {
//CHECK-NEXT:   %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
//CHECK-NEXT:   %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
//CHECK-NEXT:   %c2_ui2 = firrtl.constant 2 : !firrtl.uint<2>
//CHECK-NEXT:   %c3_ui2 = firrtl.constant 3 : !firrtl.uint<2>
//CHECK-NEXT:   %0 = firrtl.and %p0, %p1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
//CHECK-NEXT:   %1 = firrtl.not %p1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
//CHECK-NEXT:   %2 = firrtl.and %p0, %1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
//CHECK-NEXT:   %3 = firrtl.mux(%p1, %c0_ui2, %c1_ui2) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
//CHECK-NEXT:   %4 = firrtl.not %p0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
//CHECK-NEXT:   %5 = firrtl.and %4, %p1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
//CHECK-NEXT:   %6 = firrtl.not %p1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
//CHECK-NEXT:   %7 = firrtl.and %4, %6 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
//CHECK-NEXT:   %8 = firrtl.mux(%p1, %c2_ui2, %c3_ui2) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
//CHECK-NEXT:   %9 = firrtl.mux(%p0, %3, %8) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
//CHECK-NEXT:   firrtl.connect %out, %9 : !firrtl.uint<2>, !firrtl.uint<2>
//CHECK-NEXT: }

// Test invalid value optimization
// CHECK-LABEL: firrtl.module @InvalidValues
firrtl.module @InvalidValues(in %p: !firrtl.uint<1>, out %out0: !firrtl.uint<2>, out %out1: !firrtl.uint<2>, out %out2: !firrtl.uint<2>, out %out3: !firrtl.uint<2>, out %out4: !firrtl.uint<2>, out %out5: !firrtl.uint<2>) {
  %c2_ui2 = firrtl.constant 2 : !firrtl.uint<2>
  %invalid_ui2 = firrtl.invalidvalue : !firrtl.uint<2>

  firrtl.when %p : !firrtl.uint<1> {
    firrtl.connect %out0, %c2_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  } else  {
    firrtl.connect %out0, %invalid_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  // CHECK: firrtl.connect %out0, %c2_ui2

  firrtl.when %p : !firrtl.uint<1> {
    firrtl.connect %out1, %invalid_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  } else  {
    firrtl.connect %out1, %c2_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  // CHECK: firrtl.connect %out1, %c2_ui2

  firrtl.connect %out2, %invalid_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  firrtl.when %p : !firrtl.uint<1> {
    firrtl.connect %out2, %c2_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  // CHECK: firrtl.connect %out2, %c2_ui2

  firrtl.connect %out3, %invalid_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  firrtl.when %p : !firrtl.uint<1> {
    firrtl.skip
  } else  {
    firrtl.connect %out3, %c2_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  // CHECK: firrtl.connect %out3, %c2_ui2

  firrtl.connect %out4, %c2_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  firrtl.when %p : !firrtl.uint<1> {
    firrtl.connect %out4, %invalid_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  // CHECK: firrtl.connect %out4, %c2_ui2

  firrtl.connect %out5, %c2_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  firrtl.when %p : !firrtl.uint<1> {
    firrtl.skip
  } else  {
    firrtl.connect %out5, %invalid_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  // CHECK: firrtl.connect %out5, %c2_ui2
}

// Test that registers are multiplexed with themselves.
firrtl.module @register_mux(in %p : !firrtl.uint<1>, in %clock: !firrtl.clock) {
  %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
  %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>

  // CHECK: %reg0 = firrtl.reg %clock
  // CHECK: firrtl.connect %reg0, %reg0
  %reg0 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<2>

  // CHECK: %reg1 = firrtl.reg %clock
  // CHECK: firrtl.connect %reg1, %c0_ui2
  %reg1 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<2>
  firrtl.connect %reg1, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>

  // CHECK: %reg2 = firrtl.reg %clock
  // CHECK: [[MUX:%.+]] = firrtl.mux(%p, %c0_ui2, %reg2)
  // CHECK: firrtl.connect %reg2, [[MUX]]
  %reg2 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<2>
  firrtl.when %p : !firrtl.uint<1> {
    firrtl.connect %reg2, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }

  // CHECK: %reg3 = firrtl.reg %clock
  // CHECK: [[MUX:%.+]] = firrtl.mux(%p, %c0_ui2, %c1_ui2)
  // CHECK: firrtl.connect %reg3, [[MUX]]
  %reg3 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<2>
  firrtl.when %p : !firrtl.uint<1> {
    firrtl.connect %reg3, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  } else {
    firrtl.connect %reg3, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
}


// Test that bundle types are supported.
firrtl.module @bundle_types(in %p : !firrtl.uint<1>, in %clock: !firrtl.clock) {

  %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
  %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
  %w = firrtl.wire  : !firrtl.bundle<a: uint<2>, b flip: uint<2>>

  // CHECK: [[W_A:%.*]] = firrtl.subfield %w[a]
  // CHECK: [[MUX:%.*]] = firrtl.mux(%p, %c1_ui2, %c0_ui2)
  // CHECK: firrtl.connect [[W_A]], [[MUX]]
  firrtl.when %p : !firrtl.uint<1> {
    %w_a = firrtl.subfield %w[a] : !firrtl.bundle<a : uint<2>, b flip: uint<2>>
    firrtl.connect %w_a, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  } else {
    %w_a = firrtl.subfield %w[a] : !firrtl.bundle<a : uint<2>, b flip: uint<2>>
    firrtl.connect %w_a, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }

  // CHECK: [[W_B:%.*]] = firrtl.subfield %w[b]
  // CHECK: [[MUX:%.*]] = firrtl.mux(%p, %c1_ui2, %c0_ui2)
  // CHECK: firrtl.connect [[W_B]], [[MUX]]
  %w_b0 = firrtl.subfield %w[b] : !firrtl.bundle<a : uint<2>, b flip: uint<2>>
  firrtl.connect %w_b0, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  firrtl.when %p : !firrtl.uint<1> {
  } else {
    %w_b1 = firrtl.subfield %w[b] : !firrtl.bundle<a : uint<2>, b flip: uint<2>>
    firrtl.connect %w_b1, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
}


// This is exercising a bug in field reference creation when the bundle is
// wrapped in an outer flip. See https://github.com/llvm/circt/issues/1172.
firrtl.module @simple(in %in : !firrtl.bundle<a: uint<1>>) { }
firrtl.module @bundle_ports() {
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %simple_in = firrtl.instance test0 @simple(in in : !firrtl.bundle<a: uint<1>>)
  %0 = firrtl.subfield %simple_in[a] : !firrtl.bundle<a: uint<1>>
  firrtl.connect %0, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
}

// This that types are converted to passive when they are muxed together.
firrtl.module @simple2(in %in : !firrtl.uint<3>) { }
firrtl.module @as_passive(in %p : !firrtl.uint<1>) {
  %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
  %c3_ui3 = firrtl.constant 3 : !firrtl.uint<3>
  %simple0_in = firrtl.instance test0 @simple2(in in : !firrtl.uint<3>)
  firrtl.connect %simple0_in, %c2_ui3 : !firrtl.uint<3>, !firrtl.uint<3>

  %simple1_in = firrtl.instance test0 @simple2(in in : !firrtl.uint<3>)
  firrtl.when %p : !firrtl.uint<1> {
    // This is the tricky part, connect the input ports together.
    firrtl.connect %simple1_in, %simple0_in : !firrtl.uint<3>, !firrtl.uint<3>
  } else {
    firrtl.connect %simple1_in, %c3_ui3 : !firrtl.uint<3>, !firrtl.uint<3>
  }
  // CHECK: [[MUX:%.*]] = firrtl.mux(%p, %test0_in, %c3_ui3) : (!firrtl.uint<1>, !firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>
  // CHECK: firrtl.connect %test0_in_0, [[MUX]] : !firrtl.uint<3>, !firrtl.uint<3>
}


// Test that analog types are not tracked by ExpandWhens
firrtl.module @analog(out %analog : !firrtl.analog<1>) {
  // Should not complain about the output

  // Should not complain about the embeded analog.
  %c1 = firrtl.constant 0 : !firrtl.uint<1>
  %w = firrtl.wire : !firrtl.bundle<a: uint<1>, b: analog<1>>
  %w_a = firrtl.subfield %w[a] : !firrtl.bundle<a : uint<1>, b : analog<1>>
  firrtl.connect %w_a, %c1 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: @vector_simple
firrtl.module @vector_simple(in %clock: !firrtl.clock, out %ret: !firrtl.vector<uint<1>, 1>) {
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %0 = firrtl.subindex %ret[0] : !firrtl.vector<uint<1>, 1>
  firrtl.connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %0, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:      %0 = firrtl.subindex %ret[0] : !firrtl.vector<uint<1>, 1>
  // CHECK-NEXT: firrtl.connect %0, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1
}

// CHECK-LABEL: @shadow_when_vector
firrtl.module @shadow_when_vector(in %p : !firrtl.uint<1>) {
  %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
  %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
  firrtl.when %p : !firrtl.uint<1> {
    %w = firrtl.wire : !firrtl.vector<uint<2>, 1>
    %0 = firrtl.subindex %w[0] : !firrtl.vector<uint<2>, 1>
    firrtl.connect %0, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  // CHECK:      %w = firrtl.wire  : !firrtl.vector<uint<2>, 1>
  // CHECK-NEXT: %0 = firrtl.subindex %w[0] : !firrtl.vector<uint<2>, 1>
  // CHECK-NEXT: firrtl.connect %0, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
}

// CHECK-LABEL: @multi_dim_vector
firrtl.module @multi_dim_vector(in %p : !firrtl.uint<1>) {
  %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
  %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
  %w = firrtl.wire : !firrtl.vector<vector<uint<2>, 2>, 1>
  %0 = firrtl.subindex %w[0] : !firrtl.vector<vector<uint<2>, 2>, 1>
  %1 = firrtl.subindex %0[0] : !firrtl.vector<uint<2>, 2>
  %2 = firrtl.subindex %0[1] : !firrtl.vector<uint<2>, 2>
  firrtl.when %p : !firrtl.uint<1> {
    firrtl.connect %1, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
    firrtl.connect %2, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  } else {
    firrtl.connect %1, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
    firrtl.connect %2, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  // CHECK:      [[MUX1:%.*]] = firrtl.mux(%p, %c0_ui2, %c1_ui2) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
  // CHECK-NEXT: firrtl.connect %1, [[MUX1]] : !firrtl.uint<2>, !firrtl.uint<2>
  // CHECK-NEXT: [[MUX2:%.*]] = firrtl.mux(%p, %c1_ui2, %c0_ui2) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
  // CHECK-NEXT: firrtl.connect %2, [[MUX2]] : !firrtl.uint<2>, !firrtl.uint<2>
}

// CHECK-LABEL: @vector_of_bundle
firrtl.module @vector_of_bundle(in %p : !firrtl.uint<1>, out %ret: !firrtl.vector<bundle<a:uint<1>>, 1>) {
  %0 = firrtl.subindex %ret[0] : !firrtl.vector<bundle<a:uint<1>>, 1>
  %1 = firrtl.subfield %0[a] : !firrtl.bundle<a:uint<1>>
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  firrtl.connect %1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %1, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NOT: firrtl.connect %1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:     firrtl.connect %1, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: @aggregate_register
firrtl.module @aggregate_register(in %clock: !firrtl.clock) {
  %0 = firrtl.reg %clock : !firrtl.clock, !firrtl.bundle<a : uint<1>, b : uint<1>>
  // CHECK:      %1 = firrtl.subfield %0[a]
  // CHECK-NEXT: firrtl.connect %1, %1
  // CHECK-NEXT: %2 = firrtl.subfield %0[b]
  // CHECK-NEXT: firrtl.connect %2, %2
}

// CHECK-LABEL: @aggregate_regreset
firrtl.module @aggregate_regreset(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %resetval: !firrtl.vector<uint<1>, 2>) {
  %0 = firrtl.regreset %clock, %reset, %resetval : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  // CHECK:      %1 = firrtl.subindex %0[0]
  // CHECK-NEXT: firrtl.connect %1, %1
  // CHECK-NEXT: %2 = firrtl.subindex %0[1]
  // CHECK-NEXT: firrtl.connect %2, %2
}

// CHECK-LABEL: @refdefine
firrtl.module @refdefine(in %x : !firrtl.uint<1>, out %out : !firrtl.probe<uint<1>>) {
  // CHECK-NEXT: %[[REF:.+]] = firrtl.ref.send %x
  // CHECK-NEXT: firrtl.ref.define %out, %[[REF]]
  // CHECK-NEXT: }
  firrtl.when %x : !firrtl.uint<1> {
    %ref = firrtl.ref.send %x : !firrtl.uint<1>
    firrtl.ref.define %out, %ref : !firrtl.probe<uint<1>>
  }
}

// CHECK-LABEL: @WhenCForce
firrtl.module @WhenCForce(in %c: !firrtl.uint<1>, in %clock : !firrtl.clock, in %x: !firrtl.uint<4>) {
  // CHECK-NOT: when
  %n, %n_ref = firrtl.node %x forceable : !firrtl.uint<4>
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  firrtl.when %c : !firrtl.uint<1> {
    // CHECK: firrtl.ref.force %clock, %c, %n_ref, %x :
    firrtl.ref.force %clock, %c1_ui1, %n_ref, %x : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<4>
    // CHECK: firrtl.ref.force_initial %c, %n_ref, %x :
    firrtl.ref.force_initial %c1_ui1, %n_ref, %x : !firrtl.uint<1>, !firrtl.uint<4>
    // CHECK: firrtl.ref.release %clock, %c, %n_ref :
    firrtl.ref.release %clock, %c1_ui1, %n_ref : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>
    // CHECK: firrtl.ref.release_initial %c, %n_ref :
    firrtl.ref.release_initial %c1_ui1, %n_ref : !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>
  }
}

// Check that propassign initialized output ports.
// CHECK-LABEL: firrtl.module @PropInitOut(out %out: !firrtl.string)
firrtl.module @PropInitOut(out %out : !firrtl.string) {
  %0 = firrtl.string "hello"
  // CHECK: firrtl.propassign %out, %0 : !firrtl.string
  firrtl.propassign %out, %0 : !firrtl.string
}

// Check that expand whens works for groups.
firrtl.declgroup @GroupFoo bind {}
// CHECK-LABEL: firrtl.module @WhenInGroup
// CHECK-NOT:   firrtl.when
firrtl.module @WhenInGroup(in %cond : !firrtl.uint<1>) {
  firrtl.group @GroupFoo {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %a = firrtl.wire : !firrtl.uint<1>
    firrtl.strictconnect %a, %c0_ui1 : !firrtl.uint<1>
    firrtl.when %cond : !firrtl.uint<1> {
      firrtl.strictconnect %a, %c1_ui1 : !firrtl.uint<1>
    }
  }
}

}
