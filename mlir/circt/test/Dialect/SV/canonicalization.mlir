// RUN: circt-opt -canonicalize='top-down=true region-simplify=true' %s | FileCheck %s

// CHECK-LABEL: func @if_dead_condition(%arg0: i1) {
// CHECK-NEXT:    [[FD:%.*]] = hw.constant -2147483646 : i32
// CHECK-NEXT:    sv.always posedge %arg0  {
// CHECK-NEXT:      sv.fwrite [[FD]], "Reachable1"
// CHECK-NEXT:      sv.fwrite [[FD]], "Reachable2"
// CHECK-NEXT:      sv.fwrite [[FD]], "Reachable3"
// CHECK-NEXT:      sv.fwrite [[FD]], "Reachable4"
// CHECK-NEXT:      sv.if
// CHECK-NOT:       else
// CHECK:           return
// CHECK-NEXT:  }

func.func @if_dead_condition(%arg0: i1) {
  %fd = hw.constant 0x80000002 : i32

  sv.always posedge %arg0 {
    %true = hw.constant true
    %false = hw.constant false

    sv.if %true {}

    sv.if %false {
        sv.fwrite %fd, "Unreachable0"
    }

    sv.if %true {
      sv.fwrite %fd, "Reachable1"
    }

    sv.if %true {
      sv.fwrite %fd, "Reachable2"
    } else {
      sv.fwrite %fd, "Unreachable2"
    }

    sv.if %false {
      sv.fwrite %fd, "Unreachable3"
    } else {
      sv.fwrite %fd, "Reachable3"
    }

    sv.if %false {
      sv.fwrite %fd, "Unreachable4"
    } else {
      sv.fwrite %fd, "Reachable4"
    }

    sv.if %arg0 {
      sv.verbatim "foo"
    } else {}

  }

  return
}

// CHECK-LABEL: func @empy_op(%arg0: i1) {
// CHECK-NOT:     sv.if
// CHECK-NOT:     sv.ifdef
// CHECK-NOT:     sv.ifdef.procedural
// CHECK-NOT:     sv.always
// CHECK-NOT:     sv.initial
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @empy_op(%arg0: i1) {
  sv.initial {
    sv.if %arg0 {}
    sv.if %arg0 {} else {}
    sv.ifdef.procedural "SYNTHESIS" {}
    sv.ifdef.procedural "SYNTHESIS" {} else {}
  }
  sv.ifdef "SYNTHESIS" {}
  sv.ifdef "SYNTHESIS" {} else {}
  sv.always posedge %arg0 {}
  sv.initial {}
  return
}

// CHECK-LABEL: func @invert_if(%arg0: i1, %arg1: i1) {
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    [[FD:%.*]] = hw.constant -2147483646 : i32
// CHECK-NEXT:    sv.initial  {
// CHECK-NEXT:        %0 = comb.xor %arg0, %arg1, %true : i1
// CHECK-NEXT:      sv.if %0  {
// CHECK-NEXT:        sv.fwrite [[FD]], "Foo"
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @invert_if(%arg0: i1, %arg1 : i1) {
  sv.initial {
    %false = hw.constant false
    %0 = comb.xor bin %arg0, %arg1 : i1
    sv.if %0 {
    } else {
      %fd = hw.constant 0x80000002 : i32
      sv.fwrite %fd, "Foo"
    }
  }
  return
}

// CHECK-LABEL: func @no_invert_if(%arg0: i1) {
// CHECK-NEXT:    [[FD:%.*]] = hw.constant -2147483646 : i32
// CHECK-NEXT:    sv.initial  {
// CHECK-NEXT:      sv.if %arg0  {
// CHECK-NEXT:      } else {  
// CHECK-NEXT:        sv.fwrite [[FD]], "Foo"
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @no_invert_if(%arg0: i1) {
  sv.initial {
    sv.if %arg0 {
    } else {
      %fd = hw.constant 0x80000002 : i32
      sv.fwrite %fd, "Foo"
    }
  }
  return
}

// CHECK-LABEL: @assert_canonicalization
hw.module @assert_canonicalization(%clock: i1) {
  %true = hw.constant 1 : i1
  %false = hw.constant 0 : i1
  sv.always posedge %clock {
    // CHECK-NOT: sv.assert
    sv.assert %true, immediate message "assert"
    // CHECK-NOT: sv.assume
    sv.assume %true, immediate message "assume"
    // CHECK-NOT: sv.cover
    sv.cover %false, immediate
  }

  // CHECK-NOT: sv.assert.concurrent
  sv.assert.concurrent posedge %clock, %true
  // CHECK-NOT: sv.assume.concurrent
  sv.assume.concurrent posedge %clock, %true
  // CHECK-NOT: sv.cover.concurrent
  sv.cover.concurrent posedge %clock, %false
}


// CHECK-LABEL: hw.module @svAttrPreventsCanonicalization(%arg0: i1) {
hw.module @svAttrPreventsCanonicalization(%arg0: i1) {
  %0 = sv.wire : !hw.inout<i1>
  // CHECK:      %0 = sv.wire : !hw.inout<i1>
  // CHECK-NEXT: sv.assign %0, %arg0 {sv.attributes = [#sv.attribute<"attr">]} : i1
  sv.assign %0, %arg0 {sv.attributes = [#sv.attribute<"attr">]} : i1
}

// CHECK-LABEL: @case_stmt
hw.module @case_stmt(%arg: i3) {
  %fd = hw.constant 0x80000002 : i32
  sv.initial {
    // CHECK: sv.case %arg
    sv.case casez %arg : i3
    case b000: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }

    // CHECK: sv.case %arg
    sv.case casez %arg : i3
    case b00x: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }

    // CHECK: sv.case casez %arg
    sv.case casez %arg : i3
    case b00z: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }

    // CHECK: sv.case casez %arg
    sv.case casez %arg : i3
    case b00z: {
      sv.fwrite %fd, "x"
    }
    case b00x: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }


    // CHECK: sv.case %arg
    sv.case casex %arg : i3
    case b000: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }

    // CHECK: sv.case casex %arg
    sv.case casex %arg : i3
    case b00x: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }

    // CHECK: sv.case casez %arg
    sv.case casex %arg : i3
    case b00z: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }

    // CHECK: sv.case casex %arg
    sv.case casex %arg : i3
    case b00z: {
      sv.fwrite %fd, "x"
    }
    case b00x: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }


    // CHECK: sv.case %arg
    sv.case %arg : i3
    case b000: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }

    // CHECK: sv.case %arg
    sv.case %arg : i3
    case b00x: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }

    // CHECK: sv.case %arg
    sv.case %arg : i3
    case b00z: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }

    // CHECK: sv.case %arg
    sv.case %arg : i3
    case b00z: {
      sv.fwrite %fd, "x"
    }
    case b00x: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
    }

  }

  }

// CHECK-LABEL: MergeAssignments
hw.module @MergeAssignments(%a: !hw.array<4xi1>, %clock: i1) -> (d: !hw.array<4xi1>) {
  %c-1_i2 = hw.constant -1 : i2
  %c-2_i2 = hw.constant -2 : i2
  %c1_i2 = hw.constant 1 : i2
  %c0_i2 = hw.constant 0 : i2
  %v0 = hw.array_get %a[%c0_i2] : !hw.array<4xi1>, i2
  %v1 = hw.array_get %a[%c1_i2] : !hw.array<4xi1>, i2
  %v2 = hw.array_get %a[%c-2_i2] : !hw.array<4xi1>, i2
  %r = sv.reg  : !hw.inout<array<4xi1>>
  %i0 = sv.array_index_inout %r[%c0_i2] : !hw.inout<array<4xi1>>, i2
  %i1 = sv.array_index_inout %r[%c1_i2] : !hw.inout<array<4xi1>>, i2
  %i2 = sv.array_index_inout %r[%c-2_i2] : !hw.inout<array<4xi1>>, i2
  %read = sv.read_inout %r : !hw.inout<array<4xi1>>
  sv.always posedge %clock {
    // CHECK: sv.always
    // CHECK-NEXT:  %[[index:.+]] = sv.indexed_part_select_inout %r[%c0_i2 : 3] : !hw.inout<array<4xi1>>, i2
    // CHECK-NEXT:  %[[slice:.+]] = hw.array_slice %a[%c0_i2] : (!hw.array<4xi1>) -> !hw.array<3xi1>
    // CHECK-NEXT:  sv.passign %[[index]], %[[slice]]
    // CHECK-NEXT:  %[[index:.+]] = sv.indexed_part_select_inout %r[%c0_i2 : 3] : !hw.inout<array<4xi1>>, i2
    // CHECK-NEXT:  %[[slice:.+]] = hw.array_slice %a[%c0_i2] : (!hw.array<4xi1>) -> !hw.array<3xi1>
    // CHECK-NEXT:  sv.bpassign %[[index]], %[[slice]]
    sv.passign %i0, %v0 : i1
    sv.passign %i1, %v1 : i1
    sv.passign %i2, %v2 : i1
    sv.bpassign %i0, %v0 : i1
    sv.bpassign %i1, %v1 : i1
    sv.bpassign %i2, %v2 : i1
  }
  hw.output %read : !hw.array<4xi1>
}

// CHECK-LABEL: @Sampled
// CHECK-NEXT: hw.output
hw.module @Sampled(%in: i1) {
  %2 = sv.system.sampled %in : i1
  hw.output
}
