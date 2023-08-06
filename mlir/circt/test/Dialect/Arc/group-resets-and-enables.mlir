// RUN: circt-opt %s --arc-group-resets-and-enables | FileCheck %s

// CHECK-LABEL: arc.model "BasicResetGrouping"
arc.model "BasicResetGrouping" {
^bb0(%arg0: !arc.storage):
  %c0_i4 = hw.constant 0 : i4
  %in_clock = arc.root_input "clock", %arg0 : (!arc.storage) -> !arc.state<i1>
  %in_i0 = arc.root_input "i0", %arg0 : (!arc.storage) -> !arc.state<i4>
  %in_i1 = arc.root_input "i1", %arg0 : (!arc.storage) -> !arc.state<i4>
  %in_reset0 = arc.root_input "reset0", %arg0 : (!arc.storage) -> !arc.state<i1>
  %in_reset1 = arc.root_input "reset1", %arg0 : (!arc.storage) -> !arc.state<i1>
  %0 = arc.state_read %in_clock : <i1>
  // Group resets:
  arc.clock_tree %0 {
    //  CHECK: [[IN_RESET0:%.+]] = arc.state_read %in_reset0
    %3 = arc.state_read %in_reset0 : <i1>
    //  CHECK-NEXT: scf.if [[IN_RESET0]] {
    scf.if %3 {
      //   CHECK-NEXT:  arc.state_write [[FOO_ALLOC:%.+]] = %c0_i4
      //   CHECK-NEXT:  arc.state_write [[BAR_ALLOC:%.+]] = %c0_i4
      arc.state_write %1 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT:  [[IN_I0:%.+]] = arc.state_read %in_i0
      // CHECK-NEXT:  arc.state_write [[FOO_ALLOC]] = [[IN_I0]]
      // CHECK-NEXT:  [[IN_I1:%.+]] = arc.state_read %in_i1
      // CHECK-NEXT:  arc.state_write [[BAR_ALLOC]] = [[IN_I1]]
      %4 = arc.state_read %in_i0 : <i4>
      arc.state_write %1 = %4 : <i4>
      // CHECK-NEXT: }
    }
    scf.if %3 {
      arc.state_write %2 = %c0_i4 : <i4>
    } else {
      %5 = arc.state_read %in_i1 : <i4>
      arc.state_write %2 = %5 : <i4>
    }
    // CHECK-NEXT: }
  }
  // Don't group resets that don't match:
  arc.clock_tree %0 {
    //  CHECK: [[IN_RESET0_1:%.+]] = arc.state_read %in_reset0
    %6 = arc.state_read %in_reset0 : <i1>
    //  CHECK-NEXT: [[IN_RESET1_1:%.+]] = arc.state_read %in_reset1
    %7 = arc.state_read %in_reset1 : <i1>
    //  CHECK-NEXT: scf.if [[IN_RESET0_1]] {
    scf.if %6 {
      //   CHECK-NEXT:  arc.state_write [[FOO_ALLOC]] = %c0_i4
      arc.state_write %1 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT:  [[IN_I0_1:%.+]] = arc.state_read %in_i0
      // CHECK-NEXT:  arc.state_write [[FOO_ALLOC]] = [[IN_I0_1]]
      %8 = arc.state_read %in_i0 : <i4>
      arc.state_write %1 = %8 : <i4>
      // CHECK-NEXT: }
    }
    //  CHECK-NEXT: scf.if [[IN_RESET1_1]] {
    scf.if %7 {
      //   CHECK-NEXT:  arc.state_write [[BAR_ALLOC]] = %c0_i4
      arc.state_write %2 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT:  [[IN_I1_1:%.+]] = arc.state_read %in_i1
      // CHECK-NEXT:  arc.state_write [[BAR_ALLOC]] = [[IN_I1_1]]
      %9 = arc.state_read %in_i1 : <i4>
      arc.state_write %2 = %9 : <i4>
    }
    // CHECK-NEXT: }
  // CHECK-NEXT: }
  }
  // Don't group IfOps with return values:
  arc.clock_tree %0 {
    //  CHECK: [[IN_RESET0:%.+]] = arc.state_read %in_reset0
    %10 = arc.state_read %in_reset0 : <i1>
    //  CHECK-NEXT: scf.if [[IN_RESET0]] {
    scf.if %10 {
      //   CHECK-NEXT:  arc.state_write [[FOO_ALLOC]] = %c0_i4
      arc.state_write %1 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT:  [[IN_I0:%.+]] = arc.state_read %in_i0
      // CHECK-NEXT:  arc.state_write [[FOO_ALLOC]] = [[IN_I0]]
      %11 = arc.state_read %in_i0 : <i4>
      arc.state_write %1 = %11 : <i4>
      // CHECK-NEXT: }
    }
    //  CHECK-NEXT: [[IF_RESULT:%.+]] scf.if [[IN_RESET0]] -> (i4) {
    %res = scf.if %10 -> (i4) {
      //   CHECK-NEXT:  arc.state_write [[BAR_ALLOC]] = %c0_i4
      //   CHECK-NEXT:  scf.yield %c0_i4 : i4
      arc.state_write %2 = %c0_i4 : <i4>
      scf.yield %c0_i4 : i4
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT:  [[IN_I1:%.+]] = arc.state_read %in_i1
      // CHECK-NEXT:  arc.state_write [[BAR_ALLOC]] = [[IN_I1]]
      //   CHECK-NEXT:  scf.yield %c0_i4 : i4
      %12 = arc.state_read %in_i1 : <i4>
      arc.state_write %2 = %12 : <i4>
      scf.yield %c0_i4 : i4
    }
    // CHECK-NEXT: }
  // CHECK-NEXT: }
  }
  // Group resets with no else in an early block (that has its contents moved):
  arc.clock_tree %0 {
    //  CHECK: [[IN_RESET0:%.+]] = arc.state_read %in_reset0
    %13 = arc.state_read %in_reset0 : <i1>
    //  CHECK-NEXT: scf.if [[IN_RESET0]] {
    scf.if %13 {
      //   CHECK-NEXT:  arc.state_write [[FOO_ALLOC:%.+]] = %c0_i4
      //   CHECK-NEXT:  arc.state_write [[BAR_ALLOC:%.+]] = %c0_i4
      arc.state_write %1 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
      // CHECK-NEXT:  [[IN_I1:%.+]] = arc.state_read %in_i1
      // CHECK-NEXT:  arc.state_write [[BAR_ALLOC]] = [[IN_I1]]
      // CHECK-NEXT: }
    }
    scf.if %13 {
      arc.state_write %2 = %c0_i4 : <i4>
    } else {
      %14 = arc.state_read %in_i1 : <i4>
      arc.state_write %2 = %14 : <i4>
    }
    // CHECK-NEXT: }
  }
  // Group resets with no else in the last if (where contents are moved to):
  arc.clock_tree %0 {
    //  CHECK: [[IN_RESET0:%.+]] = arc.state_read %in_reset0
    %15 = arc.state_read %in_reset0 : <i1>
    //  CHECK-NEXT: scf.if [[IN_RESET0]] {
    scf.if %15 {
      //   CHECK-NEXT:  arc.state_write [[FOO_ALLOC:%.+]] = %c0_i4
      //   CHECK-NEXT:  arc.state_write [[BAR_ALLOC:%.+]] = %c0_i4
      arc.state_write %1 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT:  [[IN_I0:%.+]] = arc.state_read %in_i0
      // CHECK-NEXT:  arc.state_write [[FOO_ALLOC]] = [[IN_I0]]
      %16 = arc.state_read %in_i0 : <i4>
      arc.state_write %1 = %16 : <i4>
      // CHECK-NEXT: }
    }
    scf.if %15 {
      arc.state_write %2 = %c0_i4 : <i4>
    }
    // CHECK-NEXT: }
  }
  // CHECK-NEXT: [[FOO_ALLOC]] = arc.alloc_state %arg0 {name = "foo"} : (!arc.storage) -> !arc.state<i4>
  // CHECK-NEXT: [[BAR_ALLOC]] = arc.alloc_state %arg0 {name = "bar"} : (!arc.storage) -> !arc.state<i4>
  %1 = arc.alloc_state %arg0 {name = "foo"} : (!arc.storage) -> !arc.state<i4>
  %2 = arc.alloc_state %arg0 {name = "bar"} : (!arc.storage) -> !arc.state<i4>
}

// CHECK-LABEL: arc.model "BasicEnableGrouping"
arc.model "BasicEnableGrouping" {
^bb0(%arg0: !arc.storage):
  %c0_i4 = hw.constant 0 : i4
  %in_clock = arc.root_input "clock", %arg0 : (!arc.storage) -> !arc.state<i1>
  %in_i0 = arc.root_input "i0", %arg0 : (!arc.storage) -> !arc.state<i4>
  %in_i1 = arc.root_input "i1", %arg0 : (!arc.storage) -> !arc.state<i4>
  %in_en0 = arc.root_input "en0", %arg0 : (!arc.storage) -> !arc.state<i1>
  %in_en1 = arc.root_input "en1", %arg0 : (!arc.storage) -> !arc.state<i1>
  %0 = arc.state_read %in_clock : <i1>
  // Group enables:
  arc.clock_tree %0 {
    //  CHECK: [[IN_EN0:%.+]] = arc.state_read %in_en0
    %3 = arc.state_read %in_en0 : <i1>
    //   CHECK-NEXT: arc.state_write [[FOO_ALLOC:%.+]] = %c0_i4
    //   CHECK-NEXT: arc.state_write [[BAR_ALLOC:%.+]] = %c0_i4
    arc.state_write %1 = %c0_i4 : <i4>
    arc.state_write %2 = %c0_i4 : <i4>
    // CHECK-NEXT:   scf.if [[IN_EN0]] {
    // state_reads are pulled in:
    // CHECK-NEXT:   [[IN_I0:%.+]] = arc.state_read %in_i0
    // CHECK-NEXT:    arc.state_write [[FOO_ALLOC]] = [[IN_I0]]
    // CHECK-NEXT:   [[IN_I1:%.+]] = arc.state_read %in_i1
    // CHECK-NEXT:    arc.state_write [[BAR_ALLOC]] = [[IN_I1]]
    %4 = arc.state_read %in_i0 : <i4>
    arc.state_write %1 = %4 if %3 : <i4>
    %5 = arc.state_read %in_i1 : <i4>
    arc.state_write %2 = %5 if %3 : <i4>
    // CHECK-NEXT:  }
  // CHECK-NEXT: }
  }
  // Don't group non-matching enables:
  arc.clock_tree %0 {
    //  CHECK: [[IN_EN0_1:%.+]] = arc.state_read %in_en0
    %6 = arc.state_read %in_en0 : <i1>
    //  CHECK-NEXT: [[IN_EN1_1:%.+]] = arc.state_read %in_en1
    %7 = arc.state_read %in_en1 : <i1>
    //   CHECK-NEXT: arc.state_write [[FOO_ALLOC]] = %c0_i4
    //   CHECK-NEXT: arc.state_write [[BAR_ALLOC]] = %c0_i4
    arc.state_write %1 = %c0_i4 : <i4>
    arc.state_write %2 = %c0_i4 : <i4>
    // CHECK-NEXT:   [[IN_I0_1:%.+]] = arc.state_read %in_i0
    // CHECK-NEXT:   arc.state_write [[FOO_ALLOC]] = [[IN_I0_1]] if [[IN_EN0_1]]
    // CHECK-NEXT:   [[IN_I1_1:%.+]] = arc.state_read %in_i1
    // CHECK-NEXT:   arc.state_write [[BAR_ALLOC]] = [[IN_I1_1]] if [[IN_EN1_1]]
    %8 = arc.state_read %in_i0 : <i4>
    arc.state_write %1 = %8 if %6 : <i4>
    %9 = arc.state_read %in_i1 : <i4>
    arc.state_write %2 = %9 if %7 : <i4>
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: [[FOO_ALLOC]] = arc.alloc_state %arg0 {name = "foo"} : (!arc.storage) -> !arc.state<i4>
  // CHECK-NEXT: [[BAR_ALLOC]] = arc.alloc_state %arg0 {name = "bar"} : (!arc.storage) -> !arc.state<i4>
  %1 = arc.alloc_state %arg0 {name = "foo"} : (!arc.storage) -> !arc.state<i4>
  %2 = arc.alloc_state %arg0 {name = "bar"} : (!arc.storage) -> !arc.state<i4>
}

// CHECK-LABEL: arc.model "GroupAssignmentsInIfTesting"
arc.model "GroupAssignmentsInIfTesting" {
^bb0(%arg0: !arc.storage):
  %in_clock = arc.root_input "clock", %arg0 : (!arc.storage) -> !arc.state<i1>
  %in_i1 = arc.root_input "i1", %arg0 : (!arc.storage) -> !arc.state<i4>
  %in_i2 = arc.root_input "i2", %arg0 : (!arc.storage) -> !arc.state<i4>
  %in_cond0 = arc.root_input "cond0", %arg0 : (!arc.storage) -> !arc.state<i1>
  %in_cond1 = arc.root_input "cond1", %arg0 : (!arc.storage) -> !arc.state<i1>
  %0 = arc.state_read %in_clock : <i1>
  // Do pull value in (1st and 2nd layer)
  arc.clock_tree %0 {
    // CHECK: [[IN_COND0:%.+]] = arc.state_read %in_cond0
    %3 = arc.state_read %in_cond0 : <i1>
    %4 = arc.state_read %in_i1 : <i4>
    %5 = arc.state_read %in_i2 : <i4>
    // CHECK-NEXT: scf.if [[IN_COND0]] {
    scf.if %3 {
      // CHECK-NEXT: [[IN_I1:%.+]] = arc.state_read %in_i1
      // CHECK-NEXT: arc.state_write [[FOO_ALLOC:%.+]] = [[IN_I1]]
      arc.state_write %1 = %4 : <i4>
      // CHECK: [[IN_COND1:%.+]] = arc.state_read %in_cond1
      %6 = arc.state_read %in_cond1 : <i1>
      // CHECK-NEXT: scf.if [[IN_COND1]] {
      scf.if %6 {
        // CHECK-NEXT: [[IN_I2:%.+]] = arc.state_read %in_i2
        // CHECK-NEXT: arc.state_write [[BAR_ALLOC:%.+]] = [[IN_I2]]
        arc.state_write %2 = %5 : <i4>
        // CHECK-NEXT: }
      }
      // CHECK-NEXT: }
    }
  }
  // CHECK-NEXT: }
  // Don't pull value in
  arc.clock_tree %0 {
    // CHECK: [[IN_COND0:%.+]] = arc.state_read %in_cond0
    %5 = arc.state_read %in_cond0 : <i1>
    // CHECK-NEXT: [[IN_I1:%.+]] = arc.state_read %in_i1
    %6 = arc.state_read %in_i1 : <i4>
    // CHECK-NEXT: scf.if [[IN_COND0]] {
    scf.if %5 {
      // CHECK-NEXT: arc.state_write [[FOO_ALLOC:%.+]] = [[IN_I1]]
      arc.state_write %1 = %6 : <i4>
      // CHECK-NEXT: }
    }
    // CHECK-NEXT: arc.state_write [[BAR_ALLOC:%.+]] = [[IN_I1]]
    arc.state_write %2 = %6 : <i4>
  // CHECK-NEXT: }
  }
  // Pull multi-use value into first if only
  arc.clock_tree %0 {
    // CHECK: [[IN_COND0:%.+]] = arc.state_read %in_cond0
    %5 = arc.state_read %in_cond0 : <i1>
    %6 = arc.state_read %in_cond1 : <i1>
    %7 = arc.state_read %in_i1 : <i4>
    // CHECK-NEXT: scf.if [[IN_COND0]] {
    scf.if %5 {
      // CHECK-NEXT: [[IN_I1:%.+]] = arc.state_read %in_i1
      // CHECK-NEXT: arc.state_write [[FOO_ALLOC:%.+]] = [[IN_I1]]
      arc.state_write %1 = %7 : <i4>
      // CHECK-NEXT: [[IN_COND1:%.+]] = arc.state_read %in_cond1
      // CHECK-NEXT: scf.if [[IN_COND1]] {
      scf.if %6 {
        // CHECK-NEXT: arc.state_write [[BAR_ALLOC:%.+]] = [[IN_I1]]
        arc.state_write %2 = %7 : <i4>
        // CHECK-NEXT: }
      }
      // CHECK-NEXT: }
    }
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: [[FOO_ALLOC]] = arc.alloc_state %arg0 {name = "foo"} : (!arc.storage) -> !arc.state<i4>
  // CHECK-NEXT: [[BAR_ALLOC]] = arc.alloc_state %arg0 {name = "bar"} : (!arc.storage) -> !arc.state<i4>
  %1 = arc.alloc_state %arg0 {name = "foo"} : (!arc.storage) -> !arc.state<i4>
  %2 = arc.alloc_state %arg0 {name = "bar"} : (!arc.storage) -> !arc.state<i4>
}

// CHECK-LABEL: arc.model "ResetAndEnableGrouping"
arc.model "ResetAndEnableGrouping" {
^bb0(%arg0: !arc.storage):
  %c0_i4 = hw.constant 0 : i4
  %in_clock = arc.root_input "clock", %arg0 : (!arc.storage) -> !arc.state<i1>
  %in_i0 = arc.root_input "i0", %arg0 : (!arc.storage) -> !arc.state<i4>
  %in_i1 = arc.root_input "i1", %arg0 : (!arc.storage) -> !arc.state<i4>
  %in_reset = arc.root_input "reset", %arg0 : (!arc.storage) -> !arc.state<i1>
  %in_en0 = arc.root_input "en0", %arg0 : (!arc.storage) -> !arc.state<i1>
  %in_en1 = arc.root_input "en1", %arg0 : (!arc.storage) -> !arc.state<i1>
  %0 = arc.state_read %in_clock : <i1>
  // Group enables inside resets (and pull in reads):
  arc.clock_tree %0 {
    //  CHECK: [[IN_RESET:%.+]] = arc.state_read %in_reset
    %3 = arc.state_read %in_reset : <i1>
    //  CHECK-NEXT: scf.if [[IN_RESET]] {
    scf.if %3 {
      //   CHECK-NEXT: arc.state_write [[FOO_ALLOC:%.+]] = %c0_i4
      //   CHECK-NEXT: arc.state_write [[BAR_ALLOC:%.+]] = %c0_i4
      arc.state_write %1 = %c0_i4 : <i4>
      arc.state_write %2 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT:   [[IN_EN:%.+]] = arc.state_read %in_en1
      %4 = arc.state_read %in_en1 : <i1>
      // CHECK-NEXT:   scf.if [[IN_EN]] {
      // CHECK-NEXT:   [[IN_I0:%.+]] = arc.state_read %in_i0
      // CHECK-NEXT:    arc.state_write [[FOO_ALLOC]] = [[IN_I0]]
      // CHECK-NEXT:   [[IN_I1:%.+]] = arc.state_read %in_i1
      // CHECK-NEXT:    arc.state_write [[BAR_ALLOC]] = [[IN_I1]]
      %5 = arc.state_read %in_i0 : <i4>
      arc.state_write %1 = %5 if %4 : <i4>
      %6 = arc.state_read %in_i1 : <i4>
      arc.state_write %2 = %6 if %4 : <i4>
      // CHECK-NEXT:   }
    // CHECK-NEXT:  }
    }
  // CHECK-NEXT: }
  }
  // Group both resets and enables (and pull in reads):
  arc.clock_tree %0 {
    //  CHECK: [[IN_RESET:%.+]] = arc.state_read %in_reset
    %7 = arc.state_read %in_reset : <i1>
    %8 = arc.state_read %in_en0 : <i1>
    //  CHECK-NEXT: scf.if [[IN_RESET]] {
    scf.if %7 {
      //   CHECK-NEXT:  arc.state_write [[FOO_ALLOC]] = %c0_i4
      //   CHECK-NEXT:  arc.state_write [[BAR_ALLOC]] = %c0_i4
      arc.state_write %1 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT: [[IN_EN0:%.+]] = arc.state_read %in_en0
      // State reads are pulled in
      // CHECK-NEXT:   scf.if [[IN_EN0]] {
      // CHECK-NEXT:   [[IN_I0:%.+]] = arc.state_read %in_i0
      // CHECK-NEXT:    arc.state_write [[FOO_ALLOC]] = [[IN_I0]]
      // CHECK-NEXT:   [[IN_I1:%.+]] = arc.state_read %in_i1
      // CHECK-NEXT:    arc.state_write [[BAR_ALLOC]] = [[IN_I1]]
      // CHECK-NEXT:   }
      %9 = arc.state_read %in_i0 : <i4>
      arc.state_write %1 = %9 if %8 : <i4>
      // CHECK-NEXT: }
    }
    scf.if %7 {
      arc.state_write %2 = %c0_i4 : <i4>
    } else {
      %10 = arc.state_read %in_i1 : <i4>
      arc.state_write %2 = %10 if %8 : <i4>
    }
    // CHECK-NEXT: }
  }
  // Group resets that are separated by an enable read (and pull in reads):
  arc.clock_tree %0 {
    //  CHECK: [[IN_RESET:%.+]] = arc.state_read %in_reset
    %11 = arc.state_read %in_reset : <i1>
    //  CHECK-NEXT: scf.if [[IN_RESET]] {
    scf.if %11 {
      //   CHECK-NEXT:  arc.state_write [[FOO_ALLOC]] = %c0_i4
      //   CHECK-NEXT:  arc.state_write [[BAR_ALLOC]] = %c0_i4
      arc.state_write %1 = %c0_i4 : <i4>
      //   CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT: [[IN_EN0:%.+]] = arc.state_read %in_en0
      // CHECK-NEXT:  [[IN_I0:%.+]] = arc.state_read %in_i0
      %12 = arc.state_read %in_en0 : <i1>
      // CHECK-NEXT:  arc.state_write [[FOO_ALLOC]] = [[IN_I0]] if [[IN_EN0]]
      // CHECK-NEXT:  [[IN_I1:%.+]] = arc.state_read %in_i1
      // CHECK-NEXT:  [[IN_EN1:%.+]] = arc.state_read %in_en1
      // CHECK-NEXT:  arc.state_write [[BAR_ALLOC]] = [[IN_I1]] if [[IN_EN1]]
      %13 = arc.state_read %in_i0 : <i4>
      arc.state_write %1 = %13 if %12 : <i4>
      // CHECK-NEXT: }
    }
    %14 = arc.state_read %in_en1 : <i1>
    scf.if %11 {
      arc.state_write %2 = %c0_i4 : <i4>
    } else {
      %15 = arc.state_read %in_i1 : <i4>
      arc.state_write %2 = %15 if %14 : <i4>
    }
    // CHECK-NEXT: }
  }
  // CHECK-NEXT: [[FOO_ALLOC]] = arc.alloc_state %arg0 {name = "foo"} : (!arc.storage) -> !arc.state<i4>
  // CHECK-NEXT: [[BAR_ALLOC]] = arc.alloc_state %arg0 {name = "bar"} : (!arc.storage) -> !arc.state<i4>
  %1 = arc.alloc_state %arg0 {name = "foo"} : (!arc.storage) -> !arc.state<i4>
  %2 = arc.alloc_state %arg0 {name = "bar"} : (!arc.storage) -> !arc.state<i4>
}
