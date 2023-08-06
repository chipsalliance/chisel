// RUN: circt-opt -hw-cleanup %s | FileCheck %s
// RUN: circt-opt -hw-cleanup %s | FileCheck %s
// RUN: circt-opt -hw-cleanup="merge-always-blocks=false" %s | FileCheck %s --check-prefix=SEPARATE

//CHECK-LABEL: hw.module @alwaysff_basic(%arg0: i1, %arg1: i1) {
//CHECK-NEXT:   [[FD:%.*]] = hw.constant -2147483646 : i32
//CHECK-NEXT:   sv.initial {
//CHECK-NEXT:     sv.fwrite [[FD]], "Middle\0A"
//CHECK-NEXT:   }
//CHECK-NEXT:   sv.alwaysff(posedge %arg0)  {
//CHECK-NEXT:     sv.fwrite [[FD]], "A1"
//CHECK-NEXT:     sv.fwrite [[FD]], "A2"
//CHECK-NEXT:   }
//CHECK-NEXT:   sv.alwaysff(posedge %arg1)  {
//CHECK-NEXT:     sv.fwrite [[FD]], "B1"
//CHECK-NEXT:     sv.fwrite [[FD]], "B2"
//CHECK-NEXT:   }
//CHECK-NEXT:   hw.output
//CHECK-NEXT: }

//SEPARATE-LABEL: hw.module @alwaysff_basic(%arg0: i1, %arg1: i1) {
//SEPARATE-NEXT:   [[FD:%.*]] = hw.constant -2147483646 : i32
//SEPARATE-NEXT:   sv.alwaysff(posedge %arg0)  {
//SEPARATE-NEXT:     sv.fwrite [[FD]], "A1"
//SEPARATE-NEXT:   }
//SEPARATE-NEXT:   sv.alwaysff(posedge %arg1)  {
//SEPARATE-NEXT:     sv.fwrite [[FD]], "B1"
//SEPARATE-NEXT:   }
//SEPARATE-NEXT:   sv.initial {
//SEPARATE-NEXT:     sv.fwrite [[FD]], "Middle\0A"
//SEPARATE-NEXT:   }
//SEPARATE-NEXT:   sv.alwaysff(posedge %arg0)  {
//SEPARATE-NEXT:     sv.fwrite [[FD]], "A2"
//SEPARATE-NEXT:   }
//SEPARATE-NEXT:   sv.alwaysff(posedge %arg1)  {
//SEPARATE-NEXT:     sv.fwrite [[FD]], "B2"
//SEPARATE-NEXT:   }
//SEPARATE-NEXT:   hw.output
//SEPARATE-NEXT: }

hw.module @alwaysff_basic(%arg0: i1, %arg1: i1) {
  %fd = hw.constant 0x80000002 : i32

  sv.alwaysff(posedge %arg0) {
    sv.fwrite %fd, "A1"
  }
  sv.alwaysff(posedge %arg1) {
    sv.fwrite %fd, "B1"
  }
  sv.initial {
    sv.fwrite %fd, "Middle\n"
  }
  sv.alwaysff(posedge %arg0) {
    sv.fwrite %fd, "A2"
  }
  sv.alwaysff(posedge %arg1) {
    sv.fwrite %fd, "B2"
  }
  hw.output
}

// CHECK-LABEL: hw.module @alwaysff_basic_reset(%arg0: i1, %arg1: i1) {
// CHECK-NEXT:   [[FD:%.*]] = hw.constant -2147483646 : i32
// CHECK-NEXT:   sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:     sv.fwrite [[FD]], "A1"
// CHECK-NEXT:     sv.fwrite [[FD]], "A2"
// CHECK-NEXT:   }(asyncreset : negedge %arg1)  {
// CHECK-NEXT:     sv.fwrite [[FD]], "B1"
// CHECK-NEXT:     sv.fwrite [[FD]], "B2"
// CHECK-NEXT:   }
// CHECK-NEXT:   hw.output
// CHECK-NEXT: }

hw.module @alwaysff_basic_reset(%arg0: i1, %arg1: i1) {
  %fd = hw.constant 0x80000002 : i32

  sv.alwaysff (posedge %arg0) {
    sv.fwrite %fd, "A1"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite %fd, "B1"
  }
  sv.alwaysff (posedge %arg0) {
    sv.fwrite %fd, "A2"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite %fd, "B2"
  }
  hw.output
}


// CHECK-LABEL: hw.module @alwaysff_different_reset(%arg0: i1, %arg1: i1) {
// CHECK-NEXT:   [[FD:%.*]] = hw.constant -2147483646 : i32
// CHECK-NEXT:   sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:     sv.fwrite [[FD]], "A1"
// CHECK-NEXT:     sv.fwrite [[FD]], "A2"
// CHECK-NEXT:   }(asyncreset : negedge %arg1)  {
// CHECK-NEXT:     sv.fwrite [[FD]], "B1"
// CHECK-NEXT:     sv.fwrite [[FD]], "B2"
// CHECK-NEXT:   }
// CHECK-NEXT:   sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:     sv.fwrite [[FD]], "C1"
// CHECK-NEXT:     sv.fwrite [[FD]], "C2"
// CHECK-NEXT:   }(asyncreset : posedge %arg1)  {
// CHECK-NEXT:     sv.fwrite [[FD]], "D1"
// CHECK-NEXT:     sv.fwrite [[FD]], "D2"
// CHECK-NEXT:   }
// CHECK-NEXT:   hw.output
// CHECK-NEXT: }

hw.module @alwaysff_different_reset(%arg0: i1, %arg1: i1) {
  %fd = hw.constant 0x80000002 : i32

  sv.alwaysff (posedge %arg0) {
    sv.fwrite %fd, "A1"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite %fd, "B1"
  }
  sv.alwaysff (posedge %arg0) {
    sv.fwrite %fd, "C1"
  } ( asyncreset : posedge %arg1) {
    sv.fwrite %fd, "D1"
  }
  sv.alwaysff (posedge %arg0) {
    sv.fwrite %fd, "A2"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite %fd, "B2"
  }
  sv.alwaysff (posedge %arg0) {
    sv.fwrite %fd, "C2"
  } ( asyncreset : posedge %arg1) {
    sv.fwrite %fd, "D2"
  }
  hw.output
}

//CHECK-LABEL: hw.module @alwaysff_ifdef(%arg0: i1) {
//CHECK-NEXT:  [[FD:%.*]] = hw.constant -2147483646 : i32
//CHECK-NEXT:  sv.ifdef "FOO" {
//CHECK-NEXT:     sv.alwaysff(posedge %arg0)  {
//CHECK-NEXT:       sv.fwrite [[FD]], "A1"
//CHECK-NEXT:       sv.fwrite [[FD]], "B1"
//CHECK-NEXT:     }
//CHECK-NEXT:   }
//CHECK-NEXT:   hw.output
//CHECK-NEXT: }

hw.module @alwaysff_ifdef(%arg0: i1) {
  %fd = hw.constant 0x80000002 : i32

  sv.ifdef "FOO" {
    sv.alwaysff(posedge %arg0) {
      sv.fwrite %fd, "A1"
    }
    sv.alwaysff(posedge %arg0) {
      sv.fwrite %fd, "B1"
    }
  }
  hw.output
}

// CHECK-LABEL: hw.module @ifdef_merge(%arg0: i1) {
// CHECK-NEXT:    [[FD:%.*]] = hw.constant -2147483646 : i32
// CHECK-NEXT:    sv.ifdef "FOO"  {
// CHECK-NEXT:      sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:        sv.fwrite [[FD]], "A1"
// CHECK-NEXT:        sv.fwrite [[FD]], "B1"
// CHECK-NEXT:      }
// CHECK-NEXT:    }
hw.module @ifdef_merge(%arg0: i1) {
  %fd = hw.constant 0x80000002 : i32

  sv.ifdef "FOO" {
    sv.alwaysff(posedge %arg0) {
      sv.fwrite %fd, "A1"
    }
  }
  sv.ifdef "FOO" {
    sv.alwaysff(posedge %arg0) {
      sv.fwrite %fd, "B1"
    }
  }
  hw.output
}

// CHECK-LABEL: hw.module @ifdef_proc_merge(%arg0: i1) {
// CHECK-NEXT:    [[FD:%.*]] = hw.constant -2147483646 : i32
// CHECK-NEXT:    sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:      %true = hw.constant true
// CHECK-NEXT:      [[XOR:%.*]] = comb.xor %arg0, %true : i1
// CHECK-NEXT:      sv.ifdef.procedural "FOO"  {
// CHECK-NEXT:        sv.fwrite [[FD]], "A1"
// CHECK-NEXT:        sv.fwrite [[FD]], "%x"([[XOR]]) : i1
// CHECK-NEXT:      }
// CHECK-NEXT:      sv.ifdef.procedural "BAR"  {
// CHECK-NEXT:        sv.fwrite [[FD]], "B1"
// CHECK-NEXT:      }
// CHECK-NEXT:    }
hw.module @ifdef_proc_merge(%arg0: i1) {
  %fd = hw.constant 0x80000002 : i32

  sv.alwaysff(posedge %arg0) {
    sv.ifdef.procedural "FOO" {
      sv.fwrite %fd, "A1"
    }
    %true = hw.constant true
    %0 = comb.xor %arg0, %true : i1
    sv.ifdef.procedural "FOO" {
       sv.fwrite %fd, "%x"(%0) : i1
    }
     sv.ifdef.procedural "BAR" {
       sv.fwrite %fd, "B1"
    }
  }
  hw.output
}

// CHECK-LABEL: hw.module @if_merge(%arg0: i1, %arg1: i1) {
// CHECK-NEXT:    [[FD:%.*]] = hw.constant -2147483646 : i32
// CHECK-NEXT:    sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:      %true = hw.constant true
// CHECK-NEXT:      [[XOR:%.*]] = comb.xor %arg1, %true : i1
// CHECK-NEXT:      sv.if %arg1  {
// CHECK-NEXT:        sv.fwrite [[FD]], "A1"
// CHECK-NEXT:        sv.fwrite [[FD]], "%x"([[XOR:%.*]]) : i1
// CHECK-NEXT:      }
// CHECK-NEXT:      sv.if [[XOR]] {
// CHECK-NEXT:        sv.fwrite [[FD]], "B1"
// CHECK-NEXT:      }
// CHECK-NEXT:    }
hw.module @if_merge(%arg0: i1, %arg1: i1) {
  %fd = hw.constant 0x80000002 : i32

  sv.alwaysff(posedge %arg0) {
    sv.if %arg1 {
      sv.fwrite %fd, "A1"
    }
    %true = hw.constant true
    %0 = comb.xor %arg1, %true : i1
    sv.if %arg1 {
      sv.fwrite %fd, "%x"(%0) : i1
    }
    sv.if %0 {
      sv.fwrite %fd, "B1"
    }
  }
  hw.output
}


// CHECK-LABEL: hw.module @initial_merge(%arg0: i1) {
// CHECK-NEXT:    [[FD:%.*]] = hw.constant -2147483646 : i32
// CHECK-NEXT:    sv.initial {
// CHECK-NEXT:      sv.fwrite [[FD]], "A1"
// CHECK-NEXT:      sv.fwrite [[FD]], "B1"
// CHECK-NEXT:    }
hw.module @initial_merge(%arg0: i1) {
  %fd = hw.constant 0x80000002 : i32

  sv.initial {
    sv.fwrite %fd, "A1"
  }
  sv.initial {
    sv.fwrite %fd, "B1"
  }
  hw.output
}

//CHECK-LABEL: hw.module @always_basic(%arg0: i1, %arg1: i1) {
//CHECK-NEXT:   [[FD:%.*]] = hw.constant -2147483646 : i32
//CHECK-NEXT:   sv.initial {
//CHECK-NEXT:     sv.fwrite [[FD]], "Middle\0A"
//CHECK-NEXT:   }
//CHECK-NEXT:   sv.always   posedge %arg0   {
//CHECK-NEXT:     sv.fwrite [[FD]], "A1"
//CHECK-NEXT:     sv.fwrite [[FD]], "A2"
//CHECK-NEXT:   }
//CHECK-NEXT:   sv.always   posedge %arg1   {
//CHECK-NEXT:     sv.fwrite [[FD]], "B1"
//CHECK-NEXT:     sv.fwrite [[FD]], "B2"
//CHECK-NEXT:   }
//CHECK-NEXT:   hw.output
//CHECK-NEXT: }

//SEPARATE-LABEL: hw.module @always_basic(%arg0: i1, %arg1: i1) {
//SEPARATE-NEXT:   [[FD:%.*]] = hw.constant -2147483646 : i32
//SEPARATE-NEXT:   sv.always   posedge %arg0   {
//SEPARATE-NEXT:     sv.fwrite [[FD]], "A1"
//SEPARATE-NEXT:   }
//SEPARATE-NEXT:   sv.always   posedge %arg1   {
//SEPARATE-NEXT:     sv.fwrite [[FD]], "B1"
//SEPARATE-NEXT:   }
//SEPARATE-NEXT:   sv.initial {
//SEPARATE-NEXT:     sv.fwrite [[FD]], "Middle\0A"
//SEPARATE-NEXT:   }
//SEPARATE-NEXT:   sv.always   posedge %arg0   {
//SEPARATE-NEXT:     sv.fwrite [[FD]], "A2"
//SEPARATE-NEXT:   }
//SEPARATE-NEXT:   sv.always   posedge %arg1   {
//SEPARATE-NEXT:     sv.fwrite [[FD]], "B2"
//SEPARATE-NEXT:   }
//SEPARATE-NEXT:   hw.output
//SEPARATE-NEXT: }

hw.module @always_basic(%arg0: i1, %arg1: i1) {
  %fd = hw.constant 0x80000002 : i32

  sv.always posedge %arg0 {
    sv.fwrite %fd, "A1"
  }
  sv.always posedge %arg1 {
    sv.fwrite %fd, "B1"
  }
  sv.initial {
    sv.fwrite %fd, "Middle\n"
  }
  sv.always posedge %arg0 {
    sv.fwrite %fd, "A2"
  }
  sv.always posedge %arg1 {
    sv.fwrite %fd, "B2"
  }
  hw.output
}


// CHECK-LABEL: hw.module @nested_regions(
// CHECK-NEXT:  [[FD:%.*]] = hw.constant -2147483646 : i32
// CHECK-NEXT:  sv.initial  {
// CHECK-NEXT:    sv.ifdef.procedural "L1"  {
// CHECK-NEXT:      sv.ifdef.procedural "L2"  {
// CHECK-NEXT:        sv.ifdef.procedural "L3"  {
// CHECK-NEXT:          sv.fwrite [[FD]], "A"
// CHECK-NEXT:          sv.fwrite [[FD]], "B"
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
hw.module @nested_regions() {
  %fd = hw.constant 0x80000002 : i32

  sv.initial {
    sv.ifdef.procedural "L1" {
      sv.ifdef.procedural "L2" {
        sv.ifdef.procedural "L3" {
          sv.fwrite %fd, "A"
        }
      }
    }
  }
  sv.initial {
    sv.ifdef.procedural "L1" {
      sv.ifdef.procedural "L2" {
        sv.ifdef.procedural "L3" {
          sv.fwrite %fd, "B"
        }
      }
    }
  }
}

// CHECK-LABEL: hw.module @sv_attributes(
hw.module @sv_attributes() {
  %fd = hw.constant 0x80000002 : i32
  // Check that initials are not merged.
  // CHECK: sv.initial
  // CHECK: sv.initial
  sv.initial  {
    sv.fwrite %fd, "A"
  } {sv.attributes = [#sv.attribute<"dont_merge">]}

  sv.initial  {
    sv.fwrite %fd, "B"
  } {sv.attributes = [#sv.attribute<"dont_merge">]}
}
