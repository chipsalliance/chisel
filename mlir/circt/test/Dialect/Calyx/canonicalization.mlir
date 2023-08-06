// RUN: circt-opt %s -canonicalize -split-input-file | FileCheck %s

// Nested SeqOps are collapsed.
module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    // CHECK-LABEL: calyx.control {
    // CHECK-NEXT:    calyx.seq {
    // CHECK-NEXT:      calyx.enable @A
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }
    calyx.control {
      calyx.seq {
        calyx.seq {
          calyx.enable @A
        }
      }
    }
  }
}

// -----

// Nested ParOps are collapsed.
module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    // CHECK-LABEL: calyx.control {
    // CHECK-NEXT:    calyx.par {
    // CHECK-NEXT:      calyx.enable @A
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }
    calyx.control {
      calyx.par {
        calyx.par {
          calyx.enable @A
        }
      }
    }
  }
}

// -----

// IfOp nested in SeqOp removes common tail from within SeqOps.
module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %eq.left, %eq.right, %eq.out = calyx.std_eq @eq : i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.comb_group @Cond {
        calyx.assign %eq.left =  %c1_1 : i1
        calyx.assign %eq.right = %c1_1 : i1
      }
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
      calyx.group @B {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
      calyx.group @C {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    // CHECK-LABEL: calyx.control {
    // CHECK-NEXT:    calyx.seq {
    // CHECK-NEXT:      calyx.if %eq.out with @Cond {
    // CHECK-NEXT:        calyx.seq {
    // CHECK-NEXT:          calyx.enable @B
    // CHECK-NEXT:        }
    // CHECK-NEXT:      } else {
    // CHECK-NEXT:        calyx.seq {
    // CHECK-NEXT:          calyx.enable @C
    // CHECK-NEXT:        }
    // CHECK-NEXT:     }
    // CHECK-NEXT:     calyx.enable @A
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    calyx.control {
      calyx.seq {
        calyx.if %eq.out with @Cond {
          calyx.seq {
            calyx.enable @B
            calyx.enable @A
          }
        } else {
          calyx.seq {
            calyx.enable @C
            calyx.enable @A
          }
        }
      }
    }
  }
}

// -----

// IfOp nested in ParOp removes common tails from within ParOps.
module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %eq.left, %eq.right, %eq.out = calyx.std_eq @eq : i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.comb_group @Cond {
        calyx.assign %eq.left =  %c1_1 : i1
        calyx.assign %eq.right = %c1_1 : i1
      }
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
      calyx.group @B {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
      calyx.group @C {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
      calyx.group @D {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    // CHECK-LABEL: calyx.control {
    // CHECK-NEXT:    calyx.par {
    // CHECK-NEXT:      calyx.if %eq.out with @Cond {
    // CHECK-NEXT:        calyx.par {
    // CHECK-NEXT:          calyx.enable @A
    // CHECK-NEXT:        }
    // CHECK-NEXT:      } else {
    // CHECK-NEXT:        calyx.par {
    // CHECK-NEXT:          calyx.enable @B
    // CHECK-NEXT:        }
    // CHECK-NEXT:     }
    // CHECK-NEXT:     calyx.enable @C
    // CHECK-NEXT:     calyx.enable @D
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    calyx.control {
      calyx.par {
        calyx.if %eq.out with @Cond {
          calyx.par {
            calyx.enable @A
            calyx.enable @C
            calyx.enable @D
          }
        } else {
          calyx.par {
            calyx.enable @B
            calyx.enable @C
            calyx.enable @D
          }
        }
      }
    }
  }
}

// -----

// IfOp nested in ParOp removes common tail from within SeqOps. The important check
// here is ensuring the removed EnableOps are still computed sequentially.
module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %eq.left, %eq.right, %eq.out = calyx.std_eq @eq : i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.comb_group @Cond {
        calyx.assign %eq.left =  %c1_1 : i1
        calyx.assign %eq.right = %c1_1 : i1
      }
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
      calyx.group @B {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
      calyx.group @C {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    // CHECK-LABEL: calyx.control {
    // CHECK-NEXT:    calyx.par {
    // CHECK-NEXT:      calyx.seq {
    // CHECK-NEXT:        calyx.if %eq.out with @Cond {
    // CHECK-NEXT:          calyx.seq {
    // CHECK-NEXT:            calyx.enable @B
    // CHECK-NEXT:          }
    // CHECK-NEXT:        } else {
    // CHECK-NEXT:          calyx.seq {
    // CHECK-NEXT:            calyx.enable @C
    // CHECK-NEXT:          }
    // CHECK-NEXT:        }
    // CHECK-NEXT:        calyx.enable @A
    // CHECK-NEXT:      }
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }
    calyx.control {
      calyx.par {
        calyx.if %eq.out with @Cond {
          calyx.seq {
            calyx.enable @B
            calyx.enable @A
          }
        } else {
          calyx.seq {
            calyx.enable @C
            calyx.enable @A
          }
        }
      }
    }
  }
}

// -----

// IfOp nested in ParOp removes common tail from within StaticSeqOps. The important check
// here is ensuring the removed EnableOps are still computed sequentially.
module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %eq.in, %eq.write_en, %eq.clk, %eq.reset, %eq.out, %eq.done = calyx.register @eq : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.static_group latency<1> @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
      }
      calyx.static_group latency<1>  @B {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
      }
      calyx.static_group latency<1>  @C {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
      }
    }
    // CHECK-LABEL: calyx.control {
    // CHECK-NEXT:    calyx.par {
    // CHECK-NEXT:      calyx.static_seq {
    // CHECK-NEXT:        calyx.static_if %eq.out {
    // CHECK-NEXT:          calyx.static_seq {
    // CHECK-NEXT:            calyx.enable @B
    // CHECK-NEXT:          }
    // CHECK-NEXT:        } else {
    // CHECK-NEXT:          calyx.static_seq {
    // CHECK-NEXT:            calyx.enable @C
    // CHECK-NEXT:          }
    // CHECK-NEXT:        }
    // CHECK-NEXT:        calyx.enable @A
    // CHECK-NEXT:      }
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }
    calyx.control {
      calyx.par {
        calyx.static_if %eq.out {
          calyx.static_seq {
            calyx.enable @B
            calyx.enable @A
          }
        } else {
          calyx.static_seq {
            calyx.enable @C
            calyx.enable @A
          }
        }
      }
    }
  }
}

// -----

// IfOp nested in SeqOp removes common tail from within ParOps. The important check
// here is ensuring the removed EnableOps are still computed in parallel.
module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %eq.left, %eq.right, %eq.out = calyx.std_eq @eq : i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.comb_group @Cond {
        calyx.assign %eq.left =  %c1_1 : i1
        calyx.assign %eq.right = %c1_1 : i1
      }
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
      calyx.group @B {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
      calyx.group @C {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
      calyx.group @D {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    // CHECK-LABEL: calyx.control {
    // CHECK-NEXT:    calyx.seq {
    // CHECK-NEXT:      calyx.par {
    // CHECK-NEXT:        calyx.if %eq.out with @Cond {
    // CHECK-NEXT:          calyx.par {
    // CHECK-NEXT:            calyx.enable @A
    // CHECK-NEXT:          }
    // CHECK-NEXT:        } else {
    // CHECK-NEXT:          calyx.par {
    // CHECK-NEXT:            calyx.enable @B
    // CHECK-NEXT:          }
    // CHECK-NEXT:        }
    // CHECK-NEXT:        calyx.enable @C
    // CHECK-NEXT:        calyx.enable @D
    // CHECK-NEXT:      }
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }
    calyx.control {
      calyx.seq {
        calyx.if %eq.out with @Cond {
          calyx.par {
            calyx.enable @A
            calyx.enable @C
            calyx.enable @D
          }
        } else {
          calyx.par {
            calyx.enable @B
            calyx.enable @C
            calyx.enable @D
          }
        }
      }
    }
  }
}

// -----

// StaticIfOp nested in SeqOp removes common tail from within StaticParOps. The important check
// here is ensuring the removed EnableOps are still computed in parallel.
module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %eq.in, %eq.write_en, %eq.clk, %eq.reset, %eq.out, %eq.done = calyx.register @eq : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.static_group latency<1> @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
      }
      calyx.static_group latency<1> @B {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
      }
      calyx.static_group latency<1> @C {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
      }
      calyx.static_group latency<1> @D {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
      }
    }
    // CHECK-LABEL: calyx.control {
    // CHECK-NEXT:    calyx.seq {
    // CHECK-NEXT:      calyx.static_par {
    // CHECK-NEXT:        calyx.static_if %eq.out {
    // CHECK-NEXT:          calyx.static_par {
    // CHECK-NEXT:            calyx.enable @A
    // CHECK-NEXT:          }
    // CHECK-NEXT:        } else {
    // CHECK-NEXT:          calyx.static_par {
    // CHECK-NEXT:            calyx.enable @B
    // CHECK-NEXT:          }
    // CHECK-NEXT:        }
    // CHECK-NEXT:        calyx.enable @C
    // CHECK-NEXT:        calyx.enable @D
    // CHECK-NEXT:      }
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }
    calyx.control {
      calyx.seq {
        calyx.static_if %eq.out {
          calyx.static_par {
            calyx.enable @A
            calyx.enable @C
            calyx.enable @D
          }
        } else {
          calyx.static_par {
            calyx.enable @B
            calyx.enable @C
            calyx.enable @D
          }
        }
      }
    }
  }
}

// -----

// Empty Then and Else regions lead to the removal of the IfOp (as well as unused cells and groups).
module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    // CHECK-NOT: %eq.left, %eq.right, %eq.out = calyx.std_eq @eq : i1, i1, i1
    %eq.left, %eq.right, %eq.out = calyx.std_eq @eq : i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      // CHECK-NOT: calyx.comp_group @Cond
      calyx.comb_group @Cond {
        calyx.assign %eq.left =  %c1_1 : i1
        calyx.assign %eq.right = %c1_1 : i1
      }
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    // CHECK-LABEL: calyx.control {
    // CHECK-NEXT:    calyx.seq {
    // CHECK-NEXT:      calyx.enable @A
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }
    calyx.control {
      calyx.seq {
        calyx.if %eq.out with @Cond {
          calyx.seq {
            calyx.enable @A
          }
        } else {
          calyx.seq {
            calyx.enable @A
          }
        }
      }
    }
  }
}

// -----

// Empty Then region and no Else region leads to removal of IfOp (as well as unused cells).
module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    // CHECK-NOT: %eq.left, %eq.right, %eq.out = calyx.std_eq @eq : i1, i1, i1
    %eq.left, %eq.right, %eq.out = calyx.std_eq @eq : i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    // CHECK-LABEL: calyx.control {
    // CHECK-NEXT:    calyx.seq {
    // CHECK-NEXT:      calyx.enable @A
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }
    calyx.control {
      calyx.seq {
        calyx.enable @A
        calyx.if %eq.out {}
      }
    }
  }
}

// -----

// Empty Then region and no Else region leads to removal of StaticIfOp (as well as unused cells).
module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    // CHECK-NOT: %eq.in, %eq.write_en, %eq.clk, %eq.reset, %eq.out, %eq.done = calyx.register @eq : i1, i1, i1, i1, i1, i1
    %eq.in, %eq.write_en, %eq.clk, %eq.reset, %eq.out, %eq.done = calyx.register @eq : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    // CHECK-LABEL: calyx.control {
    // CHECK-NEXT:    calyx.seq {
    // CHECK-NEXT:      calyx.enable @A
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }
    calyx.control {
      calyx.seq {
        calyx.enable @A
        calyx.static_if %eq.out {}
      }
    }
  }
}

// -----

// Empty body leads to removal of WhileOp (as well as unused cells and groups).
module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    // CHECK: %eq.left, %eq.right, %eq.out = calyx.std_eq @eq : i1, i1, i1
    %eq.left, %eq.right, %eq.out = calyx.std_eq @eq : i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      // CHECK-NOT: calyx.comp_group @Cond
      calyx.comb_group @Cond {
        calyx.assign %eq.left =  %c1_1 : i1
        calyx.assign %eq.right = %c1_1 : i1
      }
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        // Use the `std_eq` here to verify it is not removed.
        calyx.assign %r.write_en = %eq.out : i1
        calyx.group_done %r.done : i1
      }
    }
    // CHECK-LABEL: calyx.control {
    // CHECK-NEXT:    calyx.seq {
    // CHECK-NEXT:      calyx.enable @A
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }
    calyx.control {
      calyx.seq {
        calyx.enable @A
        calyx.while %eq.out with @Cond {}
      }
    }
  }
}

// -----

// Empty ParOp and SeqOp are removed.
module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    // CHECK-LABEL: calyx.control {
    // CHECK-NEXT:    calyx.seq {
    // CHECK-NEXT:      calyx.enable @A
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }
    calyx.control {
      calyx.seq {
        calyx.enable @A
        calyx.seq { calyx.seq {} }
        calyx.par { calyx.seq {} }
      }
    }
  }
}

// -----

// Unary control operations are collapsed.
module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
      calyx.group @B {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    // CHECK-LABEL: calyx.control {
    // CHECK-NEXT:    calyx.seq {
    // CHECK-NEXT:      calyx.enable @B
    // CHECK-NEXT:      calyx.enable @A
    // CHECK-NEXT:      calyx.enable @B
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }
    calyx.control {
      calyx.seq {
        calyx.enable @B
        calyx.par {
          calyx.seq {
            calyx.enable @A
          }
        }
        calyx.enable @B
      }
    }
  }
}
