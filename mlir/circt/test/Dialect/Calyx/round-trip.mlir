// RUN: circt-opt %s --verify-diagnostics --split-input-file | circt-opt --verify-diagnostics | FileCheck %s

// CHECK: module attributes {calyx.entrypoint = "main"} {
module attributes {calyx.entrypoint = "main"} {
  // CHECK-LABEL: calyx.component @A(%in: i8, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i8, %done: i1 {done}) {
  calyx.component @A(%in: i8, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i8, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }

  // CHECK-LABEL: calyx.component @B(%in: i8, %clk: i1 {clk}, %go: i1 {go}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
  calyx.component @B (%in: i8, %clk: i1 {clk}, %go: i1 {go}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }

  // CHECK-LABEL:   calyx.component @main(%clk: i1 {clk}, %go: i1 {go}, %reset: i1 {reset}) -> (%done: i1 {done}) {
  calyx.component @main(%clk: i1 {clk}, %go: i1 {go}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    // CHECK:      %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i8, i1, i1, i1, i8, i1
    // CHECK-NEXT: %r2.in, %r2.write_en, %r2.clk, %r2.reset, %r2.out, %r2.done = calyx.register @r2 : i1, i1, i1, i1, i1, i1
    // CHECK-NEXT: %mu.clk, %mu.reset, %mu.go, %mu.left, %mu.right, %mu.out, %mu.done = calyx.std_mult_pipe @mu : i1, i1, i1, i32, i32, i32, i1
    // CHECK-NEXT: %du.clk, %du.reset, %du.go, %du.left, %du.right, %du.out_quotient, %du.done = calyx.std_divu_pipe @du : i1, i1, i1, i32, i32, i32, i1
    // CHECK-NEXT: %m.addr0, %m.addr1, %m.write_data, %m.write_en, %m.clk, %m.read_data, %m.done = calyx.memory @m <[64, 64] x 8> [6, 6] : i6, i6, i8, i1, i1, i8, i1
    // CHECK-NEXT: %c0.in, %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance @c0 of @A : i8, i1, i1, i1, i8, i1
    // CHECK-NEXT: %c1.in, %c1.go, %c1.clk, %c1.reset, %c1.out, %c1.done = calyx.instance @c1 of @A : i8, i1, i1, i1, i8, i1
    // CHECK-NEXT: %c2.in, %c2.clk, %c2.go, %c2.reset, %c2.out, %c2.done = calyx.instance @c2 of @B : i8, i1, i1, i1, i1, i1
    // CHECK-NEXT: %adder.left, %adder.right, %adder.out = calyx.std_add @adder : i8, i8, i8
    // CHECK-NEXT: %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1
    // CHECK-NEXT: %pad.in, %pad.out = calyx.std_pad @pad : i8, i9
    // CHECK-NEXT: %slice.in, %slice.out = calyx.std_slice @slice : i8, i7
    // CHECK-NEXT: %not.in, %not.out = calyx.std_not @not : i1, i1
    // CHECK-NEXT: %wire.in, %wire.out = calyx.std_wire @wire : i8, i8
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i8, i1, i1, i1, i8, i1
    %r2.in, %r2.write_en, %r2.clk, %r2.reset, %r2.out, %r2.done = calyx.register @r2 : i1, i1, i1, i1, i1, i1
    %mu.clk, %mu.reset, %mu.go, %mu.lhs, %mu.rhs, %mu.out, %mu.done = calyx.std_mult_pipe @mu : i1, i1, i1, i32, i32, i32, i1
    %du.clk, %du.reset, %du.go, %du.left, %du.right, %du.out, %du.done = calyx.std_divu_pipe @du : i1, i1, i1, i32, i32, i32, i1
    %m.addr0, %m.addr1, %m.write_data, %m.write_en, %m.clk, %m.read_data, %m.done = calyx.memory @m <[64, 64] x 8> [6, 6] : i6, i6, i8, i1, i1, i8, i1
    %c0.in, %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance @c0 of @A : i8, i1, i1, i1, i8, i1
    %c1.in, %c1.go, %c1.clk, %c1.reset, %c1.out, %c1.done = calyx.instance @c1 of @A : i8, i1, i1, i1, i8, i1
    %c2.in, %c2.clk, %c2.go, %c2.reset, %c2.out, %c2.done = calyx.instance @c2 of @B : i8, i1, i1, i1, i1, i1
    %adder.left, %adder.right, %adder.out = calyx.std_add @adder : i8, i8, i8
    %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1
    %pad.in, %pad.out = calyx.std_pad @pad : i8, i9
    %slice.in, %slice.out = calyx.std_slice @slice : i8, i7
    %not.in, %not.out = calyx.std_not @not : i1, i1
    %wire.in, %wire.out = calyx.std_wire @wire : i8, i8
    %c1_i1 = hw.constant 1 : i1
    %c0_i1 = hw.constant 0 : i1
    %c0_i6 = hw.constant 0 : i6
    %c0_i8 = hw.constant 0 : i8

    calyx.wires {
      // CHECK:      calyx.assign %not.in = %r2.out : i1
      // CHECK-NEXT: calyx.assign %gt.left = %r2.out ? %adder.out : i8
      // CHECK-NEXT: calyx.assign %gt.left = %not.out ? %adder.out : i8
      // CHECK-NEXT: calyx.assign %r.in = %0 ? %c0_i8 : i8
      // CHECK-NEXT: %0 = comb.and %true, %true : i1
      calyx.assign %not.in = %r2.out : i1
      calyx.assign %gt.left = %r2.out ? %adder.out : i8
      calyx.assign %gt.left = %not.out ? %adder.out : i8
      calyx.assign %r.in = %0 ? %c0_i8 : i8
      %0 = comb.and %c1_i1, %c1_i1 : i1

      // CHECK: calyx.group @Group1 {
      calyx.group @Group1 {
        // CHECK: calyx.assign %c1.in = %c0.out : i8
        // CHECK-NEXT: calyx.group_done %c1.done : i1
        calyx.assign %c1.in = %c0.out : i8
        calyx.group_done %c1.done : i1
      }
      calyx.comb_group @ReadMemory {
        // CHECK: calyx.assign %m.addr0 = %c0_i6 : i6
        // CHECK-NEXT: calyx.assign %m.addr1 = %c0_i6 : i6
        // CHECK-NEXT: calyx.assign %gt.left = %m.read_data : i8
        // CHECK-NEXT: calyx.assign %gt.right = %c0_i8 : i8
        calyx.assign %m.addr0 = %c0_i6 : i6
        calyx.assign %m.addr1 = %c0_i6 : i6
        calyx.assign %gt.left = %m.read_data : i8
        calyx.assign %gt.right = %c0_i8 : i8
      }
      calyx.group @Group3 {
        calyx.assign %r.in = %c0.out : i8
        calyx.assign %r.write_en = %c1_i1 : i1
        calyx.group_done %r.done : i1
      }
    }
    calyx.control {
      // CHECK:      calyx.seq {
      // CHECK-NEXT: calyx.seq {
      // CHECK-NEXT: calyx.enable @Group1
      // CHECK-NEXT: calyx.enable @Group3
      // CHECK-NEXT: calyx.seq {
      // CHECK-NEXT: calyx.if %gt.out with @ReadMemory {
      // CHECK-NEXT: calyx.enable @Group1
      // CHECK-NEXT: } else {
      // CHECK-NEXT: calyx.enable @Group3
      // CHECK-NEXT: }
      // CHECK-NEXT: calyx.if %c2.out {
      // CHECK-NEXT: calyx.enable @Group1
      // CHECK-NEXT: }
      // CHECK-NEXT: calyx.while %gt.out with @ReadMemory {
      // CHECK-NEXT: calyx.while %c2.out {
      // CHECK-NEXT: calyx.enable @Group1
      // CHECK:      calyx.par {
      // CHECK-NEXT: calyx.enable @Group1
      // CHECK-NEXT: calyx.enable @Group3
      calyx.seq {
        calyx.seq {
          calyx.enable @Group1
          calyx.enable @Group3
          calyx.seq {
            calyx.if %gt.out with @ReadMemory {
              calyx.enable @Group1
            } else {
              calyx.enable @Group3
            }
            calyx.if %c2.out {
              calyx.enable @Group1
            }
            calyx.while %gt.out with @ReadMemory {
              calyx.while %c2.out {
                calyx.enable @Group1
              }
            }
          }
        }
        calyx.par {
          calyx.enable @Group1
          calyx.enable @Group3
        }
      }
    }
  }
}

// -----
// CHECK: module attributes {calyx.entrypoint = "A"} {
module attributes {calyx.entrypoint = "A"} {
  // CHECK: hw.module.extern @prim(%in: i32) -> (out: i32) attributes {filename = "test.v"}
  hw.module.extern @prim(%in: i32) -> (out: i32) attributes {filename = "test.v"}

  // CHECK: hw.module.extern @params<WIDTH: i32>(%in: !hw.int<#hw.param.decl.ref<"WIDTH">>) -> (out: !hw.int<#hw.param.decl.ref<"WIDTH">>) attributes {filename = "test.v"}
  hw.module.extern @params<WIDTH: i32>(%in: !hw.int<#hw.param.decl.ref<"WIDTH">>) -> (out: !hw.int<#hw.param.decl.ref<"WIDTH">>) attributes {filename = "test.v"}

  // CHECK-LABEL: calyx.component @A(%in_0: i32, %in_1: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out_0: i32, %out_1: i32, %done: i1 {done})
  calyx.component @A(%in_0: i32, %in_1: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out_0: i32, %out_1: i32, %done: i1 {done}) {
    // CHECK: %true = hw.constant true
    %c1_1 = hw.constant 1 : i1
    // CHECK-NEXT: %params_0.in, %params_0.out = calyx.primitive @params_0 of @params<WIDTH: i32 = 32> : i32, i32
    %params_0.in, %params_0.out = calyx.primitive @params_0 of @params<WIDTH: i32 = 32> : i32, i32
    // CHECK-NEXT: %prim_0.in, %prim_0.out = calyx.primitive @prim_0 of @prim : i32, i32
    %prim_0.in, %prim_0.out = calyx.primitive @prim_0 of @prim : i32, i32

    calyx.wires {
      // CHECK: calyx.assign %done = %true : i1
      calyx.assign %done = %c1_1 : i1
      // CHECK-NEXT: calyx.assign %params_0.in = %in_0 : i32
      calyx.assign %params_0.in = %in_0 : i32
      // CHECK-NEXT: calyx.assign %out_0 = %params_0.out : i32
      calyx.assign %out_0 = %params_0.out : i32
      // CHECK-NEXT: calyx.assign %prim_0.in = %in_1 : i32
      calyx.assign %prim_0.in = %in_1 : i32
      // CHECK-NEXT: calyx.assign %out_1 = %prim_0.out : i32
      calyx.assign %out_1 = %prim_0.out : i32
    }
    calyx.control {}
  } {static = 1}
}

// -----
// CHECK: module attributes {calyx.entrypoint = "A"} {
module attributes {calyx.entrypoint = "A"} {
  // CHECK: hw.module.extern @params<WIDTH: i32>(%in: !hw.int<#hw.param.decl.ref<"WIDTH">>, %clk: i1 {calyx.clk}, %go: i1 {calyx.go}) -> (out: !hw.int<#hw.param.decl.ref<"WIDTH">>, done: i1 {calyx.done}) attributes {filename = "test.v"}
  hw.module.extern @params<WIDTH: i32>(%in: !hw.int<#hw.param.decl.ref<"WIDTH">>, %clk: i1 {calyx.clk}, %go: i1 {calyx.go}) -> (out: !hw.int<#hw.param.decl.ref<"WIDTH">>, done: i1 {calyx.done}) attributes {filename = "test.v"}

  // CHECK-LABEL: calyx.component @A(%in_0: i32, %in_1: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out_0: i32, %out_1: i32, %done: i1 {done})
  calyx.component @A(%in_0: i32, %in_1: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out_0: i32, %out_1: i32, %done: i1 {done}) {
    // CHECK: %true = hw.constant true
    %c1_1 = hw.constant 1 : i1
    // CHECK-NEXT: %params_0.in, %params_0.clk, %params_0.go, %params_0.out, %params_0.done = calyx.primitive @params_0 of @params<WIDTH: i32 = 32> : i32, i1, i1, i32, i1
    %params_0.in, %params_0.clk, %params_0.go, %params_0.out, %params_0.done = calyx.primitive @params_0 of @params<WIDTH: i32 = 32> : i32, i1, i1, i32, i1

    calyx.wires {
      // CHECK: calyx.assign %done = %true : i1
      calyx.assign %done = %c1_1 : i1
      // CHECK-NEXT: calyx.assign %params_0.in = %in_0 : i32
      calyx.assign %params_0.in = %in_0 : i32
      // CHECK-NEXT: calyx.assign %out_0 = %params_0.out : i32
      calyx.assign %out_0 = %params_0.out : i32
    }
    calyx.control {}
  } {static = 1}
}

// -----
module attributes {calyx.entrypoint = "main"} {
  // CHECK-LABEL: calyx.comb_component @A(%in: i32) -> (%out: i32) {
  calyx.comb_component @A(%in: i32) -> (%out: i32) {
    // CHECK: %c1_i32 = hw.constant 1 : i32
    %0 = hw.constant 1 : i32
    // CHECK: %add0.left, %add0.right, %add0.out = calyx.std_add @add0 : i32, i32, i32
    %1:3 = calyx.std_add @add0 : i32, i32, i32
    calyx.wires {
      // CHECK: calyx.assign %add0.left = %in : i32
      calyx.assign %1#0 = %in : i32
      // CHECK: calyx.assign %add0.right = %c1_i32 : i32
      calyx.assign %1#1 = %0 : i32
      // CHECK: calyx.assign %out = %add0.out : i32
      calyx.assign %out = %1#2 : i32
    }
  }

  // CHECK-LABEL: calyx.component @main(%in: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i32, %done: i1 {done}) {
  calyx.component @main(%in: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i32, %done: i1 {done}) {
    // CHECK: %true = hw.constant true
    %c1_1 = hw.constant 1 : i1
    // CHECK: %A_0.in, %A_0.out = calyx.instance @A_0 of @A : i32, i32
    %A_0.in, %A_0.out = calyx.instance @A_0 of @A : i32, i32

    calyx.wires {
      // CHECK: calyx.assign %done = %true : i1
      calyx.assign %done = %c1_1 : i1
      // CHECK: calyx.assign %A_0.in = %in : i32
      calyx.assign %A_0.in = %in : i32
      // CHECK: calyx.assign %out = %A_0.out : i32
      calyx.assign %out = %A_0.out : i32
    }
    calyx.control {}
  } {static = 1}
}

// -----
module attributes {calyx.entrypoint = "main"} {
  // CHECK-LABEL: calyx.component @main(%in: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i32, %done: i1 {done}) {
  calyx.component @main(%in: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i32, %done: i1 {done}) {
    // CHECK: %true = hw.constant true
    %c1_1 = hw.constant 1 : i1
    // CHECK: %m1.addr0, %m1.write_data, %m1.write_en, %m1.write_done, %m1.clk, %m1.read_data, %m1.read_en, %m1.read_done = calyx.seq_mem @m1 <[64] x 32> [6] : i6, i32, i1, i1, i1, i32, i1, i1
    %m1.addr0, %m1.write_data, %m1.write_en, %m1.write_done, %m1.clk, %m1.read_data, %m1.read_en, %m1.read_done = calyx.seq_mem @m1 <[64] x 32> [6] : i6, i32, i1, i1, i1, i32, i1, i1

    calyx.wires {
      // CHECK: calyx.assign %done = %m1.write_done : i1
      calyx.assign %done = %m1.write_done : i1
      // CHECK: calyx.assign %m1.write_en = %go : i1
      calyx.assign %m1.write_en = %go : i1
      // CHECK: calyx.assign %m1.write_data = %in : i32
      calyx.assign %m1.write_data = %in : i32
      // CHECK: calyx.assign %out = %m1.read_data : i32
      calyx.assign %out = %m1.read_data : i32
    }
    calyx.control {}
  } {static = 1}
}

// -----
module attributes {calyx.entrypoint = "main"} {
  // CHECK-LABEL: calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %p.in, %p.write_en, %p.clk, %p.reset, %p.out, %p.done = calyx.register @p : i3, i1, i1, i1, i3, i1
    %incr.left, %incr.right, %incr.out = calyx.std_add @incr : i3, i3, i3
    %l.left, %l.right, %l.out = calyx.std_lt @l : i3, i3, i1
    %c1_3 = hw.constant 1 : i3
    %c1_1 = hw.constant 1 : i1
    %c6_3 = hw.constant 6 : i3

    calyx.wires {
      // CHECK: calyx.static_group latency<1> @A {
      calyx.static_group latency<1> @A {
        calyx.assign %incr.left = %p.out : i3
        calyx.assign %incr.right = %c1_3 : i3
        calyx.assign %p.in = %incr.out : i3
        // CHECK: %0 = calyx.cycle 0
        %0 = calyx.cycle 0
        calyx.assign %p.write_en = %0 : i1
      }
      calyx.assign %l.left = %p.out : i3
      calyx.assign %l.right = %c6_3 : i3
    }
    calyx.control {
      calyx.while %l.out {
        calyx.enable @A
      }
    }
  }
}

// -----
module attributes {calyx.entrypoint = "main"} {
  // CHECK-LABEL: calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %p.in, %p.write_en, %p.clk, %p.reset, %p.out, %p.done = calyx.register @p : i3, i1, i1, i1, i3, i1
    %incr.left, %incr.right, %incr.out = calyx.std_add @incr : i3, i3, i3
    %l.left, %l.right, %l.out = calyx.std_lt @l : i3, i3, i1
    %c1_3 = hw.constant 1 : i3
    %c1_1 = hw.constant 1 : i1
    %c6_3 = hw.constant 6 : i3

    calyx.wires {
      // CHECK: calyx.static_group latency<1> @A {
      calyx.static_group latency<1> @A {
        calyx.assign %incr.left = %p.out : i3
        calyx.assign %incr.right = %c1_3 : i3
        calyx.assign %p.in = %incr.out : i3
        // CHECK: %0 = calyx.cycle 0
        %0 = calyx.cycle 0
        calyx.assign %p.write_en = %0 : i1
      }
      calyx.assign %l.left = %p.out : i3
      calyx.assign %l.right = %c6_3 : i3
    }
    calyx.control {
      // CHECK: calyx.static_repeat 10 {
      calyx.static_repeat 10 {
        calyx.enable @A
      }
    }
  }
}
        
// -----
module attributes {calyx.entrypoint = "main"} {
  // CHECK-LABEL: calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %a.in, %a.write_en, %a.clk, %a.reset, %a.out, %a.done = calyx.register @a : i2, i1, i1, i1, i2, i1
    %b.in, %b.write_en, %b.clk, %b.reset, %b.out, %b.done = calyx.register @b : i2, i1, i1, i1, i2, i1
    %c.in, %c.write_en, %c.clk, %c.reset, %c.out, %c.done = calyx.register @c : i2, i1, i1, i1, i2, i1
    %c0_2 = hw.constant 0 : i2
    %c1_2 = hw.constant 1 : i2
    %c2_2 = hw.constant 2 : i2
    %c1_1 = hw.constant 1 : i1

    calyx.wires {
      // CHECK: calyx.static_group latency<2> @A {
      calyx.static_group latency<2> @A {
        calyx.assign %a.in = %c0_2 : i2
        // CHECK: %0 = calyx.cycle 0
        %0 = calyx.cycle 0
        calyx.assign %a.write_en = %0 ? %c1_1 : i1
        calyx.assign %b.in = %c1_2 : i2
        // CHECK: %1 = calyx.cycle 1
        %1 = calyx.cycle 1
        calyx.assign %b.write_en = %1 ? %c1_1 : i1
      }

      // CHECK: calyx.static_group latency<1> @C {
      calyx.static_group latency<1> @C {
        calyx.assign %c.in = %c2_2 : i2
        // CHECK: %0 = calyx.cycle 0
        %0 = calyx.cycle 0
        calyx.assign %c.write_en = %0 ? %c1_1 : i1
      }

    }
    calyx.control {
      // CHECK: calyx.static_par {
      calyx.static_par {
        calyx.enable @A
        calyx.enable @C
      }
    }
  }
}

// -----
module attributes {calyx.entrypoint = "main"} {
  // CHECK-LABEL: calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %a.in, %a.write_en, %a.clk, %a.reset, %a.out, %a.done = calyx.register @a : i2, i1, i1, i1, i2, i1
    %b.in, %b.write_en, %b.clk, %b.reset, %b.out, %b.done = calyx.register @b : i2, i1, i1, i1, i2, i1

    %c0_2 = hw.constant 0 : i2
    %c2_2 = hw.constant 0 : i2
    %c1_1 = hw.constant 1 : i1

    calyx.wires {
      // CHECK: calyx.static_group latency<1> @A {
      calyx.static_group latency<1> @A {
        calyx.assign %a.in =%c0_2  : i2
        // CHECK: %0 = calyx.cycle 0
        %0 = calyx.cycle 0
        calyx.assign %a.write_en = %0 ? %c1_1 : i1
      }
      // CHECK: calyx.static_group latency<1> @B {
      calyx.static_group latency<1> @B {
        calyx.assign %b.in =%c2_2  : i2
        // CHECK: %0 = calyx.cycle 0
        %0 = calyx.cycle 0
        calyx.assign %b.write_en = %0 ? %c1_1 : i1
      }
    }
    calyx.control {
      // CHECK: calyx.static_seq {
      calyx.static_seq { 
        calyx.enable @A
        calyx.enable @B
      }
    }
  }
}


// -----
module attributes {calyx.entrypoint = "main"} {
  // CHECK-LABEL: calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %d.in, %d.write_en, %d.clk, %d.reset, %d.out, %d.done = calyx.register @d : i2, i1, i1, i1, i2, i1
    %r1.in, %r1.write_en, %r1.clk, %r1.reset, %r1.out, %r1.done = calyx.register @r1 : i1, i1, i1, i1, i1, i1
    %c.in, %c.write_en, %c.clk, %c.reset, %c.out, %c.done = calyx.register @c : i2, i1, i1, i1, i2, i1

    %c0_2 = hw.constant 0 : i2
    %c1_2 = hw.constant 1 : i2
    %c1_1 = hw.constant 1 : i1


    calyx.wires {
      // CHECK: calyx.static_group latency<1> @C {
      calyx.static_group latency<1> @C {
        calyx.assign %c.in = %c0_2 : i2
        // CHECK: %0 = calyx.cycle 0
        %0 = calyx.cycle 0
        calyx.assign %c.write_en = %0 ? %c1_1 : i1
      }

      // CHECK: calyx.static_group latency<1> @D {
      calyx.static_group latency<1> @D {
        calyx.assign %d.in = %c1_2 : i2
        // CHECK: %0 = calyx.cycle 0
        %0 = calyx.cycle 0
        calyx.assign %d.write_en = %0 ? %c1_1 : i1
      }

    }
    calyx.control {
      // CHECK: calyx.static_if %r1.out {
      calyx.static_if %r1.out {
        calyx.enable @C
      } else {
          calyx.enable @D
      }
    }
  }
}
