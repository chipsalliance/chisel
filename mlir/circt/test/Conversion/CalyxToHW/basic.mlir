// RUN: circt-opt %s -lower-calyx-to-hw | FileCheck %s

// Sample program:
//
// component main(a: 32, b: 32) -> (out: 32) {
//   cells {
//     add = std_add(32);
//     buf = std_reg(32);
//   }
//
//   wires {
//     out = buf.out;
//     group g0 {
//       add.left = a;
//       add.right = b;
//       buf.in = add.out;
//       buf.write_en = 1'd1;
//       g0[done] = buf.done;
//     }
//   }
//
//   control {
//     g0;
//   }
// }
//
// Compiled with:
//
// futil -p pre-opt -p compile -p post-opt -p lower -p lower-guards -b mlir

// module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%a: i32, %b: i32, %go: i1 {go = 1 : i64}, %clk: i1 {clk = 1 : i64}, %reset: i1 {reset = 1 : i64}) -> (%out: i32, %done: i1 {done = 1 : i64}) {
    // CHECK-DAG:  %out = sv.wire
    // CHECK-DAG:  %[[OUT_VAL:.+]] = sv.read_inout %out
    // CHECK-DAG:  %done = sv.wire
    // CHECK-DAG:  %[[DONE_VAL:.+]] = sv.read_inout %done
    // CHECK-DAG:  %add_left = sv.wire
    // CHECK-DAG:  %[[ADD_LEFT_VAL:.+]] = sv.read_inout %add_left
    // CHECK-DAG:  %add_right = sv.wire
    // CHECK-DAG:  %[[ADD_RIGHT_VAL:.+]] = sv.read_inout %add_right
    // CHECK-DAG:  %[[ADD:.+]] = comb.add %[[ADD_LEFT_VAL]], %[[ADD_RIGHT_VAL]]
    // CHECK-DAG:  %add_out = sv.wire
    // CHECK-DAG:  sv.assign %add_out, %[[ADD]]
    // CHECK-DAG:  %[[ADD_OUT_VAL:.+]] = sv.read_inout %add_out
    %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

    // CHECK-DAG:  %buf_in = sv.wire
    // CHECK-DAG:  %[[BUF_IN_VAL:.+]] = sv.read_inout %buf_in
    // CHECK-DAG:  %buf_write_en = sv.wire
    // CHECK-DAG:  %[[BUF_WRITE_EN_VAL:.+]] = sv.read_inout %buf_write_en
    // CHECK-DAG:  %buf_clk = sv.wire
    // CHECK-DAG:  %[[BUF_CLK_VAL:.+]] = sv.read_inout %buf_clk
    // CHECK-DAG:  %buf_reset = sv.wire
    // CHECK-DAG:  %[[BUF_RESET_VAL:.+]] = sv.read_inout %buf_reset
    // CHECK-DAG:  %[[FALSE:.+]] = hw.constant false
    // CHECK-DAG:  %[[BUF_DONE_REG:.+]] = seq.compreg sym @buf_done_reg %[[BUF_WRITE_EN_VAL]], %[[BUF_CLK_VAL]], %[[BUF_RESET_VAL]], %[[FALSE]]  : i1
    // CHECK-DAG:  %buf_done = sv.wire
    // CHECK-DAG:  sv.assign %buf_done, %[[BUF_DONE_REG]]
    // CHECK-DAG:  %[[BUF_DONE_VAL:.+]] = sv.read_inout %buf_done
    // CHECK-DAG:  %[[TRUE:.+]] = hw.constant true
    // CHECK-DAG:  %[[BUF_DONE_VAL_NEG:.+]] = comb.xor %[[BUF_DONE_VAL]], %true : i1
    // CHECK-DAG:  %[[BUF_REG_WRITE_EN:.+]] = comb.and %[[BUF_WRITE_EN_VAL]], %[[BUF_DONE_VAL_NEG]] : i1
    // CHECK-DAG:  %[[C0_I32:.+]] = hw.constant 0 : i32
    // CHECK-DAG:  %[[BUF_REG:.+]] = seq.compreg.ce sym @buf_reg %[[BUF_IN_VAL]], %[[BUF_CLK_VAL]], %[[BUF_REG_WRITE_EN]], %[[BUF_RESET_VAL]], %[[C0_I32]]
    // CHECK-DAG:  %buf = sv.wire
    // CHECK-DAG:  sv.assign %buf, %[[BUF_REG]]
    // CHECK-DAG:  %[[BUF_VAL:.+]] = sv.read_inout %buf
    %buf.in, %buf.write_en, %buf.clk, %buf.reset, %buf.out, %buf.done = calyx.register @buf : i32, i1, i1, i1, i32, i1

    // CHECK-DAG:  %g0_go = sv.wire
    // CHECK-DAG:  %[[G0_GO_VAL:.+]] = sv.read_inout %g0_go
    %g0_go.in, %g0_go.out = calyx.std_wire @g0_go {generated = 1 : i64} : i1, i1

    // CHECK-DAG:  %g0_done = sv.wire
    // CHECK-DAG:  %[[G0_DONE_VAL:.+]] = sv.read_inout %g0_done
    %g0_done.in, %g0_done.out = calyx.std_wire @g0_done {generated = 1 : i64} : i1, i1

    calyx.wires {
      // CHECK-DAG:  %[[TRUE:.+]] = hw.constant true
      %true = hw.constant true

      // CHECK-DAG:  %[[FALSE_0:.+]] = hw.constant false
      // CHECK-DAG:  %[[VAL_TO_DONE:.+]] = comb.mux %[[G0_DONE_VAL]], %[[TRUE]], %[[FALSE_0]]
      // CHECK-DAG:  sv.assign %done, %[[VAL_TO_DONE]]
      calyx.assign %done = %g0_done.out ? %true : i1

      // CHECK-DAG:  %[[C0_I32_0:.+]] = hw.constant 0
      // CHECK-DAG:  %[[VAL_TO_OUT:.+]] = comb.mux %[[TRUE]], %[[BUF_VAL]], %[[C0_I32_0]]
      // CHECK-DAG:  sv.assign %out, %[[VAL_TO_OUT]]
      calyx.assign %out = %true ? %buf.out : i32

      // CHECK-DAG:  %[[C0_I32_1:.+]] = hw.constant 0
      // CHECK-DAG:  %[[ADD_LEFT_IN:.+]] = comb.mux %[[G0_GO_VAL]], %a, %[[C0_I32_1]]
      // CHECK-DAG:  sv.assign %add_left, %[[ADD_LEFT_IN]]
      calyx.assign %add.left = %g0_go.out ? %a : i32

      // CHECK-DAG:  %[[C0_I32_2:.+]] = hw.constant 0
      // CHECK-DAG:  %[[ADD_RIGHT_IN:.+]] = comb.mux %[[G0_GO_VAL]], %b, %[[C0_I32_2]]
      // CHECK-DAG:  sv.assign %add_right, %[[ADD_RIGHT_IN]]
      calyx.assign %add.right = %g0_go.out ? %b : i32

      // CHECK-DAG:  %[[FALSE_1:.+]] = hw.constant false
      // CHECK-DAG:  %[[BUF_CLK_IN:.+]] = comb.mux %[[TRUE]], %clk, %[[FALSE_1]]
      // CHECK-DAG:  sv.assign %buf_clk, %[[BUF_CLK_IN]] : i1
      calyx.assign %buf.clk = %true ? %clk : i1

      // CHECK-DAG:  %[[C0_I32_3:.+]] = hw.constant 0
      // CHECK-DAG:  %[[BUF_IN_IN:.+]] = comb.mux %[[G0_GO_VAL]], %[[ADD_OUT_VAL]], %[[C0_I32_3]]
      // CHECK-DAG:  sv.assign %buf_in, %[[BUF_IN_IN]]
      calyx.assign %buf.in = %g0_go.out ? %add.out : i32

      // CHECK-DAG:  %[[FALSE_2:.+]] = hw.constant false
      // CHECK-DAG:  %[[BUF_RESET_IN:.+]] = comb.mux %[[TRUE]], %reset, %[[FALSE_2]]
      // CHECK-DAG:  sv.assign %buf_reset, %[[BUF_RESET_IN]]
      calyx.assign %buf.reset = %true ? %reset : i1

      // CHECK-DAG:  %[[FALSE_3:.+]] = hw.constant false
      // CHECK-DAG:  %[[BUF_WRITE_EN_IN:.+]] = comb.mux %[[G0_GO_VAL]], %[[TRUE]], %[[FALSE_3]]
      // CHECK-DAG:  sv.assign %buf_write_en, %[[BUF_WRITE_EN_IN]]
      calyx.assign %buf.write_en = %g0_go.out ? %true : i1

      // CHECK-DAG:  %[[FALSE_4:.+]] = hw.constant false
      // CHECK-DAG:  %[[G0_DONE_IN:.+]] = comb.mux %[[TRUE]], %[[BUF_DONE_VAL]], %[[FALSE_4]]
      // CHECK-DAG:  sv.assign %g0_done, %[[G0_DONE_IN]]
      calyx.assign %g0_done.in = %true ? %buf.done : i1

      // CHECK-DAG:  %[[FALSE_5:.+]] = hw.constant false
      // CHECK-DAG:  %[[G0_GO_IN:.+]] = comb.mux %[[TRUE]], %go, %[[FALSE_5]]
      // CHECK-DAG:  sv.assign %g0_go, %[[G0_GO_IN]]
      calyx.assign %g0_go.in = %true ? %go : i1

      // CHECK-DAG:  hw.output %[[OUT_VAL]], %[[DONE_VAL]]
    }
    calyx.control {
    }
  }
// }
