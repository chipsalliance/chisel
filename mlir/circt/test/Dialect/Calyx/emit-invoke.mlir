// RUN: circt-translate --export-calyx --split-input-file --verify-diagnostics %s | FileCheck %s --strict-whitespace

module attributes {calyx.entrypoint = "main"} { 
calyx.component @identity(%in: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i32, %done: i1 {done}) {
  %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i32, i1, i1, i1, i32, i1
  calyx.wires {
    calyx.assign %out = %r.out :i32
  }

  calyx.control {
    calyx.seq {
      // CHECK: invoke r(in = in)();
      calyx.invoke @r(%r.in = %in) -> (i32)
    }
  }
}

calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}, %out : i32) {
  %id.in, %id.go, %id.clk, %id.reset, %id.out, %id.done = calyx.instance @id of @identity : i32, i1, i1, i1, i32, i1
  %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i32, i1, i1, i1, i32, i1 
  %c1_10 = hw.constant 10 : i32
    calyx.wires {

    }

    calyx.control {
      calyx.seq {
        // CHECK: invoke id(in = 32'd10)();
        calyx.invoke @id(%id.in = %c1_10) -> (i32)
        // CHECK: invoke r(in = id.out)(out = out);
        calyx.invoke @r(%r.in = %id.out, %out = %r.out) -> (i32, i32) 
      }
    }
  }
}
