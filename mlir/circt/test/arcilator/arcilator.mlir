// RUN: arcilator %s --inline=0 --until-before=llvm-lowering | FileCheck %s
// RUN: arcilator %s | FileCheck %s --check-prefix=LLVM
// RUN: arcilator --print-debug-info %s | FileCheck %s --check-prefix=LLVM-DEBUG

// CHECK:      arc.define @[[XOR_ARC:.+]](
// CHECK-NEXT:   comb.xor
// CHECK-NEXT:   arc.output
// CHECK-NEXT: }

// CHECK:      arc.define @[[ADD_ARC:.+]](
// CHECK-NEXT:   comb.add
// CHECK-NEXT:   arc.output
// CHECK-NEXT: }

// CHECK:      arc.define @[[MUL_ARC:.+]](
// CHECK-NEXT:   comb.mul
// CHECK-NEXT:   arc.output
// CHECK-NEXT: }

// CHECK-NOT: hw.module @Top
// CHECK-LABEL: arc.model "Top" {
// CHECK-NEXT: ^bb0(%arg0: !arc.storage<6>):
hw.module @Top(%clock: i1, %i0: i4, %i1: i4) -> (out: i4) {
  // CHECK-DAG: arc.root_input "clock", %arg0 {offset = 0
  // CHECK-DAG: arc.root_input "i0", %arg0 {offset = 1
  // CHECK-DAG: arc.root_input "i1", %arg0 {offset = 2
  // CHECK-DAG: arc.root_output "out", %arg0 {offset = 3
  // CHECK-DAG: arc.alloc_state %arg0 {name = "foo", offset = 4
  // CHECK-DAG: arc.alloc_state %arg0 {name = "bar", offset = 5

  // CHECK-DAG: arc.passthrough {
  // CHECK-DAG:   [[FOO:%.+]] = arc.storage.get %arg0[4]
  // CHECK-DAG:   [[READ_FOO:%.+]] = arc.state_read [[FOO]]
  // CHECK-DAG:   [[BAR:%.+]] = arc.storage.get %arg0[5]
  // CHECK-DAG:   [[READ_BAR:%.+]] = arc.state_read [[BAR]]
  // CHECK-DAG:   [[MUL:%.+]] = arc.state @[[MUL_ARC]]([[READ_FOO]], [[READ_BAR]]) lat 0
  // CHECK-DAG:   [[PTR_OUT:%.+]] = arc.storage.get %arg0[3]
  // CHECK-DAG:   arc.state_write [[PTR_OUT]] = [[MUL]]
  // CHECK-DAG: }

  // CHECK-DAG: [[CLOCK:%.+]] = arc.storage.get %arg0[0]
  // CHECK-DAG: [[READ_CLOCK:%.+]] = arc.state_read [[CLOCK]]
  // CHECK-DAG:  arc.clock_tree [[READ_CLOCK]] {
  // CHECK-DAG:   [[I0:%.+]] = arc.storage.get %arg0[1]
  // CHECK-DAG:   [[READ_I0:%.+]] = arc.state_read [[I0]]
  // CHECK-DAG:   [[I1:%.+]] = arc.storage.get %arg0[2]
  // CHECK-DAG:   [[READ_I1:%.+]] = arc.state_read [[I1]]
  // CHECK-DAG:   [[ADD:%.+]] = arc.state @[[ADD_ARC]]([[READ_I0]], [[READ_I1]]) lat 0
  // CHECK-DAG:   [[XOR1:%.+]] = arc.state @[[XOR_ARC]]([[ADD]], [[READ_I0]]) lat 0
  // CHECK-DAG:   [[XOR2:%.+]] = arc.state @[[XOR_ARC]]([[ADD]], [[READ_I1]]) lat 0
  // CHECK-DAG:   [[FOO:%.+]] = arc.storage.get %arg0[4]
  // CHECK-DAG:   arc.state_write [[FOO]] = [[XOR1]]
  // CHECK-DAG:   [[BAR:%.+]] = arc.storage.get %arg0[5]
  // CHECK-DAG:   arc.state_write [[BAR]] = [[XOR2]]
  // CHECK-DAG:  }

  %0 = comb.add %i0, %i1 : i4
  %1 = comb.xor %0, %i0 : i4
  %2 = comb.xor %0, %i1 : i4
  %foo = seq.compreg %1, %clock : i4
  %bar = seq.compreg %2, %clock : i4
  %3 = comb.mul %foo, %bar : i4
  hw.output %3 : i4
}

// LLVM: define void @Top_passthrough(ptr %0)
// LLVM:   mul i4
// LLVM: define void @Top_clock(ptr %0)
// LLVM:   add i4
// LLVM:   xor i4
// LLVM:   xor i4

// LLVM-DEBUG: define void @Top_passthrough(ptr %0){{.*}}!dbg
// LLVM-DEBUG:   mul i4{{.*}}!dbg
// LLVM-DEBUG: define void @Top_clock(ptr %0){{.*}}!dbg
// LLVM-DEBUG:   add i4{{.*}}!dbg
// LLVM-DEBUG:   xor i4{{.*}}!dbg
// LLVM-DEBUG:   xor i4{{.*}}!dbg
