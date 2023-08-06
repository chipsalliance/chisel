// RUN: circt-opt %s --arc-allocate-state | FileCheck %s

// CHECK-LABEL: arc.model "test"
arc.model "test" {
^bb0(%arg0: !arc.storage):
  // CHECK-NEXT: ([[PTR:%.+]]: !arc.storage<5724>):

  // CHECK-NEXT: arc.alloc_storage [[PTR]][0] : (!arc.storage<5724>) -> !arc.storage<1143>
  // CHECK-NEXT: arc.passthrough {
  arc.passthrough {
    // CHECK-NEXT: [[SUBPTR:%.+]] = arc.storage.get [[PTR]][0] : !arc.storage<5724> -> !arc.storage<1143>
    %0 = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i1>
    arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i8>
    arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i16>
    arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i32>
    arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i64>
    arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i9001>
    %1 = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i1>
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 0 : i32}
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 1 : i32}
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 2 : i32}
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 4 : i32}
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 8 : i32}
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 16 : i32}
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 1142 : i32}
    // CHECK-NEXT: scf.execute_region {
    scf.execute_region {
      arc.state_read %0 : <i1>
      // CHECK-NEXT: [[SUBPTR:%.+]] = arc.storage.get [[PTR]][0] : !arc.storage<5724> -> !arc.storage<1143>
      // CHECK-NEXT: [[STATE:%.+]] = arc.storage.get [[SUBPTR]][0] : !arc.storage<1143> -> !arc.state<i1>
      // CHECK-NEXT: arc.state_read [[STATE]] : <i1>
      arc.state_read %1 : <i1>
      // CHECK-NEXT: [[STATE:%.+]] = arc.storage.get [[SUBPTR]][1142] : !arc.storage<1143> -> !arc.state<i1>
      // CHECK-NEXT: arc.state_read [[STATE]] : <i1>
      scf.yield
      // CHECK-NEXT: scf.yield
    }
    // CHECK-NEXT: }
  }
  // CHECK-NEXT: }

  // CHECK-NEXT: arc.alloc_storage [[PTR]][1144] : (!arc.storage<5724>) -> !arc.storage<4577>
  // CHECK-NEXT: arc.passthrough {
  arc.passthrough {
    // CHECK-NEXT: [[SUBPTR:%.+]] = arc.storage.get [[PTR]][1144] : !arc.storage<5724> -> !arc.storage<4577>
    arc.alloc_memory %arg0 : (!arc.storage) -> !arc.memory<4 x i1, i1>
    arc.alloc_memory %arg0 : (!arc.storage) -> !arc.memory<4 x i8, i1>
    arc.alloc_memory %arg0 : (!arc.storage) -> !arc.memory<4 x i16, i1>
    arc.alloc_memory %arg0 : (!arc.storage) -> !arc.memory<4 x i32, i1>
    arc.alloc_memory %arg0 : (!arc.storage) -> !arc.memory<4 x i64, i1>
    arc.alloc_memory %arg0 : (!arc.storage) -> !arc.memory<4 x i9001, i1>
    arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i1>
    // CHECK-NEXT: arc.alloc_memory [[SUBPTR]] {offset = 0 : i32, stride = 1 : i32}
    // CHECK-SAME: -> !arc.memory<4 x i1, i1>
    // CHECK-NEXT: arc.alloc_memory [[SUBPTR]] {offset = 4 : i32, stride = 1 : i32}
    // CHECK-SAME: -> !arc.memory<4 x i8, i1>
    // CHECK-NEXT: arc.alloc_memory [[SUBPTR]] {offset = 8 : i32, stride = 2 : i32}
    // CHECK-SAME: -> !arc.memory<4 x i16, i1>
    // CHECK-NEXT: arc.alloc_memory [[SUBPTR]] {offset = 16 : i32, stride = 4 : i32}
    // CHECK-SAME: -> !arc.memory<4 x i32, i1>
    // CHECK-NEXT: arc.alloc_memory [[SUBPTR]] {offset = 32 : i32, stride = 8 : i32}
    // CHECK-SAME: -> !arc.memory<4 x i64, i1>
    // CHECK-NEXT: arc.alloc_memory [[SUBPTR]] {offset = 64 : i32, stride = 1128 : i32}
    // CHECK-SAME: -> !arc.memory<4 x i9001, i1>
    // CHECK-NEXT: arc.alloc_state [[SUBPTR]] {offset = 4576 : i32}
  }
  // CHECK-NEXT: }

  // CHECK-NEXT: arc.alloc_storage [[PTR]][5722] : (!arc.storage<5724>) -> !arc.storage<2>
  // CHECK-NEXT: arc.passthrough {
  arc.passthrough {
    arc.root_input "x", %arg0 : (!arc.storage) -> !arc.state<i1>
    arc.root_output "y", %arg0 : (!arc.storage) -> !arc.state<i1>
    // CHECK-NEXT: [[SUBPTR:%.+]] = arc.storage.get [[PTR]][5722] : !arc.storage<5724> -> !arc.storage<2>
    // CHECK-NEXT: arc.root_input "x", [[SUBPTR]] {offset = 0 : i32}
    // CHECK-NEXT: arc.root_output "y", [[SUBPTR]] {offset = 1 : i32}
  }
  // CHECK-NEXT: }
}
