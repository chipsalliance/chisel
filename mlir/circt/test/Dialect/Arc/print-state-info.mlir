// RUN: circt-opt %s --arc-print-state-info=state-file=%t
// RUN: cat %t | FileCheck %s

// CHECK-LABEL: "name": "Foo"
// CHECK-DAG: "numStateBytes": 5724
arc.model "Foo" {
^bb0(%arg0: !arc.storage<5724>):
  // CHECK:      "name": "a"
  // CHECK-NEXT: "offset": 0
  // CHECK-NEXT: "numBits": 19
  // CHECK-NEXT: "type": "input"
  arc.root_input "a", %arg0 {offset = 0} : (!arc.storage<5724>) -> !arc.state<i19>

  // CHECK:      "name": "b"
  // CHECK-NEXT: "offset": 16
  // CHECK-NEXT: "numBits": 42
  // CHECK-NEXT: "type": "output"
  arc.root_output "b", %arg0 {offset = 16} : (!arc.storage<5724>) -> !arc.state<i42>
}

// CHECK-LABEL: "name": "Bar"
// CHECK-DAG: "numStateBytes": 9001
arc.model "Bar" {
^bb0(%arg0: !arc.storage<9001>):
  // CHECK-NOT: "offset": "420"
  arc.alloc_state %arg0 {offset = 420} : (!arc.storage<9001>) -> !arc.state<i11>

  // CHECK:      "name": "x"
  // CHECK-NEXT: "offset": 24
  // CHECK-NEXT: "numBits": 63
  // CHECK-NEXT: "type": "register"
  arc.alloc_state %arg0 {name = "x", offset = 24} : (!arc.storage<9001>) -> !arc.state<i63>

  // CHECK:      "name": "y"
  // CHECK-NEXT: "offset": 48
  // CHECK-NEXT: "numBits": 17
  // CHECK-NEXT: "type": "memory"
  // CHECK-NEXT: "stride": 3
  // CHECK-NEXT: "depth": 5
  arc.alloc_memory %arg0 {name = "y", offset = 48, stride = 3} : (!arc.storage<9001>) -> !arc.memory<5 x i17, i3>

  // CHECK:      "name": "z"
  // CHECK-NEXT: "offset": 92
  // CHECK-NEXT: "numBits": 1337
  // CHECK-NEXT: "type": "wire"
  arc.alloc_state %arg0 tap {name = "z", offset = 92} : (!arc.storage<9001>) -> !arc.state<i1337>
}
