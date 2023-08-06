// RUN: circt-opt %s --arc-legalize-state-update | FileCheck %s

// CHECK-LABEL: func.func @Unaffected
func.func @Unaffected(%arg0: !arc.storage, %arg1: i4) -> i4 {
  %0 = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i4>
  %1 = arc.state_read %0 : <i4>
  arc.state_write %0 = %arg1 : <i4>
  return %1 : i4
  // CHECK-NEXT: arc.alloc_state
  // CHECK-NEXT: arc.state_read
  // CHECK-NEXT: arc.state_write
  // CHECK-NEXT: return
}
// CHECK-NEXT: }

// CHECK-LABEL: func.func @SameBlock
func.func @SameBlock(%arg0: !arc.storage, %arg1: i4) -> i4 {
  %0 = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i4>
  %1 = arc.state_read %0 : <i4>
  // CHECK-NEXT: [[STATE:%.+]] = arc.alloc_state
  // CHECK-NEXT: arc.state_read [[STATE]]

  arc.state_write %0 = %arg1 : <i4>
  // CHECK-NEXT: [[TMP:%.+]] = arc.alloc_state
  // CHECK-NEXT: [[CURRENT:%.+]] = arc.state_read [[STATE]]
  // CHECK-NEXT: arc.state_write [[TMP]] = [[CURRENT]]
  // CHECK-NEXT: arc.state_write [[STATE]] = %arg1

  %2 = arc.state_read %0 : <i4>
  %3 = arc.state_read %0 : <i4>
  %4 = comb.xor %1, %2, %3 : i4
  return %4 : i4
  // CHECK-NEXT: arc.state_read [[TMP]]
  // CHECK-NEXT: arc.state_read [[TMP]]
  // CHECK-NEXT: comb.xor
  // CHECK-NEXT: return
}
// CHECK-NEXT: }

// CHECK-LABEL: func.func @FuncLegal
func.func @FuncLegal(%arg0: !arc.storage, %arg1: i4) -> i4 {
  %0 = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i4>
  %1 = call @ReadFunc(%0) : (!arc.state<i4>) -> i4
  call @WriteFunc(%0, %arg1) : (!arc.state<i4>, i4) -> ()
  return %1 : i4
  // CHECK-NEXT: arc.alloc_state
  // CHECK-NEXT: call @ReadFunc
  // CHECK-NEXT: call @WriteFunc
  // CHECK-NEXT: return
}
// CHECK-NEXT: }

// CHECK-LABEL: func.func @FuncIllegal
func.func @FuncIllegal(%arg0: !arc.storage, %arg1: i4) -> i4 {
  %0 = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i4>
  %1 = call @ReadFunc(%0) : (!arc.state<i4>) -> i4
  // CHECK-NEXT: [[STATE:%.+]] = arc.alloc_state
  // CHECK-NEXT: call @ReadFunc

  call @WriteFunc(%0, %arg1) : (!arc.state<i4>, i4) -> ()
  // CHECK-NEXT: [[TMP:%.+]] = arc.alloc_state
  // CHECK-NEXT: [[CURRENT:%.+]] = arc.state_read [[STATE]]
  // CHECK-NEXT: arc.state_write [[TMP]] = [[CURRENT]]
  // CHECK-NEXT: call @WriteFunc

  %2 = call @ReadFunc(%0) : (!arc.state<i4>) -> i4
  %3 = call @ReadFunc(%0) : (!arc.state<i4>) -> i4
  %4 = comb.xor %1, %2, %3 : i4
  return %4 : i4
  // CHECK-NEXT: call @ReadFunc([[TMP]])
  // CHECK-NEXT: call @ReadFunc([[TMP]])
  // CHECK-NEXT: comb.xor
  // CHECK-NEXT: return
}
// CHECK-NEXT: }

// CHECK-LABEL: func.func @NestedBlocks
func.func @NestedBlocks(%arg0: !arc.storage, %arg1: i4) -> i4 {
  %0 = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i4>
  %11 = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i4>
  // CHECK-NEXT: [[S0:%.+]] = arc.alloc_state
  // CHECK-NEXT: [[S1:%.+]] = arc.alloc_state

  // CHECK-NEXT: scf.execute_region
  %10 = scf.execute_region -> i4 {
    // CHECK-NEXT: [[TMP0:%.+]] = arc.alloc_state
    // CHECK-NEXT: [[CURRENT:%.+]] = arc.state_read [[S0]]
    // CHECK-NEXT: arc.state_write [[TMP0]] = [[CURRENT]]
    // CHECK-NEXT: [[TMP1:%.+]] = arc.alloc_state
    // CHECK-NEXT: [[CURRENT:%.+]] = arc.state_read [[S1]]
    // CHECK-NEXT: arc.state_write [[TMP1]] = [[CURRENT]]
    // CHECK-NEXT: scf.execute_region
    %3 = scf.execute_region -> i4 {
      // CHECK-NEXT: scf.execute_region
      %1 = scf.execute_region -> i4 {
        %2 = arc.state_read %0 : <i4>
        scf.yield %2 : i4
        // CHECK-NEXT: arc.state_read [[TMP0]]
        // CHECK-NEXT: scf.yield
      }
      // CHECK-NEXT: }
      // CHECK-NEXT: scf.execute_region
      scf.execute_region {
        arc.state_write %0 = %arg1 : <i4>
        arc.state_write %11 = %arg1 : <i4>
        scf.yield
        // CHECK-NEXT: arc.state_write [[S0]]
        // CHECK-NEXT: arc.state_write [[S1]]
        // CHECK-NEXT: scf.yield
      }
      // CHECK-NEXT: }
      scf.yield %1 : i4
      // CHECK-NEXT: scf.yield
    }
    // CHECK-NEXT: }
    func.call @WriteFunc(%0, %arg1) : (!arc.state<i4>, i4) -> ()
    // CHECK-NEXT: func.call @WriteFunc([[S0]], %arg1)
    // CHECK-NEXT: scf.execute_region
    %7, %8 = scf.execute_region -> (i4, i4) {
      // CHECK-NEXT: scf.execute_region
      %4 = scf.execute_region -> i4 {
        %5 = func.call @ReadFunc(%0) : (!arc.state<i4>) -> i4
        scf.yield %5 : i4
        // CHECK-NEXT: func.call @ReadFunc([[TMP0]])
        // CHECK-NEXT: scf.yield
      }
      // CHECK-NEXT: }
      %6 = arc.state_read %0 : <i4>
      %12 = arc.state_read %11 : <i4>
      scf.yield %4, %6 : i4, i4
      // CHECK-NEXT: arc.state_read [[TMP0]]
      // CHECK-NEXT: arc.state_read [[TMP1]]
      // CHECK-NEXT: scf.yield
    }
    // CHECK-NEXT: }
    %9 = comb.xor %3, %7, %8 : i4
    scf.yield %9 : i4
    // CHECK-NEXT: comb.xor
    // CHECK-NEXT: scf.yield
  }
  // CHECK-NEXT: }
  return %10 : i4
  // CHECK-NEXT: return
}

func.func @ReadFunc(%arg0: !arc.state<i4>) -> i4 {
  %0 = func.call @InnerReadFunc(%arg0) : (!arc.state<i4>) -> i4
  return %0 : i4
}

func.func @WriteFunc(%arg0: !arc.state<i4>, %arg1: i4) {
  func.call @InnerWriteFunc(%arg0, %arg1) : (!arc.state<i4>, i4) -> ()
  return
}

func.func @InnerReadFunc(%arg0: !arc.state<i4>) -> i4 {
  %0 = arc.state_read %arg0 : <i4>
  return %0 : i4
}

func.func @InnerWriteFunc(%arg0: !arc.state<i4>, %arg1: i4) {
  arc.state_write %arg0 = %arg1 : <i4>
  return
}

// State legalization should not happen across clock trees and passthrough ops.
// CHECK-LABEL: arc.model "DontLeakThroughClockTreeOrPassthrough"
arc.model "DontLeakThroughClockTreeOrPassthrough" {
^bb0(%arg0: !arc.storage):
  %false = hw.constant false
  %in_a = arc.root_input "a", %arg0 : (!arc.storage) -> !arc.state<i1>
  %out_b = arc.root_output "b", %arg0 : (!arc.storage) -> !arc.state<i1>
  // CHECK: arc.alloc_state %arg0 {foo}
  %0 = arc.alloc_state %arg0 {foo} : (!arc.storage) -> !arc.state<i1>
  // CHECK-NOT: arc.alloc_state
  // CHECK-NOT: arc.state_read
  // CHECK-NOT: arc.state_write
  // CHECK: arc.clock_tree
  arc.clock_tree %false {
    %1 = arc.state_read %in_a : <i1>
    arc.state_write %0 = %1 : <i1>
  }
  // CHECK: arc.passthrough
  arc.passthrough {
    %1 = arc.state_read %0 : <i1>
    arc.state_write %out_b = %1 : <i1>
  }
}

// CHECK-LABEL: arc.model "Memory"
arc.model "Memory" {
^bb0(%arg0: !arc.storage):
  %false = hw.constant false
  // CHECK: arc.clock_tree %false attributes {ct1}
  arc.clock_tree %false attributes {ct1} {
    // CHECK-NEXT: arc.state_read
    // CHECK-NEXT: arc.memory_read [[MEM1:%.+]][%false]
    // CHECK-NEXT: arc.memory_write [[MEM1]]
    // CHECK-NEXT: arc.memory_read [[MEM2:%.+]][%false]
    // CHECK-NEXT: arc.memory_write [[MEM2]]
    %r1 = arc.state_read %s1 : <i32>
    arc.memory_write %mem2[%false], %r1 : <2 x i32, i1>
    arc.memory_write %mem1[%false], %r1 : <2 x i32, i1>
    %mr1 = arc.memory_read %mem1[%false] : <2 x i32, i1>
    %mr2 = arc.memory_read %mem2[%false] : <2 x i32, i1>
  // CHECK-NEXT: }
  }
  // CHECK: arc.clock_tree %false attributes {ct2}
  arc.clock_tree %false attributes {ct2} {
    // CHECK-NEXT: arc.state_read
    // CHECK-NEXT: arc.memory_read
    // CHECK-NEXT: scf.if %false {
    // CHECK-NEXT:   arc.memory_read
    // CHECK-NEXT: }
    // CHECK-NEXT: arc.memory_write
    %r1 = arc.state_read %s1 : <i32>
    arc.memory_write %mem1[%false], %r1 : <2 x i32, i1>
    %mr1 = arc.memory_read %mem1[%false] : <2 x i32, i1>
    scf.if %false {
      %mr2 = arc.memory_read %mem1[%false] : <2 x i32, i1>
    }
  // CHECK-NEXT: }
  }
  // CHECK: arc.clock_tree %false attributes {ct3}
  arc.clock_tree %false attributes {ct3} {
    // CHECK-NEXT: arc.memory_read [[MEM1]]
    // CHECK-NEXT: arc.memory_read [[MEM2]]
    // CHECK-NEXT: scf.if %false {
    // CHECK-NEXT:   arc.state_read
    // CHECK-NEXT:   scf.if %false {
    // CHECK-NEXT:     arc.memory_write [[MEM2]]
    // CHECK-NEXT:     arc.memory_read [[MEM1]]
    // CHECK-NEXT:   }
    // CHECK-NEXT:   arc.memory_write [[MEM1]]
    // CHECK-NEXT: }
    scf.if %false {
      %r1 = arc.state_read %s1 : <i32>
      arc.memory_write %mem1[%false], %r1 : <2 x i32, i1>
      scf.if %false {
        arc.memory_write %mem2[%false], %r1 : <2 x i32, i1>
        %mr3 = arc.memory_read %mem1[%false] : <2 x i32, i1>
      }
    }
    %mr1 = arc.memory_read %mem1[%false] : <2 x i32, i1>
    %mr2 = arc.memory_read %mem2[%false] : <2 x i32, i1>
  // CHECK-NEXT: }
  }
  // CHECK: [[MEM1]] = arc.alloc_memory %arg0 :
  // CHECK: [[MEM2]] = arc.alloc_memory %arg0 :
  %mem1 = arc.alloc_memory %arg0 : (!arc.storage) -> !arc.memory<2 x i32, i1>
  %mem2 = arc.alloc_memory %arg0 : (!arc.storage) -> !arc.memory<2 x i32, i1>
  %s1 = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i32>
}
