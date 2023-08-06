// RUN: circt-opt %s --arc-lower-clocks-to-funcs --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @Trivial_clock(%arg0: !arc.storage<42>) {
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    %c0_i9001 = hw.constant 0 : i9001
// CHECK-NEXT:    %0 = comb.mux %true, %c0_i9001, %c0_i9001 : i9001
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK-LABEL: func.func @Trivial_passthrough(%arg0: !arc.storage<42>) {
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    %c1_i9001 = hw.constant 1 : i9001
// CHECK-NEXT:    %0 = comb.mux %true, %c1_i9001, %c1_i9001 : i9001
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK-LABEL: arc.model "Trivial" {
// CHECK-NEXT:  ^bb0(%arg0: !arc.storage<42>):
// CHECK-NEXT:    %false = hw.constant false
// CHECK-NEXT:    func.call @Trivial_clock(%arg0) : (!arc.storage<42>) -> ()
// CHECK-NEXT:    func.call @Trivial_passthrough(%arg0) : (!arc.storage<42>) -> ()
// CHECK-NEXT:  }

arc.model "Trivial" {
^bb0(%arg0: !arc.storage<42>):
  %true = hw.constant true
  %false = hw.constant false
  arc.clock_tree %true {
    %c0_i9001 = hw.constant 0 : i9001
    %0 = comb.mux %true, %c0_i9001, %c0_i9001 : i9001
  }
  arc.passthrough {
    %c1_i9001 = hw.constant 1 : i9001
    %0 = comb.mux %true, %c1_i9001, %c1_i9001 : i9001
  }
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @NestedRegions_passthrough(%arg0: !arc.storage<42>) {
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    scf.if %true {
// CHECK-NEXT:      hw.constant 1337
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK-LABEL: arc.model "NestedRegions" {
// CHECK-NEXT:  ^bb0(%arg0: !arc.storage<42>):
// CHECK-NEXT:    func.call @NestedRegions_passthrough(%arg0) : (!arc.storage<42>) -> ()
// CHECK-NEXT:  }

arc.model "NestedRegions" {
^bb0(%arg0: !arc.storage<42>):
  arc.passthrough {
    %true = hw.constant true
    scf.if %true {
      %0 = hw.constant 1337 : i42
    }
  }
}

//===----------------------------------------------------------------------===//

// The constants should copied to the top of the clock function body, not in
// front of individual users, to prevent issues with caching and nested regions.
// https://github.com/llvm/circt/pull/4685#discussion_r1132913165

// CHECK-LABEL: func.func @InsertionOrderProblem_passthrough(%arg0: !arc.storage<42>) {
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    %false = hw.constant false
// CHECK-NEXT:    scf.if %true {
// CHECK-NEXT:      comb.add %true, %false
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK-LABEL: arc.model "InsertionOrderProblem" {
arc.model "InsertionOrderProblem" {
^bb0(%arg0: !arc.storage<42>):
  %true = hw.constant true
  %false = hw.constant false
  arc.passthrough {
    scf.if %true {
      comb.add %true, %false : i1
    }
  }
}
