// RUN: circt-opt %s --arc-simplify-variadic-ops | FileCheck %s

// Should convert variadic comb ops with arbitrarily shuffled inputs into a
// reasonable tree with individual nodes as close to their operand definitions
// as possible.
// CHECK-LABEL: func @BalancedTree
func.func @BalancedTree(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1) -> i1 {
  // CHECK-NEXT: comb.or
  // CHECK-NEXT: comb.or
  // CHECK-NEXT: comb.or
  %0 = builtin.unrealized_conversion_cast to i1 {a}
  // CHECK-NEXT: unrealized_conversion_cast
  // CHECK-NEXT: comb.or
  %1 = builtin.unrealized_conversion_cast to i1 {b}
  // CHECK-NEXT: unrealized_conversion_cast
  // CHECK-NEXT: comb.or
  %2 = builtin.unrealized_conversion_cast to i1 {c}
  // CHECK-NEXT: unrealized_conversion_cast
  // CHECK-NEXT: comb.or
  %3 = builtin.unrealized_conversion_cast to i1 {d}
  // CHECK-NEXT: unrealized_conversion_cast
  // CHECK-NEXT: comb.or
  %4 = builtin.unrealized_conversion_cast to i1 {e}
  // CHECK-NEXT: unrealized_conversion_cast
  // CHECK-NEXT: comb.or
  %5 = builtin.unrealized_conversion_cast to i1 {f}
  // CHECK-NEXT: unrealized_conversion_cast
  // CHECK-NEXT: comb.or
  %6 = builtin.unrealized_conversion_cast to i1 {g}
  // CHECK-NEXT: unrealized_conversion_cast
  // CHECK-NEXT: comb.or
  %7 = builtin.unrealized_conversion_cast to i1 {h}
  // CHECK-NEXT: unrealized_conversion_cast
  // CHECK-NEXT: comb.or
  %8 = comb.or %7, %2, %arg3, %arg1, %5, %1, %arg2, %0, %arg0, %3, %4, %6 : i1
  return %8 : i1
}

// Should not touch variadic ops if they cross block boundaries.
// CHECK-LABEL: func @SkipBlockBoundaries
func.func @SkipBlockBoundaries() {
  %0 = builtin.unrealized_conversion_cast to i1 {a}
  %1 = builtin.unrealized_conversion_cast to i1 {b}
  scf.execute_region {
    %2 = builtin.unrealized_conversion_cast to i1 {c}
    // CHECK: comb.or %0, %1, %2 : i1
    %3 = comb.or %0, %1, %2 : i1
    scf.yield
  }
  return
}
