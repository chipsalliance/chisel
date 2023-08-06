// RUN: circt-opt -ssp-print %s 2>&1 | FileCheck %s

// CHECK: digraph G {
ssp.instance @Foo of "Problem" {
  library {
    operator_type @Bar [latency<1>]
  }
  graph {
// CHECK: op0 -> op1
// CHECK: op0 -> op2
// CHECK: op1 -> op3
// CHECK: op2 -> op3
    %0 = operation<@Bar>()
    %1 = operation<@Bar>(%0)
    %2 = operation<@Bar>(%0)
    operation<@Bar>(%1, %2)
  }
}
