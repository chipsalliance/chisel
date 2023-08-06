// RUN: circt-opt %s  --lower-scf-to-calyx='cider-source-location-metadata' -canonicalize | FileCheck %s

module {
// CHECK: module attributes {calyx.entrypoint = "main", calyx.metadata = ["loc({{.*}}11:5)", "loc({{.*}}13:5)"]}
  func.func @main(%a0 : i32, %a1 : i32, %a2 : i32) -> i32 {
    %0 = arith.addi %a0, %a1 : i32
    %1 = arith.addi %0, %a1 : i32
    %b = arith.cmpi uge, %1, %a2 : i32
    cf.cond_br %b, ^bb1, ^bb2
  ^bb1:
    return %a1 : i32
  ^bb2:
    return %a2 : i32
  }
}
