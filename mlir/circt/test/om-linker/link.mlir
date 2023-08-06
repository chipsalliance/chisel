// RUN: om-linker %S/Inputs/a.mlir %S/Inputs/b.mlir | FileCheck %s
// CHECK:      module {
// CHECK-NEXT:   om.class @A(%arg: i1) {
// CHECK-NEXT:   }
// CHECK-NEXT:   om.class @Conflict_a() {
// CHECK-NEXT:   }
// CHECK-NEXT:   om.class @B(%arg: i2) {
// CHECK-NEXT:   }
// CHECK-NEXT:   om.class @Conflict_b() {
// CHECK-NEXT:   }
// CHECK-NEXT: }
