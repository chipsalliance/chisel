// RUN: circt-opt --hw-flatten-io="recursive=true" %s | FileCheck %s

// Ensure that non-struct-using modules pass cleanly through the pass.

// CHECK-LABEL: hw.module @level0(%arg0: i32) -> (out0: i32) {
// CHECK-NEXT:    hw.output %arg0 : i32
// CHECK-NEXT:  }
hw.module @level0(%arg0 : i32) -> (out0 : i32) {
    hw.output %arg0: i32
}

// CHECK-LABEL: hw.module @level1(%arg0: i32, %in.a: i1, %in.b: i2, %arg1: i32) -> (out0: i32, out.a: i1, out.b: i2, out1: i32) {
// CHECK-NEXT:    %0 = hw.struct_create (%in.a, %in.b) : !hw.struct<a: i1, b: i2>
// CHECK-NEXT:    %a, %b = hw.struct_explode %0 : !hw.struct<a: i1, b: i2>
// CHECK-NEXT:    hw.output %arg0, %a, %b, %arg1 : i32, i1, i2, i32
// CHECK-NEXT:  }
!Struct1 = !hw.struct<a: i1, b: i2>
hw.module @level1(%arg0 : i32, %in : !Struct1, %arg1: i32) -> (out0 : i32,out: !Struct1, out1: i32) {
    hw.output %arg0, %in, %arg1 : i32, !Struct1, i32
}

// CHECK-LABEL: hw.module @level2(%in.aa.a: i1, %in.aa.b: i2, %in.bb.a: i1, %in.bb.b: i2) -> (out.aa.a: i1, out.aa.b: i2, out.bb.a: i1, out.bb.b: i2) {
// CHECK-NEXT:    %0 = hw.struct_create (%in.aa.a, %in.aa.b) : !hw.struct<a: i1, b: i2>
// CHECK-NEXT:    %1 = hw.struct_create (%in.bb.a, %in.bb.b) : !hw.struct<a: i1, b: i2>
// CHECK-NEXT:    %2 = hw.struct_create (%0, %1) : !hw.struct<aa: !hw.struct<a: i1, b: i2>, bb: !hw.struct<a: i1, b: i2>>
// CHECK-NEXT:    %aa, %bb = hw.struct_explode %2 : !hw.struct<aa: !hw.struct<a: i1, b: i2>, bb: !hw.struct<a: i1, b: i2>>
// CHECK-NEXT:    %a, %b = hw.struct_explode %aa : !hw.struct<a: i1, b: i2>
// CHECK-NEXT:    %a_0, %b_1 = hw.struct_explode %bb : !hw.struct<a: i1, b: i2>
// CHECK-NEXT:    hw.output %a, %b, %a_0, %b_1 : i1, i2, i1, i2
// CHECK-NEXT:  }
!Struct2 = !hw.struct<aa: !Struct1, bb: !Struct1>
hw.module @level2(%in : !Struct2) -> (out: !Struct2) {
    hw.output %in : !Struct2
}

// CHECK-LABEL: hw.module.extern @level1_extern(%arg0: i32, %in.a: i1, %in.b: i2, %arg1: i32) -> (out0: i32, out.a: i1, out.b: i2, out1: i32)
hw.module.extern @level1_extern(%arg0 : i32, %in : !Struct1, %arg1: i32) -> (out0 : i32,out: !Struct1, out1: i32)


hw.type_scope @foo {
  hw.typedecl @bar : !Struct1
}
!ScopedStruct = !hw.typealias<@foo::@bar,!Struct1>

// CHECK-LABEL: hw.module @scoped(%arg0: i32, %in.a: i1, %in.b: i2, %arg1: i32) -> (out0: i32, out.a: i1, out.b: i2, out1: i32) {
// CHECK-NEXT:    %0 = hw.struct_create (%in.a, %in.b) : !hw.struct<a: i1, b: i2>
// CHECK-NEXT:    %a, %b = hw.struct_explode %0 : !hw.struct<a: i1, b: i2>
// CHECK-NEXT:    hw.output %arg0, %a, %b, %arg1 : i32, i1, i2, i32
// CHECK-NEXT:  }
hw.module @scoped(%arg0 : i32, %in : !ScopedStruct, %arg1: i32) -> (out0 : i32,out: !ScopedStruct, out1: i32) {
  hw.output %arg0, %in, %arg1 : i32, !ScopedStruct, i32
}

// CHECK-LABEL: hw.module @instance(%arg0: i32, %arg1.a: i1, %arg1.b: i2) -> (out.a: i1, out.b: i2) {
// CHECK-NEXT:   %0 = hw.struct_create (%arg1.a, %arg1.b) : !hw.struct<a: i1, b: i2>
// CHECK-NEXT:   %a, %b = hw.struct_explode %0 : !hw.struct<a: i1, b: i2>
// CHECK-NEXT:   %l1.out0, %l1.out.a, %l1.out.b, %l1.out1 = hw.instance "l1" @level1(arg0: %arg0: i32, in.a: %a: i1, in.b: %b: i2, arg1: %arg0: i32) -> (out0: i32, out.a: i1, out.b: i2, out1: i32)
// CHECK-NEXT:   %1 = hw.struct_create (%l1.out.a, %l1.out.b) : !hw.struct<a: i1, b: i2>
// CHECK-NEXT:   %a_0, %b_1 = hw.struct_explode %1 : !hw.struct<a: i1, b: i2>
// CHECK-NEXT:   hw.output %a_0, %b_1 : i1, i2
// CHECK-NEXT: }
hw.module @instance(%arg0 : i32, %arg1 : !Struct1) -> (out : !Struct1) {
  %0:3 = hw.instance "l1" @level1(arg0: %arg0 : i32, in: %arg1 : !Struct1, arg1: %arg0 : i32) -> (out0: i32, out: !Struct1, out1: i32)
  hw.output %0#1 : !Struct1
}

