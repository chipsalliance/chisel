// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt %s --ibis-call-prep | circt-opt | FileCheck %s --check-prefix=PREP


// CHECK-LABEL: ibis.class @C {
// CHECK:         ibis.method @getAndSet(%x: ui32) -> ui32 {
// CHECK:           ibis.return %x : ui32
// CHECK:         ibis.method @returnNothing() {
// CHECK:           ibis.return
// CHECK:         ibis.method @returnNothingWithRet() {
// CHECK:           ibis.return

// PREP-LABEL: ibis.class @C {
// PREP:         ibis.method @getAndSet(%arg: !hw.struct<x: ui32>) -> ui32 {
// PREP:           %x = hw.struct_explode %arg : !hw.struct<x: ui32>
// PREP:           ibis.return %x : ui32
// PREP:         ibis.method @returnNothing(%arg: !hw.struct<>) {
// PREP:           hw.struct_explode %arg : !hw.struct<>
// PREP:           ibis.return
// PREP:         ibis.method @returnNothingWithRet(%arg: !hw.struct<>) {
// PREP:           hw.struct_explode %arg : !hw.struct<>
// PREP:           ibis.return
ibis.class @C {
  ibis.method @getAndSet(%x: ui32) -> ui32 {
    ibis.return %x : ui32
  }
  ibis.method @returnNothing() {}
  ibis.method @returnNothingWithRet() {
    ibis.return
  }
}

// CHECK-LABEL: ibis.class @User {
// CHECK:         ibis.instance @c, @C
// CHECK:         ibis.method @getAndSetWrapper(%new_value: ui32) -> ui32 {
// CHECK:           [[x:%.+]] = ibis.call @c::@getAndSet(%new_value) : (ui32) -> ui32
// CHECK:           ibis.return [[x]] : ui32
// CHECK:         ibis.method @getAndSetDup(%new_value: ui32) -> ui32 {
// CHECK:           [[x:%.+]] = ibis.call @c::@getAndSet(%new_value) : (ui32) -> ui32
// CHECK:           ibis.return [[x]] : ui32


// PREP-LABEL: ibis.class @User {
// PREP:         ibis.instance @c, @C
// PREP:         ibis.method @getAndSetWrapper(%arg: !hw.struct<new_value: ui32>) -> ui32 {
// PREP:           %new_value = hw.struct_explode %arg : !hw.struct<new_value: ui32>
// PREP:           %0 = hw.struct_create (%new_value) {sv.namehint = "getAndSet_args_called_from_getAndSetWrapper"} : !hw.struct<x: ui32>
// PREP:           %1 = ibis.call @c::@getAndSet(%0) : (!hw.struct<x: ui32>) -> ui32
// PREP:         ibis.method @getAndSetDup(%arg: !hw.struct<new_value: ui32>) -> ui32 {
// PREP:           %new_value = hw.struct_explode %arg : !hw.struct<new_value: ui32>
// PREP:           %0 = hw.struct_create (%new_value) {sv.namehint = "getAndSet_args_called_from_getAndSetDup"} : !hw.struct<x: ui32>
// PREP:           %1 = ibis.call @c::@getAndSet(%0) : (!hw.struct<x: ui32>) -> ui32
// PREP:           ibis.return %1 : ui32
ibis.class @User {
  ibis.instance @c, @C
  ibis.method @getAndSetWrapper(%new_value: ui32) -> ui32 {
    %x = ibis.call @c::@getAndSet(%new_value): (ui32) -> ui32
    ibis.return %x : ui32
  }

  ibis.method @getAndSetDup(%new_value: ui32) -> ui32 {
    %x = ibis.call @c::@getAndSet(%new_value): (ui32) -> ui32
    ibis.return %x : ui32
  }
}
