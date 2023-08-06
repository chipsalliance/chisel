// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK:       hw.module @foo(%arg0: !dc.token, %arg1: !dc.value<i1>, %arg2: i32) attributes {argNames = ["", "", ""]} {
// CHECK-NEXT:    %0 = dc.buffer[2] %arg0 : !dc.token
// CHECK-NEXT:    %1 = dc.buffer[2] %arg1 [1, 2] : !dc.value<i1>
// CHECK-NEXT:    %2:2 = dc.fork [2] %arg0 
// CHECK-NEXT:    %3 = dc.pack %arg0, %arg2 : i32
// CHECK-NEXT:    %4 = dc.merge %0, %arg0
// CHECK-NEXT:    %token, %output = dc.unpack %arg1 : !dc.value<i1>
// CHECK-NEXT:    %5 = dc.to_esi %arg0 : !dc.token
// CHECK-NEXT:    %6 = dc.to_esi %arg1 : !dc.value<i1>
// CHECK-NEXT:    %7 = dc.from_esi %5 : <i0>
// CHECK-NEXT:    %8 = dc.from_esi %6 : <i1>
// CHECK-NEXT:    hw.output
// CHECK-NEXT:  }

hw.module @foo(%0 : !dc.token, %1 : !dc.value<i1>, %2 : i32) {
  %buffer = dc.buffer [2] %0 : !dc.token
  %bufferInit = dc.buffer [2] %1 [1, 2] : !dc.value<i1>
  %f1, %f2 = dc.fork [2] %0
  %pack = dc.pack %0, %2 : i32
  %merge = dc.merge %buffer, %0
  %unpack_token, %unpack_value = dc.unpack %1 : !dc.value<i1>
  %esi_token = dc.to_esi %0 : !dc.token
  %esi_value = dc.to_esi %1 : !dc.value<i1>
  %from_esi_token = dc.from_esi %esi_token : !esi.channel<i0>
  %from_esi_value = dc.from_esi %esi_value : !esi.channel<i1>
}
