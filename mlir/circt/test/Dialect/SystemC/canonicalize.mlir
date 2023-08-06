// RUN: circt-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: @removeEmptySensitivityList
systemc.module @removeEmptySensitivityList(%in: !systemc.in<i1>) {
  // CHECK-NEXT: systemc.ctor {
  systemc.ctor {
    systemc.sensitive
    // CHECK-NEXT: systemc.sensitive %in : !systemc.in<i1>
    systemc.sensitive %in : !systemc.in<i1>
  // CHECK-NEXT: }
  }
}

// CHECK-LABEL: @convertOpFolder
systemc.module @convertOpFolder() {
  %scuint32 = systemc.cpp.variable : !systemc.uint<32>
  %uint16 = systemc.cpp.variable : i16
  %uint32 = systemc.cpp.variable : i32
  %uint65 = systemc.cpp.variable : i65
  %scint32 = systemc.cpp.variable : !systemc.int<32>
  %logic = systemc.cpp.variable : !systemc.logic
  %bool = systemc.cpp.variable : i1
  %uint_base = systemc.cpp.variable : !systemc.uint_base
  // CHECK: systemc.cpp.variable : !systemc.uint_base

  // same input and output type
  %0 = systemc.convert %scuint32 : (!systemc.uint<32>) -> !systemc.uint<32>
  // CHECK-NEXT: systemc.cpp.assign %scuint32 = %scuint32 : !systemc.uint<32>
  systemc.cpp.assign %scuint32 = %0 : !systemc.uint<32>

  %1 = systemc.convert %scuint32 : (!systemc.uint<32>) -> i32
  %c1 = systemc.convert %1 : (i32) -> !systemc.uint<32>
  // CHECK-NEXT: systemc.cpp.assign %scuint32 = %scuint32 : !systemc.uint<32>
  systemc.cpp.assign %scuint32 = %c1 : !systemc.uint<32>

  %2 = systemc.convert %uint32 : (i32) -> !systemc.uint<64>
  %c2 = systemc.convert %2 : (!systemc.uint<64>) -> i32
  // CHECK-NEXT: systemc.cpp.assign %uint32 = %uint32 : i32
  systemc.cpp.assign %uint32 = %c2 : i32

  // input and result types do not match -> not folded
  // CHECK-NEXT: [[V0:%.+]] = systemc.convert %uint32 : (i32) -> !systemc.uint<32>
  // CHECK-NEXT: [[V1:%.+]] = systemc.convert [[V0]] : (!systemc.uint<32>) -> i16
  %3 = systemc.convert %uint32 : (i32) -> !systemc.uint<32>
  %c3 = systemc.convert %3 : (!systemc.uint<32>) -> i16
  // CHECK-NEXT: systemc.cpp.assign %uint16 = [[V1]] : i16
  systemc.cpp.assign %uint16 = %c3 : i16

  // signed integers
  %4 = systemc.convert %scint32 : (!systemc.int<32>) -> !systemc.bigint<128>
  %c4 = systemc.convert %4 : (!systemc.bigint<128>) -> !systemc.int<32>
  // CHECK-NEXT: systemc.cpp.assign %scint32 = %scint32 : !systemc.int<32>
  systemc.cpp.assign %scint32 = %c4 : !systemc.int<32>

  // mixed signed and unsigned integers -> not folded
  // CHECK-NEXT: [[V2:%.+]] = systemc.convert %scint32 : (!systemc.int<32>) -> !systemc.uint<32>
  // CHECK-NEXT: [[V3:%.+]] = systemc.convert [[V2]] : (!systemc.uint<32>) -> !systemc.int<32>
  %5 = systemc.convert %scint32 : (!systemc.int<32>) -> !systemc.uint<32>
  %c5 = systemc.convert %5 : (!systemc.uint<32>) -> !systemc.int<32>
  // CHECK-NEXT: systemc.cpp.assign %scint32 = [[V3]] : !systemc.int<32>
  systemc.cpp.assign %scint32 = %c5 : !systemc.int<32>

  // mixed signed and unsigned integers -> not folded
  // CHECK-NEXT: [[V4:%.+]] = systemc.convert %scuint32 : (!systemc.uint<32>) -> !systemc.int<32>
  // CHECK-NEXT: [[V5:%.+]] = systemc.convert [[V4]] : (!systemc.int<32>) -> !systemc.uint<32>
  %6 = systemc.convert %scuint32 : (!systemc.uint<32>) -> !systemc.int<32>
  %c6 = systemc.convert %6 : (!systemc.int<32>) -> !systemc.uint<32>
  // CHECK-NEXT: systemc.cpp.assign %scuint32 = [[V5]] : !systemc.uint<32>
  systemc.cpp.assign %scuint32 = %c6 : !systemc.uint<32>

  // 4-valued to 2-valued and back -> not folded
  // CHECK-NEXT: [[V6:%.+]] = systemc.convert %logic : (!systemc.logic) -> i1
  // CHECK-NEXT: [[V7:%.+]] = systemc.convert [[V6]] : (i1) -> !systemc.logic
  %7 = systemc.convert %logic : (!systemc.logic) -> i1
  %c7 = systemc.convert %7 : (i1) -> !systemc.logic
  // CHECK-NEXT: systemc.cpp.assign %logic = [[V7]] : !systemc.logic
  systemc.cpp.assign %logic = %c7 : !systemc.logic

  // 2-valued to 4-valued and back
  %8 = systemc.convert %bool : (i1) -> !systemc.logic
  %c8 = systemc.convert %8 : (!systemc.logic) -> i1
  // CHECK-NEXT: systemc.cpp.assign %bool = %bool : i1
  systemc.cpp.assign %bool = %c8 : i1

  // intermediate value has smaller bit-width -> not folded
  // CHECK-NEXT: [[V8:%.+]] = systemc.convert %uint32 : (i32) -> !systemc.uint<16>
  // CHECK-NEXT: [[V9:%.+]] = systemc.convert [[V8]] : (!systemc.uint<16>) -> i32
  %9 = systemc.convert %uint32 : (i32) -> !systemc.uint<16>
  %c9 = systemc.convert %9 : (!systemc.uint<16>) -> i32
  // CHECK-NEXT: systemc.cpp.assign %uint32 = [[V9]] : i32
  systemc.cpp.assign %uint32 = %c9 : i32

  %10 = systemc.convert %logic : (!systemc.logic) -> !systemc.lv<16>
  %c10 = systemc.convert %10 : (!systemc.lv<16>) -> !systemc.logic
  // CHECK-NEXT: systemc.cpp.assign %logic = %logic : !systemc.logic
  systemc.cpp.assign %logic = %c10 : !systemc.logic

  %11 = systemc.convert %uint_base : (!systemc.uint_base) -> i64
  %c11 = systemc.convert %11 : (i64) -> !systemc.uint_base
  // CHECK-NEXT: systemc.cpp.assign %uint_base = %uint_base : !systemc.uint_base
  systemc.cpp.assign %uint_base = %c11 : !systemc.uint_base

  // intermediate type must have at least a bit-width of 64 -> not folded
  // CHECK-NEXT: [[V10:%.+]] = systemc.convert %uint_base : (!systemc.uint_base) -> i60
  // CHECK-NEXT: [[V11:%.+]] = systemc.convert [[V10]] : (i60) -> !systemc.uint_base
  %12 = systemc.convert %uint_base : (!systemc.uint_base) -> i60
  %c12 = systemc.convert %12 : (i60) -> !systemc.uint_base
  // CHECK-NEXT: systemc.cpp.assign %uint_base = [[V11]] : !systemc.uint_base
  systemc.cpp.assign %uint_base = %c12 : !systemc.uint_base

  %13 = systemc.convert %uint_base : (!systemc.uint_base) -> !systemc.bv_base
  %c13 = systemc.convert %13 : (!systemc.bv_base) -> !systemc.uint_base
  // CHECK-NEXT: systemc.cpp.assign %uint_base = %uint_base : !systemc.uint_base
  systemc.cpp.assign %uint_base = %c13 : !systemc.uint_base

  %14 = systemc.convert %uint32 : (i32) -> !systemc.bv_base
  %c14 = systemc.convert %14 : (!systemc.bv_base) -> i32
  // CHECK-NEXT: systemc.cpp.assign %uint32 = %uint32 : i32
  systemc.cpp.assign %uint32 = %c14 : i32

  %15 = systemc.convert %uint_base : (!systemc.uint_base) -> !systemc.unsigned
  %c15 = systemc.convert %15 : (!systemc.unsigned) -> !systemc.uint_base
  // CHECK-NEXT: systemc.cpp.assign %uint_base = %uint_base : !systemc.uint_base
  systemc.cpp.assign %uint_base = %c15 : !systemc.uint_base

  %16 = systemc.convert %uint32 : (i32) -> !systemc.unsigned
  %c16 = systemc.convert %16 : (!systemc.unsigned) -> i32
  // CHECK-NEXT: systemc.cpp.assign %uint32 = %uint32 : i32
  systemc.cpp.assign %uint32 = %c16 : i32

  %17 = systemc.convert %uint32 : (i32) -> !systemc.uint_base
  %c17 = systemc.convert %17 : (!systemc.uint_base) -> i32
  // CHECK-NEXT: systemc.cpp.assign %uint32 = %uint32 : i32
  systemc.cpp.assign %uint32 = %c17 : i32

  // We don't necessarily know the max bit-width of unsigned, thus we should not fold above 64 bits
  // CHECK-NEXT: [[V12:%.+]] = systemc.convert %uint65 : (i65) -> !systemc.unsigned
  // CHECK-NEXT: [[V13:%.+]] = systemc.convert [[V12]] : (!systemc.unsigned) -> i65
  %18 = systemc.convert %uint65 : (i65) -> !systemc.unsigned
  %c18 = systemc.convert %18 : (!systemc.unsigned) -> i65
  // CHECK-NEXT: systemc.cpp.assign %uint65 = [[V13]] : i65
  systemc.cpp.assign %uint65 = %c18 : i65
}
