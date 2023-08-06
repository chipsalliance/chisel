// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s -verify-diagnostics --msft-lower-constructs | FileCheck %s --check-prefix=LOWER

// LOWER-LABEL: hw.module @SAExample1(%clk: i1) -> (out: !hw.array<2xarray<3xi8>>) {
// LOWER:         %c0_i8 = hw.constant 0 : i8
// LOWER:         %c1_i8 = hw.constant 1 : i8
// LOWER:         %c2_i8 = hw.constant 2 : i8
// LOWER:         [[INROW:%.+]] = hw.array_create %c0_i8, %c1_i8 : i8
// LOWER:         [[INCOL:%.+]] = hw.array_create %c0_i8, %c1_i8, %c2_i8 : i8
// LOWER:         %false = hw.constant false
// LOWER:         [[ROW0:%.+]] = hw.array_get [[INROW]][%false] {sv.namehint = "row_0"} : !hw.array<2xi8>
// LOWER:         %true = hw.constant true
// LOWER:         [[ROW1:%.+]] = hw.array_get [[INROW]][%true] {sv.namehint = "row_1"} : !hw.array<2xi8>
// LOWER:         %c0_i2 = hw.constant 0 : i2
// LOWER:         [[COL0:%.+]] = hw.array_get [[INCOL]][%c0_i2] {sv.namehint = "col_0"} : !hw.array<3xi8>
// LOWER:         %c1_i2 = hw.constant 1 : i2
// LOWER:         [[COL1:%.+]] = hw.array_get [[INCOL]][%c1_i2] {sv.namehint = "col_1"} : !hw.array<3xi8>
// LOWER:         %c-2_i2 = hw.constant -2 : i2
// LOWER:         [[COL2:%.+]] = hw.array_get [[INCOL]][%c-2_i2] {sv.namehint = "col_2"} : !hw.array<3xi8>
// LOWER:         [[PEOUT_0_0:%.+]] = hw.instance "pe" @PE(clk: %clk: i1, a: [[ROW0]]: i8, b: [[COL0]]: i8) -> (sum: i8)
// LOWER:         [[PEOUT_0_1:%.+]] = hw.instance "pe" @PE(clk: %clk: i1, a: [[ROW0]]: i8, b: [[COL1]]: i8) -> (sum: i8)
// LOWER:         [[PEOUT_0_2:%.+]] = hw.instance "pe" @PE(clk: %clk: i1, a: [[ROW0]]: i8, b: [[COL2]]: i8) -> (sum: i8)
// LOWER:         [[PEROW0:%.+]] = hw.array_create [[PEOUT_0_2]], [[PEOUT_0_1]], [[PEOUT_0_0]] : i8
// LOWER:         [[PEOUT_1_0:%.+]] = hw.instance "pe" @PE(clk: %clk: i1, a: [[ROW1]]: i8, b: [[COL0]]: i8) -> (sum: i8)
// LOWER:         [[PEOUT_1_1:%.+]] = hw.instance "pe" @PE(clk: %clk: i1, a: [[ROW1]]: i8, b: [[COL1]]: i8) -> (sum: i8)
// LOWER:         [[PEOUT_1_2:%.+]] = hw.instance "pe" @PE(clk: %clk: i1, a: [[ROW1]]: i8, b: [[COL2]]: i8) -> (sum: i8)
// LOWER:         [[PEROW1:%.+]] = hw.array_create [[PEOUT_1_2]], [[PEOUT_1_1]], [[PEOUT_1_0]] : i8
// LOWER:         %9 = hw.array_create [[PEROW1]], [[PEROW0]] : !hw.array<3xi8>
// LOWER:         hw.output %9 : !hw.array<2xarray<3xi8>>

hw.module @SAExample1 (%clk : i1) -> (out: !hw.array<2 x array<3 x i8>>) {
  %c0_8 = hw.constant 0 : i8
  %c1_8 = hw.constant 1 : i8
  %c2_8 = hw.constant 2 : i8
  %rowInputs = hw.array_create %c0_8, %c1_8 : i8
  %colInputs = hw.array_create %c0_8, %c1_8, %c2_8 : i8

  // CHECK: msft.systolic.array [%{{.+}} : 2 x i8] [%{{.+}} : 3 x i8] pe (%{{.+}}, %{{.+}}) -> (i8)
  %peOuts = msft.systolic.array [%rowInputs : 2 x i8] [%colInputs : 3 x i8]
    pe (%row, %col) -> (i8) {
      %sum = hw.instance "pe" @PE (clk: %clk: i1, a: %row: i8, b: %col: i8) -> (sum: i8)
      msft.pe.output %sum : i8
    }
  hw.output %peOuts : !hw.array<2 x array<3 x i8>>
}

hw.module @PE(%clk: i1, %a: i8, %b: i8) -> (sum: i8) {
  %sum = comb.add %a, %b: i8
  %sumDelay1 = seq.compreg %sum, %clk : i8
  hw.output %sumDelay1 : i8
}

// CHECK-LABEL: msft.module @ChannelExample {} (%clk: i1, %a: i8) -> (out: i8) {
// CHECK:         [[REG0:%.+]] = msft.constructs.channel %a %clk "chEx"(2) : i8
// CHECK:         msft.output [[REG0]] : i8
// LOWER-LABEL: msft.module @ChannelExample {} (%clk: i1, %a: i8) -> (out: i8) {
// LOWER:         %chEx_0 = seq.compreg sym @chEx_0 %a, %clk : i8
// LOWER:         %chEx_1 = seq.compreg sym @chEx_1 %chEx_0, %clk : i8
// LOWER:         msft.output %chEx_1 : i8
msft.module @ChannelExample {} (%clk: i1, %a : i8) -> (out: i8) {
  %out = msft.constructs.channel %a %clk "chEx" (2) : i8
  msft.output %out : i8
}

// CHECK-LABEL: msft.module @foo {} (%in0: i32, %in1: i32, %in2: i32, %clk: i1) -> (out: i32) {
// CHECK:       %0 = msft.hlc.linear clock %clk : i32 {
// CHECK:         %1 = comb.mul %in0, %in1 : i32
// CHECK:         %2 = comb.add %1, %in2 : i32
// CHECK:         msft.output %2 : i32
// CHECK:       }
// CHECK:       msft.output %0 : i32
msft.module @foo {} (%in0 : i32, %in1 : i32, %in2 : i32, %clk : i1) -> (out: i32) {
  %0 = msft.hlc.linear clock %clk : i32 {
    %0 = comb.mul %in0, %in1 : i32
    %1 = comb.add %0, %in2 : i32
    msft.output %1 : i32
  }
  msft.output %0 : i32
}
