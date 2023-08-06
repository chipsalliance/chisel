// RUN: circt-opt -split-input-file -lower-hwarith-to-hw %s | FileCheck %s

// CHECK: hw.module @constant() -> (out: i32) {
// CHECK:   %c0_i32 = hw.constant 0 : i32
// CHECK:   hw.output %c0_i32 : i32

hw.module @constant() -> (out: i32) {
  %0 = hwarith.constant 0 : si32
  %out = hwarith.cast %0 : (si32) -> i32
  hw.output %out : i32
}

// -----

// CHECK: hw.module @add(%op0: i32, %op1: i32) -> (sisi: i32, siui: i32, uisi: i32, uiui: i32) {
hw.module @add(%op0: i32, %op1: i32) -> (sisi: i32, siui: i32, uisi: i32, uiui: i32) {
  %op0Signed = hwarith.cast %op0 : (i32) -> si32
  %op0Unsigned = hwarith.cast %op0 : (i32) -> ui32
  %op1Signed = hwarith.cast %op1 : (i32) -> si32
  %op1Unsigned = hwarith.cast %op1 : (i32) -> ui32

// CHECK:   %[[SIGN_BIT_OP0:.*]] = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[SIGN_BIT_OP0]], %op0 : i1, i32
// CHECK:   %[[SIGN_BIT_OP1:.*]] = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %[[SIGN_BIT_OP1]], %op1 : i1, i32
// CHECK:   %[[SISI_RES:.*]] = comb.add %[[OP0_PADDED]], %[[OP1_PADDED]] : i33
  %sisi = hwarith.add %op0Signed, %op1Signed : (si32, si32) -> si33

// CHECK:   %[[SIGN_BIT_OP0:.*]] = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:   %[[SIGN_EXTEND:.*]] = comb.replicate %[[SIGN_BIT_OP0]] : (i1) -> i2
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[SIGN_EXTEND]], %op0 : i2, i32
// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant 0 : i2
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op1 : i2, i32
// CHECK:   %[[SIUI_RES:.*]] = comb.add %[[OP0_PADDED]], %[[OP1_PADDED]] : i34
  %siui = hwarith.add %op0Signed, %op1Unsigned : (si32, ui32) -> si34

// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant 0 : i2
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op0 : i2, i32
// CHECK:   %[[SIGN_BIT_OP1:.*]] = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:   %[[SIGN_EXTEND:.*]] = comb.replicate %[[SIGN_BIT_OP1]] : (i1) -> i2
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %[[SIGN_EXTEND]], %op1 : i2, i32
// CHECK:   %[[UISI_RES:.*]] = comb.add %[[OP0_PADDED]], %[[OP1_PADDED]] : i34
  %uisi = hwarith.add %op0Unsigned, %op1Signed : (ui32, si32) -> si34

// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant false
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op0 : i1, i32
// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant false
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op1 : i1, i32
// CHECK:   %[[UIUI_RES:.*]] = comb.add %[[OP0_PADDED]], %[[OP1_PADDED]] : i33
  %uiui = hwarith.add %op0Unsigned, %op1Unsigned : (ui32, ui32) -> ui33

// CHECK:   %[[SISI_OUT:.*]] = comb.extract %[[SISI_RES]] from 0 : (i33) -> i32
// CHECK:   %[[SIUI_OUT:.*]] = comb.extract %[[SIUI_RES]] from 0 : (i34) -> i32
// CHECK:   %[[UISI_OUT:.*]] = comb.extract %[[UISI_RES]] from 0 : (i34) -> i32
// CHECK:   %[[UIUI_OUT:.*]] = comb.extract %[[UIUI_RES]] from 0 : (i33) -> i32
  %sisiOut = hwarith.cast %sisi : (si33) -> i32
  %siuiOut = hwarith.cast %siui : (si34) -> i32
  %uisiOut = hwarith.cast %uisi : (si34) -> i32
  %uiuiOut = hwarith.cast %uiui : (ui33) -> i32

// CHECK:   hw.output %[[SISI_OUT]], %[[SIUI_OUT]], %[[UISI_OUT]], %[[UIUI_OUT]] : i32, i32, i32, i32
  hw.output %sisiOut, %siuiOut, %uisiOut, %uiuiOut : i32, i32, i32, i32
}

// -----

// CHECK: hw.module @sub(%op0: i32, %op1: i32) -> (sisi: i32, siui: i32, uisi: i32, uiui: i32) {
hw.module @sub(%op0: i32, %op1: i32) -> (sisi: i32, siui: i32, uisi: i32, uiui: i32) {
  %op0Signed = hwarith.cast %op0 : (i32) -> si32
  %op0Unsigned = hwarith.cast %op0 : (i32) -> ui32
  %op1Signed = hwarith.cast %op1 : (i32) -> si32
  %op1Unsigned = hwarith.cast %op1 : (i32) -> ui32

// CHECK:   %[[SIGN_BIT_OP0_1:.*]] = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[SIGN_BIT_OP0_1]], %op0 : i1, i32
// CHECK:   %[[SIGN_BIT_OP1_1:.*]] = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %[[SIGN_BIT_OP1_1]], %op1 : i1, i32
// CHECK:   %[[SISI_RES:.*]] = comb.sub %[[OP0_PADDED]], %[[OP1_PADDED]] : i33
  %sisi = hwarith.sub %op0Signed, %op1Signed : (si32, si32) -> si33

// CHECK:   %[[SIGN_BIT_OP0_2:.*]] = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:   %[[SIGN_EXTEND:.*]] = comb.replicate %[[SIGN_BIT_OP0]] : (i1) -> i2
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[SIGN_EXTEND]], %op0 : i2, i32
// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant 0 : i2
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op1 : i2, i32
// CHECK:   %[[SIUI_RES:.*]] = comb.sub %[[OP0_PADDED]], %[[OP1_PADDED]] : i34
  %siui = hwarith.sub %op0Signed, %op1Unsigned : (si32, ui32) -> si34

// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant 0 : i2
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op0 : i2, i32
// CHECK:   %[[SIGN_BIT_OP1:.*]] = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:   %[[SIGN_EXTEND:.*]] = comb.replicate %[[SIGN_BIT_OP1]] : (i1) -> i2
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %[[SIGN_EXTEND]], %op1 : i2, i32
// CHECK:   %[[UISI_RES:.*]] = comb.sub %[[OP0_PADDED]], %[[OP1_PADDED]] : i34
  %uisi = hwarith.sub %op0Unsigned, %op1Signed : (ui32, si32) -> si34

// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant false
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op0 : i1, i32
// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant false
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op1 : i1, i32
// CHECK:   %[[UIUI_RES:.*]] = comb.sub %[[OP0_PADDED]], %[[OP1_PADDED]] : i33
  %uiui = hwarith.sub %op0Unsigned, %op1Unsigned : (ui32, ui32) -> si33

// CHECK:   %[[SISI_OUT:.*]] = comb.extract %[[SISI_RES]] from 0 : (i33) -> i32
  %sisiOut = hwarith.cast %sisi : (si33) -> i32
// CHECK:   %[[SIUI_OUT:.*]] = comb.extract %[[SIUI_RES]] from 0 : (i34) -> i32
  %siuiOut = hwarith.cast %siui : (si34) -> i32
// CHECK:   %[[UISI_OUT:.*]] = comb.extract %[[UISI_RES]] from 0 : (i34) -> i32
  %uisiOut = hwarith.cast %uisi : (si34) -> i32
// CHECK:   %[[UIUI_OUT:.*]] = comb.extract %[[UIUI_RES]] from 0 : (i33) -> i32
  %uiuiOut = hwarith.cast %uiui : (si33) -> i32

// CHECK:   hw.output %[[SISI_OUT]], %[[SIUI_OUT]], %[[UISI_OUT]], %[[UIUI_OUT]] : i32, i32, i32, i32
  hw.output %sisiOut, %siuiOut, %uisiOut, %uiuiOut : i32, i32, i32, i32
}

// -----

// CHECK: hw.module @mul(%op0: i32, %op1: i32) -> (sisi: i32, siui: i32, uisi: i32, uiui: i32) {
hw.module @mul(%op0: i32, %op1: i32) -> (sisi: i32, siui: i32, uisi: i32, uiui: i32) {

  %op0Signed = hwarith.cast %op0 : (i32) -> si32
  %op0Unsigned = hwarith.cast %op0 : (i32) -> ui32
  %op1Signed = hwarith.cast %op1 : (i32) -> si32
  %op1Unsigned = hwarith.cast %op1 : (i32) -> ui32

// CHECK:  %[[SIGN_BIT_OP0:.*]] = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:  %[[SIGN_EXTEND:.*]] = comb.replicate %[[SIGN_BIT_OP0]] : (i1) -> i32
// CHECK:  %[[OP0_PADDED:.*]] = comb.concat %[[SIGN_EXTEND]], %op0 : i32, i32
// CHECK:  %[[SIGN_BIT_OP1:.*]] = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:  %[[SIGN_EXTEND:.*]] = comb.replicate %[[SIGN_BIT_OP1]] : (i1) -> i32
// CHECK:  %[[OP1_PADDED:.*]] = comb.concat %[[SIGN_EXTEND]], %op1 : i32, i32
// CHECK:  %[[SISI_RES:.*]] = comb.mul %[[OP0_PADDED]], %[[OP1_PADDED]] : i64
  %sisi = hwarith.mul %op0Signed, %op1Signed : (si32, si32) -> si64

// CHECK:  %[[SIGN_BIT_OP0:.*]] = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:  %[[SIGN_EXTEND:.*]] = comb.replicate %[[SIGN_BIT_OP0]] : (i1) -> i32
// CHECK:  %[[OP0_PADDED:.*]] = comb.concat %[[SIGN_EXTEND]], %op0 : i32, i32
// CHECK:  %[[ZERO_EXTEND:.*]] = hw.constant 0 : i32
// CHECK:  %[[OP1_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op1 : i32, i32
// CHECK:  %[[SIUI_RES:.*]] = comb.mul %[[OP0_PADDED]], %[[OP1_PADDED]] : i64
  %siui = hwarith.mul %op0Signed, %op1Unsigned : (si32, ui32) -> si64

// CHECK:  %[[ZERO_EXTEND:.*]] = hw.constant 0 : i32
// CHECK:  %[[OP0_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op0 : i32, i32
// CHECK:  %[[SIGN_BIT_OP1:.*]] = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:  %[[SIGN_EXTEND:.*]] = comb.replicate %[[SIGN_BIT_OP1]] : (i1) -> i32
// CHECK:  %[[OP1_PADDED:.*]] = comb.concat %[[SIGN_EXTEND]], %op1 : i32, i32
// CHECK:  %[[UISI_RES:.*]] = comb.mul %[[OP0_PADDED]], %[[OP1_PADDED]] : i64
  %uisi = hwarith.mul %op0Unsigned, %op1Signed : (ui32, si32) -> si64

// CHECK:  %[[ZERO_EXTEND:.*]] = hw.constant 0 : i32
// CHECK:  %[[OP0_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op0 : i32, i32
// CHECK:  %[[ZERO_EXTEND:.*]] = hw.constant 0 : i32
// CHECK:  %[[OP1_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op1 : i32, i32
// CHECK:  %[[UIUI_RES:.*]] = comb.mul %[[OP0_PADDED]], %[[OP1_PADDED]] : i64
  %uiui = hwarith.mul %op0Unsigned, %op1Unsigned : (ui32, ui32) -> ui64

// CHECK:  %[[SISI_OUT:.*]] = comb.extract %[[SISI_RES]] from 0 : (i64) -> i32
  %sisiOut = hwarith.cast %sisi : (si64) -> i32
// CHECK:  %[[SIUI_OUT:.*]] = comb.extract %[[SIUI_RES]] from 0 : (i64) -> i32
  %siuiOut = hwarith.cast %siui : (si64) -> i32
// CHECK:  %[[UISI_OUT:.*]] = comb.extract %[[UISI_RES]] from 0 : (i64) -> i32
  %uisiOut = hwarith.cast %uisi : (si64) -> i32
// CHECK:  %[[UIUI_OUT:.*]] = comb.extract %[[UIUI_RES]] from 0 : (i64) -> i32
  %uiuiOut = hwarith.cast %uiui : (ui64) -> i32

// CHECK:   hw.output %[[SISI_OUT]], %[[SIUI_OUT]], %[[UISI_OUT]], %[[UIUI_OUT]] : i32, i32, i32, i32
  hw.output %sisiOut, %siuiOut, %uisiOut, %uiuiOut : i32, i32, i32, i32
}

// -----

// CHECK: hw.module @div(%op0: i32, %op1: i32) -> (sisi: i32, siui: i32, uisi: i32, uiui: i32) {
hw.module @div(%op0: i32, %op1: i32) -> (sisi: i32, siui: i32, uisi: i32, uiui: i32) {
  %op0Signed = hwarith.cast %op0 : (i32) -> si32
  %op0Unsigned = hwarith.cast %op0 : (i32) -> ui32
  %op1Signed = hwarith.cast %op1 : (i32) -> si32
  %op1Unsigned = hwarith.cast %op1 : (i32) -> ui32

// CHECK:   %[[SIGN_BIT_OP0:.*]] = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[SIGN_BIT_OP0]], %op0 : i1, i32
// CHECK:   %[[SIGN_BIT_OP1:.*]] = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %[[SIGN_BIT_OP1]], %op1 : i1, i32
// CHECK:   %[[SISI_RES:.*]] = comb.divs %[[OP0_PADDED]], %[[OP1_PADDED]] : i33
  %sisi = hwarith.div %op0Signed, %op1Signed : (si32, si32) -> si33

// CHECK:   %[[SIGN_BIT_OP0:.*]] = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[SIGN_BIT_OP0]], %op0 : i1, i32
// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant false
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op1 : i1, i32
// CHECK:   %[[SIUI_RES_IMM:.*]] = comb.divs %[[OP0_PADDED]], %[[OP1_PADDED]] : i33
// CHECK:   %[[SIUI_RES:.*]] = comb.extract %[[SIUI_RES_IMM]] from 0 : (i33) -> i32
  %siui = hwarith.div %op0Signed, %op1Unsigned : (si32, ui32) -> si32

// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant false
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op0 : i1, i32
// CHECK:   %[[SIGN_BIT_OP1:.*]] = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %[[SIGN_BIT_OP1]], %op1 : i1, i32
// CHECK:   %[[UISI_RES:.*]] = comb.divs %[[OP0_PADDED]], %[[OP1_PADDED]] : i33
  %uisi = hwarith.div %op0Unsigned, %op1Signed : (ui32, si32) -> si33

// CHECK:   %[[UIUI_RES:.*]] = comb.divu %op0, %op1 : i32
  %uiui = hwarith.div %op0Unsigned, %op1Unsigned : (ui32, ui32) -> ui32

// CHECK:   %[[SISI_OUT:.*]] = comb.extract %[[SISI_RES]] from 0 : (i33) -> i32
  %sisiOut = hwarith.cast %sisi : (si33) -> i32
  %siuiOut = hwarith.cast %siui : (si32) -> i32
// CHECK:   %[[UISI_OUT:.*]] = comb.extract %[[UISI_RES]] from 0 : (i33) -> i32
  %uisiOut = hwarith.cast %uisi : (si33) -> i32
  %uiuiOut = hwarith.cast %uiui : (ui32) -> i32

// CHECK:   hw.output %[[SISI_OUT]], %[[SIUI_RES]], %[[UISI_OUT]], %[[UIUI_RES]] : i32, i32, i32, i32
  hw.output %sisiOut, %siuiOut, %uisiOut, %uiuiOut : i32, i32, i32, i32
}

// -----

// CHECK: hw.module @icmp(%op0: i32, %op1: i32) -> (sisi: i1, siui: i1, uisi: i1, uiui: i1) {
hw.module @icmp(%op0: i32, %op1: i32) -> (sisi: i1, siui: i1, uisi: i1, uiui: i1) {
  %op0Signed = hwarith.cast %op0 : (i32) -> si32
  %op0Unsigned = hwarith.cast %op0 : (i32) -> ui32
  %op1Signed = hwarith.cast %op1 : (i32) -> si32
  %op1Unsigned = hwarith.cast %op1 : (i32) -> ui32

// CHECK:   %[[SISI_OUT:.*]] = comb.icmp slt %op0, %op1 : i32
  %sisi = hwarith.icmp lt %op0Signed, %op1Signed : si32, si32

// CHECK:   %[[SIGN_BIT_OP0:.*]] = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[SIGN_BIT_OP0]], %op0 : i1, i32
// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant false
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op1 : i1, i32
// CHECK:   %[[SIUI_OUT:.*]] = comb.icmp slt %[[OP0_PADDED]], %[[OP1_PADDED]] : i33
  %siui = hwarith.icmp lt %op0Signed, %op1Unsigned : si32, ui32

// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant false
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op0 : i1, i32
// CHECK:   %[[SIGN_BIT_OP1:.*]] = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %[[SIGN_BIT_OP1]], %op1 : i1, i32
// CHECK:   %[[UISI_OUT:.*]] = comb.icmp slt %[[OP0_PADDED]], %[[OP1_PADDED]] : i33
  %uisi = hwarith.icmp lt %op0Unsigned, %op1Signed : ui32, si32

// CHECK:   %[[UIUI_OUT:.*]] = comb.icmp ult %op0, %op1 : i32
  %uiui = hwarith.icmp lt %op0Unsigned, %op1Unsigned : ui32, ui32

  %sisiOut = hwarith.cast %sisi : (ui1) -> i1
  %siuiOut = hwarith.cast %siui : (ui1) -> i1
  %uisiOut = hwarith.cast %uisi : (ui1) -> i1
  %uiuiOut = hwarith.cast %uiui : (ui1) -> i1

// CHECK:   hw.output %[[SISI_OUT]], %[[SIUI_OUT]], %[[UISI_OUT]], %[[UIUI_OUT]] : i1, i1, i1, i1
  hw.output %sisiOut, %siuiOut, %uisiOut, %uiuiOut : i1, i1, i1, i1
}

// -----

// CHECK: hw.module @icmp_mixed_width(%op0: i5, %op1: i7) -> (sisi: i1, siui: i1, uisi: i1, uiui: i1) {
hw.module @icmp_mixed_width(%op0: i5, %op1: i7) -> (sisi: i1, siui: i1, uisi: i1, uiui: i1) {
  %op0Signed = hwarith.cast %op0 : (i5) -> si5
  %op0Unsigned = hwarith.cast %op0 : (i5) -> ui5
  %op1Signed = hwarith.cast %op1 : (i7) -> si7
  %op1Unsigned = hwarith.cast %op1 : (i7) -> ui7

// CHECK:   %[[SIGN_BIT_OP0:.*]] = comb.extract %op0 from 4 : (i5) -> i1
// CHECK:   %[[SIGN_EXTEND:.*]] = comb.replicate %[[SIGN_BIT_OP0]] : (i1) -> i2
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[SIGN_EXTEND]], %op0 : i2, i5
// CHECK:   %[[SISI_OUT:.*]] = comb.icmp slt %[[OP0_PADDED]], %op1 : i7
  %sisi = hwarith.icmp lt %op0Signed, %op1Signed : si5, si7

// CHECK:   %[[SIGN_BIT_OP0:.*]] = comb.extract %op0 from 4 : (i5) -> i1
// CHECK:   %[[SIGN_EXTEND:.*]] = comb.replicate %[[SIGN_BIT_OP0]] : (i1) -> i3
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[SIGN_EXTEND]], %op0 : i3, i5
// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant false
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op1 : i1, i7
// CHECK:   %[[SIUI_OUT:.*]] = comb.icmp slt %[[OP0_PADDED]], %[[OP1_PADDED]] : i8
  %siui = hwarith.icmp lt %op0Signed, %op1Unsigned : si5, ui7

// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant 0 : i2
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op0 : i2, i5
// CHECK:   %[[UISI_OUT:.*]] = comb.icmp slt %[[OP0_PADDED]], %op1 : i7
  %uisi = hwarith.icmp lt %op0Unsigned, %op1Signed : ui5, si7

// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant 0 : i2
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op0 : i2, i5
// CHECK:   %[[UIUI_OUT:.*]] = comb.icmp ult %[[OP0_PADDED]], %op1 : i7
  %uiui = hwarith.icmp lt %op0Unsigned, %op1Unsigned : ui5, ui7

  %sisiOut = hwarith.cast %sisi : (ui1) -> i1
  %siuiOut = hwarith.cast %siui : (ui1) -> i1
  %uisiOut = hwarith.cast %uisi : (ui1) -> i1
  %uiuiOut = hwarith.cast %uiui : (ui1) -> i1

// CHECK:   hw.output %[[SISI_OUT]], %[[SIUI_OUT]], %[[UISI_OUT]], %[[UIUI_OUT]] : i1, i1, i1, i1
  hw.output %sisiOut, %siuiOut, %uisiOut, %uiuiOut : i1, i1, i1, i1
}

// -----

// Signature conversion and other-dialect operations using signedness values.
// CHECK:      hw.module @sigAndOps(%a: i8, %b: i8, %cond: i1, %clk: i1) -> (out: i8) {
// CHECK-NEXT:   %[[MUX_OUT:.*]] = comb.mux %cond, %a, %b : i8
// CHECK-NEXT:   %[[REG_OUT:.*]] = seq.compreg %[[MUX_OUT]], %clk : i8
// CHECK-NEXT:   hw.output %[[REG_OUT]] : i8
// CHECK-NEXT: }
hw.module @sigAndOps(%a: ui8, %b: ui8, %cond: i1, %clk : i1) -> (out: ui8)  {
    %0 = comb.mux %cond, %a, %b : ui8
    %1 = seq.compreg %0, %clk: ui8
    hw.output %1 : ui8
}

// -----

// Type conversions of struct and array ops.
// CHECK:      hw.module @structAndArrays(%a: i8, %b: i8) -> (out: !hw.struct<foo: !hw.array<2xi8>>) {
// CHECK-NEXT:   %[[ARRAY:.*]] = hw.array_create %a, %b : i8
// CHECK-NEXT:   %[[STRUCT:.*]] = hw.struct_create (%[[ARRAY]]) : !hw.struct<foo: !hw.array<2xi8>>
// CHECK-NEXT:   hw.output %[[STRUCT]] : !hw.struct<foo: !hw.array<2xi8>>
// CHECK-NEXT: }
hw.module @structAndArrays(%a: ui8, %b: ui8) -> (out: !hw.struct<foo: !hw.array<2xui8>>)  {
    %2 = hw.array_create %a, %b : ui8
    %3 = hw.struct_create (%2) : !hw.struct<foo: !hw.array<2xui8>>
    hw.output %3 : !hw.struct<foo: !hw.array<2xui8>>
}

// -----

// CHECK: msft.module.extern @externModule(%a: i8, %b: i8) -> (out: !hw.struct<foo: !hw.array<2xi8>>)
msft.module.extern @externModule(%a: ui8, %b: ui8) -> (out: !hw.struct<foo: !hw.array<2xui8>>) 

// -----


// CHECK-LABEL:   hw.module @backedges() {
// CHECK-NEXT:      %[[VAL_0:.*]] = hw.constant false
// CHECK-NEXT:      %[[VAL_1:.*]] = comb.concat %[[VAL_0]], %[[VAL_2:.*]] : i1, i1
// CHECK-NEXT:      %[[VAL_3:.*]] = hw.constant false
// CHECK-NEXT:      %[[VAL_4:.*]] = comb.concat %[[VAL_3]], %[[VAL_2]] : i1, i1
// CHECK-NEXT:      %[[VAL_5:.*]] = comb.add %[[VAL_1]], %[[VAL_4]] : i2
// CHECK-NEXT:      %[[VAL_2]] = hw.constant true
// CHECK-NEXT:      hw.output
// CHECK-NEXT:    }
hw.module @backedges() {
  %res = hwarith.add %arg, %arg : (ui1, ui1) -> ui2
  %arg = hwarith.constant 1 : ui1
}

// -----

// CHECK-LABEL:   hw.module @wires() {
// CHECK:           %[[VAL_0:.*]] = sv.wire  : !hw.inout<i2>
// CHECK:           %[[VAL_1:.*]] = sv.reg  : !hw.inout<i2>
// CHECK:           %[[VAL_2:.*]] = sv.read_inout %[[VAL_0]] : !hw.inout<i2>
// CHECK:           %[[VAL_3:.*]] = sv.read_inout %[[VAL_1]] : !hw.inout<i2>
// CHECK:           %[[VAL_4:.*]] = comb.icmp eq %[[VAL_3]], %[[VAL_2]] : i2
// CHECK:           %[[VAL_5:.*]] = hw.constant -2 : i2
// CHECK:           sv.assign %[[VAL_0]], %[[VAL_5]] : i2
// CHECK:           sv.assign %[[VAL_1]], %[[VAL_5]] : i2
// CHECK:           hw.output
// CHECK:         }
hw.module @wires () -> () {
  %r52 = sv.wire  : !hw.inout<ui2>
  %r53 = sv.reg : !hw.inout<ui2>
  %0 = sv.read_inout %r52 : !hw.inout<ui2>
  %1 = sv.read_inout %r53 : !hw.inout<ui2>
  %33 = hwarith.cast %0 : (ui2) -> i2
  %34 = hwarith.cast %1 : (ui2) -> i2
  %35 = comb.icmp eq %34, %33 : i2

  %c0_ui2 = hwarith.constant 2 : ui2
  sv.assign %r52, %c0_ui2 : ui2
  sv.assign %r53, %c0_ui2 : ui2
}
