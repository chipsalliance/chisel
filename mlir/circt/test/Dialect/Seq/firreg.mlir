// RUN: circt-opt %s -verify-diagnostics --lower-seq-firrtl-init-to-sv --lower-seq-firrtl-to-sv | FileCheck %s --check-prefixes=CHECK,COMMON
// RUN: circt-opt %s -verify-diagnostics --pass-pipeline="builtin.module(lower-seq-firrtl-init-to-sv, hw.module(lower-seq-firrtl-to-sv{disable-reg-randomization}))" | FileCheck %s --check-prefix COMMON --implicit-check-not RANDOMIZE_REG
// RUN: circt-opt %s -verify-diagnostics --pass-pipeline="builtin.module(lower-seq-firrtl-init-to-sv, hw.module(lower-seq-firrtl-to-sv{emit-separate-always-blocks}))" | FileCheck %s --check-prefixes SEPARATE

// COMMON-LABEL: hw.module @lowering
// SEPARATE-LABEL: hw.module @lowering
hw.module @lowering(%clk: i1, %rst: i1, %in: i32) -> (a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) {
  %cst0 = hw.constant 0 : i32

  // CHECK: %rA = sv.reg sym @regA : !hw.inout<i32>
  // CHECK: [[VAL_A:%.+]] = sv.read_inout %rA : !hw.inout<i32>
  %rA = seq.firreg %in clock %clk sym @regA : i32

  // CHECK: %rB = sv.reg sym @regB : !hw.inout<i32>
  // CHECK: [[VAL_B:%.+]] = sv.read_inout %rB : !hw.inout<i32>
  %rB = seq.firreg %in clock %clk sym @regB reset sync %rst, %cst0 : i32

  // CHECK: %rC = sv.reg sym @regC : !hw.inout<i32>
  // CHECK: [[VAL_C:%.+]] = sv.read_inout %rC : !hw.inout<i32>
  %rC = seq.firreg %in clock %clk sym @regC reset async %rst, %cst0 : i32

  // CHECK: %rD = sv.reg sym @regD : !hw.inout<i32>
  // CHECK: [[VAL_D:%.+]] = sv.read_inout %rD : !hw.inout<i32>
  %rD = seq.firreg %in clock %clk sym @regD : i32

  // CHECK: %rE = sv.reg sym @regE : !hw.inout<i32>
  // CHECK: [[VAL_E:%.+]] = sv.read_inout %rE : !hw.inout<i32>
  %rE = seq.firreg %in clock %clk sym @regE reset sync %rst, %cst0 : i32

  // CHECK: %rF = sv.reg sym @regF : !hw.inout<i32>
  // CHECK: [[VAL_F:%.+]] = sv.read_inout %rF : !hw.inout<i32>
  %rF = seq.firreg %in clock %clk sym @regF reset async %rst, %cst0 : i32

  // CHECK: %rAnamed = sv.reg sym @regA : !hw.inout<i32>
  %r = seq.firreg %in clock %clk sym @regA { "name" = "rAnamed" }: i32

  // CHECK: %rNoSym = sv.reg : !hw.inout<i32>
  %rNoSym = seq.firreg %in clock %clk : i32

  // CHECK:      sv.always posedge %clk {
  // CHECK-NEXT:   sv.passign %rA, %in : i32
  // CHECK-NEXT:   sv.passign %rD, %in : i32
  // CHECK-NEXT:   sv.passign %rAnamed, %in : i32
  // CHECK-NEXT:   sv.passign %rNoSym, %in : i32
  // CHECK-NEXT: }
  // CHECK-NEXT: sv.always posedge %clk {
  // CHECK-NEXT:   sv.if %rst {
  // CHECK-NEXT:     sv.passign %rB, %c0_i32 : i32
  // CHECK-NEXT:     sv.passign %rE, %c0_i32 : i32
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     sv.passign %rB, %in : i32
  // CHECK-NEXT:     sv.passign %rE, %in : i32
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: sv.always posedge %clk, posedge %rst {
  // CHECK-NEXT:   sv.if %rst {
  // CHECK-NEXT:     sv.passign %rC, %c0_i32 : i32
  // CHECK-NEXT:     sv.passign %rF, %c0_i32 : i32
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     sv.passign %rC, %in : i32
  // CHECK-NEXT:     sv.passign %rF, %in : i32
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  // SEPARATE:      sv.always posedge %clk {
  // SEPARATE-NEXT:   sv.passign %rA, %in : i32
  // SEPARATE-NEXT: }
  // SEPARATE-NEXT: sv.always posedge %clk {
  // SEPARATE-NEXT:   sv.if %rst {
  // SEPARATE-NEXT:     sv.passign %rB, %c0_i32 : i32
  // SEPARATE-NEXT:   } else {
  // SEPARATE-NEXT:     sv.passign %rB, %in : i32
  // SEPARATE-NEXT:   }
  // SEPARATE-NEXT: }
  // SEPARATE-NEXT: sv.always posedge %clk, posedge %rst {
  // SEPARATE-NEXT:   sv.if %rst {
  // SEPARATE-NEXT:     sv.passign %rC, %c0_i32 : i32
  // SEPARATE-NEXT:   } else {
  // SEPARATE-NEXT:     sv.passign %rC, %in : i32
  // SEPARATE-NEXT:   }
  // SEPARATE-NEXT: }
  // SEPARATE-NEXT: sv.always posedge %clk {
  // SEPARATE-NEXT:   sv.passign %rD, %in : i32
  // SEPARATE-NEXT: }
  // SEPARATE-NEXT: sv.always posedge %clk {
  // SEPARATE-NEXT:   sv.if %rst {
  // SEPARATE-NEXT:     sv.passign %rE, %c0_i32 : i32
  // SEPARATE-NEXT:   } else {
  // SEPARATE-NEXT:     sv.passign %rE, %in : i32
  // SEPARATE-NEXT:   }
  // SEPARATE-NEXT: }
  // SEPARATE-NEXT: sv.always posedge %clk, posedge %rst {
  // SEPARATE-NEXT:   sv.if %rst {
  // SEPARATE-NEXT:     sv.passign %rF, %c0_i32 : i32
  // SEPARATE-NEXT:   } else {
  // SEPARATE-NEXT:     sv.passign %rF, %in : i32
  // SEPARATE-NEXT:   }
  // SEPARATE-NEXT: }
  // SEPARATE-NEXT: sv.always posedge %clk {
  // SEPARATE-NEXT:   sv.passign %rAnamed, %in : i32
  // SEPARATE-NEXT: }
  // SEPARATE-NEXT: sv.always posedge %clk {
  // SEPARATE-NEXT:   sv.passign %rNoSym, %in : i32
  // SEPARATE-NEXT: }

  // CHECK:      sv.ifdef  "ENABLE_INITIAL_REG_" {
  // CHECK-NEXT:   sv.ordered {
  // CHECK-NEXT:     sv.ifdef  "FIRRTL_BEFORE_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_BEFORE_INITIAL"
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.initial {
  // CHECK-NEXT:       sv.ifdef.procedural "INIT_RANDOM_PROLOG_" {
  // CHECK-NEXT:         sv.verbatim "`INIT_RANDOM_PROLOG_"
  // CHECK-NEXT:       }
  // CHECK-NEXT:       sv.ifdef.procedural  "RANDOMIZE_REG_INIT" {
  // CHECK-NEXT:         %_RANDOM = sv.logic : !hw.inout<uarray<8xi32>>
  // CHECK-NEXT:         sv.for %i = %c0_i4 to %c-8_i4 step %c1_i4 : i4 {
  // CHECK-NEXT:           %RANDOM = sv.macro.ref.se @RANDOM() : () -> i32
  // CHECK-NEXT:           %24 = comb.extract %i from 0 : (i4) -> i3
  // CHECK-NEXT:           %25 = sv.array_index_inout %_RANDOM[%24] : !hw.inout<uarray<8xi32>>, i3
  // CHECK-NEXT:           sv.bpassign %25, %RANDOM : i32
  // CHECK-NEXT:         }
  // CHECK-NEXT:         %8 = sv.array_index_inout %_RANDOM[%c0_i3] : !hw.inout<uarray<8xi32>>, i3
  // CHECK-NEXT:         %9 = sv.array_index_inout %_RANDOM[%c1_i3] : !hw.inout<uarray<8xi32>>, i3
  // CHECK-NEXT:         %10 = sv.array_index_inout %_RANDOM[%c2_i3] : !hw.inout<uarray<8xi32>>, i3
  // CHECK-NEXT:         %11 = sv.array_index_inout %_RANDOM[%c3_i3] : !hw.inout<uarray<8xi32>>, i3
  // CHECK-NEXT:         %12 = sv.array_index_inout %_RANDOM[%c-4_i3] : !hw.inout<uarray<8xi32>>, i3
  // CHECK-NEXT:         %13 = sv.array_index_inout %_RANDOM[%c-3_i3] : !hw.inout<uarray<8xi32>>, i3
  // CHECK-NEXT:         %14 = sv.array_index_inout %_RANDOM[%c-2_i3] : !hw.inout<uarray<8xi32>>, i3
  // CHECK-NEXT:         %15 = sv.array_index_inout %_RANDOM[%c-1_i3] : !hw.inout<uarray<8xi32>>, i3
  // CHECK-NEXT:         %16 = sv.read_inout %8 : !hw.inout<i32>
  // CHECK-NEXT:         sv.bpassign %rA, %16 : i32
  // CHECK-NEXT:         %17 = sv.read_inout %9 : !hw.inout<i32>
  // CHECK-NEXT:         sv.bpassign %rB, %17 : i32
  // CHECK-NEXT:         %18 = sv.read_inout %10 : !hw.inout<i32>
  // CHECK-NEXT:         sv.bpassign %rC, %18 : i32
  // CHECK-NEXT:         %19 = sv.read_inout %11 : !hw.inout<i32>
  // CHECK-NEXT:         sv.bpassign %rD, %19 : i32
  // CHECK-NEXT:         %20 = sv.read_inout %12 : !hw.inout<i32>
  // CHECK-NEXT:         sv.bpassign %rE, %20 : i32
  // CHECK-NEXT:         %21 = sv.read_inout %13 : !hw.inout<i32>
  // CHECK-NEXT:         sv.bpassign %rF, %21 : i32
  // CHECK-NEXT:         %22 = sv.read_inout %14 : !hw.inout<i32>
  // CHECK-NEXT:         sv.bpassign %rAnamed, %22 : i32
  // CHECK-NEXT:         %23 = sv.read_inout %15 : !hw.inout<i32>
  // CHECK-NEXT:         sv.bpassign %rNoSym, %23 : i32
  // CHECK-NEXT:       }
  // CHECK-NEXT:       sv.if %rst {
  // CHECK-NEXT:         sv.bpassign %rC, %c0_i32 : i32
  // CHECK-NEXT:         sv.bpassign %rF, %c0_i32 : i32
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.ifdef  "FIRRTL_AFTER_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_AFTER_INITIAL"
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  // CHECK: hw.output [[VAL_A]], [[VAL_B]], [[VAL_C]], [[VAL_D]], [[VAL_E]], [[VAL_F]] : i32, i32, i32, i32, i32, i32
  hw.output %rA, %rB, %rC, %rD, %rE, %rF : i32, i32, i32, i32, i32, i32
}

// COMMON-LABEL: hw.module private @UninitReg1(%clock: i1, %reset: i1, %cond: i1, %value: i2) {
hw.module private @UninitReg1(%clock: i1, %reset: i1, %cond: i1, %value: i2) {
  // CHECK: %c0_i2 = hw.constant 0 : i2
  %c0_i2 = hw.constant 0 : i2
  // CHECK-NEXT: %count = sv.reg sym @count : !hw.inout<i2>
  // CHECK-NEXT: %0 = sv.read_inout %count : !hw.inout<i2>
  // CHECK-NEXT: %1 = comb.mux bin %cond, %value, %0 : i2
  // CHECK-NEXT: %2 = comb.mux bin %reset, %c0_i2, %1 : i2
  // CHECK-NEXT: sv.always posedge %clock {
  // CHECK-NEXT:   sv.if %reset {
  // CHECK-NEXT:     sv.passign %count, %c0_i2
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     sv.if %cond {
  // CHECK-NEXT:       sv.passign %count, %value
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  %count = seq.firreg %2 clock %clock sym @count : i2
  %1 = comb.mux bin %cond, %value, %count : i2
  %2 = comb.mux bin %reset, %c0_i2, %1 : i2

  // CHECK-NEXT: sv.ifdef "ENABLE_INITIAL_REG_"  {
  // CHECK-NEXT:   sv.ordered {
  // CHECK-NEXT:     sv.ifdef "FIRRTL_BEFORE_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_BEFORE_INITIAL"
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.initial {
  // CHECK-NEXT:       sv.ifdef.procedural "INIT_RANDOM_PROLOG_" {
  // CHECK-NEXT:         sv.verbatim "`INIT_RANDOM_PROLOG_"
  // CHECK-NEXT:       }
  // CHECK-NEXT:       sv.ifdef.procedural "RANDOMIZE_REG_INIT"  {
  // CHECK-NEXT:         %_RANDOM = sv.logic : !hw.inout<uarray<1xi32>>
  // CHECK:              sv.for %i = %{{false.*}} to %{{true.*}} step %{{true.*}} : i1 {
  // CHECK-NEXT:           %RANDOM = sv.macro.ref.se @RANDOM() : () -> i32
  // CHECK-NEXT:           %6 = comb.extract %i from 0 : (i1) -> i0
  // CHECK-NEXT:           %7 = sv.array_index_inout %_RANDOM[%6] : !hw.inout<uarray<1xi32>>, i0
  // CHECK-NEXT:           sv.bpassign %7, %RANDOM : i32
  // CHECK-NEXT:         }
  // CHECK-NEXT:         %3 = sv.array_index_inout %_RANDOM[%c0_i0] : !hw.inout<uarray<1xi32>>, i0
  // CHECK-NEXT:         %4 = sv.read_inout %3 : !hw.inout<i32>
  // CHECK-NEXT:         %5 = comb.extract %4 from 0 : (i32) -> i2
  // CHECK-NEXT:         sv.bpassign %count, %5 : i2
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.ifdef "FIRRTL_AFTER_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_AFTER_INITIAL"
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  // CHECK: hw.output
  hw.output
}
// COMMON-LABEL: hw.module private @UninitReg1_nonbin(%clock: i1, %reset: i1, %cond: i1, %value: i2) {
hw.module private @UninitReg1_nonbin(%clock: i1, %reset: i1, %cond: i1, %value: i2) {
  // CHECK: %c0_i2 = hw.constant 0 : i2
  %c0_i2 = hw.constant 0 : i2
  // CHECK-NEXT: %count = sv.reg sym @count : !hw.inout<i2>
  // CHECK-NEXT: %0 = sv.read_inout %count : !hw.inout<i2>
  // CHECK-NEXT: %1 = comb.mux %cond, %value, %0 : i2
  // CHECK-NEXT: %2 = comb.mux %reset, %c0_i2, %1 : i2
  // CHECK-NEXT: sv.always posedge %clock {
  // CHECK-NEXT:   sv.passign %count, %2
  // CHECK-NEXT: }

  %count = seq.firreg %2 clock %clock sym @count : i2
  %1 = comb.mux %cond, %value, %count : i2
  %2 = comb.mux %reset, %c0_i2, %1 : i2
  // CHECK: hw.output
  hw.output
}


// module InitReg1 :
//     input clock : Clock
//     input reset : UInt<1>
//     input io_d : UInt<32>
//     output io_q : UInt<32>
//     input io_en : UInt<1>
//
//     node _T = asAsyncReset(reset)
//     reg reg : UInt<32>, clock with :
//       reset => (_T, UInt<32>("h0"))
//     io_q <= reg
//     reg <= mux(io_en, io_d, reg)

// COMMON-LABEL: hw.module private @InitReg1(
hw.module private @InitReg1(%clock: i1, %reset: i1, %io_d: i32, %io_en: i1) -> (io_q: i32) {
  %false = hw.constant false
  %c1_i32 = hw.constant 1 : i32
  %c0_i32 = hw.constant 0 : i32
  %reg = seq.firreg %4 clock %clock sym @__reg__ reset async %reset, %c0_i32 : i32
  %reg2 = seq.firreg %reg2 clock %clock sym @__reg2__ reset sync %reset, %c0_i32 : i32
  %reg3 = seq.firreg %reg3 clock %clock sym @__reg3__ reset async %reset, %c1_i32 : i32
  %0 = comb.concat %false, %reg : i1, i32
  %1 = comb.concat %false, %reg2 : i1, i32
  %2 = comb.add %0, %1 : i33
  %3 = comb.extract %2 from 1 : (i33) -> i32
  %4 = comb.mux bin %io_en, %io_d, %3 : i32

  // COMMON:       %reg = sv.reg sym @[[reg_sym:.+]] : !hw.inout<i32>
  // COMMON-NEXT:  %0 = sv.read_inout %reg : !hw.inout<i32>
  // COMMON-NEXT:  %reg2 = sv.reg sym @[[reg2_sym:.+]] : !hw.inout<i32>
  // COMMON-NEXT:  %1 = sv.read_inout %reg2 : !hw.inout<i32>
  // COMMON-NEXT:  %reg3 = sv.reg sym @[[reg3_sym:.+]] : !hw.inout<i32
  // COMMON-NEXT:  %2 = sv.read_inout %reg3 : !hw.inout<i32>
  // COMMON-NEXT:  %3 = comb.concat %false, %0 : i1, i32
  // COMMON-NEXT:  %4 = comb.concat %false, %1 : i1, i32
  // COMMON-NEXT:  %5 = comb.add %3, %4 : i33
  // COMMON-NEXT:  %6 = comb.extract %5 from 1 : (i33) -> i32
  // COMMON-NEXT:  %7 = comb.mux bin %io_en, %io_d, %6 : i32
  // COMMON-NEXT:  sv.always posedge %clock, posedge %reset  {
  // COMMON-NEXT:    sv.if %reset {
  // COMMON-NEXT:      sv.passign %reg, %c0_i32 : i32
  // COMMON-NEXT:      sv.passign %reg3, %c1_i32 : i32
  // COMMON-NEXT:    } else {
  // COMMON-NEXT:      sv.if %io_en {
  // COMMON-NEXT:        sv.passign %reg, %io_d : i32
  // COMMON-NEXT:      } else {
  // COMMON-NEXT:        sv.passign %reg, %6 : i32
  // COMMON-NEXT:      }
  // COMMON-NEXT:      sv.passign %reg3, %2 : i32
  // COMMON-NEXT:    }
  // COMMON-NEXT:  }
  // COMMON-NEXT:  sv.always posedge %clock  {
  // COMMON-NEXT:    sv.if %reset  {
  // COMMON-NEXT:      sv.passign %reg2, %c0_i32 : i32
  // COMMON-NEXT:    } else  {
  // COMMON-NEXT:    }
  // COMMON-NEXT:  }
  // COMMON-NEXT:  sv.ifdef "ENABLE_INITIAL_REG_"  {
  // COMMON-NEXT:    sv.ordered {
  // COMMON-NEXT:      sv.ifdef  "FIRRTL_BEFORE_INITIAL" {
  // COMMON-NEXT:        sv.verbatim "`FIRRTL_BEFORE_INITIAL"
  // COMMON-NEXT:      }
  // COMMON-NEXT:      sv.initial {
  // CHECK:            sv.ifdef.procedural "INIT_RANDOM_PROLOG_" {
  // CHECK-NEXT:         sv.verbatim "`INIT_RANDOM_PROLOG_"
  // CHECK-NEXT:       }
  // CHECK-NEXT:       sv.ifdef.procedural "RANDOMIZE_REG_INIT"  {
  // CHECK-NEXT:          %_RANDOM = sv.logic : !hw.inout<uarray<3xi32>>
  // CHECK-NEXT:          sv.for %i = %c0_i2 to %c-1_i2 step %c1_i2 : i2 {
  // CHECK-NEXT:            %RANDOM = sv.macro.ref.se @RANDOM() : () -> i32
  // CHECK-NEXT:            %14 = sv.array_index_inout %_RANDOM[%i] : !hw.inout<uarray<3xi32>>, i2
  // CHECK-NEXT:            sv.bpassign %14, %RANDOM : i32
  // CHECK-NEXT:          }
  // CHECK-NEXT:          %8 = sv.array_index_inout %_RANDOM[%c0_i2] : !hw.inout<uarray<3xi32>>, i2
  // CHECK-NEXT:          %9 = sv.array_index_inout %_RANDOM[%c1_i2] : !hw.inout<uarray<3xi32>>, i2
  // CHECK-NEXT:          %10 = sv.array_index_inout %_RANDOM[%c-2_i2] : !hw.inout<uarray<3xi32>>, i2
  // CHECK-NEXT:          %11 = sv.read_inout %8 : !hw.inout<i32>
  // CHECK-NEXT:          sv.bpassign %reg, %11 : i32
  // CHECK-NEXT:          %12 = sv.read_inout %9 : !hw.inout<i32>
  // CHECK-NEXT:          sv.bpassign %reg2, %12 : i32
  // CHECK-NEXT:          %13 = sv.read_inout %10 : !hw.inout<i32>
  // CHECK-NEXT:          sv.bpassign %reg3, %13 : i32
  // CHECK-NEXT:       }
  // COMMON-NEXT:      sv.if %reset {
  // COMMON-NEXT:        sv.bpassign %reg, %c0_i32 : i32
  // COMMON-NEXT:        sv.bpassign %reg3, %c1_i32 : i32
  // COMMON-NEXT:      }
  // COMMON-NEXT:    }
  // COMMON-NEXT:    sv.ifdef  "FIRRTL_AFTER_INITIAL" {
  // COMMON-NEXT:      sv.verbatim "`FIRRTL_AFTER_INITIAL"
  // COMMON-NEXT:    }
  // COMMON-NEXT:  }
  // COMMON-NEXT: }
  // COMMON-NEXT: hw.output %0 : i32
  hw.output %reg : i32
}

// COMMON-LABEL: hw.module private @UninitReg42(%clock: i1, %reset: i1, %cond: i1, %value: i42) {
hw.module private @UninitReg42(%clock: i1, %reset: i1, %cond: i1, %value: i42) {
  %c0_i42 = hw.constant 0 : i42
  %count = seq.firreg %1 clock %clock sym @count : i42
  %0 = comb.mux %cond, %value, %count : i42
  %1 = comb.mux %reset, %c0_i42, %0 : i42

  // CHECK:      %count = sv.reg sym @count : !hw.inout<i42>
  // CHECK:      sv.ifdef "ENABLE_INITIAL_REG_"  {
  // CHECK-NEXT:   sv.ordered {
  // CHECK-NEXT:     sv.ifdef  "FIRRTL_BEFORE_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_BEFORE_INITIAL"
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.initial {
  // CHECK-NEXT:       sv.ifdef.procedural "INIT_RANDOM_PROLOG_" {
  // CHECK-NEXT:         sv.verbatim "`INIT_RANDOM_PROLOG_"
  // CHECK-NEXT:       }
  // CHECK-NEXT:       sv.ifdef.procedural  "RANDOMIZE_REG_INIT" {
  // CHECK-NEXT:         %_RANDOM = sv.logic : !hw.inout<uarray<2xi32>>
  // CHECK-NEXT:         sv.for %i = %c0_i2 to %c-2_i2 step %c1_i2 : i2 {
  // CHECK-NEXT:           %RANDOM = sv.macro.ref.se @RANDOM() : () -> i32
  // CHECK-NEXT:           %9 = comb.extract %i from 0 : (i2) -> i1
  // CHECK-NEXT:           %10 = sv.array_index_inout %_RANDOM[%9] : !hw.inout<uarray<2xi32>>, i1
  // CHECK-NEXT:           sv.bpassign %10, %RANDOM : i32
  // CHECK-NEXT:         }
  // CHECK-NEXT:         %3 = sv.array_index_inout %_RANDOM[%false] : !hw.inout<uarray<2xi32>>, i1
  // CHECK-NEXT:         %4 = sv.array_index_inout %_RANDOM[%true] : !hw.inout<uarray<2xi32>>, i1
  // CHECK-NEXT:         %5 = sv.read_inout %3 : !hw.inout<i32>
  // CHECK-NEXT:         %6 = sv.read_inout %4 : !hw.inout<i32>
  // CHECK-NEXT:         %7 = comb.extract %6 from 0 : (i32) -> i10
  // CHECK-NEXT:         %8 = comb.concat %5, %7 : i32, i10
  // CHECK-NEXT:         sv.bpassign %count, %8 : i42
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.ifdef  "FIRRTL_AFTER_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_AFTER_INITIAL"
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  hw.output
}

// COMMON-LABEL: hw.module private @init1DVector
hw.module private @init1DVector(%clock: i1, %a: !hw.array<2xi1>) -> (b: !hw.array<2xi1>) {
  %r = seq.firreg %a clock %clock sym @__r__ : !hw.array<2xi1>

  // CHECK:      %r = sv.reg sym @[[r_sym:[_A-Za-z0-9]+]]

  // CHECK:      sv.always posedge %clock  {
  // CHECK-NEXT:   sv.passign %r, %a : !hw.array<2xi1>
  // CHECK-NEXT: }

  // CHECK:      sv.ifdef "ENABLE_INITIAL_REG_" {
  // CHECK-NEXT:   sv.ordered {
  // CHECK-NEXT:     sv.ifdef  "FIRRTL_BEFORE_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_BEFORE_INITIAL"
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.initial {
  // CHECK-NEXT:       sv.ifdef.procedural "INIT_RANDOM_PROLOG_" {
  // CHECK-NEXT:         sv.verbatim "`INIT_RANDOM_PROLOG_"
  // CHECK-NEXT:       }
  // CHECK-NEXT:       sv.ifdef.procedural "RANDOMIZE_REG_INIT"  {
  // CHECK-NEXT:       %_RANDOM = sv.logic : !hw.inout<uarray<1xi32>>
  // CHECK-NEXT:       sv.for %i = %false to %true step %true : i1 {
  // CHECK-NEXT:         %RANDOM = sv.macro.ref.se @RANDOM() : () -> i32
  // CHECK-NEXT:         %8 = comb.extract %i from 0 : (i1) -> i0
  // CHECK-NEXT:         %9 = sv.array_index_inout %_RANDOM[%8] : !hw.inout<uarray<1xi32>>, i0
  // CHECK-NEXT:         sv.bpassign %9, %RANDOM : i32
  // CHECK-NEXT:       }
  // CHECK-NEXT:       %1 = sv.array_index_inout %_RANDOM[%c0_i0] : !hw.inout<uarray<1xi32>>, i0
  // CHECK-NEXT:       %2 = sv.read_inout %1 : !hw.inout<i32>
  // CHECK-NEXT:       %3 = comb.extract %2 from 0 : (i32) -> i2
  // CHECK-NEXT:       %4 = sv.array_index_inout %r[%false] : !hw.inout<array<2xi1>>, i1
  // CHECK-NEXT:       %5 = comb.extract %3 from 1 : (i2) -> i1
  // CHECK-NEXT:       sv.bpassign %4, %5 : i1
  // CHECK-NEXT:       %6 = sv.array_index_inout %r[%true] : !hw.inout<array<2xi1>>, i1
  // CHECK-NEXT:       %7 = comb.extract %3 from 0 : (i2) -> i1
  // CHECK-NEXT:       sv.bpassign %6, %7 : i1

  // CHECK:            }
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.ifdef "FIRRTL_AFTER_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_AFTER_INITIAL"
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: hw.output %0 : !hw.array<2xi1>

  hw.output %r : !hw.array<2xi1>
}

// COMMON-LABEL: hw.module private @init2DVector
hw.module private @init2DVector(%clock: i1, %a: !hw.array<1xarray<1xi1>>) -> (b: !hw.array<1xarray<1xi1>>) {
  %r = seq.firreg %a clock %clock sym @__r__ : !hw.array<1xarray<1xi1>>

  // CHECK:      sv.always posedge %clock  {
  // CHECK-NEXT:   sv.passign %r, %a : !hw.array<1xarray<1xi1>>
  // CHECK-NEXT: }
  // CHECK-NEXT: sv.ifdef  "ENABLE_INITIAL_REG_" {
  // CHECK-NEXT:   sv.ordered {
  // CHECK-NEXT:     sv.ifdef  "FIRRTL_BEFORE_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_BEFORE_INITIAL"
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.initial {
  // CHECK-NEXT:       sv.ifdef.procedural "INIT_RANDOM_PROLOG_" {
  // CHECK-NEXT:         sv.verbatim "`INIT_RANDOM_PROLOG_"
  // CHECK-NEXT:       }
  // CHECK-NEXT:       sv.ifdef.procedural  "RANDOMIZE_REG_INIT" {
  // CHECK-NEXT:         %_RANDOM = sv.logic : !hw.inout<uarray<1xi32>>
  // CHECK-NEXT:         sv.for %i = %false to %true step %true : i1 {
  // CHECK-NEXT:           %RANDOM = sv.macro.ref.se @RANDOM() : () -> i32
  // CHECK-NEXT:           %6 = comb.extract %i from 0 : (i1) -> i0
  // CHECK-NEXT:           %7 = sv.array_index_inout %_RANDOM[%6] : !hw.inout<uarray<1xi32>>, i0
  // CHECK-NEXT:           sv.bpassign %7, %RANDOM : i32
  // CHECK-NEXT:         }
  // CHECK-NEXT:         %1 = sv.array_index_inout %_RANDOM[%c0_i0] : !hw.inout<uarray<1xi32>>, i0
  // CHECK-NEXT:         %2 = sv.read_inout %1 : !hw.inout<i32>
  // CHECK-NEXT:         %3 = comb.extract %2 from 0 : (i32) -> i1
  // CHECK-NEXT:         %4 = sv.array_index_inout %r[%c0_i0] : !hw.inout<array<1xarray<1xi1>>>, i0
  // CHECK-NEXT:         %5 = sv.array_index_inout %4[%c0_i0] : !hw.inout<array<1xi1>>, i0
  // CHECK-NEXT:         sv.bpassign %5, %3 : i1
  // CHECK:            }
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.ifdef "FIRRTL_AFTER_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_AFTER_INITIAL"
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  hw.output %r : !hw.array<1xarray<1xi1>>
  // CHECK: hw.output %0 : !hw.array<1xarray<1xi1>>
}

// COMMON-LABEL: hw.module private @initStruct
hw.module private @initStruct(%clock: i1) {
  %r = seq.firreg %r clock %clock sym @__r__ : !hw.struct<a: i1>

  // CHECK:      %r = sv.reg sym @[[r_sym:[_A-Za-z0-9]+]]
  // CHECK:      sv.ifdef "ENABLE_INITIAL_REG_" {
  // CHECK-NEXT:   sv.ordered {
  // CHECK-NEXT:     sv.ifdef  "FIRRTL_BEFORE_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_BEFORE_INITIAL"
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.initial {
  // CHECK-NEXT:       sv.ifdef.procedural "INIT_RANDOM_PROLOG_" {
  // CHECK-NEXT:         sv.verbatim "`INIT_RANDOM_PROLOG_"
  // CHECK-NEXT:       }
  // CHECK-NEXT:       sv.ifdef.procedural "RANDOMIZE_REG_INIT"  {
  // CHECK:              %[[EXTRACT:.*]] = comb.extract %{{.*}} from 0 : (i32) -> i1
  // CHECK-NEXT:         %[[INOUT:.*]] = sv.struct_field_inout %r["a"] : !hw.inout<struct<a: i1>>
  // CHECK-NEXT:         sv.bpassign %[[INOUT]], %[[EXTRACT]] : i1
  // CHECK:            }
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.ifdef "FIRRTL_AFTER_INITIAL" {
  // CHECK-NEXT:       sv.verbatim "`FIRRTL_AFTER_INITIAL"
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  hw.output
}

// COMMON-LABEL: issue1594
// Make sure LowerToHW's merging of always blocks kicks in for this example.
hw.module @issue1594(%clock: i1, %reset: i1, %a: i1) -> (b: i1) {
  %true = hw.constant true
  %false = hw.constant false
  %reset_n = sv.wire sym @__issue1594__reset_n  : !hw.inout<i1>
  %0 = sv.read_inout %reset_n : !hw.inout<i1>
  %1 = comb.xor %reset, %true : i1
  sv.assign %reset_n, %1 : i1
  %r = seq.firreg %a clock %clock sym @__r__ reset sync %0, %false : i1
  // CHECK: sv.always posedge %clock
  // CHECK-NOT: sv.always
  // CHECK: hw.output
  hw.output %r : i1
}

// Check that deeply nested if statement creation doesn't cause any issue.
// COMMON-LABEL: @DeeplyNestedIfs
// CHECK-COUNT-17: sv.if
hw.module @DeeplyNestedIfs(%a_0: i1, %a_1: i1, %a_2: i1, %c_0_0: i1, %c_0_1: i1, %c_1_0: i1, %c_1_1: i1, %c_2_0: i1, %c_2_1: i1, %clock: i1) -> (out_0: i1, out_1: i1) {
  %r_0 = seq.firreg %25 clock %clock {firrtl.random_init_start = 0 : ui64} : i1
  %r_1 = seq.firreg %51 clock %clock {firrtl.random_init_start = 1 : ui64} : i1
  %0 = comb.mux bin %a_1, %c_1_0, %c_0_0 : i1
  %1 = comb.mux bin %a_0, %0, %c_2_0 : i1
  %2 = comb.mux bin %a_2, %1, %c_1_0 : i1
  %3 = comb.mux bin %a_1, %2, %c_0_0 : i1
  %4 = comb.mux bin %a_0, %3, %c_2_0 : i1
  %5 = comb.mux bin %a_2, %4, %c_1_0 : i1
  %6 = comb.mux bin %a_1, %5, %c_0_0 : i1
  %7 = comb.mux bin %a_0, %6, %c_2_0 : i1
  %8 = comb.mux bin %a_2, %7, %c_1_0 : i1
  %9 = comb.mux bin %a_1, %8, %c_0_0 : i1
  %10 = comb.mux bin %a_0, %9, %c_2_0 : i1
  %11 = comb.mux bin %a_2, %10, %c_1_0 : i1
  %12 = comb.mux bin %a_1, %11, %c_0_0 : i1
  %13 = comb.mux bin %a_0, %12, %c_2_0 : i1
  %14 = comb.mux bin %a_2, %13, %c_1_0 : i1
  %15 = comb.mux bin %a_1, %14, %c_0_0 : i1
  %16 = comb.mux bin %a_0, %15, %c_2_0 : i1
  %17 = comb.mux bin %a_2, %16, %c_1_0 : i1
  %18 = comb.mux bin %a_1, %17, %c_0_0 : i1
  %19 = comb.mux bin %a_0, %18, %c_2_0 : i1
  %20 = comb.mux bin %a_2, %19, %c_1_0 : i1
  %21 = comb.mux bin %a_1, %20, %c_0_0 : i1
  %22 = comb.mux bin %a_0, %21, %c_2_0 : i1
  %23 = comb.mux bin %a_2, %22, %c_1_0 : i1
  %24 = comb.mux bin %a_1, %23, %c_0_0 : i1
  %25 = comb.mux bin %a_0, %24, %r_0 : i1
  %26 = comb.mux bin %a_1, %c_1_1, %c_0_1 : i1
  %27 = comb.mux bin %a_0, %26, %c_2_1 : i1
  %28 = comb.mux bin %a_2, %27, %c_1_1 : i1
  %29 = comb.mux bin %a_1, %28, %c_0_1 : i1
  %30 = comb.mux bin %a_0, %29, %c_2_1 : i1
  %31 = comb.mux bin %a_2, %30, %c_1_1 : i1
  %32 = comb.mux bin %a_1, %31, %c_0_1 : i1
  %33 = comb.mux bin %a_0, %32, %c_2_1 : i1
  %34 = comb.mux bin %a_2, %33, %c_1_1 : i1
  %35 = comb.mux bin %a_1, %34, %c_0_1 : i1
  %36 = comb.mux bin %a_0, %35, %c_2_1 : i1
  %37 = comb.mux bin %a_2, %36, %c_1_1 : i1
  %38 = comb.mux bin %a_1, %37, %c_0_1 : i1
  %39 = comb.mux bin %a_0, %38, %c_2_1 : i1
  %40 = comb.mux bin %a_2, %39, %c_1_1 : i1
  %41 = comb.mux bin %a_1, %40, %c_0_1 : i1
  %42 = comb.mux bin %a_0, %41, %c_2_1 : i1
  %43 = comb.mux bin %a_2, %42, %c_1_1 : i1
  %44 = comb.mux bin %a_1, %43, %c_0_1 : i1
  %45 = comb.mux bin %a_0, %44, %c_2_1 : i1
  %46 = comb.mux bin %a_2, %45, %c_1_1 : i1
  %47 = comb.mux bin %a_1, %46, %c_0_1 : i1
  %48 = comb.mux bin %a_0, %47, %c_2_1 : i1
  %49 = comb.mux bin %a_2, %48, %c_1_1 : i1
  %50 = comb.mux bin %a_1, %49, %c_0_1 : i1
  %51 = comb.mux bin %a_0, %50, %r_1 : i1
  hw.output %r_0, %r_1 : i1, i1
}

// COMMON-LABEL: @ArrayElements
hw.module @ArrayElements(%a: !hw.array<2xi1>, %clock: i1, %cond: i1) -> (b: !hw.array<2xi1>) {
  %false = hw.constant false
  %true = hw.constant true
  %0 = hw.array_get %a[%true] : !hw.array<2xi1>, i1
  %1 = hw.array_get %a[%false] : !hw.array<2xi1>, i1
  %r = seq.firreg %6 clock %clock {firrtl.random_init_start = 0 : ui64} : !hw.array<2xi1>
  %2 = hw.array_get %r[%true] : !hw.array<2xi1>, i1
  %3 = hw.array_get %r[%false] : !hw.array<2xi1>, i1
  %4 = comb.mux bin %cond, %1, %3 : i1
  %5 = comb.mux bin %cond, %0, %2 : i1
  %6 = hw.array_create %5, %4 : i1
  hw.output %r : !hw.array<2xi1>
  // CHECK:      %[[r1:.+]] = sv.array_index_inout %r[%false] : !hw.inout<array<2xi1>>, i1
  // CHECK-NEXT: %[[r2:.+]] = sv.array_index_inout %r[%true] : !hw.inout<array<2xi1>>, i1
  // CHECK:      sv.always posedge %clock {
  // CHECK-NEXT:   sv.if %cond {
  // CHECK-NEXT:     sv.passign %[[r1]], %1 : i1
  // CHECK-NEXT:     sv.passign %[[r2]], %0 : i1
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
}

// Explicitly check that an asynchronous reset register with no driver other
// than the reset is given a self-reset.  This avoids lint errors around
// inferred latches that would otherwise happen.
//
// COMMON-LABEL: @AsyncResetUndriven
hw.module @AsyncResetUndriven(%clock: i1, %reset: i1) -> (q: i32) {
  %c0_i32 = hw.constant 0 : i32
  %r = seq.firreg %r clock %clock sym @r reset async %reset, %c0_i32 {firrtl.random_init_start = 0 : ui64} : i32
  hw.output %r : i32
  // CHECK:      %[[regRead:[a-zA-Z0-9_]+]] = sv.read_inout %r
  // CHECK-NEXT: sv.always posedge %clock, posedge %reset
  // CHECK-NEXT:   sv.if %reset {
  // CHECK-NEXT:     sv.passign %r, %c0_i32
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     sv.passign %r, %[[regRead]]
}

// CHECK-LABEL: @Subaccess
hw.module @Subaccess(%clock: i1, %en: i1, %addr: i2, %data: i32) -> (out: !hw.array<3xi32>) {
  %c0_i2 = hw.constant 0 : i2
  %c1_i2 = hw.constant 1 : i2
  %c-2_i2 = hw.constant -2 : i2
  %r = seq.firreg %12 clock %clock {firrtl.random_init_start = 0 : ui64} : !hw.array<3xi32>
  %0 = hw.array_get %r[%c0_i2] : !hw.array<3xi32>, i2
  %1 = hw.array_get %r[%c1_i2] : !hw.array<3xi32>, i2
  %2 = hw.array_get %r[%c-2_i2] : !hw.array<3xi32>, i2
  %3 = comb.icmp bin eq %addr, %c0_i2 : i2
  %4 = comb.and bin %en, %3 : i1
  %5 = comb.mux bin %4, %data, %0 : i32
  %6 = comb.icmp bin eq %addr, %c1_i2 : i2
  %7 = comb.and bin %en, %6 : i1
  %8 = comb.mux bin %7, %data, %1 : i32
  %9 = comb.icmp bin eq %addr, %c-2_i2 : i2
  %10 = comb.and bin %en, %9 : i1
  %11 = comb.mux bin %10, %data, %2 : i32
  %12 = hw.array_create %11, %8, %5 : i32
  hw.output %r : !hw.array<3xi32>
  // CHECK:     %[[IDX:.+]] = sv.array_index_inout %r[%addr] : !hw.inout<array<3xi32>>, i2
  // CHECK:        sv.always posedge %clock {
  // CHECK-NEXT:     sv.if %en {
  // CHECK-NEXT:       sv.passign %[[IDX]], %data : i32
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:     }
  // CHECK-NEXT: }
}

// CHECK-LABEL: @NestedSubaccess
// Check subaccess is restored for nested whens.
// The following test case is generated from:
//  when en_0:
//    when en_1:
//      r[addr_0] <= data_0
//    else when en_2:
//      r[addr_1] <= data_1
//    else:
//      r[addr_2] <= data_2
//  else:
//    r[addr_3] <= data_3
//
hw.module @NestedSubaccess(%clock: i1, %en_0: i1, %en_1: i1, %en_2: i1, %addr_0: i2, %addr_1: i2, %addr_2: i2, %addr_3: i2, %data_0: i32, %data_1: i32, %data_2: i32, %data_3: i32) -> () {
  %c0_i2 = hw.constant 0 : i2
  %c1_i2 = hw.constant 1 : i2
  %c-2_i2 = hw.constant -2 : i2
  %r = seq.firreg %33 clock %clock : !hw.array<3xi32>
  %0 = hw.array_get %r[%c0_i2] : !hw.array<3xi32>, i2
  %1 = hw.array_get %r[%c1_i2] : !hw.array<3xi32>, i2
  %2 = hw.array_get %r[%c-2_i2] : !hw.array<3xi32>, i2
  %3 = comb.icmp bin eq %addr_0, %c0_i2 : i2
  %4 = comb.mux bin %3, %data_0, %0 : i32
  %5 = comb.icmp bin eq %addr_0, %c1_i2 : i2
  %6 = comb.mux bin %5, %data_0, %1 : i32
  %7 = comb.icmp bin eq %addr_0, %c-2_i2 : i2
  %8 = comb.mux bin %7, %data_0, %2 : i32
  %9 = comb.icmp bin eq %addr_1, %c0_i2 : i2
  %10 = comb.mux bin %9, %data_1, %0 : i32
  %11 = comb.icmp bin eq %addr_1, %c1_i2 : i2
  %12 = comb.mux bin %11, %data_1, %1 : i32
  %13 = comb.icmp bin eq %addr_1, %c-2_i2 : i2
  %14 = comb.mux bin %13, %data_1, %2 : i32
  %15 = comb.icmp bin eq %addr_2, %c0_i2 : i2
  %16 = comb.mux bin %15, %data_2, %0 : i32
  %17 = comb.icmp bin eq %addr_2, %c1_i2 : i2
  %18 = comb.mux bin %17, %data_2, %1 : i32
  %19 = comb.icmp bin eq %addr_2, %c-2_i2 : i2
  %20 = comb.mux bin %19, %data_2, %2 : i32
  %21 = comb.icmp bin eq %addr_3, %c0_i2 : i2
  %22 = comb.mux bin %21, %data_3, %0 : i32
  %23 = comb.icmp bin eq %addr_3, %c1_i2 : i2
  %24 = comb.mux bin %23, %data_3, %1 : i32
  %25 = comb.icmp bin eq %addr_3, %c-2_i2 : i2
  %26 = comb.mux bin %25, %data_3, %2 : i32
  %27 = hw.array_create %8, %6, %4 : i32
  %28 = hw.array_create %14, %12, %10 : i32
  %29 = hw.array_create %20, %18, %16 : i32
  %30 = comb.mux bin %en_2, %28, %29 : !hw.array<3xi32>
  %31 = comb.mux bin %en_1, %27, %30 : !hw.array<3xi32>
  %32 = hw.array_create %26, %24, %22 : i32
  %33 = comb.mux bin %en_0, %31, %32 : !hw.array<3xi32>
  // CHECK:        %[[IDX1:.+]] = sv.array_index_inout %r[%addr_0] : !hw.inout<array<3xi32>>, i2
  // CHECK:        %[[IDX2:.+]] = sv.array_index_inout %r[%addr_1] : !hw.inout<array<3xi32>>, i2
  // CHECK:        %[[IDX3:.+]] = sv.array_index_inout %r[%addr_2] : !hw.inout<array<3xi32>>, i2
  // CHECK:        %[[IDX4:.+]] = sv.array_index_inout %r[%addr_3] : !hw.inout<array<3xi32>>, i2
  // CHECK:        sv.always posedge %clock {
  // CHECK-NEXT:   sv.if %en_0 {
  // CHECK-NEXT:     sv.if %en_1 {
  // CHECK-NEXT:       sv.if %true {
  // CHECK-NEXT:         sv.passign %[[IDX1]], %data_0 : i32
  // CHECK-NEXT:       } else {
  // CHECK-NEXT:       }
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       sv.if %en_2 {
  // CHECK-NEXT:         sv.if %true {
  // CHECK-NEXT:           sv.passign %[[IDX2]], %data_1 : i32
  // CHECK-NEXT:         } else {
  // CHECK-NEXT:         }
  // CHECK-NEXT:       } else {
  // CHECK-NEXT:         sv.if %true {
  // CHECK-NEXT:           sv.passign %[[IDX3]], %data_2 : i32
  // CHECK-NEXT:         } else {
  // CHECK-NEXT:         }
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     sv.if %true {
  // CHECK-NEXT:       sv.passign %[[IDX4]], %data_3 : i32
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  hw.output
}

// CHECK-LABEL: @with_preset
hw.module @with_preset(%clock: i1, %reset: i1, %next32: i32, %next16: i16) -> () {
  %preset_0 = seq.firreg %next32 clock %clock preset 0 : i32
  %preset_42 = seq.firreg %next16 clock %clock preset 42 : i16

  // CHECK: %c42_i16 = hw.constant 42 : i16
  // CHECK: %c0_i32 = hw.constant 0 : i32
  // CHECK: sv.ordered {
  // CHECK:   sv.initial {
  // CHECK:     sv.bpassign %preset_0, %c0_i32 : i32
  // CHECK:     sv.bpassign %preset_42, %c42_i16 : i16
  // CHECK:   }
  // CHECK: }
}
