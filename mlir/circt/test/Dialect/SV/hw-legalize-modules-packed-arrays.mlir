// RUN: circt-opt -split-input-file -hw-legalize-modules -verify-diagnostics %s | FileCheck %s

module attributes {circt.loweringOptions = "disallowPackedArrays"} {
hw.module @reject_arrays(%arg0: i8, %arg1: i8, %arg2: i8,
                         %arg3: i8, %sel: i2, %clock: i1)
   -> (a: !hw.array<4xi8>) {
  // This needs full-on "legalize types" for the HW dialect.
  
  %reg = sv.reg  : !hw.inout<array<4xi8>>
  sv.alwaysff(posedge %clock)  {
    // expected-error @+1 {{unsupported packed array expression}}
    %0 = hw.array_create %arg0, %arg1, %arg2, %arg3 : i8
    sv.passign %reg, %0 : !hw.array<4xi8>
  }

  // expected-error @+1 {{unsupported packed array expression}}
  %1 = sv.read_inout %reg : !hw.inout<array<4xi8>>
  hw.output %1 : !hw.array<4xi8>
}
}

// -----
module attributes {circt.loweringOptions = "disallowPackedArrays"} {
// CHECK-LABEL: hw.module @array_create_get_comb
hw.module @array_create_get_comb(%arg0: i8, %arg1: i8, %arg2: i8, %arg3: i8,
                                 %sel: i2)
   -> (a: i8) {
  // CHECK: %casez_tmp = sv.reg  : !hw.inout<i8>
  // CHECK: sv.alwayscomb  {
  // CHECK:   sv.case casez %sel : i2
  // CHECK:   case b00: {
  // CHECK:     sv.bpassign %casez_tmp, %arg0 : i8
  // CHECK:   }
  // CHECK:   case b01: {
  // CHECK:     sv.bpassign %casez_tmp, %arg1 : i8
  // CHECK:   }
  // CHECK:   case b10: {
  // CHECK:     sv.bpassign %casez_tmp, %arg2 : i8
  // CHECK:   }
  // CHECK:   default: {
  // CHECK:     sv.bpassign %casez_tmp, %arg3 : i8
  // CHECK:   }
  // CHECK: }
  %0 = hw.array_create %arg3, %arg2, %arg1, %arg0 : i8

  // CHECK: %0 = sv.read_inout %casez_tmp : !hw.inout<i8>
  %1 = hw.array_get %0[%sel] : !hw.array<4xi8>, i2

  // CHECK: hw.output %0 : i8
  hw.output %1 : i8
}

// CHECK-LABEL: hw.module @array_create_get_default
hw.module @array_create_get_default(%arg0: i8, %arg1: i8, %arg2: i8, %arg3: i8,
                            %sel: i2) {
  // CHECK: %casez_tmp = sv.reg  : !hw.inout<i8>
  // CHECK: sv.initial  {
  sv.initial {
    // CHECK:   %x_i8 = sv.constantX : i8
    // CHECK:   sv.case casez %sel : i2
    // CHECK:   case b00: {
    // CHECK:     sv.bpassign %casez_tmp, %arg0 : i8
    // CHECK:   }
    // CHECK:   case b01: {
    // CHECK:     sv.bpassign %casez_tmp, %arg1 : i8
    // CHECK:   }
    // CHECK:   case b10: {
    // CHECK:     sv.bpassign %casez_tmp, %arg2 : i8
    // CHECK:   }
    // CHECK:   default: {
    // CHECK:     sv.bpassign %casez_tmp, %x_i8 : i8
    // CHECK:   }
    %three_array = hw.array_create %arg2, %arg1, %arg0 : i8

    // CHECK:   %0 = sv.read_inout %casez_tmp : !hw.inout<i8>
    %2 = hw.array_get %three_array[%sel] : !hw.array<3xi8>, i2

    // CHECK:   %1 = comb.icmp eq %0, %arg2 : i8
    // CHECK:   sv.if %1  {
    %cond = comb.icmp eq %2, %arg2 : i8
    sv.if %cond {
      sv.fatal 1
    }
  }
}

// CHECK-LABEL: hw.module @array_constant_get_comb
hw.module @array_constant_get_comb(%sel: i2)
   -> (a: i8) {
  // CHECK: %casez_tmp = sv.reg  : !hw.inout<i8>
  // CHECK: sv.alwayscomb  {
  // CHECK:   sv.case casez %sel : i2
  // CHECK:   case b00: {
  // CHECK:     sv.bpassign %casez_tmp, %c3_i8 : i8
  // CHECK:   }
  // CHECK:   case b01: {
  // CHECK:     sv.bpassign %casez_tmp, %c2_i8 : i8
  // CHECK:   }
  // CHECK:   case b10: {
  // CHECK:     sv.bpassign %casez_tmp, %c1_i8 : i8
  // CHECK:   }
  // CHECK:   default: {
  // CHECK:     sv.bpassign %casez_tmp, %c0_i8 : i8
  // CHECK:   }
  // CHECK: }
  %0 = hw.aggregate_constant [0 : i8, 1 : i8, 2 : i8, 3 : i8] : !hw.array<4xi8>
  // CHECK: %0 = sv.read_inout %casez_tmp : !hw.inout<i8>
  %1 = hw.array_get %0[%sel] : !hw.array<4xi8>, i2

  // CHECK: hw.output %0 : i8
  hw.output %1 : i8
}

}  // end builtin.module
