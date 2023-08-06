// REQUIRES: iverilog,cocotb

// RUN: hlstool %s --dynamic-hw --buffering-strategy=cycles --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=max --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// RUN: hlstool %s --dynamic-hw --buffering-strategy=all --verilog --lowering-options=disallowLocalVariables > %t.sv && \
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=max --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// CHECK:      ** TEST
// CHECK-NEXT: ********************************
// CHECK-NEXT: ** max.oneInput
// CHECK-NEXT: ** max.multipleInputs
// CHECK-NEXT: ********************************
// CHECK-NEXT: ** TESTS=2 PASS=2 FAIL=0 SKIP=0
// CHECK-NEXT: ********************************

// Computes the maximum of all inputs
func.func @top(%in0: i64, %in1: i64, %in2: i64, %in3: i64, %in4: i64, %in5: i64, %in6: i64, %in7: i64) -> i64 {
  %c0 = arith.cmpi sge, %in0, %in1 : i64
  %t0 = arith.select %c0, %in0, %in1 : i64
  %c1 = arith.cmpi sge, %in2, %in3 : i64
  %t1 = arith.select %c1, %in2, %in3 : i64
  %c2 = arith.cmpi sge, %in4, %in5 : i64
  %t2 = arith.select %c2, %in4, %in5 : i64
  %c3 = arith.cmpi sge, %in6, %in7 : i64
  %t3 = arith.select %c3, %in6, %in7 : i64

  %c4 = arith.cmpi sge, %t0, %t1 : i64
  %t4 = arith.select %c4, %t0, %t1 : i64
  %c5 = arith.cmpi sge, %t2, %t3 : i64
  %t5 = arith.select %c5, %t2, %t3 : i64

  %c6 = arith.cmpi sge, %t4, %t5 : i64
  %t6 = arith.select %c6, %t4, %t5 : i64
  return %t6 : i64
}
