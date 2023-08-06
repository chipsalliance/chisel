// These tests will be only enabled if circt-lec is built.
// REQUIRES: circt-lec

hw.module @basic(%in: i1) -> (out: i1) {
  hw.output %in : i1
}

// hw.constant
//  RUN: circt-lec %s -c1=basic -c2=notnot -v=false | FileCheck %s --check-prefix=HW_CONSTANT
//  HW_CONSTANT: c1 == c2

hw.module @onePlusTwo() -> (out: i2) {
  %one = hw.constant 1 : i2
  %two = hw.constant 2 : i2
  %three = comb.add bin %one, %two : i2
  hw.output %three : i2
}

hw.module @three() -> (out: i2) {
  %three = hw.constant 3 : i2
  hw.output %three : i2
}

// hw.instance
//  RUN: circt-lec %s -c1=basic -c2=notnot -v=false | FileCheck %s --check-prefix=HW_INSTANCE
//  HW_INSTANCE: c1 == c2

hw.module @not(%in: i1) -> (out: i1) {
  %true = hw.constant true
  %out = comb.xor bin %in, %true : i1
  hw.output %out : i1
}

hw.module @notnot(%in: i1) -> (out: i1) {
  %n = hw.instance "n" @not(in: %in: i1) -> (out: i1)
  %nn = hw.instance "nn" @not(in: %n: i1) -> (out: i1)
  hw.output %nn : i1
}

// hw.output
//  RUN: circt-lec %s -c1=basic -c2=basic -v=false | FileCheck %s --check-prefix=HW_OUTPUT
//  HW_OUTPUT: c1 == c2


hw.module @constZeroZero(%in: i1) -> (o1: i1, o2: i1) {
  %zero = hw.constant 0 : i1
  hw.output %zero, %zero : i1, i1
}

hw.module @xorZeroZero(%in: i1) -> (o1: i1, o2: i1) {
  %zero = comb.xor bin %in, %in : i1
  hw.output %zero, %zero : i1, i1
}

hw.module @constZeroOne(%in: i1) -> (o1: i1, o2: i1) {
  %zero = hw.constant 0 : i1
  %one  = hw.constant 1 : i1
  hw.output %zero, %one : i1, i1
}

// Equivalent modules with two outputs
//  RUN: circt-lec %s -c1=constZeroZero -c2=xorZeroZero -v=false | FileCheck %s --check-prefix=TWOOUTPUTS
//  TWOOUTPUTS: c1 == c2

// Modules with one equivalent and one non-equivalent output
//  RUN: not circt-lec %s -c1=constZeroZero -c2=constZeroOne -v=false | FileCheck %s --check-prefix=TWOOUTPUTSFAIL
//  TWOOUTPUTSFAIL: c1 != c2
