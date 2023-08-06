// RUN: circt-opt %s --canonicalize | FileCheck %s

func.func private @Bool(%arg0: i1)
func.func private @Seq(%arg0: !ltl.sequence)
func.func private @Prop(%arg0: !ltl.property)

// CHECK-LABEL: @DelayFolds
func.func @DelayFolds(%arg0: !ltl.sequence) {
  // delay(s, 0, 0) -> s
  // CHECK-NEXT: call @Seq(%arg0)
  %0 = ltl.delay %arg0, 0, 0 : !ltl.sequence
  call @Seq(%0) : (!ltl.sequence) -> ()

  // delay(delay(s, 1), 2) -> delay(s, 3)
  // CHECK-NEXT: ltl.delay %arg0, 3 :
  // CHECK-NEXT: call
  %1 = ltl.delay %arg0, 1 : !ltl.sequence
  %2 = ltl.delay %1, 2 : !ltl.sequence
  call @Seq(%2) : (!ltl.sequence) -> ()

  // delay(delay(s, 1, N), 2) -> delay(s, 3)
  // N is dropped
  // CHECK-NEXT: ltl.delay %arg0, 3 :
  // CHECK-NEXT: ltl.delay %arg0, 3 :
  // CHECK-NEXT: call
  // CHECK-NEXT: call
  %3 = ltl.delay %arg0, 1, 0 : !ltl.sequence
  %4 = ltl.delay %arg0, 1, 42 : !ltl.sequence
  %5 = ltl.delay %3, 2 : !ltl.sequence
  %6 = ltl.delay %4, 2 : !ltl.sequence
  call @Seq(%5) : (!ltl.sequence) -> ()
  call @Seq(%6) : (!ltl.sequence) -> ()

  // delay(delay(s, 1), 2, N) -> delay(s, 3)
  // N is dropped
  // CHECK-NEXT: ltl.delay %arg0, 3 :
  // CHECK-NEXT: ltl.delay %arg0, 3 :
  // CHECK-NEXT: call
  // CHECK-NEXT: call
  %7 = ltl.delay %arg0, 1 : !ltl.sequence
  %8 = ltl.delay %arg0, 1 : !ltl.sequence
  %9 = ltl.delay %7, 2, 0 : !ltl.sequence
  %10 = ltl.delay %8, 2, 42 : !ltl.sequence
  call @Seq(%9) : (!ltl.sequence) -> ()
  call @Seq(%10) : (!ltl.sequence) -> ()

  // delay(delay(s, 1, 2), 3, 0) -> delay(s, 4, 2)
  // delay(delay(s, 1, 2), 3, 5) -> delay(s, 4, 7)
  // CHECK-NEXT: ltl.delay %arg0, 4, 2 :
  // CHECK-NEXT: ltl.delay %arg0, 4, 7 :
  // CHECK-NEXT: call
  // CHECK-NEXT: call
  %11 = ltl.delay %arg0, 1, 2 : !ltl.sequence
  %12 = ltl.delay %arg0, 1, 2 : !ltl.sequence
  %13 = ltl.delay %11, 3, 0 : !ltl.sequence
  %14 = ltl.delay %12, 3, 5 : !ltl.sequence
  call @Seq(%13) : (!ltl.sequence) -> ()
  call @Seq(%14) : (!ltl.sequence) -> ()
  return
}

// CHECK-LABEL: @ConcatFolds
func.func @ConcatFolds(%arg0: !ltl.sequence, %arg1: !ltl.sequence, %arg2: !ltl.sequence) {
  // concat(s) -> s
  // CHECK-NEXT: call @Seq(%arg0)
  %0 = ltl.concat %arg0 : !ltl.sequence
  call @Seq(%0) : (!ltl.sequence) -> ()

  // concat(concat(s0, s1), s2) -> concat(s0, s1, s2)
  // concat(s0, concat(s1, s2)) -> concat(s0, s1, s2)
  // concat(concat(s0, s1), s2, s0, concat(s1, s2)) -> concat(s0, s1, s2, s0, s1, s2)
  // CHECK-NEXT: ltl.concat %arg0, %arg1, %arg2 :
  // CHECK-NEXT: ltl.concat %arg0, %arg1, %arg2 :
  // CHECK-NEXT: ltl.concat %arg0, %arg1, %arg2, %arg0, %arg1, %arg2 :
  // CHECK-NEXT: call
  // CHECK-NEXT: call
  // CHECK-NEXT: call
  %1 = ltl.concat %arg0, %arg1 : !ltl.sequence, !ltl.sequence
  %2 = ltl.concat %1, %arg2 : !ltl.sequence, !ltl.sequence
  %3 = ltl.concat %arg1, %arg2 : !ltl.sequence, !ltl.sequence
  %4 = ltl.concat %arg0, %3 : !ltl.sequence, !ltl.sequence
  %5 = ltl.concat %1, %arg2, %arg0, %3 : !ltl.sequence, !ltl.sequence, !ltl.sequence, !ltl.sequence
  call @Seq(%2) : (!ltl.sequence) -> ()
  call @Seq(%4) : (!ltl.sequence) -> ()
  call @Seq(%5) : (!ltl.sequence) -> ()

  // delay(concat(s0, s1), N, M) -> concat(delay(s0, N, M), s1)
  // CHECK-NEXT: [[TMP:%.+]] = ltl.delay %arg0, 2, 3 :
  // CHECK-NEXT: ltl.concat [[TMP]], %arg1 :
  // CHECK-NEXT: call
  %6 = ltl.concat %arg0, %arg1 : !ltl.sequence, !ltl.sequence
  %7 = ltl.delay %6, 2, 3 : !ltl.sequence
  call @Seq(%7) : (!ltl.sequence) -> ()
  return
}

// CHECK-LABEL: @ClockingFolds
func.func @ClockingFolds(%arg0: !ltl.property) {
  // disable(p, false) -> p
  // CHECK-NEXT: call @Prop(%arg0)
  %false = hw.constant false
  %0 = ltl.disable %arg0 if %false : !ltl.property
  call @Prop(%0) : (!ltl.property) -> ()
  return
}
