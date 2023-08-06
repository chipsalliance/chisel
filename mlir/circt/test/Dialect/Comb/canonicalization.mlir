// RUN: circt-opt %s -canonicalize='top-down=true region-simplify=true' | FileCheck %s

// CHECK-LABEL: @narrowMux
hw.module @narrowMux(%a: i8, %b: i8, %c: i1) -> (o: i4) {
// CHECK-NEXT: %0 = comb.extract %a from 1 : (i8) -> i4
// CHECK-NEXT: %1 = comb.extract %b from 1 : (i8) -> i4
// CHECK-NEXT: %2 = comb.mux %c, %0, %1 : i4
  %0 = comb.mux %c, %a, %b : i8
  %1 = comb.extract %0 from 1 : (i8) -> i4
  hw.output %1 : i4
}

// CHECK-LABEL: @muxConstantInputs
hw.module @muxConstantInputs(%cond: i1) -> (o: i2) {
// CHECK-NEXT: %false = hw.constant false
// CHECK-NEXT: %0 = comb.concat %cond, %false : i1, i1
  %c0 = hw.constant 2 : i2
  %c1 = hw.constant 0 : i2
  %0 = comb.mux %cond, %c0, %c1 : i2
  hw.output %0 : i2
}

// CHECK-LABEL: @muxConstantInputs2
hw.module @muxConstantInputs2(%cond: i1) -> (o: i2) {
// CHECK-NEXT: %true = hw.constant true
// CHECK-NEXT: %false = hw.constant false
// CHECK-NEXT: %0 = comb.xor %cond, %true : i1
// CHECK-NEXT: %1 = comb.concat %0, %false : i1, i1
  %c0 = hw.constant 0 : i2
  %c1 = hw.constant 2 : i2
  %0 = comb.mux %cond, %c0, %c1: i2
  hw.output %0 : i2
}

// CHECK-LABEL: @muxTF
hw.module @muxTF(%cond: i1) -> (o: i1) {
// CHECK-NEXT: hw.output %cond
  %c0 = hw.constant 0 : i1
  %c1 = hw.constant 1 : i1
  %0 = comb.mux %cond, %c1, %c0: i1
  hw.output %0 : i1
}

// CHECK-LABEL: @muxConstantInputsNegated
hw.module @muxConstantInputsNegated(%cond: i1) -> (o: i2) {
// CHECK-NEXT: %true = hw.constant true
// CHECK-NEXT: %0 = comb.xor %cond, %true : i1
// CHECK-NEXT: %1 = comb.concat %true, %0 : i1, i1
  %c0 = hw.constant 2 : i2
  %c1 = hw.constant 3 : i2
  %0 = comb.mux %cond, %c0, %c1: i2
  hw.output %0 : i2
}

// CHECK-LABEL: @notMux
hw.module @notMux(%a: i4, %b: i4, %cond: i1, %cond2: i1) -> (o: i4, o2: i4) {
  // CHECK-NEXT: comb.mux bin %cond, %b, %a : i4
  %c1 = hw.constant 1 : i1
  %0 = comb.xor %cond, %c1 : i1
  %1 = comb.mux bin %0, %a, %b : i4

  // CHECK-NEXT: %1 = comb.and %cond, %cond2 : i1
  // CHECK-NEXT: %2 = comb.mux %1, %b, %a : i4
  %2 = comb.xor %cond2, %c1 : i1
  %3 = comb.or %0, %2 : i1
  %4 = comb.mux %3, %a, %b : i4

  hw.output %1, %4 : i4, i4
}

// mux(a, 0, 1) -> ~a
// CHECK-LABEL: @notMuxResult
hw.module @notMuxResult(%a: i1) -> (o: i1) {
  // CHECK-NEXT: %true = hw.constant true
  // CHECK-NEXT: %0 = comb.xor %a, %true : i1
  // CHECK-NEXT: hw.output %0
  %c0 = hw.constant 0 : i1
  %c1 = hw.constant 1 : i1
  %0 = comb.mux %a, %c0, %c1 : i1
  hw.output %0 : i1
}

// mux(a, 0, b) -> and(~a, b)
// CHECK-LABEL: @muxSingleBitConstantInputs
hw.module @muxSingleBitConstantInputs(%a: i1, %b: i1) -> (o: i1) {
  // CHECK-NEXT: %true = hw.constant true
  // CHECK-NEXT: %0 = comb.xor %a, %true : i1
  // CHECK-NEXT: %1 = comb.and %0, %b : i1
  // CHECK-NEXT: hw.output %1
  %c0 = hw.constant 0 : i1
  %0 = comb.mux %a, %c0, %b : i1
  hw.output %0 : i1
}

// mux(a, 1, b) -> or(a, b)
// CHECK-LABEL: @muxSingleBitConstantInputs2
hw.module @muxSingleBitConstantInputs2(%a: i1, %b: i1) -> (o: i1) {
  // CHECK-NEXT: %0 = comb.or %a, %b : i1
  // CHECK-NEXT: hw.output %0
  %c0 = hw.constant 1 : i1
  %0 = comb.mux %a, %c0, %b : i1
  hw.output %0 : i1
}

// mux(a, b, 0) -> and(a, b)
// CHECK-LABEL: @muxSingleBitConstantInputs3
hw.module @muxSingleBitConstantInputs3(%a: i1, %b: i1) -> (o: i1) {
  // CHECK-NEXT: %0 = comb.and %a, %b : i1
  // CHECK-NEXT: hw.output %0
  %c0 = hw.constant 0 : i1
  %0 = comb.mux %a, %b, %c0 : i1
  hw.output %0 : i1
}

// mux(a, b, 1) -> or(~a, b)
// CHECK-LABEL: @muxSingleBitConstantInputs4
hw.module @muxSingleBitConstantInputs4(%cond: i1, %arg0: i1) -> (o: i1) {
  // CHECK-NEXT: %true = hw.constant true
  // CHECK-NEXT: %0 = comb.xor %cond, %true : i1
  // CHECK-NEXT: %1 = comb.or %0, %arg0 : i1
  // CHECK-NEXT: hw.output %1
  %c0 = hw.constant 1 : i1
  %0 = comb.mux %cond, %arg0, %c0 : i1
  hw.output %0 : i1
}


// CHECK-LABEL: @notNot
hw.module @notNot(%a: i1) -> (o: i1) {
// CHECK-NEXT: hw.output %a
  %c1 = hw.constant 1 : i1
  %0 = comb.xor %a, %c1 : i1
  %1 = comb.xor %0, %c1 : i1
  hw.output %1 : i1
}


// CHECK-LABEL: @andCancel
hw.module @andCancel(%a: i4, %b : i4) -> (o1: i4, o2: i4) {
// CHECK-NEXT: hw.constant 0 : i4
// CHECK-NEXT: hw.output %c0_i4, %c0_i4 : i4, i4
  %c1 = hw.constant 15 : i4
  %anot = comb.xor %a, %c1 : i4
  %1 = comb.and %a, %anot : i4
  %2 = comb.and %b, %a, %b, %anot, %b : i4
  hw.output %1, %2 : i4, i4
}


// CHECK-LABEL: hw.module @idempotentDeduped1(%arg0: i7, %arg1: i7)
hw.module @idempotentDeduped1(%arg0: i7, %arg1: i7) -> (resAnd: i7, resOr: i7) {
// CHECK-NEXT:    %0 = comb.and %arg0, %arg1 : i7
// CHECK-NEXT:    %1 = comb.or %arg0, %arg1 : i7
// CHECK-NEXT:    hw.output %0, %1 : i7, i7
  %0 = comb.and %arg0    : i7
  %1 = comb.and %0, %arg1: i7
  %2 = comb.or %arg0    : i7
  %3 = comb.or %0, %arg1: i7
  hw.output %1, %3 : i7, i7
}

// CHECK-LABEL: hw.module @idempotentDeduped2(%arg0: i7, %arg1: i7)
hw.module @idempotentDeduped2(%arg0: i7, %arg1: i7) -> (resAnd: i7, resOr: i7) {
// CHECK-NEXT:    %0 = comb.and %arg0, %arg1 : i7
// CHECK-NEXT:    %1 = comb.or %arg0, %arg1 : i7
// CHECK-NEXT:    hw.output %0, %1 : i7, i7
  %0 = comb.and %arg0, %arg0: i7
  %1 = comb.and %0, %arg1: i7
  %2 = comb.or %arg0, %arg0: i7
  %3 = comb.or %0, %arg1: i7
  hw.output %1, %3 : i7, i7
}

// CHECK-LABEL: hw.module @dedupLong(%arg0: i7, %arg1: i7, %arg2: i7)
hw.module @dedupLong(%arg0: i7, %arg1: i7, %arg2: i7) -> (resAnd: i7, resOr: i7) {
// CHECK-NEXT:    %0 = comb.and %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    %1 = comb.or %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    hw.output %0, %1 : i7, i7
  %0 = comb.and %arg0, %arg1, %arg2, %arg0: i7
  %1 = comb.or %arg0, %arg1, %arg2, %arg0: i7
  hw.output %0, %1 : i7, i7
}

// CHECK-LABEL: hw.module @orExclusiveConcats
hw.module @orExclusiveConcats(%arg0: i6, %arg1: i2) -> (o: i9) {
  // CHECK-NEXT:    %false = hw.constant false
  // CHECK-NEXT:    %0 = comb.concat %arg1, %false, %arg0 : i2, i1, i6
  // CHECK-NEXT:    hw.output %0 : i9
  %c0 = hw.constant 0 : i3
  %0 = comb.concat %c0, %arg0 : i3, i6
  %c1 = hw.constant 0 : i7
  %1 = comb.concat %arg1, %c1 : i2, i7
  %2 = comb.or %0, %1 : i9
  hw.output %2 : i9
}

// When two concats are or'd together and have mutually-exclusive fields, they
// can be merged together into a single concat.
// concat0: 0aaa aaa0 0000 0bb0
// concat1: 0000 0000 ccdd d000
// merged:  0aaa aaa0 ccdd dbb0
// CHECK-LABEL: hw.module @orExclusiveConcats2
hw.module @orExclusiveConcats2(%arg0: i6, %arg1: i2, %arg2: i2, %arg3: i3) -> (o: i16) {
  // CHECK-NEXT:    %false = hw.constant false
  // CHECK-NEXT:    %0 = comb.concat %false, %arg0, %false, %arg2, %arg3, %arg1, %false : i1, i6, i1, i2, i3, i2, i1
  // CHECK-NEXT:    hw.output %0 : i16
  %c0 = hw.constant 0 : i1
  %c1 = hw.constant 0 : i6
  %c2 = hw.constant 0 : i1
  %0 = comb.concat %c0, %arg0, %c1, %arg1, %c2: i1, i6, i6, i2, i1
  %c3 = hw.constant 0 : i8
  %c4 = hw.constant 0 : i3
  %1 = comb.concat %c3, %arg2, %arg3, %c4 : i8, i2, i3, i3
  %2 = comb.or %0, %1 : i16
  hw.output %2 : i16
}

// When two concats are or'd together and have mutually-exclusive fields, they
// can be merged together into a single concat.
// concat0: aaaa 1111
// concat1: 1111 10bb
// merged:  1111 1111
// CHECK-LABEL: hw.module @orExclusiveConcats3
hw.module @orExclusiveConcats3(%arg0: i4, %arg1: i2) -> (o: i8) {
  // CHECK-NEXT:    [[RES:%[a-z0-9_-]+]] = hw.constant -1 : i8
  // CHECK-NEXT:    hw.output [[RES]] : i8
  %c0 = hw.constant -1 : i4
  %0 = comb.concat %arg0, %c0: i4, i4
  %c1 = hw.constant -1 : i5
  %c2 = hw.constant 0 : i1
  %1 = comb.concat %c1, %c2, %arg1 : i5, i1, i2
  %2 = comb.or %0, %1 : i8
  hw.output %2 : i8
}

// CHECK-LABEL: hw.module @orMultipleExclusiveConcats
hw.module @orMultipleExclusiveConcats(%arg0: i2, %arg1: i2, %arg2: i2) -> (o: i6) {
  // CHECK-NEXT:    %0 = comb.concat %arg0, %arg1, %arg2 : i2, i2, i2
  // CHECK-NEXT:    hw.output %0 : i6
  %c2 = hw.constant 0 : i2
  %c4 = hw.constant 0 : i4
  %0 = comb.concat %arg0, %c4: i2, i4
  %1 = comb.concat %c2, %arg1, %c2: i2, i2, i2
  %2 = comb.concat %c4, %arg2: i4, i2
  %out = comb.or %0, %1, %2 : i6
  hw.output %out : i6
}

// CHECK-LABEL: hw.module @orConcatsWithMux
hw.module @orConcatsWithMux(%bit: i1, %cond: i1) -> (o: i6) {
  // CHECK-NEXT:    [[RES:%[a-z0-9_-]+]] = hw.constant 0 : i4
  // CHECK-NEXT:    %0 = comb.concat [[RES]], %cond, %bit : i4, i1, i1
  // CHECK-NEXT:    hw.output %0 : i6
  %c0 = hw.constant 0 : i5
  %0 = comb.concat %c0, %bit: i5, i1
  %c1 = hw.constant 0 : i4
  %c2 = hw.constant 2 : i2
  %c3 = hw.constant 0 : i2
  %1 = comb.mux %cond, %c2, %c3 : i2
  %2 = comb.concat %c1, %1 : i4, i2
  %3 = comb.or %0, %2 : i6
  hw.output %3 : i6
}

// CHECK-LABEL: @extractNested
hw.module @extractNested(%0: i5) -> (o1 : i1) {
// Multiple layers of nested extract is a weak evidence that the cannonicalization
// operates recursively.
// CHECK-NEXT: %0 = comb.extract %arg0 from 4 : (i5) -> i1
  %1 = comb.extract %0 from 1 : (i5) -> i4
  %2 = comb.extract %1 from 2 : (i4) -> i2
  %3 = comb.extract %2 from 1 : (i2) -> i1
  hw.output %3 : i1
}

// CHECK-LABEL: @flattenMuxTrue
hw.module @flattenMuxTrue(%arg0: i1, %arg1: i8, %arg2: i8, %arg3: i8, %arg4 : i8) -> (o1 : i8) {
// CHECK-NEXT:    [[RET:%[0-9]+]] = comb.mux %arg0, %arg1, %arg4
// CHECK-NEXT:    hw.output [[RET]]
  %0 = comb.mux %arg0, %arg1, %arg2 : i8
  %1 = comb.mux %arg0, %0   , %arg3 : i8
  %2 = comb.mux %arg0, %1   , %arg4 : i8
  hw.output %2 : i8
}

// CHECK-LABEL: @flattenMuxFalse
hw.module @flattenMuxFalse(%arg0: i1, %arg1: i8, %arg2: i8, %arg3: i8, %arg4 : i8) -> (o1 : i8) {
// CHECK-NEXT:    [[RET:%[0-9]+]] = comb.mux %arg0, %arg4, %arg2
// CHECK-NEXT:    hw.output [[RET]]
  %0 = comb.mux %arg0, %arg1, %arg2 : i8
  %1 = comb.mux %arg0, %arg3, %0    : i8
  %2 = comb.mux %arg0, %arg4, %1    : i8
  hw.output %2 : i8
}

// CHECK-LABEL: @flattenMuxMixed
hw.module @flattenMuxMixed(%arg0: i1, %arg1: i8, %arg2: i8, %arg3: i8, %arg4 : i8) -> (o1 : i8) {
// CHECK-NEXT:    [[RET:%[0-9]+]] = comb.mux %arg0, %arg1, %arg4
// CHECK-NEXT:    hw.output [[RET]]
  %0 = comb.mux %arg0, %arg1, %arg2 : i8
  %1 = comb.mux %arg0, %arg3, %arg4 : i8
  %2 = comb.mux %arg0, %0   , %1    : i8
  hw.output %2 : i8
}

// CHECK-LABEL: @flattenNotOnDifferentCond
hw.module @flattenNotOnDifferentCond(%arg0: i1, %arg1: i1, %arg2: i1,
 %arg3: i8, %arg4 : i8, %arg5: i8, %arg6: i8) -> (o1 : i8) {
// CHECK-NEXT:    %0 = comb.mux %arg0, %arg3, %arg4 : i8
// CHECK-NEXT:    %1 = comb.mux %arg1, %0, %arg5 : i8
// CHECK-NEXT:    %2 = comb.mux %arg2, %1, %arg6 : i8
// CHECK-NEXT:    hw.output %2 : i8
  %0 = comb.mux %arg0, %arg3, %arg4 : i8
  %1 = comb.mux %arg1, %0,    %arg5 : i8
  %2 = comb.mux %arg2, %1,    %arg6 : i8
  hw.output %2 : i8
}

// CHECK-LABEL: @subCst
hw.module @subCst(%a: i4) -> (o1: i4) {
// CHECK-NEXT: %c-4_i4 = hw.constant -4 : i4
// CHECK-NEXT: %0 = comb.add %a, %c-4_i4 : i4
  %c1 = hw.constant 4 : i4
  %b = comb.sub %a, %c1 : i4
  hw.output %b : i4
}

// CHECK-LABEL: @addConstAndConst
hw.module @addConstAndConst(%a: i4) -> (o1: i4, o2: i4) {
// CHECK: %c3_i4 = hw.constant 3 : i4
// CHECK: [[RESULT:%.+]] = comb.add %a, %c3_i4 : i4
// CHECK: hw.output [[RESULT]]
  %c1 = hw.constant 1 : i4
  %c2 = hw.constant 2 : i4
  %b = comb.add %a, %c1 : i4
  %c = comb.add %b, %c2 : i4
  hw.output %c, %b : i4, i4
}

// Validates that when there is a matching suffix, and prefix, both of them are removed
// appropriately, and strips of an unnecessary Cat where possible.
// CHECK-LABEL: hw.module @compareStrengthReductionRemoveSuffixAndPrefix
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.icmp uge %arg0, %arg1 : i9
// CHECK-NEXT:    hw.output [[RES]] : i1
hw.module @compareStrengthReductionRemoveSuffixAndPrefix(%arg0: i9, %arg1: i9) -> (o : i1) {
  %0 = comb.concat %arg0, %arg0, %arg1: i9, i9, i9
  %1 = comb.concat %arg0, %arg1, %arg1: i9, i9, i9
  %2 = comb.icmp uge %0, %1 : i27
  hw.output %2 : i1
}

// Validates that comparison strength reduction will retain the concatenation operator
// when there is >1 elements left in one of them, and doens't spuriously remove all non-matching
// suffices
// CHECK-LABEL: hw.module @compareStrengthReductionRetainCat
// CHECK-NEXT:    [[ARG0:%[0-9]+]] = comb.concat %arg0, %arg1 : i9, i9
// CHECK-NEXT:    [[ARG1:%[0-9]+]] = comb.concat %arg1, %arg0 : i9, i9
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.icmp uge [[ARG0]], [[ARG1]] : i18
// CHECK-NEXT:    hw.output [[RES]] : i1
hw.module @compareStrengthReductionRetainCat(%arg0: i9, %arg1: i9) -> (o : i1) {
  %0 = comb.concat %arg0, %arg0, %arg1 : i9, i9, i9
  %1 = comb.concat %arg0, %arg1, %arg0 : i9, i9, i9
  %2 = comb.icmp uge %0, %1 : i27
  hw.output %2 : i1
}

// Validates that narrowing signed comparisons without stripping the common suffix
// must not pad an additional sign bit.
// CHECK-LABEL: hw.module @compareStrengthSignedCommonSuffix
// CHECK-NEXT:    [[ARG0:%[0-9]+]] = comb.replicate %arg0 : (i9) -> i18
// CHECK-NEXT:    [[ARG1:%[0-9]+]] = comb.replicate %arg1 : (i9) -> i18
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.icmp sge [[ARG0]], [[ARG1]] : i18
// CHECK-NEXT:    hw.output [[RES]] : i1
hw.module @compareStrengthSignedCommonSuffix(%arg0: i9, %arg1: i9) -> (o : i1) {
  %0 = comb.concat %arg0, %arg0, %arg1 : i9, i9, i9
  %1 = comb.replicate %arg1 : (i9) -> i27
  %2 = comb.icmp sge %0, %1 : i27
  hw.output %2 : i1
}

// Validates that narrowing signed comparisons that strips of the common suffix
// must add the sign-bit.
// CHECK-LABEL: hw.module @compareStrengthSignedCommonPrefix
// CHECK-NEXT:    [[SIGNBIT:%[0-9]+]] = comb.extract %arg0 from 2 : (i3) -> i1
// CHECK-NEXT:    [[ARG1:%[0-9]+]] = comb.concat [[SIGNBIT]], %arg1 : i1, i9
// CHECK-NEXT:    [[ARG2:%[0-9]+]] = comb.concat [[SIGNBIT]], %arg2 : i1, i9
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.icmp sge [[ARG1]], [[ARG2]] : i10
// CHECK-NEXT:    hw.output [[RES]] : i1
hw.module @compareStrengthSignedCommonPrefix(%arg0 : i3, %arg1: i9, %arg2: i9) -> (o : i1) {
  %0 = comb.concat %arg0, %arg1 : i3, i9
  %1 = comb.concat %arg0, %arg2 : i3, i9
  %2 = comb.icmp sge %0, %1 : i12
  hw.output %2 : i1
}

// Validates that narrowing signed comparisons that strips of the common suffix
// must add the sign-bit. The sign bit if the leading common element has a length of 1.
// CHECK-LABEL: hw.module @compareStrengthSignedCommonPrefixNoExtract
// CHECK-NEXT:    [[ARG1:%[0-9]+]] = comb.concat %arg0, %arg2 : i1, i9
// CHECK-NEXT:    [[ARG2:%[0-9]+]] = comb.concat %arg0, %arg3 : i1, i9
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.icmp sge [[ARG1]], [[ARG2]] : i10
// CHECK-NEXT:    hw.output [[RES]] : i1
hw.module @compareStrengthSignedCommonPrefixNoExtract(%arg0 : i1, %arg1 : i3, %arg2: i9, %arg3: i9) -> (o : i1) {
  %0 = comb.concat %arg0, %arg1, %arg2 : i1, i3, i9
  %1 = comb.concat %arg0, %arg1, %arg3 : i1, i3, i9
  %2 = comb.icmp sge %0, %1 : i13
  hw.output %2 : i1
}

// Validates that cmp(concat(..), concat(...)) that should be simplified to true
// are indeed so.
// CHECK-LABEL: hw.module @compareConcatEliminationTrueCases
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    hw.output %true : i1
hw.module @compareConcatEliminationTrueCases(%arg0 : i4, %arg1: i9, %arg2: i7) -> (o : i1) {
  %0 = comb.concat %arg0, %arg1, %arg2 : i4, i9, i7
  %1 = comb.concat %arg0, %arg1, %arg2 : i4, i9, i7
  %2 = comb.icmp sle %0, %1 : i20
  %3 = comb.icmp sge %0, %1 : i20
  %4 = comb.icmp ule %0, %1 : i20
  %5 = comb.icmp uge %0, %1 : i20
  %6 = comb.icmp  eq %0, %1 : i20
  %o = comb.and %2, %3, %4, %5, %6 : i1
  hw.output %o : i1
}

// Validates cases of cmp(concat(..), concat(...)) that should be simplified to false
// are indeed so.
// CHECK-LABEL: hw.module @compareConcatEliminationFalseCases
// CHECK-NEXT:    %false = hw.constant false
// CHECK-NEXT:    hw.output %false : i1
hw.module @compareConcatEliminationFalseCases(%arg0 : i4, %arg1: i9, %arg2: i7) -> (o : i1) {
  %0 = comb.concat %arg0, %arg1, %arg2 : i4, i9, i7
  %1 = comb.concat %arg0, %arg1, %arg2 : i4, i9, i7
  %2 = comb.icmp slt %0, %1 : i20
  %3 = comb.icmp sgt %0, %1 : i20
  %4 = comb.icmp ult %0, %1 : i20
  %5 = comb.icmp ugt %0, %1 : i20
  %6 = comb.icmp  ne %0, %1 : i20
  %o = comb.or %2, %3, %4, %5, %6 : i1
  hw.output %o : i1
}

// CHECK-LABEL: @compareExtractFold
hw.module @compareExtractFold(%arg0: i8) -> (o1: i1, o2: i1, o3: i1) {
  // x > 0b00000011 -> extract(x, 2..8) != 0b000000
  %c3_i8 = hw.constant 3 : i8
  %0 = comb.icmp ugt %arg0, %c3_i8 : i8
  // CHECK: %0 = comb.extract %arg0 from 2 : (i8) -> i6
  // CHECK: %1 = comb.icmp ne %0, %c0_i6 : i6 

  // x < 0b11000000 -> extract(x, 6..8) != 0b11
  %c192_i8 = hw.constant 192 : i8
  %1 = comb.icmp ult %arg0, %c192_i8 : i8
  // CHECK: %2 = comb.extract %arg0 from 6 : (i8) -> i2
  // CHECK: %3 = comb.icmp ne %2, %c-1_i2 : i2

  // The following used to be erroneously folded to `%arg0`.
  %c0_i2 = hw.constant 0 : i2
  %2 = comb.concat %c0_i2, %arg0 : i2, i8
  %c768_i10 = hw.constant 768 : i10
  %3 = comb.icmp ult %2, %c768_i10 : i10

  // CHECK: hw.output %1, %3, %true :
  hw.output %0, %1, %3 : i1, i1, i1
}


// Validates that extract(cat(a, b, c)) -> cat(b, c) when it aligns with the exact elements, or simply
// a when it is a single full element.
// CHECK-LABEL: hw.module @extractCatAlignWithExactElements
hw.module @extractCatAlignWithExactElements(%arg0: i8, %arg1: i9, %arg2: i10) -> (o1 : i17, o2: i19, o3: i9, o4: i10) {
  %0 = comb.concat %arg0, %arg1, %arg2 : i8, i9, i10

  // CHECK-NEXT:    [[R0:%.+]] = comb.concat %arg0, %arg1
  %1 = comb.extract %0 from 10 : (i27) -> i17

  // CHECK-NEXT:    [[R1:%.+]] = comb.concat %arg1, %arg2
  %2 = comb.extract %0 from 0 : (i27) -> i19
  %3 = comb.extract %0 from 10 : (i27) -> i9
  %4 = comb.extract %0 from 0 : (i27) -> i10

  // CHECK-NEXT:    hw.output [[R0]], [[R1]], %arg1, %arg2
  hw.output %1, %2, %3, %4 : i17, i19, i9, i10
}

// Validates that extract(cat(a, b, c)) -> cat(extract(b)) when it matches only on a single
// partial element
// CHECK-LABEL: hw.module @extractCatOnSinglePartialElement
hw.module @extractCatOnSinglePartialElement(%arg0: i8, %arg1: i9, %arg2: i10) -> (o1 : i1, o2: i1, o3: i1, o4: i1) {
  %0 = comb.concat %arg0, %arg1, %arg2 : i8, i9, i10

  // From the first bit position
  // CHECK-NEXT:    [[R0:%.+]] = comb.extract %arg2 from 0 : (i10) -> i1
  %1 = comb.extract %0 from 0 : (i27) -> i1

  // From the last bit position
  // CHECK-NEXT:    [[R1:%.+]] = comb.extract %arg2 from 9 : (i10) -> i1
  %2 = comb.extract %0 from 9 : (i27) -> i1

  // From some middling position
  // CHECK-NEXT:    [[R2:%.+]] = comb.extract %arg2 from 5 : (i10) -> i1
  %3 = comb.extract %0 from 5 : (i27) -> i1

  // From the first bit position on non-first element.
  // CHECK-NEXT:    [[R3:%.]] = comb.extract %arg1 from 0 : (i9) -> i1
  %4 = comb.extract %0 from 10 : (i27) -> i1

  // CHECK-NEXT:    hw.output [[R0]], [[R1]], [[R2]], [[R3]]
  hw.output %1, %2, %3, %4 : i1, i1, i1, i1
}

// Validates that extract(cat(a, b, c)) -> cat(extract(..), .., extract(..))
// containing a mix of full elements and extract elements.
// A few things to look out here:
// - extract is only inserted at elements that require it
// - no zero-elements introduced
// - the order of the elements are correct.
// CHECK-LABEL: hw.module @extractCatOnMultiplePartialElements
hw.module @extractCatOnMultiplePartialElements(%arg0: i8, %arg1: i9, %arg2: i10) -> (o1 : i11, o2 : i5) {
  %0 = comb.concat %arg0, %arg1, %arg2 : i8, i9, i10

  // Part of arg0, all of arg1, part of arg2
  // CHECK-NEXT: [[FROMARG2:%.+]] = comb.extract %arg2 from 9 : (i10) -> i1
  // CHECK-NEXT: [[FROMARG0:%.+]] = comb.extract %arg0 from 0 : (i8) -> i1
  // CHECK-NEXT: [[RESULT1:%.+]] = comb.concat [[FROMARG0]], %arg1, [[FROMARG2]] : i1, i9, i1
  %1 = comb.extract %0 from 9 : (i27) -> i11

  // Part of arg1 and part of arg2
  // CHECK-NEXT: [[FROMARG2:%.+]] = comb.extract %arg2 from 9 : (i10) -> i1
  // CHECK-NEXT: [[FROMARG1:%.+]] = comb.extract %arg1 from 0 : (i9) -> i4
  // CHECK-NEXT: [[RESULT2:%.+]] = comb.concat [[FROMARG1]], [[FROMARG2]] : i4, i1
  %2 = comb.extract %0 from 9 : (i27) -> i5

  // CHECK-NEXT: hw.output [[RESULT1:%.+]], [[RESULT2:%.+]]
  hw.output %1, %2 : i11, i5
}

// Validates that addition narrows the operand widths to the width of the
// single extract usage.
// CHECK-LABEL: hw.module @narrowAdditionSingleExtractUse
hw.module @narrowAdditionSingleExtractUse(%x: i8, %y: i8) -> (z1: i6) {
  // CHECK-NEXT: [[RX:%.+]] = comb.extract %x from 0 : (i8) -> i6
  // CHECK-NEXT: [[RY:%.+]] = comb.extract %y from 0 : (i8) -> i6
  // CHECK-NEXT: [[RESULT:%.+]] = comb.add [[RX]], [[RY]] : i6
  // CHECK-NEXT: hw.output [[RESULT]]

  %false = hw.constant false
  %0 = comb.concat %false, %x : i1, i8
  %1 = comb.concat %false, %y : i1, i8
  %2 = comb.add %0, %1 : i9
  %3 = comb.extract %2 from 0 : (i9) -> i6
  hw.output %3 : i6
}

// Validates that addition narrows to the element itself without an extract
// where possible.
// CHECK-LABEL: hw.module @narrowAdditionToDirectAddition
hw.module @narrowAdditionToDirectAddition(%x: i8, %y: i8) -> (z1: i8) {
  // CHECK-NEXT: [[RESULT:%.+]] = comb.add %x, %y : i8
  // CHECK-NEXT: hw.output [[RESULT]]

  %false = hw.constant false
  %0 = comb.concat %x, %x : i8, i8
  %1 = comb.concat %y, %y : i8, i8
  %2 = comb.add %0, %1 : i16
  %3 = comb.extract %2 from 0 : (i16) -> i8
  hw.output %3 : i8
}

// Validates that addition narrow to the widest extract
// CHECK-LABEL: hw.module @narrowAdditionToWidestExtract
hw.module @narrowAdditionToWidestExtract(%x: i8, %y: i8) -> (z1: i3, z2: i4) {
  // CHECK-NEXT: [[RX:%.+]] = comb.extract %x from 0 : (i8) -> i4
  // CHECK-NEXT: [[RY:%.+]] = comb.extract %y from 0 : (i8) -> i4
  // CHECK-NEXT: [[RESULT2:%.+]] = comb.add [[RX]], [[RY]] : i4
  // CHECK-NEXT: [[RESULT1:%.+]] = comb.extract [[RESULT2]] from 0 : (i4) -> i3
  // CHECK-NEXT: hw.output [[RESULT1]], [[RESULT2]]

  %0 = comb.concat %x, %x : i8, i8
  %1 = comb.concat %y, %y : i8, i8
  %2 = comb.add %0, %1 : i16
  %3 = comb.extract %2 from 0 : (i16) -> i3
  %4 = comb.extract %2 from 0 : (i16) -> i4
  hw.output %3, %4 : i3, i4
}

// Validates that addition narrow to the widest extract
// CHECK-LABEL: hw.module @narrowAdditionStripLeadingZero
hw.module @narrowAdditionStripLeadingZero(%x: i8, %y: i8) -> (z: i8) {
  // CHECK-NEXT: [[RESULT:%.+]] = comb.add %x, %y : i8
  // CHECK-NEXT: hw.output [[RESULT]]

  %false = hw.constant false
  %0 = comb.concat %false, %x : i1, i8
  %1 = comb.concat %false, %y : i1, i8
  %2 = comb.add %0, %1 : i9
  %3 = comb.extract %2 from 0 : (i9) -> i8
  hw.output %3 : i8
}

// Validates that addition narrowing does not happen when the width of the
// largest use is as wide as the addition result itself.
// CHECK-LABEL: hw.module @narrowAdditionRetainOriginal
hw.module @narrowAdditionRetainOriginal(%x: i8, %y: i8) -> (z0: i9, z1: i8) {
  // CHECK-NEXT: false = hw.constant false
  // CHECK-NEXT: %0 = comb.concat %false, %x : i1, i8
  // CHECK-NEXT: %1 = comb.concat %false, %y : i1, i8
  // CHECK-NEXT: %2 = comb.add %0, %1 : i9
  // CHECK-NEXT: %3 = comb.extract %2 from 0 : (i9) -> i8
  // CHECK-NEXT: hw.output %2, %3 : i9, i8

  %false = hw.constant false
  %0 = comb.concat %false, %x : i1, i8
  %1 = comb.concat %false, %y : i1, i8
  %2 = comb.add %0, %1 : i9
  %3 = comb.extract %2 from 0 : (i9) -> i8
  hw.output %2, %3 : i9, i8
}

// Validates that addition narrowing retains the lower bits when not extracting from
// zero.
// CHECK-LABEL: hw.module @narrowAdditionExtractFromNoneZero
hw.module @narrowAdditionExtractFromNoneZero(%x: i8, %y: i8) -> (z0: i4) {
  // CHECK-NEXT: [[RX:%.+]] = comb.extract %x from 0 : (i8) -> i5
  // CHECK-NEXT: [[RY:%.+]] = comb.extract %y from 0 : (i8) -> i5
  // CHECK-NEXT: [[ADD:%.+]] = comb.add [[RX]], [[RY]] : i5
  // CHECK-NEXT: [[RET:%.+]] = comb.extract [[ADD]] from 1 : (i5) -> i4
  // CHECK-NEXT: hw.output [[RET]]

  %0 = comb.add %x, %y : i8
  %1 = comb.extract %0 from 1 : (i8) -> i4
  hw.output %1 : i4
}

// Validates that subtraction narrowing retains the lower bits when not extracting from
// zero.
// CHECK-LABEL: hw.module @narrowSubExtractFromNoneZero
hw.module @narrowSubExtractFromNoneZero(%x: i8, %y: i8) -> (z0: i4) {
  // CHECK-NEXT: [[RX:%.+]] = comb.extract %x from 0 : (i8) -> i5
  // CHECK-NEXT: [[RY:%.+]] = comb.extract %y from 0 : (i8) -> i5
  // CHECK-NEXT: [[ADD:%.+]] = comb.sub [[RX]], [[RY]] : i5
  // CHECK-NEXT: [[RET:%.+]] = comb.extract [[ADD]] from 1 : (i5) -> i4
  // CHECK-NEXT: hw.output [[RET]]

  %0 = comb.sub %x, %y : i8
  %1 = comb.extract %0 from 1 : (i8) -> i4
  hw.output %1 : i4
}

// Validates that subtraction narrowing retains the lower bits when not extracting from
// zero.
// CHECK-LABEL: hw.module @narrowMulExtractFromNoneZero
hw.module @narrowMulExtractFromNoneZero(%x: i8, %y: i8) -> (z0: i4) {
  // CHECK-NEXT: [[RX:%.+]] = comb.extract %x from 0 : (i8) -> i5
  // CHECK-NEXT: [[RY:%.+]] = comb.extract %y from 0 : (i8) -> i5
  // CHECK-NEXT: [[ADD:%.+]] = comb.mul [[RX]], [[RY]] : i5
  // CHECK-NEXT: [[RET:%.+]] = comb.extract [[ADD]] from 1 : (i5) -> i4
  // CHECK-NEXT: hw.output [[RET]]

  %0 = comb.mul %x, %y : i8
  %1 = comb.extract %0 from 1 : (i8) -> i4
  hw.output %1 : i4
}

// Validates that bitwise operation does not retain the lower bit when extracting from
// non-zero.
// CHECK-LABEL: hw.module @narrowBitwiseOpsExtractFromNoneZero
hw.module @narrowBitwiseOpsExtractFromNoneZero(%a: i8, %b: i8, %c: i8, %d: i1) -> (w: i4, x: i4, y: i4, z: i4) {
  // CHECK-NEXT: [[RA:%.+]] = comb.extract %a from 1 : (i8) -> i4
  // CHECK-NEXT: [[RB:%.+]] = comb.extract %b from 1 : (i8) -> i4
  // CHECK-NEXT: [[RC:%.+]] = comb.extract %c from 1 : (i8) -> i4
  // CHECK-NEXT: [[AND:%.+]] = comb.and [[RA]], [[RB]], [[RC]] : i4
  %0 = comb.and %a, %b, %c : i8
  %1 = comb.extract %0 from 1 : (i8) -> i4

  // CHECK-NEXT: [[RA:%.+]] = comb.extract %a from 1 : (i8) -> i4
  // CHECK-NEXT: [[RB:%.+]] = comb.extract %b from 1 : (i8) -> i4
  // CHECK-NEXT: [[RC:%.+]] = comb.extract %c from 1 : (i8) -> i4
  // CHECK-NEXT: [[OR:%.+]] = comb.or [[RA]], [[RB]], [[RC]] : i4
  %2 = comb.or %a, %b, %c : i8
  %3 = comb.extract %2 from 1 : (i8) -> i4

  // CHECK-NEXT: [[RA:%.+]] = comb.extract %a from 1 : (i8) -> i4
  // CHECK-NEXT: [[RB:%.+]] = comb.extract %b from 1 : (i8) -> i4
  // CHECK-NEXT: [[RC:%.+]] = comb.extract %c from 1 : (i8) -> i4
  // CHECK-NEXT: [[XOR:%.+]] = comb.xor [[RA]], [[RB]], [[RC]] : i4
  %4 = comb.xor %a, %b, %c : i8
  %5 = comb.extract %4 from 1 : (i8) -> i4

  // CHECK-NEXT: [[RA:%.+]] = comb.extract %a from 1 : (i8) -> i4
  // CHECK-NEXT: [[RB:%.+]] = comb.extract %b from 1 : (i8) -> i4
  // CHECK-NEXT: [[MUX:%.+]] = comb.mux %d, [[RA]], [[RB]] : i4
  %6 = comb.mux %d, %a, %b : i8
  %7 = comb.extract %6 from 1 : (i8) -> i4

  // CHECK-NEXT: hw.output [[AND]], [[OR]], [[XOR]], [[MUX]]
  hw.output %1, %3, %5, %7 : i4, i4, i4, i4
}

// A regression test case that checks if the narrowed bitwise optimization sets
// insertion points appropriately on rewriting operations.
// CHECK-LABEL: hw.module @narrowBitwiseOpsInsertionPointRegression
hw.module @narrowBitwiseOpsInsertionPointRegression(%a: i8) -> (out: i1) {
  // CHECK-NEXT: [[A1:%.+]] = comb.extract %a from 4 : (i8) -> i3
  // CHECK-NEXT: [[A2:%.+]] = comb.extract %a from 0 : (i8) -> i3
  // CHECK-NEXT: [[AR:%.+]] = comb.or [[A1]], [[A2]] : i3
  %0 = comb.extract %a from 4 : (i8) -> i4
  %1 = comb.extract %a from 0 : (i8) -> i4
  %2 = comb.or %0, %1 : i4

  // CHECK-NEXT: [[B1:%.+]] = comb.extract %2 from 2 : (i3) -> i1
  // CHECK-NEXT: [[B2:%.+]] = comb.extract %2 from 0 : (i3) -> i1
  // CHECK-NEXT: [[BR:%.+]] = comb.or [[B1]], [[B2]] : i1
  %3 = comb.extract %2 from 2 : (i4) -> i2
  %4 = comb.extract %2 from 0 : (i4) -> i2
  %5 = comb.or %3, %4 : i2

  // CHECK-NEXT: hw.output [[BR]] : i1
  %6 = comb.extract %5 from 0 : (i2) -> i1
  hw.output %6 : i1
}

// CHECK-LABEL: hw.module @extract_from_replicate
hw.module @extract_from_replicate(%x: i8) -> (r0: i16, r1: i4, r2: i8) {
  // CHECK-NEXT: %0 = comb.replicate %x : (i8) -> i24
  %0 = comb.replicate %x : (i8) -> i24

  // CHECK-NEXT: %1 = comb.replicate %x : (i8) -> i16
  %r1 = comb.extract %0 from 8 : (i24) -> i16

  // CHECK-NEXT: %2 = comb.extract %x from 2 : (i8) -> i4
  %r2 = comb.extract %0 from 2 : (i24) -> i4

  // We don't know how to simplify this (yet?)
  // CHECK-NEXT: %3 = comb.extract %0 from 2 : (i24) -> i8
  %r3 = comb.extract %0 from 2 : (i24) -> i8

  // CHECK-NEXT: hw.output %1, %2, %3 : 
  hw.output %r1, %r2, %r3 : i16, i4, i8
}


// CHECK-LABEL: hw.module @narrow_extract_from_and
hw.module @narrow_extract_from_and(%arg0: i32) -> (o1: i8, o2: i14, o3: i8, o4: i8) {
  %c240_i32 = hw.constant 240 : i32  // 0xF0
  %0 = comb.and %arg0, %c240_i32 : i32
  %1 = comb.extract %0 from 3 : (i32) -> i8

  %2 = comb.extract %0 from 2 : (i32) -> i14

  // CHECK: %0 = comb.extract %arg0 from 4 : (i32) -> i4
  // CHECK: %1 = comb.concat %c0_i3, %0, %false : i3, i4, i1
  // CHECK: %2 = comb.concat %c0_i8, %0, %c0_i2 : i8, i4, i2

  %c42_i32 = hw.constant 42 : i32  // 0b101010
  %3 = comb.and %arg0, %c42_i32 : i32
  %4 = comb.extract %3 from 1 : (i32) -> i8  
  // CHECK: %3 = comb.extract %arg0 from 1 : (i32) -> i5
  // CHECK: %4 = comb.and %3, %c-11_i5 : i5
  // CHECK: %5 = comb.concat %c0_i3, %4 : i3, i5

  %c12_i8 = hw.constant 12 : i8
  %5 = comb.extract %arg0 from 23 : (i32) -> i8
  %6 = comb.and %5, %c12_i8 : i8
  // CHECK: %6 = comb.extract %arg0 from 25 : (i32) -> i2
  // CHECK: %7 = comb.concat %c0_i4, %6, %c0_i2 : i4, i2, i2
 
  hw.output %1, %2, %4, %6 : i8, i14, i8, i8
  // CHECK: hw.output %1, %2, %5, %7 : i8, i14, i8, i8
}

// CHECK-LABEL: hw.module @fold_mux_tree1
hw.module @fold_mux_tree1(%sel: i2, %a: i8, %b: i8, %c: i8, %d: i8) -> (y: i8) {
  // CHECK-NEXT: %0 = hw.array_create %d, %c, %b, %a : i8
  // CHECK-NEXT: %1 = hw.array_get %0[%sel]
  // CHECK-NEXT: hw.output %1
  %c2_i2 = hw.constant 2 : i2
  %0 = comb.icmp eq %sel, %c2_i2 : i2
  %1 = comb.mux %0, %c, %d : i8

  %c1_i2 = hw.constant 1 : i2
  %2 = comb.icmp eq %sel, %c1_i2 : i2
  %3 = comb.mux %2, %b, %1 : i8

  %c0_i2 = hw.constant 0 : i2
  %4 = comb.icmp eq %sel, %c0_i2 : i2
  %5 = comb.mux %4, %a, %3 : i8
  hw.output %5 : i8
}

// CHECK-LABEL: hw.module @fold_mux_tree1r
hw.module @fold_mux_tree1r(%sel: i2, %a: i8, %b: i8, %c: i8, %d: i8) -> (y: i8) {
  // CHECK-NEXT: %0 = hw.array_create %d, %c, %b, %a : i8
  // CHECK-NEXT: %1 = hw.array_get %0[%sel]
  // CHECK-NEXT: hw.output %1
  %c2_i2 = hw.constant 2 : i2
  %c1_i2 = hw.constant 1 : i2
  %c3_i2 = hw.constant 3 : i2

  %0 = comb.icmp eq %sel, %c1_i2 : i2
  %1 = comb.mux %0, %b, %a : i8

  %2 = comb.icmp eq %sel, %c2_i2 : i2
  %3 = comb.mux %2, %c, %1 : i8

  %4 = comb.icmp eq %sel, %c3_i2 : i2
  %5 = comb.mux %4, %d, %3 : i8
  hw.output %5 : i8
}


// CHECK-LABEL: hw.module @fold_mux_tree2
// This is a sparse tree with 5/8ths load.
hw.module @fold_mux_tree2(%sel: i3, %a: i8, %b: i8, %c: i8, %d: i8) -> (y: i8) {
  // CHECK-NEXT: %0 = hw.array_create %d, %d, %d, %b, %a, %c, %b, %a : i8
  // CHECK-NEXT: %1 = hw.array_get %0[%sel]
  // CHECK-NEXT: hw.output %1
  %c2_i2 = hw.constant 2 : i3
  %0 = comb.icmp eq %sel, %c2_i2 : i3
  %1 = comb.mux %0, %c, %d : i8

  %c1_i2 = hw.constant 1 : i3
  %2 = comb.icmp eq %sel, %c1_i2 : i3
  %3 = comb.mux %2, %b, %1 : i8

  %c0_i2 = hw.constant 0 : i3
  %4 = comb.icmp eq %sel, %c0_i2 : i3
  %5 = comb.mux %4, %a, %3 : i8

  %c3_i2 = hw.constant 3 : i3
  %6 = comb.icmp eq %sel, %c3_i2 : i3
  %7 = comb.mux %6, %a, %5 : i8

  %c4_i2 = hw.constant 4 : i3
  %8 = comb.icmp eq %sel, %c4_i2 : i3
  %9 = comb.mux %8, %b, %7 : i8
  hw.output %9 : i8
}

// CHECK-LABEL: hw.module @fold_mux_tree3
// This has two selectors for the same index value. Make sure we get the
// right one.  "%c" should not be used.
hw.module @fold_mux_tree3(%sel: i2, %a: i8, %b: i8, %c: i8, %d: i8) -> (y: i8) {
  // CHECK-NEXT: %0 = hw.array_create %d, %d, %b, %a : i8
  // CHECK-NEXT: %1 = hw.array_get %0[%sel]
  // CHECK-NEXT: hw.output %1
  %c2_i2 = hw.constant 0 : i2
  %0 = comb.icmp eq %sel, %c2_i2 : i2
  %1 = comb.mux %0, %c, %d : i8

  %c1_i2 = hw.constant 1 : i2
  %2 = comb.icmp eq %sel, %c1_i2 : i2
  %3 = comb.mux %2, %b, %1 : i8

  %c0_i2 = hw.constant 0 : i2
  %4 = comb.icmp eq %sel, %c0_i2 : i2
  %5 = comb.mux %4, %a, %3 : i8
  hw.output %5 : i8
}

// CHECK-LABEL: hw.module @fold_mux_tree4
hw.module @fold_mux_tree4(%sel: i2, %a: i8, %b: i8, %c: i8) -> (y: i8) {
  // CHECK-NEXT: %c-1_i8 = hw.constant -1 : i8
  // CHECK-NEXT: %0 = hw.array_create %c-1_i8, %c, %b, %a : i8
  // CHECK-NEXT: %1 = hw.array_get %0[%sel]
  // CHECK-NEXT: hw.output %1

  %c0_i2 = hw.constant 0 : i2
  %c1_i2 = hw.constant 1 : i2
  %c-1_i8 = hw.constant -1 : i8
  %c-2_i2 = hw.constant -2 : i2
  %0 = comb.icmp eq %sel, %c-2_i2 : i2
  %1 = comb.mux %0, %c, %c-1_i8 : i8
  %2 = comb.icmp eq %sel, %c1_i2 : i2
  %3 = comb.mux %2, %b, %1 : i8
  %4 = comb.icmp eq %sel, %c0_i2 : i2
  %5 = comb.mux %4, %a, %3 : i8
  hw.output %5 : i8
}

// CHECK-LABEL: hw.module @fold_mux_tree5
// This mux tree has an "and" of two selectors.
hw.module @fold_mux_tree5(%sel: i3, %a: i8, %b: i8, %c: i8, %d: i8) -> (y: i8) {
  // CHECK-NEXT: %0 = hw.array_create %d, %d, %d, %b, %a, %c, %b, %a : i8
  // CHECK-NEXT: %1 = hw.array_get %0[%sel]
  // CHECK-NEXT: hw.output %1
  %c-4_i3 = hw.constant -4 : i3
  %c3_i3 = hw.constant 3 : i3
  %c0_i3 = hw.constant 0 : i3
  %c1_i3 = hw.constant 1 : i3
  %c2_i3 = hw.constant 2 : i3
  %0 = comb.icmp eq %sel, %c2_i3 : i3
  %1 = comb.mux %0, %c, %d : i8
  %2 = comb.icmp eq %sel, %c1_i3 : i3
  %3 = comb.mux %2, %b, %1 : i8
  %4 = comb.icmp eq %sel, %c0_i3 : i3
  %5 = comb.icmp eq %sel, %c3_i3 : i3
  %6 = comb.or %5, %4 : i1
  %7 = comb.mux %6, %a, %3 : i8
  %8 = comb.icmp eq %sel, %c-4_i3 : i3
  %9 = comb.mux %8, %b, %7 : i8
  hw.output %9 : i8
}

// CHECK-LABEL: hw.module @dont_fold_mux_tree1
// This shouldn't be turned into an array because it is too sparse.
hw.module @dont_fold_mux_tree1(%sel: i7, %a: i8, %b: i8, %c: i8, %d: i8) -> (y: i8) {
  // CHECK-NOT: array_create
  // CHECK: hw.output
  %c0_i2 = hw.constant 0 : i7
  %c1_i2 = hw.constant 1 : i7
  %c2_i2 = hw.constant 2 : i7
  %0 = comb.icmp eq %sel, %c2_i2 : i7
  %1 = comb.mux %0, %c, %d : i8
  %2 = comb.icmp eq %sel, %c1_i2 : i7
  %3 = comb.mux %2, %b, %1 : i8
  %4 = comb.icmp eq %sel, %c0_i2 : i7
  %5 = comb.mux %4, %a, %3 : i8
  hw.output %5 : i8
}

// CHECK-LABEL: hw.module @dont_fold_mux_tree2
// This shouldn't be turned into a mux tree because its too large.
hw.module @dont_fold_mux_tree2(%sel: i64) -> (o: i3) {
  // CHECK-NOT: array_create
  // CHECK: hw.output
  %c-3_i3 = hw.constant -3 : i3
  %c-4_i3 = hw.constant -4 : i3
  %c3_i3 = hw.constant 3 : i3
  %c0_i3 = hw.constant 0 : i3
  %c14_i64 = hw.constant 14 : i64
  %0 = comb.icmp eq %sel, %c14_i64 : i64
  %1 = comb.mux %0, %c3_i3, %c0_i3 : i3
  %c15_i64 = hw.constant 15 : i64
  %2 = comb.icmp eq %sel, %c15_i64 : i64
  %3 = comb.mux %2, %c-4_i3, %1 : i3
  %c13_i64 = hw.constant 13 : i64
  %4 = comb.icmp eq %sel, %c13_i64 : i64
  %5 = comb.mux %4, %c-3_i3, %3 : i3
  hw.output %5: i3
}

// Issue 675: https://github.com/llvm/circt/issues/675
// CHECK-LABEL: hw.module @SevenSegmentDecoder
hw.module @SevenSegmentDecoder(%in: i4) -> (out: i7) {
  %c1_i4 = hw.constant 1 : i4
  %c6_i6 = hw.constant 6 : i6
  %c-1_i6 = hw.constant -1 : i6
  %c2_i4 = hw.constant 2 : i4
  %c-37_i7 = hw.constant -37 : i7
  %c3_i4 = hw.constant 3 : i4
  %c-49_i7 = hw.constant -49 : i7
  %c4_i4 = hw.constant 4 : i4
  %c-26_i7 = hw.constant -26 : i7
  %c5_i4 = hw.constant 5 : i4
  %c-19_i7 = hw.constant -19 : i7
  %c6_i4 = hw.constant 6 : i4
  %c-3_i7 = hw.constant -3 : i7
  %c7_i4 = hw.constant 7 : i4
  %c7_i7 = hw.constant 7 : i7
  %c-8_i4 = hw.constant -8 : i4
  %c-1_i7 = hw.constant -1 : i7
  %c-7_i4 = hw.constant -7 : i4
  %c-17_i7 = hw.constant -17 : i7
  %c-6_i4 = hw.constant -6 : i4
  %c-9_i7 = hw.constant -9 : i7
  %c-5_i4 = hw.constant -5 : i4
  %c-4_i7 = hw.constant -4 : i7
  %c-4_i4 = hw.constant -4 : i4
  %c57_i7 = hw.constant 57 : i7
  %c-3_i4 = hw.constant -3 : i4
  %c-34_i7 = hw.constant -34 : i7
  %c-2_i4 = hw.constant -2 : i4
  %c-7_i7 = hw.constant -7 : i7
  %c-15_i7 = hw.constant -15 : i7
  %false = hw.constant false
  %c-1_i4 = hw.constant -1 : i4
  %0 = comb.icmp eq %in, %c1_i4 : i4
  %1 = comb.mux %0, %c6_i6, %c-1_i6 : i6
  %2 = comb.icmp eq %in, %c2_i4 : i4
  %3 = comb.concat %false, %1 : i1, i6
  %4 = comb.mux %2, %c-37_i7, %3 : i7
  %5 = comb.icmp eq %in, %c3_i4 : i4
  %6 = comb.mux %5, %c-49_i7, %4 : i7
  %7 = comb.icmp eq %in, %c4_i4 : i4
  %8 = comb.mux %7, %c-26_i7, %6 : i7
  %9 = comb.icmp eq %in, %c5_i4 : i4
  %10 = comb.mux %9, %c-19_i7, %8 : i7
  %11 = comb.icmp eq %in, %c6_i4 : i4
  %12 = comb.mux %11, %c-3_i7, %10 : i7
  %13 = comb.icmp eq %in, %c7_i4 : i4
  %14 = comb.mux %13, %c7_i7, %12 : i7
  %15 = comb.icmp eq %in, %c-8_i4 : i4
  %16 = comb.mux %15, %c-1_i7, %14 : i7
  %17 = comb.icmp eq %in, %c-7_i4 : i4
  %18 = comb.mux %17, %c-17_i7, %16 : i7
  %19 = comb.icmp eq %in, %c-6_i4 : i4
  %20 = comb.mux %19, %c-9_i7, %18 : i7
  %21 = comb.icmp eq %in, %c-5_i4 : i4
  %22 = comb.mux %21, %c-4_i7, %20 : i7
  %23 = comb.icmp eq %in, %c-4_i4 : i4
  %24 = comb.mux %23, %c57_i7, %22 : i7
  %25 = comb.icmp eq %in, %c-3_i4 : i4
  %26 = comb.mux %25, %c-34_i7, %24 : i7
  %27 = comb.icmp eq %in, %c-2_i4 : i4
  %28 = comb.mux %27, %c-7_i7, %26 : i7
  %29 = comb.icmp eq %in, %c-1_i4 : i4
  %30 = comb.mux %29, %c-15_i7, %28 : i7
  // CHECK: %0 = comb.icmp eq %in, %c1_i4 : i4
  // CHECK: %1 = comb.mux %0, %c6_i6, %c-1_i6 : i6
  // CHECK: %2 = comb.concat %false, %1 : i1, i6
  // CHECK: %3 = hw.array_create %c-15_i7, %c-7_i7, %c-34_i7, %c57_i7, %c-4_i7, %c-9_i7, %c-17_i7, %c-1_i7, %c7_i7, %c-3_i7, %c-19_i7, %c-26_i7, %c-49_i7, %c-37_i7, %2, %2 : i7
  // CHECK: %4 = hw.array_get %3[%in]
  hw.output %30 : i7
}

// CHECK-LABEL: hw.module @shru_zero
// This shouldn't crash canonicalize.
hw.module @shru_zero(%a: i8) -> (y: i8) {
  // CHECK: hw.output %a : i8
  %c = hw.constant 0 : i8
  %0 = comb.shru %a, %c : i8
  hw.output %0 : i8
}

// CHECK-LABEL: hw.module @logical_concat_cst1
hw.module @logical_concat_cst1(%value: i38, %v2: i39) -> (a: i39) {
  %true = hw.constant true
  %0 = comb.concat %true, %value : i1, i38

  %c255 = hw.constant 255 : i39
  %1 = comb.and %0, %v2, %c255 : i39
  hw.output %1 : i39

  // CHECK: %false = hw.constant false
  // CHECK: %c255_i38 = hw.constant 255 : i38
  // CHECK: %0 = comb.and %value, %c255_i38 : i38
  // CHECK: %1 = comb.concat %false, %0 : i1, i38
  // CHECK: %2 = comb.and %v2, %1 : i39
  // CHECK: hw.output %2 : i39
}

// CHECK-LABEL: hw.module @logical_concat_cst2
hw.module @logical_concat_cst2(%value: i8, %v2: i16) -> (a: i16) {
  %c15 = hw.constant 15 : i8
  %0 = comb.and %value, %c15 : i8

  %1 = comb.concat %value, %0 : i8, i8

  %c7 = hw.constant 7 : i16
  %2 = comb.and %v2, %1, %c7 : i16
  hw.output %2 : i16

  // CHECK: %c0_i8 = hw.constant 0 : i8
  // CHECK: %c7_i8 = hw.constant 7 : i8
  // CHECK: %0 = comb.and %value, %c7_i8 : i8
  // CHECK: %1 = comb.concat %c0_i8, %0 : i8, i8
  // CHECK: %2 = comb.and %v2, %1 : i16
  // CHECK: hw.output %2 : i16
}

// CHECK-LABEL: hw.module @concat_fold
hw.module @concat_fold(%value: i8) -> (a: i8) {
  // CHECK: hw.output %value : i8
  %0 = comb.concat %value : i8
  hw.output %0 : i8
}


// CHECK-LABEL: hw.module @combine_icmp_compare_concat0
hw.module @combine_icmp_compare_concat0(%thing: i3) -> (a: i1) {
  %false = hw.constant false
  %0 = comb.concat %thing, %false, %thing : i3, i1, i3

  %c0 = hw.constant 0 : i7
  %1 = comb.icmp ne %0, %c0 : i7

  // CHECK: %c0_i3 = hw.constant 0 : i3
  // CHECK: %0 = comb.icmp ne %thing, %c0_i3 : i3
  // CHECK: hw.output %0 : i1
  hw.output %1 : i1
}

// CHECK-LABEL: hw.module @combine_icmp_compare_concat1
hw.module @combine_icmp_compare_concat1(%thing: i3) -> (a: i1) {
  %true = hw.constant true
  %0 = comb.concat %thing, %true, %thing : i3, i1, i3

  %c0 = hw.constant 0 : i7
  %1 = comb.icmp ne %0, %c0 : i7

  // CHECK: hw.output %true : i1
  hw.output %1 : i1
}

// CHECK-LABEL: hw.module @combine_icmp_compare_concat2
hw.module @combine_icmp_compare_concat2(%thing: i3) -> (a: i1) {
  %false = hw.constant false
  %0 = comb.concat %thing, %false, %thing, %false : i3, i1, i3, i1

  %c0 = hw.constant 0 : i8
  %1 = comb.icmp eq %0, %c0 : i8
  hw.output %1 : i1

  // CHECK: %c0_i3 = hw.constant 0 : i3
  // CHECK: %0 = comb.icmp eq %thing, %c0_i3 : i3
  // CHECK: hw.output %0 : i1
}

// CHECK-LABEL: hw.module @combine_icmp_compare_known_bits0
hw.module @combine_icmp_compare_known_bits0(%thing: i4) -> (a: i1) {
  %c5 = hw.constant 13 : i4
  %0 = comb.and %thing, %c5 : i4
  %c0 = hw.constant 0 : i4
  %1 = comb.icmp eq %0, %c0 : i4
  hw.output %1 : i1

  // CHECK:   %0 = comb.extract %thing from 2 : (i4) -> i2
  // CHECK:   %1 = comb.extract %thing from 0 : (i4) -> i1
  // CHECK:   %2 = comb.concat %0, %1 : i2, i1
  // CHECK:   %3 = comb.icmp eq %2, %c0_i3 : i3
  // CHECK:   hw.output %3 : i1
  // CHECK: }
}

// CHECK-LABEL: hw.module @not_icmp
hw.module @not_icmp(%a: i3, %b: i4, %c: i1) -> (x: i1, y: i1) {
  %true = hw.constant true

  %c0 = hw.constant 0 : i3
  %2 = comb.icmp ne %a, %c0 : i3
  %3 = comb.xor %2, %true : i1
  // CHECK: %0 = comb.icmp eq %a, %c0_i3 : i3

  %c1 = hw.constant 1 : i4
  %4 = comb.icmp slt %b, %c1 : i4
  %5 = comb.xor %4, %c, %true : i1
  // CHECK: %1 = comb.icmp sgt %b, %c0_i4 : i4
  // CHECK: %2 = comb.xor %c, %1 : i1

  hw.output %3, %5 : i1, i1
  // CHECK: hw.output %0, %2 : i1, i1
}

// CHECK-LABEL: hw.module @xorICmpConstant
hw.module @xorICmpConstant(%value: i9, %bit: i1) -> (a: i1, b: i1) {
  // This is an integration test for the testcase in Issue #1560.
  %c2_i9 = hw.constant 2 : i9
  %c0_i9 = hw.constant 0 : i9
  %1 = comb.xor %value, %c2_i9 : i9
  %2 = comb.icmp eq %1, %c0_i9 : i9
  // CHECK: %0 = comb.icmp eq %value, %c2_i9 : i9

  // Check KnownBitAnalysis for mux.
  %6 = comb.and %value, %c2_i9 : i9
  %7 = comb.mux %bit, %6, %c0_i9 : i9
  %8 = comb.icmp eq %7, %c2_i9 : i9
  // CHECK: %1 = comb.extract %value from 1 : (i9) -> i1
  // CHECK: %2 = comb.and %bit, %1 : i1

  hw.output %2, %8 : i1, i1
  // CHECK: hw.output %0, %2 : i1
}

// CHECK-LABEL: hw.module @xorICmpConstant2
// This is an integration test for the testcase in Issue #1560.
hw.module @xorICmpConstant2(%value: i9, %value2: i9) -> (a: i1, b: i9) {
  %c2_i9 = hw.constant 2 : i9
  %c0_i9 = hw.constant 0 : i9
  %1 = comb.xor %value, %value2, %c2_i9 : i9
  %2 = comb.icmp eq %1, %c0_i9 : i9
  hw.output %2, %1 : i1, i9
  // CHECK: %0 = comb.xor %value, %value2 : i9
  // CHECK: %1 = comb.xor %0, %c2_i9 : i9
  // CHECK: %2 = comb.icmp eq %0, %c2_i9 : i9
  // CHECK: hw.output %2, %1 : i1, i9
}

// CHECK-LABEL: func.func @xorICmpConstant3
// Regression check for a dominance issue in icmp(xor) refactoring.
func.func @xorICmpConstant3(%arg0: i9, %arg1: i9) -> i1 {
  %c2_i9 = hw.constant 2 : i9
  %c0_i9 = hw.constant 0 : i9
  %1 = comb.xor %arg0, %arg1, %c2_i9 : i9
  call @xorICmpConstant3Keep(%1) : (i9) -> ()
  %2 = comb.icmp eq %1, %c0_i9 : i9
  return %2 : i1
  // CHECK: %0 = comb.xor %arg0, %arg1 : i9
  // CHECK: %1 = comb.xor %0, %c2_i9 : i9
  // CHECK: call @xorICmpConstant3Keep(%1)
  // CHECK: %2 = comb.icmp eq %0, %c2_i9 : i9
  // CHECK: return %2 : i1
}

func.func private @xorICmpConstant3Keep(%arg0: i9)

// CHECK-LABEL: hw.module @test1560
// This is an integration test for the testcase in Issue #1560.
hw.module @test1560(%value: i38) -> (a: i1) {
  %c1073741824_i38 = hw.constant 1073741824 : i38
  %253 = comb.xor %value, %c1073741824_i38 : i38

  %false = hw.constant false
  %true = hw.constant true
  %254 = comb.concat %false, %253 : i1, i38

  %c-536870912_i39 = hw.constant -536870912 : i39
  %255 = comb.and %254, %c-536870912_i39 : i39

  %c0_i39 = hw.constant 0 : i39
  %256 = comb.icmp ne %255, %c0_i39 : i39
  %257 = comb.xor %256, %true : i1
  hw.output %257: i1

  // CHECK:   %0 = comb.extract %value from 29 : (i38) -> i9
  // CHECK:   %1 = comb.icmp eq %0, %c2_i9 : i9
  // CHECK:   hw.output %1 : i1
  // CHECK: }
}

// CHECK-LABEL: hw.module @extractShift
hw.module @extractShift(%arg0: i4) -> (o1 : i1, o2: i1) {
  %c1 = hw.constant 1: i4
  %0 = comb.shl %c1, %arg0 : i4

  // CHECK:  %0 = comb.icmp eq %arg0, %c0_i4 : i4
  %1 = comb.extract %0 from 0 : (i4) -> i1

  // CHECK: %1 = comb.icmp eq %arg0, %c2_i4 : i4
  %2 = comb.extract %0 from 2 : (i4) -> i1
  // CHECK: hw.output %0, %1
  hw.output %1, %2: i1, i1
}

// CHECK-LABEL: hw.module @moduloZeroDividend
hw.module @moduloZeroDividend(%arg0: i32) -> (o1: i32, o2: i32) {
  // CHECK: [[ZERO:%.*]] = hw.constant 0 : i32
  %zero = hw.constant 0 : i32
  %0 = comb.mods %zero, %arg0 : i32
  %1 = comb.modu %zero, %arg0 : i32

  // CHECK: hw.output [[ZERO]], [[ZERO]]
  hw.output %0, %1 : i32, i32
}

// CHECK-LABEL: hw.module @orWithNegation
hw.module @orWithNegation(%arg0: i32) -> (o1: i32) {
  // CHECK: [[ALLONES:%.*]] = hw.constant -1 : i32
  %allones = hw.constant -1 : i32
  %0 = comb.xor %arg0, %allones : i32
  %1 = comb.or %arg0, %0 : i32

  // CHECK: hw.output [[ALLONES]]
  hw.output %1 : i32
}

// CHECK-LABEL: hw.module @addSubParam
hw.module @addSubParam<p1: i4>(%a: i4) -> (o1: i4, o2: i4, o3: i4) {
  // CHECK-DAG: [[ADD:%.*]] = hw.param.value i4 = #hw.param.expr.add<#hw.param.expr.mul<#hw.param.decl.ref<"p1">, 2>, 4>
  %c1 = hw.constant 4 : i4
  %p = hw.param.value i4 = #hw.param.decl.ref<"p1">
  %b = comb.add %p, %c1, %p : i4

  // CHECK-DAG: [[SUB1:%.*]] = hw.param.value i4 = #hw.param.expr.add<#hw.param.decl.ref<"p1">, -4>
  %c = comb.sub %p, %c1 : i4

  // CHECK-DAG: [[SUB2:%.*]] = hw.param.value i4 = #hw.param.expr.add<#hw.param.expr.mul<#hw.param.decl.ref<"p1">, -1>, 4>
  %d = comb.sub %c1, %p : i4

  // CHECK: hw.output [[ADD]], [[SUB1]], [[SUB2]]
  hw.output %b, %c, %d : i4, i4, i4
}

// CHECK-LABEL: muxConstantsFold
hw.module @muxConstantsFold(%cond: i1) -> (o: i25) {
  // CHECK-NEXT: %0 = comb.replicate %cond : (i1) -> i25
  %c0_i25 = hw.constant 0 : i25
  %c-1_i25 = hw.constant -1 : i25
  %0 = comb.mux %cond, %c-1_i25, %c0_i25 : i25
  // CHECK-NEXT: hw.output %0
  hw.output %0 : i25
}

// CHECK-LABEL: hw.module @muxCommon
// This handles various cases of mux(cond, x, someop(x, y, z)).
hw.module @muxCommon(%cond: i1, %cond2: i1,
                     %arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32)
  -> (o1: i32, o2: i32, o3: i32, o4: i32, o5: i32, orResult: i32,
      o6: i32, o7: i32) {
  %allones = hw.constant -1 : i32
  %notArg0 = comb.xor %arg0, %allones : i32

  // CHECK: [[CONDEXT:%.*]] = comb.replicate %cond : (i1) -> i32
  // CHECK: [[O1:%.*]] = comb.xor [[CONDEXT]], %arg0 : i32
  %o1 = comb.mux %cond, %notArg0, %arg0 : i32

  // CHECK: [[CONDNOT:%.*]] = comb.xor %cond, %true : i1
  // CHECK: [[CONDEXT:%.*]] = comb.replicate [[CONDNOT]] : (i1) -> i32
  // CHECK: [[O2:%.*]] = comb.xor [[CONDEXT]], %arg0 : i32
  %o2 = comb.mux %cond, %arg0, %notArg0 : i32

  // CHECK: [[OR_PARTIAL:%.*]] = comb.or %arg1, %arg2 : i32
  // CHECK: [[CONDEXT:%.*]] = comb.replicate %cond : (i1) -> i32
  // CHECK: [[OR_PARTIAL2:%.*]] = comb.and [[CONDEXT]], [[OR_PARTIAL]] : i32
  // CHECK: [[O3:%.*]] = comb.or [[OR_PARTIAL2]], %arg0 : i32
  %or = comb.or %arg0, %arg1, %arg2 : i32
  %o3 = comb.mux %cond, %or, %arg0 : i32

  // CHECK: [[AND_PARTIAL:%.*]] = comb.and %arg1, %arg2 : i32
  // CHECK: [[CONDEXT:%.*]] = comb.replicate %cond : (i1) -> i32
  // CHECK: [[AND_PARTIAL2:%.*]] = comb.or [[CONDEXT]], [[AND_PARTIAL]] : i32
  // CHECK: [[O4:%.*]] = comb.and [[AND_PARTIAL2]], %arg0 : i32
  %and = comb.and %arg0, %arg1, %arg2 : i32
  %o4 = comb.mux %cond, %arg0, %and : i32

  // CHECK: [[CONDEXT:%.*]] = comb.replicate %cond : (i1) -> i32
  // CHECK: [[OR1:%.*]] = comb.or %arg1, %arg2, %arg3 : i32
  // CHECK: [[ORRESULT:%.*]] = comb.or [[OR1]], %arg0 : i32
  // CHECK: [[MASKED_OR:%.*]] = comb.and [[CONDEXT]], [[OR1]] : i32
  // CHECK: [[O5:%.*]] = comb.or [[MASKED_OR]], %arg0 : i32
  %orResult = comb.or %arg0, %arg1, %arg2, %arg3 : i32
  %o5 = comb.mux %cond, %orResult, %arg0 : i32

  // CHECK: [[CONDS:%.*]] = comb.or %cond2, %cond : i1
  // CHECK: [[O6:%.*]] = comb.mux [[CONDS]], %arg0, %arg1 : i32
  %0 = comb.mux %cond, %arg0, %arg1 : i32
  %o6 = comb.mux %cond2, %arg0, %0 : i32
  
  // CHECK: [[CONDS:%.*]] = comb.and %cond2, %cond : i1
  // CHECK: [[O7:%.*]] = comb.mux [[CONDS]], %arg1, %arg0 : i32
  %1 = comb.mux %cond, %arg1, %arg0 : i32
  %o7 = comb.mux %cond2, %1, %arg0 : i32

  // CHECK: hw.output [[O1]], [[O2]], [[O3]], [[O4]], [[O5]], [[ORRESULT]],
  // CHECK: [[O6]], [[O7]]
  hw.output %o1, %o2, %o3, %o4, %o5, %orResult, %o6, %o7
    : i32, i32, i32, i32, i32, i32, i32, i32
}

// CHECK-LABEL: @flatten_multi_use_and
hw.module @flatten_multi_use_and(%arg0: i8, %arg1: i8, %arg2: i8)
   -> (o1: i8, o2: i8) {
  %c1 = hw.constant 15 : i8
  %0 = comb.and %arg0, %c1 : i8
  // CHECK: %0 = comb.and %arg0, %c15_i8 : i8

  %c2 = hw.constant 7 : i8
  %1 = comb.and %arg1, %0, %arg2, %c2 : i8
  // CHECK: %1 = comb.and %arg1, %arg0, %arg2, %c7_i8 : i8

  // CHECK: hw.output %0, %1 
  hw.output %0, %1 : i8, i8
}

// CHECK-LABEL: hw.module @muxCommonOp(
// This handles various cases of mux(cond, someop(...), someop(...)).
hw.module @muxCommonOp(%cond: i1,
                       %arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32)
  -> (o1: i128, o2: i128, o3: i128) {
  %allones = hw.constant -1 : i32
  %0 = comb.concat %allones, %arg1, %arg2, %arg3 : i32, i32, i32, i32
  %1 = comb.concat %arg0, %arg2, %arg2, %arg3 : i32, i32, i32, i32
  %o1 = comb.mux %cond, %0, %1 : i128
  // CHECK: %0 = comb.concat %c-1_i32, %arg1 : i32, i32 
  // CHECK: %1 = comb.concat %arg0, %arg2 : i32, i32 
  // CHECK: %2 = comb.mux %cond, %0, %1 : i64 
  // CHECK: [[O1:%.*]] = comb.concat %2, %arg2, %arg3 : i64, i32, i32 

  %2 = comb.concat %allones, %arg1, %arg2, %arg3 : i32, i32, i32, i32
  %3 = comb.concat %allones, %arg1, %arg2, %arg3 : i32, i32, i32, i32
  %o2 = comb.mux %cond, %2, %3 : i128
  // CHECK: [[O2:%.*]] = comb.concat %c-1_i32, %arg1, %arg2, %arg3 : i32, i32, i32, i32 

  %4 = comb.concat %allones, %arg1, %arg2, %arg3 : i32, i32, i32, i32
  %5 = comb.concat %allones, %arg2, %arg2, %arg3 : i32, i32, i32, i32
  %o3 = comb.mux %cond, %4, %5 : i128
  // CHECK: [[M3:%.*]] = comb.mux %cond, %arg1, %arg2 : i32 
  // CHECK: [[O3:%.*]] = comb.concat %c-1_i32, [[M3]], %arg2, %arg3 : i32, i32, i32, i32 
 
  // CHECK: hw.output [[O1]], [[O2]], [[O3]]
  hw.output %o1, %o2, %o3 : i128, i128, i128
}

// CHECK-LABEL: hw.module @ReductionReplicate(
hw.module @ReductionReplicate(%r: i4) -> (a:i1, b:i1, c:i1, d:i1) {
  %allones = hw.constant -1 : i32
  %zero = hw.constant 0 : i32
  %0 = comb.replicate %r : (i4) -> i32
  // CHECK:      %c0_i4 = hw.constant 0 : i4
  // CHECK-NEXT: %c-1_i4 = hw.constant -1 : i4
  // CHECK-NEXT: [[A:%.+]] = comb.icmp eq %r, %c-1_i4
  // CHECK-NEXT: [[B:%.+]] = comb.icmp eq %r, %c0_i4
  // CHECK-NEXT: [[C:%.+]] = comb.icmp ne %r, %c-1_i4
  // CHECK-NEXT: [[D:%.+]] = comb.icmp ne %r, %c0_i4
  %a = comb.icmp eq %0, %allones : i32
  %b = comb.icmp eq %0, %zero : i32
  %c = comb.icmp ne %0, %allones : i32
  %d = comb.icmp ne %0, %zero : i32
  // CHECK-NEXT: hw.output [[A]], [[B]], [[C]], [[D]]
  hw.output %a, %b, %c, %d: i1, i1, i1, i1
}

// CHECK-LABEL: @propagateNamehint
hw.module @propagateNamehint(%x: i16) -> (o: i1) {
  %c0_i16 = hw.constant 0 : i16
  // swap %x and %c0_i16
  // CHECK: %0 = comb.icmp eq %x, %c0_i16 {sv.namehint = "hint"}
  %0 = comb.icmp eq %c0_i16, %x {sv.namehint = "hint"}: i16
  hw.output %0 : i1
}

// CHECK-LABEL: @extractToReductionOps
hw.module @extractToReductionOps(%a: i1, %b: i2) -> (c: i1, d: i1, e: i1) {
  // CHECK-NEXT: %c-1_i2 = hw.constant -1 : i2
  // CHECK-NEXT: %c0_i2 = hw.constant 0 : i2
  // CHECK-NEXT: %0 = comb.icmp ne %b, %c0_i2 : i2
  // CHECK-NEXT: %1 = comb.icmp eq %b, %c-1_i2 : i2
  // CHECK-NEXT: %2 = comb.parity %b : i2
  // CHECK-NEXT: hw.output %0, %1, %2 : i1, i1, i1
  %0 = comb.extract %b from 1 : (i2) -> i1
  %1 = comb.extract %b from 0 : (i2) -> i1
  %2 = comb.or %0, %1 : i1
  %3 = comb.and %0, %1 : i1
  %4 = comb.xor %0, %1 : i1

  hw.output %2, %3, %4 : i1, i1, i1
}

// https://github.com/llvm/circt/issues/2546
// CHECK-LABEL: @Issue2546
hw.module @Issue2546() -> (b: i1) {
  %true = hw.constant true
  %0 = comb.xor %0, %true : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @ArrayConcatFlatten
hw.module @ArrayConcatFlatten(%a: !hw.array<3xi1>) -> (b: i3) {
  // CHECK-NEXT:  %0 = hw.bitcast %a : (!hw.array<3xi1>) -> i3
  // CHECK-NEXT: hw.output %0 : i3
  %c-2_i2 = hw.constant -2 : i2
  %c1_i2 = hw.constant 1 : i2
  %c0_i2 = hw.constant 0 : i2
  %0 = hw.array_get %a[%c0_i2] : !hw.array<3xi1>, i2
  %1 = hw.array_get %a[%c1_i2] : !hw.array<3xi1>, i2
  %2 = hw.array_get %a[%c-2_i2] : !hw.array<3xi1>, i2
  %3 = comb.concat %1, %0 : i1, i1
  %4 = comb.concat %2, %3 : i1, i2
  hw.output %4 : i3
}

// CHECK-LABEL: hw.module @MuxSimplify
hw.module @MuxSimplify(%index: i1, %a: i1, %foo_0: i2, %foo_1: i2) -> (r_0: i2, r_1: i2, r_2 : i2, r_3: i2, r_4 : i2, r_5: i2, r_6 : i2) {
  %true = hw.constant true
  %c-2_i2 = hw.constant -2 : i2
  %c1_i2 = hw.constant 1 : i2
  %0 = comb.xor bin %index, %true : i1
  %1 = comb.mux bin %0, %c1_i2, %foo_0 : i2
  %2 = comb.mux bin %index, %c1_i2, %foo_1 : i2
  %3 = comb.mux bin %0, %c-2_i2, %foo_0 : i2
  %4 = comb.mux bin %a, %1, %3 : i2
  %5 = comb.mux bin %index, %c-2_i2, %foo_1 : i2
  %6 = comb.mux bin %a, %2, %5 : i2
  
  %7 = comb.mux bin %a, %foo_0, %foo_1 : i2
  %8 = comb.mux bin %index, %foo_0, %foo_1 : i2
  %9 = comb.xor %a, %index : i1
  %10 = comb.mux bin %9, %7, %8 : i2
  
  %11 = comb.mux bin %index, %foo_0, %foo_1 : i2
  %12 = comb.mux bin %a, %foo_0, %11 : i2

  %13 = comb.mux bin %index, %foo_1, %foo_0 : i2
  %14 = comb.mux bin %a, %foo_0, %11 : i2

  %15 = comb.mux bin %index, %foo_1, %foo_0 : i2
  %16 = comb.mux bin %a, %11, %foo_0 : i2

  %17 = comb.mux bin %index, %foo_0, %foo_1 : i2
  %18 = comb.mux bin %a, %11, %foo_0 : i2

  hw.output %4, %6, %10, %12, %14, %16, %18 : i2, i2, i2, i2, i2, i2, i2
}
// CHECK:  %0 = comb.mux %a, %c1_i2, %c-2_i2 : i2
// CHECK-NEXT:  %1 = comb.mux bin %index, %foo_0, %0 : i2
// CHECK-NEXT:  %2 = comb.mux %a, %c1_i2, %c-2_i2 : i2
// CHECK-NEXT:  %3 = comb.mux bin %index, %2, %foo_1 : i2
// CHECK-NEXT:  %4 = comb.xor %a, %index : i1 
// CHECK-NEXT:  %5 = comb.mux %4, %a, %index : i1 
// CHECK-NEXT:  %6 = comb.mux bin %5, %foo_0, %foo_1 : i2 
// CHECK-NEXT:  %7 = comb.or %a, %index : i1
// CHECK-NEXT:  %8 = comb.mux bin %7, %foo_0, %foo_1 : i2
// CHECK-NEXT:  %9 = comb.or %a, %index : i1
// CHECK-NEXT:  %10 = comb.mux bin %9, %foo_0, %foo_1 : i2
// CHECK-NEXT:  %11 = comb.xor %a, %true : i1
// CHECK-NEXT:  %12 = comb.or %11, %index : i1
// CHECK-NEXT:  %13 = comb.mux bin %12, %foo_0, %foo_1 : i2
// CHECK-NEXT:  %14 = comb.xor %a, %true : i1
// CHECK-NEXT:  %15 = comb.or %14, %index : i1
// CHECK-NEXT:  %16 = comb.mux bin %15, %foo_0, %foo_1 : i2
// CHECK-NEXT:  hw.output %1, %3, %6, %8, %10, %13, %16

// CHECK-LABEL: @twoStateICmp
hw.module @twoStateICmp(%arg: i4) -> (cond: i1) {
  // CHECK: %0 = comb.icmp bin eq %arg, %c-1_i4
  %c-1_i4 = hw.constant -1 : i4
  %0 = comb.icmp bin eq %c-1_i4, %arg : i4
  hw.output %0 : i1
}

// https://github.com/llvm/circt/issues/5531
// CHECK-LABEL: @Issue5531
hw.module @Issue5531(%arg0: i64, %arg1: i64) -> (out: i32) {
  // CHECK:  %2 = comb.mul %0, %1 {sv.namehint = "hint"} : i32
  %2 = comb.mul %arg0, %arg1 {sv.namehint = "hint"} : i64
  %3 = comb.extract %2 from 0 : (i64) -> i32
  hw.output %3 : i32
}

// or(mux(c_1, a, 0), mux(c_2, a, 0), ..., mux(c_n, a, 0)) -> mux(or(c_1, c_2, .., c_n), a, 0)
// CHECK-LABEL: OrMuxSameTrueValueAndZero
// CHECK:      %[[OR:.+]] = comb.or bin %tag_0, %tag_1, %tag_2, %tag_3 : i1
// CHECK-NEXT: %[[RESULT:.+]] = comb.mux bin %[[OR]], %in, %c0_i4 : i4
// CHECK-NEXT: hw.output %[[RESULT]]
hw.module @OrMuxSameTrueValueAndZero(%tag_0: i1, %tag_1: i1, %tag_2: i1, %tag_3: i1, %in: i4) -> (out: i4) {
  %c0_i4 = hw.constant 0 : i4
  %0 = comb.mux bin %tag_0, %in, %c0_i4 : i4
  %1 = comb.mux bin %tag_1, %in, %c0_i4 : i4
  %2 = comb.mux bin %tag_2, %in, %c0_i4 : i4
  %3 = comb.mux bin %tag_3, %in, %c0_i4 : i4
  %4 = comb.or bin %0, %1, %2, %3 : i4
  hw.output %4 : i4
}
