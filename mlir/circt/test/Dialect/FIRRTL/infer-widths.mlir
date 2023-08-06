// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-widths))' --verify-diagnostics %s | FileCheck %s

firrtl.circuit "Foo" {
  // CHECK-LABEL: @InferConstant
  // CHECK-SAME: out %out0: !firrtl.uint<42>
  // CHECK-SAME: out %out1: !firrtl.sint<42>
  firrtl.module @InferConstant(out %out0: !firrtl.uint, out %out1: !firrtl.sint) {
    %0 = firrtl.constant 1 : !firrtl.uint<42>
    %1 = firrtl.constant 2 : !firrtl.sint<42>
    // CHECK: {{.+}} = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: {{.+}} = firrtl.constant 0 : !firrtl.sint<1>
    // CHECK: {{.+}} = firrtl.constant 200 : !firrtl.uint<8>
    // CHECK: {{.+}} = firrtl.constant 200 : !firrtl.sint<9>
    // CHECK: {{.+}} = firrtl.constant -200 : !firrtl.sint<9>
    %2 = firrtl.constant 0 : !firrtl.uint
    %3 = firrtl.constant 0 : !firrtl.sint
    %4 = firrtl.constant 200 : !firrtl.uint
    %5 = firrtl.constant 200 : !firrtl.sint
    %6 = firrtl.constant -200 : !firrtl.sint
    firrtl.connect %out0, %0 : !firrtl.uint, !firrtl.uint<42>
    firrtl.connect %out1, %1 : !firrtl.sint, !firrtl.sint<42>
  }

  // CHECK-LABEL: @InferSpecialConstant
  firrtl.module @InferSpecialConstant() {
    // CHECK: %c0_clock = firrtl.specialconstant 0 : !firrtl.clock
    %c0_clock = firrtl.specialconstant 0 : !firrtl.clock
  }

  // CHECK-LABEL: @InferInvalidValue
  firrtl.module @InferInvalidValue(out %out: !firrtl.uint) {
    // CHECK: %invalid_ui6 = firrtl.invalidvalue : !firrtl.uint<6>
    %invalid_ui = firrtl.invalidvalue : !firrtl.uint
    %c42_ui = firrtl.constant 42 : !firrtl.uint
    firrtl.connect %out, %invalid_ui : !firrtl.uint, !firrtl.uint
    firrtl.connect %out, %c42_ui : !firrtl.uint, !firrtl.uint

    // Check that the invalid values are duplicated, and a corner case where the
    // wire won't be updated with a width until after updating the invalid value
    // above.
    // CHECK: %invalid_ui2 = firrtl.invalidvalue : !firrtl.uint<2>
    %w = firrtl.wire : !firrtl.uint
    %c2_ui = firrtl.constant 2 : !firrtl.uint
    firrtl.connect %w, %invalid_ui : !firrtl.uint, !firrtl.uint
    firrtl.connect %w, %c2_ui : !firrtl.uint, !firrtl.uint

    // Check that invalid values are inferred to width zero if not used in a
    // connect.
    // CHECK: firrtl.invalidvalue : !firrtl.uint<0>
    // CHECK: firrtl.invalidvalue : !firrtl.bundle<x: uint<0>>
    // CHECK: firrtl.invalidvalue : !firrtl.vector<uint<0>, 2>
    // CHECK: firrtl.invalidvalue : !firrtl.enum<a: uint<0>>
    %invalid_0 = firrtl.invalidvalue : !firrtl.uint
    %invalid_1 = firrtl.invalidvalue : !firrtl.bundle<x: uint>
    %invalid_2 = firrtl.invalidvalue : !firrtl.vector<uint, 2>
    %invalid_3 = firrtl.invalidvalue : !firrtl.enum<a: uint>
  }

  // CHECK-LABEL: @InferOutput
  // CHECK-SAME: out %out: !firrtl.uint<2>
  firrtl.module @InferOutput(in %in: !firrtl.uint<2>, out %out: !firrtl.uint) {
    firrtl.connect %out, %in : !firrtl.uint, !firrtl.uint<2>
  }

  // CHECK-LABEL: @InferOutput2
  // CHECK-SAME: out %out: !firrtl.uint<2>
  firrtl.module @InferOutput2(in %in: !firrtl.uint<2>, out %out: !firrtl.uint) {
    firrtl.connect %out, %in : !firrtl.uint, !firrtl.uint<2>
  }

  firrtl.module @InferNode() {
    %w = firrtl.wire : !firrtl.uint
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    firrtl.connect %w, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
    // CHECK: %node = firrtl.node %w : !firrtl.uint<3>
    %node = firrtl.node %w : !firrtl.uint
  }

  firrtl.module @InferNode2() {
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    %w = firrtl.wire : !firrtl.uint
    firrtl.connect %w, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>

    %node2 = firrtl.node %w : !firrtl.uint

    %w1 = firrtl.wire : !firrtl.uint
    firrtl.connect %w1, %node2 : !firrtl.uint, !firrtl.uint
  }

  // CHECK-LABEL: @AddSubOp
  firrtl.module @AddSubOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %2 = firrtl.add {{.*}} -> !firrtl.uint<4>
    // CHECK: %3 = firrtl.sub {{.*}} -> !firrtl.uint<5>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.add %0, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %3 = firrtl.sub %0, %2 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: @MulDivRemOp
  firrtl.module @MulDivRemOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %2 = firrtl.wire : !firrtl.sint<2>
    // CHECK: %3 = firrtl.wire : !firrtl.sint<3>
    // CHECK: %4 = firrtl.mul {{.*}} -> !firrtl.uint<5>
    // CHECK: %5 = firrtl.div {{.*}} -> !firrtl.uint<3>
    // CHECK: %6 = firrtl.div {{.*}} -> !firrtl.sint<4>
    // CHECK: %7 = firrtl.rem {{.*}} -> !firrtl.uint<2>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.wire : !firrtl.sint
    %3 = firrtl.wire : !firrtl.sint
    %4 = firrtl.mul %1, %0 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %5 = firrtl.div %1, %0 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %6 = firrtl.div %3, %2 : (!firrtl.sint, !firrtl.sint) -> !firrtl.sint
    %7 = firrtl.rem %1, %0 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    %c1_si2 = firrtl.constant 1 : !firrtl.sint<2>
    %c2_si3 = firrtl.constant 2 : !firrtl.sint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
    firrtl.connect %2, %c1_si2 : !firrtl.sint, !firrtl.sint<2>
    firrtl.connect %3, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @AndOrXorOp
  firrtl.module @AndOrXorOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %2 = firrtl.and {{.*}} -> !firrtl.uint<3>
    // CHECK: %3 = firrtl.or {{.*}} -> !firrtl.uint<3>
    // CHECK: %4 = firrtl.xor {{.*}} -> !firrtl.uint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.and %0, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %3 = firrtl.or %0, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %4 = firrtl.xor %0, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: @ComparisonOp
  firrtl.module @ComparisonOp(in %a: !firrtl.uint<2>, in %b: !firrtl.uint<3>) {
    // CHECK: %6 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %7 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %8 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %9 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %10 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %11 = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.leq %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %1 = firrtl.lt %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %2 = firrtl.geq %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %3 = firrtl.gt %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %4 = firrtl.eq %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %5 = firrtl.neq %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %6 = firrtl.wire : !firrtl.uint
    %7 = firrtl.wire : !firrtl.uint
    %8 = firrtl.wire : !firrtl.uint
    %9 = firrtl.wire : !firrtl.uint
    %10 = firrtl.wire : !firrtl.uint
    %11 = firrtl.wire : !firrtl.uint
    firrtl.connect %6, %0 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %7, %1 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %8, %2 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %9, %3 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %10, %4 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %11, %5 : !firrtl.uint, !firrtl.uint<1>
  }

  // CHECK-LABEL: @CatDynShiftOp
  firrtl.module @CatDynShiftOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %2 = firrtl.wire : !firrtl.sint<2>
    // CHECK: %3 = firrtl.wire : !firrtl.sint<3>
    // CHECK: %4 = firrtl.cat {{.*}} -> !firrtl.uint<5>
    // CHECK: %5 = firrtl.cat {{.*}} -> !firrtl.uint<5>
    // CHECK: %6 = firrtl.dshl {{.*}} -> !firrtl.uint<10>
    // CHECK: %7 = firrtl.dshl {{.*}} -> !firrtl.sint<10>
    // CHECK: %8 = firrtl.dshlw {{.*}} -> !firrtl.uint<3>
    // CHECK: %9 = firrtl.dshlw {{.*}} -> !firrtl.sint<3>
    // CHECK: %10 = firrtl.dshr {{.*}} -> !firrtl.uint<3>
    // CHECK: %11 = firrtl.dshr {{.*}} -> !firrtl.sint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.wire : !firrtl.sint
    %3 = firrtl.wire : !firrtl.sint
    %4 = firrtl.cat %0, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %5 = firrtl.cat %2, %3 : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint
    %6 = firrtl.dshl %1, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %7 = firrtl.dshl %3, %1 : (!firrtl.sint, !firrtl.uint) -> !firrtl.sint
    %8 = firrtl.dshlw %1, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %9 = firrtl.dshlw %3, %1 : (!firrtl.sint, !firrtl.uint) -> !firrtl.sint
    %10 = firrtl.dshr %1, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %11 = firrtl.dshr %3, %1 : (!firrtl.sint, !firrtl.uint) -> !firrtl.sint
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    %c1_si2 = firrtl.constant 1 : !firrtl.sint<2>
    %c2_si3 = firrtl.constant 2 : !firrtl.sint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
    firrtl.connect %2, %c1_si2 : !firrtl.sint, !firrtl.sint<2>
    firrtl.connect %3, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @CastOp
  firrtl.module @CastOp() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.sint<3>
    // CHECK: %4 = firrtl.asSInt {{.*}} -> !firrtl.sint<2>
    // CHECK: %5 = firrtl.asUInt {{.*}} -> !firrtl.uint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.sint
    %2 = firrtl.wire : !firrtl.clock
    %3 = firrtl.wire : !firrtl.asyncreset
    %4 = firrtl.asSInt %0 : (!firrtl.uint) -> !firrtl.sint
    %5 = firrtl.asUInt %1 : (!firrtl.sint) -> !firrtl.uint
    %6 = firrtl.asUInt %2 : (!firrtl.clock) -> !firrtl.uint<1>
    %7 = firrtl.asUInt %3 : (!firrtl.asyncreset) -> !firrtl.uint<1>
    %8 = firrtl.asClock %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.clock
    %9 = firrtl.asAsyncReset %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_si3 = firrtl.constant 2 : !firrtl.sint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @ConstCastOp
  firrtl.module @ConstCastOp() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.const.uint<1>
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.sint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.sint
    %c1 = firrtl.constant 1 : !firrtl.const.uint<2>
    %c2 = firrtl.constant 2 : !firrtl.const.sint<3>
    %3 = firrtl.constCast %c1 : (!firrtl.const.uint<2>) -> !firrtl.uint<2>
    %4 = firrtl.constCast %c2 : (!firrtl.const.sint<3>) -> !firrtl.sint<3>
    firrtl.connect %0, %3 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %4 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @CvtOp
  firrtl.module @CvtOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.sint<3>
    // CHECK: %2 = firrtl.cvt {{.*}} -> !firrtl.sint<3>
    // CHECK: %3 = firrtl.cvt {{.*}} -> !firrtl.sint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.sint
    %2 = firrtl.cvt %0 : (!firrtl.uint) -> !firrtl.sint
    %3 = firrtl.cvt %1 : (!firrtl.sint) -> !firrtl.sint
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_si3 = firrtl.constant 2 : !firrtl.sint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @NegOp
  firrtl.module @NegOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.sint<3>
    // CHECK: %2 = firrtl.neg {{.*}} -> !firrtl.sint<3>
    // CHECK: %3 = firrtl.neg {{.*}} -> !firrtl.sint<4>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.sint
    %2 = firrtl.neg %0 : (!firrtl.uint) -> !firrtl.sint
    %3 = firrtl.neg %1 : (!firrtl.sint) -> !firrtl.sint
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_si3 = firrtl.constant 2 : !firrtl.sint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @NotOp
  firrtl.module @NotOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.sint<3>
    // CHECK: %2 = firrtl.not {{.*}} -> !firrtl.uint<2>
    // CHECK: %3 = firrtl.not {{.*}} -> !firrtl.uint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.sint
    %2 = firrtl.not %0 : (!firrtl.uint) -> !firrtl.uint
    %3 = firrtl.not %1 : (!firrtl.sint) -> !firrtl.uint
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_si3 = firrtl.constant 2 : !firrtl.sint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @AndOrXorReductionOp
  firrtl.module @AndOrXorReductionOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %2 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %3 = firrtl.andr {{.*}} -> !firrtl.uint<1>
    // CHECK: %4 = firrtl.orr {{.*}} -> !firrtl.uint<1>
    // CHECK: %5 = firrtl.xorr {{.*}} -> !firrtl.uint<1>
    %c0_ui16 = firrtl.constant 0 : !firrtl.uint<16>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.wire : !firrtl.uint
    %3 = firrtl.andr %c0_ui16 : (!firrtl.uint<16>) -> !firrtl.uint<1>
    %4 = firrtl.orr %c0_ui16 : (!firrtl.uint<16>) -> !firrtl.uint<1>
    %5 = firrtl.xorr %c0_ui16 : (!firrtl.uint<16>) -> !firrtl.uint<1>
    firrtl.connect %0, %3 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %1, %4 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %2, %5 : !firrtl.uint, !firrtl.uint<1>
  }

  // CHECK-LABEL: @BitsHeadTailPadOp
  firrtl.module @BitsHeadTailPadOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %2 = firrtl.wire : !firrtl.uint<5>
    // CHECK: %3 = firrtl.wire : !firrtl.uint<5>
    // CHECK: %8 = firrtl.tail {{.*}} -> !firrtl.uint<12>
    // CHECK: %9 = firrtl.tail {{.*}} -> !firrtl.uint<12>
    // CHECK: %10 = firrtl.pad {{.*}} -> !firrtl.uint<42>
    // CHECK: %11 = firrtl.pad {{.*}} -> !firrtl.sint<42>
    // CHECK: %12 = firrtl.pad {{.*}} -> !firrtl.uint<99>
    // CHECK: %13 = firrtl.pad {{.*}} -> !firrtl.sint<99>
    %ui = firrtl.wire : !firrtl.uint
    %si = firrtl.wire : !firrtl.sint
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.wire : !firrtl.uint
    %3 = firrtl.wire : !firrtl.uint

    %4 = firrtl.bits %ui 3 to 1 : (!firrtl.uint) -> !firrtl.uint<3>
    %5 = firrtl.bits %si 3 to 1 : (!firrtl.sint) -> !firrtl.uint<3>
    %6 = firrtl.head %ui, 5 : (!firrtl.uint) -> !firrtl.uint<5>
    %7 = firrtl.head %si, 5 : (!firrtl.sint) -> !firrtl.uint<5>
    %8 = firrtl.tail %ui, 30 : (!firrtl.uint) -> !firrtl.uint
    %9 = firrtl.tail %si, 30 : (!firrtl.sint) -> !firrtl.uint
    %10 = firrtl.pad %ui, 13 : (!firrtl.uint) -> !firrtl.uint
    %11 = firrtl.pad %si, 13 : (!firrtl.sint) -> !firrtl.sint
    %12 = firrtl.pad %ui, 99 : (!firrtl.uint) -> !firrtl.uint
    %13 = firrtl.pad %si, 99 : (!firrtl.sint) -> !firrtl.sint

    firrtl.connect %0, %4 : !firrtl.uint, !firrtl.uint<3>
    firrtl.connect %1, %5 : !firrtl.uint, !firrtl.uint<3>
    firrtl.connect %2, %6 : !firrtl.uint, !firrtl.uint<5>
    firrtl.connect %3, %7 : !firrtl.uint, !firrtl.uint<5>

    %c0_ui42 = firrtl.constant 0 : !firrtl.uint<42>
    %c0_si42 = firrtl.constant 0 : !firrtl.sint<42>
    firrtl.connect %ui, %c0_ui42 : !firrtl.uint, !firrtl.uint<42>
    firrtl.connect %si, %c0_si42 : !firrtl.sint, !firrtl.sint<42>
  }

  // CHECK-LABEL: @MuxOp
  firrtl.module @MuxOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %2 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %3 = firrtl.mux{{.*}} -> !firrtl.uint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.wire : !firrtl.uint
    %3 = firrtl.mux(%2, %0, %1) : (!firrtl.uint, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
    // CHECK: %4 = firrtl.wire : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %4 = firrtl.wire : !firrtl.uint
    %5 = firrtl.mux(%4, %c1_ui1, %c1_ui1) : (!firrtl.uint, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }

  // see https://github.com/llvm/circt/issues/3070
  // CHECK-LABEL: @MuxBundle
  firrtl.module @MuxBundleOperands(in %a: !firrtl.bundle<a: uint<8>>, in %p: !firrtl.uint<1>, out %c: !firrtl.bundle<a: uint>) {
    // CHECK: %w = firrtl.wire  : !firrtl.bundle<a: uint<8>>
    %w = firrtl.wire  : !firrtl.bundle<a: uint>
    %0 = firrtl.subfield %w[a] : !firrtl.bundle<a: uint>
    %1 = firrtl.subfield %a[a] : !firrtl.bundle<a: uint<8>>
    firrtl.connect %0, %1 : !firrtl.uint, !firrtl.uint<8>
    // CHECK: %2 = firrtl.mux(%p, %a, %w) : (!firrtl.uint<1>, !firrtl.bundle<a: uint<8>>, !firrtl.bundle<a: uint<8>>) -> !firrtl.bundle<a: uint<8>>
    %2 = firrtl.mux(%p, %a, %w) : (!firrtl.uint<1>, !firrtl.bundle<a: uint<8>>, !firrtl.bundle<a: uint>) -> !firrtl.bundle<a: uint>
    firrtl.connect %c, %2 : !firrtl.bundle<a: uint>, !firrtl.bundle<a: uint>
  }

  // CHECK-LABEL: @ShlShrOp
  firrtl.module @ShlShrOp() {
    // CHECK: %0 = firrtl.shl {{.*}} -> !firrtl.uint<8>
    // CHECK: %1 = firrtl.shl {{.*}} -> !firrtl.sint<8>
    // CHECK: %2 = firrtl.shr {{.*}} -> !firrtl.uint<2>
    // CHECK: %3 = firrtl.shr {{.*}} -> !firrtl.sint<2>
    // CHECK: %4 = firrtl.shr {{.*}} -> !firrtl.uint<1>
    // CHECK: %5 = firrtl.shr {{.*}} -> !firrtl.sint<1>
    %ui = firrtl.wire : !firrtl.uint
    %si = firrtl.wire : !firrtl.sint

    %0 = firrtl.shl %ui, 3 : (!firrtl.uint) -> !firrtl.uint
    %1 = firrtl.shl %si, 3 : (!firrtl.sint) -> !firrtl.sint
    %2 = firrtl.shr %ui, 3 : (!firrtl.uint) -> !firrtl.uint
    %3 = firrtl.shr %si, 3 : (!firrtl.sint) -> !firrtl.sint
    %4 = firrtl.shr %ui, 9 : (!firrtl.uint) -> !firrtl.uint
    %5 = firrtl.shr %si, 9 : (!firrtl.sint) -> !firrtl.sint

    %c0_ui5 = firrtl.constant 0 : !firrtl.uint<5>
    %c0_si5 = firrtl.constant 0 : !firrtl.sint<5>
    firrtl.connect %ui, %c0_ui5 : !firrtl.uint, !firrtl.uint<5>
    firrtl.connect %si, %c0_si5 : !firrtl.sint, !firrtl.sint<5>
  }

  // CHECK-LABEL: @PassiveCastOp
  firrtl.module @PassiveCastOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<5>
    // CHECK: %1 = builtin.unrealized_conversion_cast %ui : !firrtl.uint<5> to !firrtl.uint<5>
    %ui = firrtl.wire : !firrtl.uint
    %0 = firrtl.wire : !firrtl.uint
    %1 = builtin.unrealized_conversion_cast %ui : !firrtl.uint to !firrtl.uint
    firrtl.connect %0, %1 : !firrtl.uint, !firrtl.uint
    %c0_ui5 = firrtl.constant 0 : !firrtl.uint<5>
    firrtl.connect %ui, %c0_ui5 : !firrtl.uint, !firrtl.uint<5>
  }

  // CHECK-LABEL: @TransparentOps
  firrtl.module @TransparentOps(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>) {
    %false = firrtl.constant 0 : !firrtl.uint<1>
    %true = firrtl.constant 1 : !firrtl.uint<1>
    %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
    %c0_ui5 = firrtl.constant 0 : !firrtl.uint<5>

    // CHECK: %ui = firrtl.wire : !firrtl.uint<5>
    %ui = firrtl.wire : !firrtl.uint

    firrtl.printf %clk, %false, "foo" : !firrtl.clock, !firrtl.uint<1>
    firrtl.skip
    firrtl.stop %clk, %false, 0 : !firrtl.clock, !firrtl.uint<1>
    firrtl.when %a : !firrtl.uint<1> {
      firrtl.connect %ui, %c0_ui4 : !firrtl.uint, !firrtl.uint<4>
    } else  {
      firrtl.connect %ui, %c0_ui5 : !firrtl.uint, !firrtl.uint<5>
    }
    firrtl.assert %clk, %true, %true, "foo" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.assume %clk, %true, %true, "foo" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.cover %clk, %true, %true, "foo" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
  }

  // Issue #1088
  // CHECK-LABEL: @Issue1088
  firrtl.module @Issue1088(out %y: !firrtl.sint<4>) {
    // CHECK: %x = firrtl.wire : !firrtl.sint<9>
    // CHECK: %c200_si9 = firrtl.constant 200 : !firrtl.sint<9>
    // CHECK: %0 = firrtl.tail %x, 5 : (!firrtl.sint<9>) -> !firrtl.uint<4>
    // CHECK: %1 = firrtl.asSInt %0 : (!firrtl.uint<4>) -> !firrtl.sint<4>
    // CHECK: firrtl.connect %y, %1 : !firrtl.sint<4>, !firrtl.sint<4>
    // CHECK: firrtl.connect %x, %c200_si9 : !firrtl.sint<9>, !firrtl.sint<9>
    %x = firrtl.wire : !firrtl.sint
    %c200_si = firrtl.constant 200 : !firrtl.sint
    firrtl.connect %y, %x : !firrtl.sint<4>, !firrtl.sint
    firrtl.connect %x, %c200_si : !firrtl.sint, !firrtl.sint
  }

  // Should truncate all the way to 0 bits if its has to.
  // CHECK-LABEL: @TruncateConnect
  firrtl.module @TruncateConnect() {
    %w = firrtl.wire  : !firrtl.uint
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.connect %w, %c1_ui1 : !firrtl.uint, !firrtl.uint<1>
    %w1 = firrtl.wire  : !firrtl.uint<0>
    // CHECK: %0 = firrtl.tail %w, 1 : (!firrtl.uint<1>) -> !firrtl.uint<0>
    // CHECK: firrtl.connect %w1, %0 : !firrtl.uint<0>, !firrtl.uint<0>
    firrtl.connect %w1, %w : !firrtl.uint<0>, !firrtl.uint
  }

  // Issue #1110: Width inference should infer 0 width when appropriate
  // CHECK-LABEL: @Issue1110
  // CHECK-SAME: out %y: !firrtl.uint<0>
  firrtl.module @Issue1110(in %x: !firrtl.uint<0>, out %y: !firrtl.uint) {
    firrtl.connect %y, %x : !firrtl.uint, !firrtl.uint<0>
  }

  // Issue #1118: Width inference should infer 0 width when appropriate
  // CHECK-LABEL: @Issue1118
  // CHECK-SAME: out %x: !firrtl.sint<13>
  firrtl.module @Issue1118(out %x: !firrtl.sint) {
    %c4232_ui = firrtl.constant 4232 : !firrtl.uint
    %0 = firrtl.asSInt %c4232_ui : (!firrtl.uint) -> !firrtl.sint
    firrtl.connect %x, %0 : !firrtl.sint, !firrtl.sint
  }

  // CHECK-LABEL: @RegSimple
  firrtl.module @RegSimple(in %clk: !firrtl.clock, in %x: !firrtl.uint<6>) {
    // CHECK: %0 = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<6>
    // CHECK: %1 = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<6>
    %0 = firrtl.reg %clk : !firrtl.clock, !firrtl.uint
    %1 = firrtl.reg %clk : !firrtl.clock, !firrtl.uint
    %2 = firrtl.wire : !firrtl.uint
    %3 = firrtl.xor %1, %2 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %0, %x : !firrtl.uint, !firrtl.uint<6>
    firrtl.connect %1, %3 : !firrtl.uint, !firrtl.uint
    firrtl.connect %2, %x : !firrtl.uint, !firrtl.uint<6>
  }

  // CHECK-LABEL: @RegShr
  firrtl.module @RegShr(in %clk: !firrtl.clock, in %x: !firrtl.uint<6>) {
    // CHECK: %0 = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<6>
    // CHECK: %1 = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<6>
    %0 = firrtl.reg %clk : !firrtl.clock, !firrtl.uint
    %1 = firrtl.reg %clk : !firrtl.clock, !firrtl.uint
    %2 = firrtl.shr %0, 0 : (!firrtl.uint) -> !firrtl.uint
    %3 = firrtl.shr %1, 3 : (!firrtl.uint) -> !firrtl.uint
    firrtl.connect %0, %x : !firrtl.uint, !firrtl.uint<6>
    firrtl.connect %1, %x : !firrtl.uint, !firrtl.uint<6>
    firrtl.connect %0, %2 : !firrtl.uint, !firrtl.uint
    firrtl.connect %1, %3 : !firrtl.uint, !firrtl.uint
  }

  // CHECK-LABEL: @RegShl
  firrtl.module @RegShl(in %clk: !firrtl.clock, in %x: !firrtl.uint<6>) {
    // CHECK: %0 = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<6>
    %0 = firrtl.reg %clk : !firrtl.clock, !firrtl.uint
    %1 = firrtl.reg %clk : !firrtl.clock, !firrtl.uint
    %2 = firrtl.reg %clk : !firrtl.clock, !firrtl.uint
    %3 = firrtl.shl %0, 0 : (!firrtl.uint) -> !firrtl.uint
    %4 = firrtl.shl %1, 3 : (!firrtl.uint) -> !firrtl.uint
    %5 = firrtl.shr %4, 3 : (!firrtl.uint) -> !firrtl.uint
    %6 = firrtl.shr %1, 3 : (!firrtl.uint) -> !firrtl.uint
    %7 = firrtl.shl %6, 3 : (!firrtl.uint) -> !firrtl.uint
    firrtl.connect %0, %x : !firrtl.uint, !firrtl.uint<6>
    firrtl.connect %1, %x : !firrtl.uint, !firrtl.uint<6>
    firrtl.connect %2, %x : !firrtl.uint, !firrtl.uint<6>
    firrtl.connect %0, %2 : !firrtl.uint, !firrtl.uint
    firrtl.connect %1, %5 : !firrtl.uint, !firrtl.uint
    firrtl.connect %2, %7 : !firrtl.uint, !firrtl.uint
  }

  // CHECK-LABEL: @RegResetSimple
  firrtl.module @RegResetSimple(
    in %clk: !firrtl.clock,
    in %rst: !firrtl.asyncreset,
    in %x: !firrtl.uint<6>
  ) {
    // CHECK: %0 = firrtl.regreset %clk, %rst, %c0_ui1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<6>
    // CHECK: %1 = firrtl.regreset %clk, %rst, %c0_ui1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<6>
    // CHECK: %2:2 = firrtl.regreset %clk, %rst, %c0_ui17 forceable : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<17>, !firrtl.uint<17>, !firrtl.rwprobe<uint<17>>
    // CHECK: %3 = firrtl.regreset %clk, %rst, %c0_ui17 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<17>, !firrtl.uint<17>
    %c0_ui = firrtl.constant 0 : !firrtl.uint
    %c0_ui17 = firrtl.constant 0 : !firrtl.uint<17>
    %0 = firrtl.regreset %clk, %rst, %c0_ui : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint, !firrtl.uint
    %1 = firrtl.regreset %clk, %rst, %c0_ui : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint, !firrtl.uint
    %2:2 = firrtl.regreset %clk, %rst, %c0_ui17 forceable : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<17>, !firrtl.uint, !firrtl.rwprobe<uint>
    %3 = firrtl.regreset %clk, %rst, %c0_ui17 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<17>, !firrtl.uint
    %4 = firrtl.wire : !firrtl.uint
    %5 = firrtl.xor %1, %4 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %0, %x : !firrtl.uint, !firrtl.uint<6>
    firrtl.connect %1, %5 : !firrtl.uint, !firrtl.uint
    firrtl.connect %2, %x : !firrtl.uint, !firrtl.uint<6>
    firrtl.connect %3, %5 : !firrtl.uint, !firrtl.uint
    firrtl.connect %4, %x : !firrtl.uint, !firrtl.uint<6>
  }

  // Inter-module width inference for one-to-one module-instance correspondence.
  // CHECK-LABEL: @InterModuleSimpleFoo
  // CHECK-SAME: in %in: !firrtl.uint<42>
  // CHECK-SAME: out %out: !firrtl.uint<43>
  // CHECK-LABEL: @InterModuleSimpleBar
  // CHECK-SAME: in %in: !firrtl.uint<42>
  // CHECK-SAME: out %out: !firrtl.uint<44>
  firrtl.module @InterModuleSimpleFoo(in %in: !firrtl.uint, out %out: !firrtl.uint) {
    %0 = firrtl.add %in, %in : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %out, %0 : !firrtl.uint, !firrtl.uint
  }
  firrtl.module @InterModuleSimpleBar(in %in: !firrtl.uint<42>, out %out: !firrtl.uint) {
    %inst_in, %inst_out = firrtl.instance inst @InterModuleSimpleFoo(in in: !firrtl.uint, out out: !firrtl.uint)
    %0 = firrtl.add %inst_out, %inst_out : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %inst_in, %in : !firrtl.uint, !firrtl.uint<42>
    firrtl.connect %out, %0 : !firrtl.uint, !firrtl.uint
  }

  // Inter-module width inference for multiple instances per module.
  // CHECK-LABEL: @InterModuleMultipleFoo
  // CHECK-SAME: in %in: !firrtl.uint<42>
  // CHECK-SAME: out %out: !firrtl.uint<43>
  // CHECK-LABEL: @InterModuleMultipleBar
  // CHECK-SAME: in %in1: !firrtl.uint<17>
  // CHECK-SAME: in %in2: !firrtl.uint<42>
  // CHECK-SAME: out %out: !firrtl.uint<43>
  firrtl.module @InterModuleMultipleFoo(in %in: !firrtl.uint, out %out: !firrtl.uint) {
    %0 = firrtl.add %in, %in : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %out, %0 : !firrtl.uint, !firrtl.uint
  }
  firrtl.module @InterModuleMultipleBar(in %in1: !firrtl.uint<17>, in %in2: !firrtl.uint<42>, out %out: !firrtl.uint) {
    %inst1_in, %inst1_out = firrtl.instance inst1 @InterModuleMultipleFoo(in in: !firrtl.uint, out out: !firrtl.uint)
    %inst2_in, %inst2_out = firrtl.instance inst2 @InterModuleMultipleFoo(in in: !firrtl.uint, out out: !firrtl.uint)
    %0 = firrtl.xor %inst1_out, %inst2_out : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %inst1_in, %in1 : !firrtl.uint, !firrtl.uint<17>
    firrtl.connect %inst2_in, %in2 : !firrtl.uint, !firrtl.uint<42>
    firrtl.connect %out, %0 : !firrtl.uint, !firrtl.uint
  }

  // CHECK-LABEL: @InferBundle
  firrtl.module @InferBundle(in %in : !firrtl.uint<3>, in %clk : !firrtl.clock) {
    // CHECK: firrtl.wire : !firrtl.bundle<a: uint<3>>
    // CHECK: firrtl.reg %clk : !firrtl.clock, !firrtl.bundle<a: uint<3>>
    %w = firrtl.wire : !firrtl.bundle<a: uint>
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.bundle<a: uint>
    %w_a = firrtl.subfield %w[a] : !firrtl.bundle<a: uint>
    %r_a = firrtl.subfield %r[a] : !firrtl.bundle<a: uint>
    firrtl.connect %w_a, %in : !firrtl.uint, !firrtl.uint<3>
    firrtl.connect %r_a, %in : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: @InferEmptyBundle
  firrtl.module @InferEmptyBundle(in %in : !firrtl.uint<3>) {
    // CHECK: %w = firrtl.wire : !firrtl.bundle<a: bundle<>, b: uint<3>>
    %w = firrtl.wire : !firrtl.bundle<a: bundle<>, b: uint>
    %w_a = firrtl.subfield %w[a] : !firrtl.bundle<a: bundle<>, b: uint>
    %w_b = firrtl.subfield %w[b] : !firrtl.bundle<a: bundle<>, b: uint>
    firrtl.connect %w_b, %in : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: @InferBundlePort
  firrtl.module @InferBundlePort(in %in: !firrtl.bundle<a: uint<2>, b: uint<3>>, out %out: !firrtl.bundle<a: uint, b: uint>) {
    // CHECK: firrtl.connect %out, %in : !firrtl.bundle<a: uint<2>, b: uint<3>>, !firrtl.bundle<a: uint<2>, b: uint<3>>
    firrtl.connect %out, %in : !firrtl.bundle<a: uint, b: uint>, !firrtl.bundle<a: uint<2>, b: uint<3>>
  }

  // CHECK-LABEL: @InferVectorSubindex
  firrtl.module @InferVectorSubindex(in %in : !firrtl.uint<4>, in %clk : !firrtl.clock) {
    // CHECK: firrtl.wire : !firrtl.vector<uint<4>, 10>
    // CHECK: firrtl.reg %clk : !firrtl.clock, !firrtl.vector<uint<4>, 10>
    %w = firrtl.wire : !firrtl.vector<uint, 10>
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.vector<uint, 10>
    %w_5 = firrtl.subindex %w[5] : !firrtl.vector<uint, 10>
    %r_5 = firrtl.subindex %r[5] : !firrtl.vector<uint, 10>
    firrtl.connect %w_5, %in : !firrtl.uint, !firrtl.uint<4>
    firrtl.connect %r_5, %in : !firrtl.uint, !firrtl.uint<4>
  }

  // CHECK-LABEL: @InferVectorSubaccess
  firrtl.module @InferVectorSubaccess(in %in : !firrtl.uint<4>, in %addr : !firrtl.uint<32>, in %clk : !firrtl.clock) {
    // CHECK: firrtl.wire : !firrtl.vector<uint<4>, 10>
    // CHECK: firrtl.reg %clk : !firrtl.clock, !firrtl.vector<uint<4>, 10>
    %w = firrtl.wire : !firrtl.vector<uint, 10>
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.vector<uint, 10>
    %w_addr = firrtl.subaccess %w[%addr] : !firrtl.vector<uint, 10>, !firrtl.uint<32>
    %r_addr = firrtl.subaccess %r[%addr] : !firrtl.vector<uint, 10>, !firrtl.uint<32>
    firrtl.connect %w_addr, %in : !firrtl.uint, !firrtl.uint<4>
    firrtl.connect %r_addr, %in : !firrtl.uint, !firrtl.uint<4>
  }

  // CHECK-LABEL: @InferVectorPort
  firrtl.module @InferVectorPort(in %in: !firrtl.vector<uint<4>, 2>, out %out: !firrtl.vector<uint, 2>) {
    // CHECK: firrtl.connect %out, %in : !firrtl.vector<uint<4>, 2>, !firrtl.vector<uint<4>, 2>
    firrtl.connect %out, %in : !firrtl.vector<uint, 2>, !firrtl.vector<uint<4>, 2>
  }

  // CHECK-LABEL: @InferVectorFancy
  firrtl.module @InferVectorFancy(in %in : !firrtl.uint<4>) {
    // CHECK: firrtl.wire : !firrtl.vector<uint<4>, 10>
    %wv = firrtl.wire : !firrtl.vector<uint, 10>
    %wv_5 = firrtl.subindex %wv[5] : !firrtl.vector<uint, 10>
    firrtl.connect %wv_5, %in : !firrtl.uint, !firrtl.uint<4>

    // CHECK: firrtl.wire : !firrtl.bundle<a: uint<4>>
    %wb = firrtl.wire : !firrtl.bundle<a: uint>
    %wb_a = firrtl.subfield %wb[a] : !firrtl.bundle<a: uint>

    %wv_2 = firrtl.subindex %wv[2] : !firrtl.vector<uint, 10>
    firrtl.connect %wb_a, %wv_2 : !firrtl.uint, !firrtl.uint
  }

  // CHECK-LABEL: InferElementAfterVector
  firrtl.module @InferElementAfterVector() {
    // CHECK: %w = firrtl.wire : !firrtl.bundle<a: vector<uint<10>, 10>, b: uint<3>>
    %w = firrtl.wire : !firrtl.bundle<a: vector<uint<10>, 10>, b :uint>
    %w_a = firrtl.subfield %w[b] : !firrtl.bundle<a: vector<uint<10>, 10>, b: uint>
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    firrtl.connect %w_a, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: @InferEnum
  firrtl.module @InferEnum(in %in : !firrtl.enum<a: uint<3>>) {
    // CHECK: %w = firrtl.wire : !firrtl.enum<a: uint<3>>
    %w = firrtl.wire : !firrtl.enum<a: uint>
    firrtl.connect %w, %in : !firrtl.enum<a: uint>, !firrtl.enum<a: uint<3>>
    // CHECK: %0 = firrtl.subtag %w[a] : !firrtl.enum<a: uint<3>>
    %0 = firrtl.subtag %w[a] : !firrtl.enum<a: uint>
  }

  // CHECK-LABEL: InferComplexBundles
  firrtl.module @InferComplexBundles() {
    // CHECK: %w = firrtl.wire : !firrtl.bundle<a: bundle<v: vector<uint<3>, 10>>, b: bundle<v: vector<uint<3>, 10>>>
    %w = firrtl.wire : !firrtl.bundle<a: bundle<v: vector<uint, 10>>, b: bundle <v: vector<uint, 10>>>
    %w_a = firrtl.subfield %w[a] : !firrtl.bundle<a: bundle<v: vector<uint, 10>>, b: bundle <v: vector<uint, 10>>>
    %w_a_v = firrtl.subfield %w_a[v] : !firrtl.bundle<v : vector<uint, 10>>
    %w_b = firrtl.subfield %w[b] : !firrtl.bundle<a: bundle<v: vector<uint, 10>>, b: bundle <v: vector<uint, 10>>>
    %w_b_v = firrtl.subfield %w_b[v] : !firrtl.bundle<v : vector<uint, 10>>
    firrtl.connect %w_a_v, %w_b_v : !firrtl.vector<uint, 10>, !firrtl.vector<uint, 10>
    %w_b_v_2 = firrtl.subindex %w_b_v[2] : !firrtl.vector<uint, 10>
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    firrtl.connect %w_b_v_2, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: InferComplexVectors
  firrtl.module @InferComplexVectors() {
    // CHECK: %w = firrtl.wire : !firrtl.vector<bundle<a: uint<3>, b: uint<3>>, 10>
    %w = firrtl.wire : !firrtl.vector<bundle<a: uint, b:uint>, 10>
    %w_2 = firrtl.subindex %w[2] : !firrtl.vector<bundle<a: uint, b:uint>, 10>
    %w_2_a = firrtl.subfield %w_2[a] : !firrtl.bundle<a: uint, b: uint>
    %w_4 = firrtl.subindex %w[4] : !firrtl.vector<bundle<a: uint, b:uint>, 10>
    %w_4_b = firrtl.subfield %w_4[b] : !firrtl.bundle<a: uint, b: uint>
    firrtl.connect %w_4_b, %w_2_a : !firrtl.uint, !firrtl.uint
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    firrtl.connect %w_2_a, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: @AttachOne
  // CHECK-SAME: in %a0: !firrtl.analog<8>
  firrtl.module @AttachOne(in %a0: !firrtl.analog<8>) {
    firrtl.attach %a0 : !firrtl.analog<8>
  }

  // CHECK-LABEL: @AttachTwo
  // CHECK-SAME: in %a0: !firrtl.analog<8>
  // CHECK-SAME: in %a1: !firrtl.analog<8>
  firrtl.module @AttachTwo(in %a0: !firrtl.analog<8>, in %a1: !firrtl.analog) {
    firrtl.attach %a0, %a1 : !firrtl.analog<8>, !firrtl.analog
  }

  // CHECK-LABEL: @AttachMany
  // CHECK-SAME: in %a0: !firrtl.analog<8>
  // CHECK-SAME: in %a1: !firrtl.analog<8>
  // CHECK-SAME: in %a2: !firrtl.analog<8>
  // CHECK-SAME: in %a3: !firrtl.analog<8>
  firrtl.module @AttachMany(
    in %a0: !firrtl.analog<8>,
    in %a1: !firrtl.analog,
    in %a2: !firrtl.analog<8>,
    in %a3: !firrtl.analog) {
    firrtl.attach %a0, %a1, %a2, %a3 : !firrtl.analog<8>, !firrtl.analog, !firrtl.analog<8>, !firrtl.analog
  }

  // CHECK-LABEL: @MemScalar
  // CHECK-SAME: out %out: !firrtl.uint<7>
  // CHECK-SAME: out %dbg: !firrtl.probe<vector<uint<7>, 8>>
  firrtl.module @MemScalar(out %out: !firrtl.uint, out %dbg: !firrtl.probe<vector<uint, 8>>) {
    // CHECK: firrtl.mem
    // CHECK-SAME: !firrtl.probe<vector<uint<7>, 8>>
    // CHECK-SAME: data flip: uint<7>
    // CHECK-SAME: data: uint<7>
    // CHECK-SAME: data: uint<7>
    %m_dbg, %m_p0, %m_p1, %m_p2 = firrtl.mem Undefined {
      depth = 8 : i64,
      name = "m",
      portNames = ["dbg", "p0", "p1", "p2"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32} :
      !firrtl.probe<vector<uint, 8>>,
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint>,
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint, mask: uint<1>>,
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, rdata flip: uint, wmode: uint<1>, wdata: uint, wmask: uint<1>>
    %m_p0_data = firrtl.subfield %m_p0[data] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint>
    %m_p1_data = firrtl.subfield %m_p1[data] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint, mask: uint<1>>
    %m_p2_wdata = firrtl.subfield %m_p2[wdata] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, rdata flip: uint, wmode: uint<1>, wdata: uint, wmask: uint<1>>
    %c0_ui5 = firrtl.constant 0 : !firrtl.uint<5>
    %c0_ui7 = firrtl.constant 0 : !firrtl.uint<7>
    firrtl.connect %m_p1_data, %c0_ui5 : !firrtl.uint, !firrtl.uint<5>
    firrtl.connect %m_p2_wdata, %c0_ui7 : !firrtl.uint, !firrtl.uint<7>
    firrtl.connect %out, %m_p0_data : !firrtl.uint, !firrtl.uint
    firrtl.ref.define %dbg, %m_dbg : !firrtl.probe<vector<uint, 8>>
    // CHECK:  firrtl.ref.define %dbg, %m_dbg : !firrtl.probe<vector<uint<7>, 8>>
  }

  // CHECK-LABEL: @MemBundle
  // CHECK-SAME: out %out: !firrtl.bundle<a: uint<7>>
  firrtl.module @MemBundle(out %out: !firrtl.bundle<a: uint>) {
    // CHECK: firrtl.mem
    // CHECK-SAME: data flip: bundle<a: uint<7>>
    // CHECK-SAME: data: bundle<a: uint<7>>
    // CHECK-SAME: data: bundle<a: uint<7>>
    %m_p0, %m_p1, %m_p2 = firrtl.mem Undefined {
      depth = 8 : i64,
      name = "m",
      portNames = ["p0", "p1", "p2"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32} :
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: bundle<a: uint>>,
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<a: uint>, mask: bundle<a: uint<1>>>,
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint>, wmode: uint<1>, wdata: bundle<a: uint>, wmask: bundle<a: uint<1>>>
    %m_p0_data = firrtl.subfield %m_p0[data] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: bundle<a: uint>>
    %m_p1_data = firrtl.subfield %m_p1[data] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<a: uint>, mask: bundle<a: uint<1>>>
    %m_p2_wdata = firrtl.subfield %m_p2[wdata] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint>, wmode: uint<1>, wdata: bundle<a: uint>, wmask: bundle<a: uint<1>>>
    %m_p1_data_a = firrtl.subfield %m_p1_data[a] : !firrtl.bundle<a: uint>
    %m_p2_wdata_a = firrtl.subfield %m_p2_wdata[a] : !firrtl.bundle<a: uint>
    %c0_ui5 = firrtl.constant 0 : !firrtl.uint<5>
    %c0_ui7 = firrtl.constant 0 : !firrtl.uint<7>
    firrtl.connect %m_p1_data_a, %c0_ui5 : !firrtl.uint, !firrtl.uint<5>
    firrtl.connect %m_p2_wdata_a, %c0_ui7 : !firrtl.uint, !firrtl.uint<7>
    firrtl.connect %out, %m_p0_data : !firrtl.bundle<a: uint>, !firrtl.bundle<a: uint>
  }

  // Breakable cycles in inter-module width inference.
  // CHECK-LABEL: @InterModuleGoodCycleFoo
  // CHECK-SAME: in %in: !firrtl.uint<42>
  // CHECK-SAME: out %out: !firrtl.uint<39>
  firrtl.module @InterModuleGoodCycleFoo(in %in: !firrtl.uint, out %out: !firrtl.uint) {
    %0 = firrtl.shr %in, 3 : (!firrtl.uint) -> !firrtl.uint
    firrtl.connect %out, %0 : !firrtl.uint, !firrtl.uint
  }
  // CHECK-LABEL: @InterModuleGoodCycleBar
  // CHECK-SAME: out %out: !firrtl.uint<39>
  firrtl.module @InterModuleGoodCycleBar(in %in: !firrtl.uint<42>, out %out: !firrtl.uint) {
    %inst_in, %inst_out = firrtl.instance inst  @InterModuleGoodCycleFoo(in in: !firrtl.uint, out out: !firrtl.uint)
    firrtl.connect %inst_in, %in : !firrtl.uint, !firrtl.uint<42>
    firrtl.connect %inst_in, %inst_out : !firrtl.uint, !firrtl.uint
    firrtl.connect %out, %inst_out : !firrtl.uint, !firrtl.uint
  }

  // CHECK-LABEL: @Issue1271
  firrtl.module @Issue1271(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>) {
    // CHECK: %a = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<2>
    // CHECK: %b = firrtl.node %0  : !firrtl.uint<3>
    // CHECK: %c = firrtl.node %1  : !firrtl.uint<2>
    %a = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.add %a, %c0_ui1 : (!firrtl.uint, !firrtl.uint<1>) -> !firrtl.uint
    %b = firrtl.node %0  : !firrtl.uint
    %1 = firrtl.tail %b, 1 : (!firrtl.uint) -> !firrtl.uint
    %c = firrtl.node %1  : !firrtl.uint
    %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
    %2 = firrtl.mux(%cond, %c0_ui2, %c) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %a, %2 : !firrtl.uint, !firrtl.uint
  }

  firrtl.module @Foo() {}

  // CHECK-LABEL: @SubRef
  // CHECK-SAME: out %x: !firrtl.probe<uint<2>>
  // CHECK-SAME: out %y: !firrtl.rwprobe<uint<2>>
  // CHECK-SAME: out %bov_ref: !firrtl.rwprobe<bundle<a: vector<uint<2>, 2>, b: uint<2>>>
  firrtl.module private @SubRef(out %x: !firrtl.probe<uint>, out %y : !firrtl.rwprobe<uint>, out %bov_ref : !firrtl.rwprobe<bundle<a: vector<uint, 2>, b : uint>>) {
    // CHECK: firrtl.wire forceable : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
    %w, %w_rw = firrtl.wire forceable : !firrtl.uint, !firrtl.rwprobe<uint>
    %bov, %bov_rw = firrtl.wire forceable : !firrtl.bundle<a: vector<uint, 2>, b flip: uint>, !firrtl.rwprobe<bundle<a: vector<uint, 2>, b : uint>>
    firrtl.ref.define %bov_ref, %bov_rw : !firrtl.rwprobe<bundle<a: vector<uint, 2>, b : uint>>

    %ref_w = firrtl.ref.send %w : !firrtl.uint
    %cast_ref_w = firrtl.ref.cast %ref_w : (!firrtl.probe<uint>) -> !firrtl.probe<uint>
    firrtl.ref.define %x, %cast_ref_w : !firrtl.probe<uint>
    firrtl.ref.define %y, %w_rw : !firrtl.rwprobe<uint>
    // CHECK: firrtl.ref.cast %w_ref : (!firrtl.rwprobe<uint<2>>) -> !firrtl.probe<uint<2>>
    %cast_w_ro = firrtl.ref.cast %w_rw : (!firrtl.rwprobe<uint>) -> !firrtl.probe<uint>

    %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
    firrtl.connect %w, %c0_ui2 : !firrtl.uint, !firrtl.uint<2>
    
    %bov_a = firrtl.subfield %bov[a] : !firrtl.bundle<a: vector<uint, 2>, b flip: uint>
    %bov_a_1 = firrtl.subindex %bov_a[1] : !firrtl.vector<uint, 2>
    %bov_b = firrtl.subfield %bov[b] : !firrtl.bundle<a: vector<uint, 2>, b flip: uint>

    firrtl.connect %w, %c0_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %bov_a_1, %c0_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %bov_b, %c0_ui2 : !firrtl.uint, !firrtl.uint<2>
  }
  // CHECK-LABEL: @Ref
  // CHECK: out x: !firrtl.probe<uint<2>>
  // CHECK-SAME: out y: !firrtl.rwprobe<uint<2>>
  // CHECK: firrtl.ref.resolve %sub_x : !firrtl.probe<uint<2>>
  // CHECK: firrtl.ref.resolve %sub_y : !firrtl.rwprobe<uint<2>>
  firrtl.module @Ref(out %r : !firrtl.uint, out %s : !firrtl.uint) {
    %sub_x, %sub_y, %sub_bov_ref = firrtl.instance sub @SubRef(out x: !firrtl.probe<uint>, out y: !firrtl.rwprobe<uint>, out bov_ref : !firrtl.rwprobe<bundle<a: vector<uint, 2>, b : uint>>)
    %res_x = firrtl.ref.resolve %sub_x : !firrtl.probe<uint>
    %res_y = firrtl.ref.resolve %sub_y : !firrtl.rwprobe<uint>
    firrtl.connect %r, %res_x : !firrtl.uint, !firrtl.uint
    firrtl.connect %s, %res_y : !firrtl.uint, !firrtl.uint

    // CHECK: !firrtl.rwprobe<bundle<a: vector<uint<2>, 2>, b: uint<2>>>
    %read_bov = firrtl.ref.resolve %sub_bov_ref : !firrtl.rwprobe<bundle<a: vector<uint, 2>, b : uint>>
    // CHECK: !firrtl.rwprobe<bundle<a: vector<uint<2>, 2>, b: uint<2>>>
    %bov_ref_a = firrtl.ref.sub %sub_bov_ref[0] : !firrtl.rwprobe<bundle<a: vector<uint, 2>, b : uint>>
    // CHECK: !firrtl.rwprobe<vector<uint<2>, 2>>
    %bov_ref_a_1 = firrtl.ref.sub %bov_ref_a[1] : !firrtl.rwprobe<vector<uint, 2>>
    // CHECK: !firrtl.rwprobe<bundle<a: vector<uint<2>, 2>, b: uint<2>>>
    %bov_ref_b  = firrtl.ref.sub %sub_bov_ref[1] : !firrtl.rwprobe<bundle<a: vector<uint, 2>, b : uint>>

    // CHECK: !firrtl.rwprobe<vector<uint<2>, 2>>
    %bov_a = firrtl.ref.resolve %bov_ref_a : !firrtl.rwprobe<vector<uint,2>>
    // CHECK: !firrtl.rwprobe<uint<2>>
    %bov_a_1 = firrtl.ref.resolve %bov_ref_a_1 : !firrtl.rwprobe<uint>
    // CHECK: !firrtl.rwprobe<uint<2>>
    %bov_b = firrtl.ref.resolve %bov_ref_b : !firrtl.rwprobe<uint>
  }

  // CHECK-LABEL: @ForeignTypes
  firrtl.module @ForeignTypes(in %a: !firrtl.uint<42>, out %b: !firrtl.uint) {
    %0 = firrtl.wire : index
    %1 = firrtl.wire : index
    firrtl.strictconnect %0, %1 : index
    firrtl.connect %b, %a : !firrtl.uint, !firrtl.uint<42>
    // CHECK-NEXT: [[W0:%.+]] = firrtl.wire : index
    // CHECK-NEXT: [[W1:%.+]] = firrtl.wire : index
    // CHECK-NEXT: firrtl.strictconnect [[W0]], [[W1]] : index
  }

  // CHECK-LABEL: @Issue4859
  firrtl.module @Issue4859() {
    %invalid = firrtl.invalidvalue : !firrtl.bundle<a: vector<uint, 2>>
    %0 = firrtl.subfield %invalid[a] : !firrtl.bundle<a: vector<uint, 2>>
    %1 = firrtl.subindex %0[0] : !firrtl.vector<uint, 2>
  }
  
  // CHECK-LABEL: @InferConst
  // CHECK-SAME: out %out: !firrtl.const.bundle<a: uint<1>, b: sint<2>, c: analog<3>, d: vector<uint<4>, 2>>
  firrtl.module @InferConst(in %a: !firrtl.const.uint<1>, in %b: !firrtl.const.sint<2>, in %c: !firrtl.const.analog<3>, in %d: !firrtl.const.vector<uint<4>, 2>,
    out %out: !firrtl.const.bundle<a: uint, b: sint, c: analog, d: vector<uint, 2>>) {
    %0 = firrtl.subfield %out[a] : !firrtl.const.bundle<a: uint, b: sint, c: analog, d: vector<uint, 2>>
    %1 = firrtl.subfield %out[b] : !firrtl.const.bundle<a: uint, b: sint, c: analog, d: vector<uint, 2>>
    %2 = firrtl.subfield %out[c] : !firrtl.const.bundle<a: uint, b: sint, c: analog, d: vector<uint, 2>>
    %3 = firrtl.subfield %out[d] : !firrtl.const.bundle<a: uint, b: sint, c: analog, d: vector<uint, 2>>

    firrtl.connect %0, %a : !firrtl.const.uint, !firrtl.const.uint<1>
    firrtl.connect %1, %b : !firrtl.const.sint, !firrtl.const.sint<2>
    firrtl.attach %2, %c : !firrtl.const.analog, !firrtl.const.analog<3>
    firrtl.connect %3, %d : !firrtl.const.vector<uint, 2>, !firrtl.const.vector<uint<4>, 2>
  }
  
  // Should not crash when encountering property types.
  // CHECK: firrtl.module @Property(in %a: !firrtl.string)
  firrtl.module @Property(in %a: !firrtl.string) { }

  // CHECK-LABEL: module @MuxIntrinsics
  // CHECK-SAME: %sel: !firrtl.uint<1>
  // CHECK-SAME: %sel2: !firrtl.uint<2>
  firrtl.module @MuxIntrinsics(in %sel: !firrtl.uint, in %sel2: !firrtl.uint, in %high: !firrtl.uint<1>, in %low: !firrtl.uint<1>, out %out1: !firrtl.uint, out %out2: !firrtl.uint) {
    %c3_ui4 = firrtl.constant 3 : !firrtl.uint<4>
    %c3_ui3 = firrtl.constant 3 : !firrtl.uint<3>
    %c2_ui2 = firrtl.constant 2 : !firrtl.uint<2>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1 = firrtl.constant 0: !firrtl.uint
    // CHECK: firrtl.int.mux2cell
    // CHECK-SAME: (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %0 = firrtl.int.mux2cell(%sel, %c0_ui1, %c1) : (!firrtl.uint, !firrtl.uint<1>, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %out1, %0: !firrtl.uint, !firrtl.uint
    // CHECK: firrtl.int.mux4cell
    // CHECK-SAME: (!firrtl.uint<2>, !firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<3>, !firrtl.uint<1>) -> !firrtl.uint<3>
    %1 = firrtl.int.mux4cell(%sel2, %c1_ui1, %c2_ui2, %c3_ui3, %c1) : (!firrtl.uint, !firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %out2, %1: !firrtl.uint, !firrtl.uint
  }
}
