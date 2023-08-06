// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-sfc-compat)))' --verify-diagnostics --split-input-file %s | FileCheck %s

firrtl.circuit "SFCCompatTests" {

  firrtl.module @SFCCompatTests() {}

  // An invalidated regreset should be converted to a reg.
  //
  // CHECK-LABEL: @InvalidValue
  firrtl.module @InvalidValue(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>) {
    // CHECK-NOT: invalid
    %invalid_ui1_dead = firrtl.invalidvalue : !firrtl.uint<1>
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    // CHECK: firrtl.reg %clock
    %r = firrtl.regreset %clock, %reset, %invalid_ui1  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A regreset invalidated through a wire should be converted to a reg.
  //
  // CHECK-LABEL: @InvalidThroughWire
  firrtl.module @InvalidThroughWire(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>) {
    %inv = firrtl.wire  : !firrtl.uint<1>
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.connect %inv, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.reg %clock
    %r = firrtl.regreset %clock, %reset, %inv  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A regreset invalidated through wires with aggregate types should be
  // converted to a reg.
  //
  // CHECK-LABEL: firrtl.module @AggregateInvalidThroughWire
  firrtl.module @AggregateInvalidThroughWire(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.vector<bundle<a: uint<1>>, 2>, out %q: !firrtl.vector<bundle<a: uint<1>>, 2>) {
    %inv = firrtl.wire : !firrtl.bundle<a: uint<1>>
    %inv_a = firrtl.subfield %inv[a] : !firrtl.bundle<a: uint<1>>
    %invalid = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.strictconnect %inv_a, %invalid : !firrtl.uint<1>

    %inv1 = firrtl.wire : !firrtl.vector<bundle<a: uint<1>>, 2>
    %inv1_0 = firrtl.subindex %inv1[0] : !firrtl.vector<bundle<a: uint<1>>, 2>
    firrtl.strictconnect %inv1_0, %inv : !firrtl.bundle<a: uint<1>>
    %inv1_1 = firrtl.subindex %inv1[0] : !firrtl.vector<bundle<a: uint<1>>, 2>
    firrtl.strictconnect %inv1_1, %inv : !firrtl.bundle<a: uint<1>>

    // CHECK: firrtl.reg %clock : !firrtl.clock, !firrtl.vector<bundle<a: uint<1>>, 2>
    %r = firrtl.regreset %clock, %reset, %inv1  : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<bundle<a: uint<1>>, 2>, !firrtl.vector<bundle<a: uint<1>>, 2>
    firrtl.strictconnect %r, %d : !firrtl.vector<bundle<a: uint<1>>, 2>
    firrtl.strictconnect %q, %r : !firrtl.vector<bundle<a: uint<1>>, 2>
  }

  // A regreset invalidated via an output port should be converted to a reg.
  //
  // CHECK-LABEL: @InvalidPort
  firrtl.module @InvalidPort(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>, out %x: !firrtl.uint<1>) {
    %inv = firrtl.wire  : !firrtl.uint<1>
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.connect %inv, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %x, %inv : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.reg %clock
    %r = firrtl.regreset %clock, %reset, %x  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A regreset invalidate via an instance input port should be converted to a
  // reg.
  //
  // CHECK-LABEL: @InvalidInstancePort
  firrtl.module @InvalidInstancePort_Submodule(in %inv: !firrtl.uint<1>) {}
  firrtl.module @InvalidInstancePort(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>) {
    %inv = firrtl.wire  : !firrtl.uint<1>
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.connect %inv, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %submodule_inv = firrtl.instance submodule  @InvalidInstancePort_Submodule(in inv: !firrtl.uint<1>)
    firrtl.connect %submodule_inv, %inv : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.reg %clock
    %r = firrtl.regreset %clock, %reset, %submodule_inv  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A primitive operation should block invalid propagation.
  firrtl.module @InvalidPrimop(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>) {
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    %0 = firrtl.not %invalid_ui1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK: firrtl.regreset %clock
    %r = firrtl.regreset %clock, %reset, %0  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A regreset invalid value should NOT propagate through a node.
  firrtl.module @InvalidNode(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<8>, out %q: !firrtl.uint<8>) {
    %inv = firrtl.wire  : !firrtl.uint<8>
    %invalid_ui8 = firrtl.invalidvalue : !firrtl.uint<8>
    firrtl.connect %inv, %invalid_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
    %_T = firrtl.node %inv  : !firrtl.uint<8>
    // CHECK: firrtl.regreset %clock
    %r = firrtl.regreset %clock, %reset, %_T  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %r, %d : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %q, %r : !firrtl.uint<8>, !firrtl.uint<8>
  }

  firrtl.module @AggregateInvalid(out %q: !firrtl.bundle<a:uint<1>>) {
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.bundle<a:uint<1>>
    firrtl.connect %q, %invalid_ui1 : !firrtl.bundle<a:uint<1>>, !firrtl.bundle<a:uint<1>>
    // CHECK: %c0_ui1 = firrtl.constant 0
    // CHECK-NEXT: %[[CAST:.+]] = firrtl.bitcast %c0_ui1
    // CHECK-NEXT: %q, %[[CAST]]
  }

  // All of these should not error as the register is initialzed to a constant
  // reset value while looking through constructs that the SFC allows.  This is
  // testing the following cases:
  //
  //   1. A wire marked don't touch driven to a constant.
  //   2. A node driven to a constant.
  //   3. A wire driven to an invalid.
  //   4. A constant that passes through SFC-approved primops.
  //
  // CHECK-LABEL: firrtl.module @ConstantAsyncReset
  firrtl.module @ConstantAsyncReset(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %r0_init = firrtl.wire sym @r0_init : !firrtl.uint<1>
    firrtl.strictconnect %r0_init, %c0_ui1 : !firrtl.uint<1>
    %r0 = firrtl.regreset %clock, %reset, %r0_init : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>

    %r1_init = firrtl.node %c0_ui1 : !firrtl.uint<1>
    %r1 = firrtl.regreset %clock, %reset, %r1_init : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>

    %inv_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    %r2_init = firrtl.wire : !firrtl.uint<1>
    firrtl.strictconnect %r2_init, %inv_ui1 : !firrtl.uint<1>
    %r2 = firrtl.regreset %clock, %reset, %r2_init : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>

    %c0_si1 = firrtl.asSInt %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.sint<1>
    %c0_clock = firrtl.asClock %c0_si1 : (!firrtl.sint<1>) -> !firrtl.clock
    %c0_asyncreset = firrtl.asAsyncReset %c0_clock : (!firrtl.clock) -> !firrtl.asyncreset
    %r3_init = firrtl.asUInt %c0_asyncreset : (!firrtl.asyncreset) -> !firrtl.uint<1>
    %r3 = firrtl.regreset %clock, %reset, %r3_init : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @TailPrimOp
  firrtl.module @TailPrimOp(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.pad %c0_ui1, 3 : (!firrtl.uint<1>) -> !firrtl.uint<3>
    %1 = firrtl.tail %0, 2 : (!firrtl.uint<3>) -> !firrtl.uint<1>
    %r0_init = firrtl.wire sym @r0_init : !firrtl.uint<1>
    firrtl.strictconnect %r0_init, %1: !firrtl.uint<1>
    %r0 = firrtl.regreset %clock, %reset, %r0_init : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "NonConstantAsyncReset_Port" {
  // expected-note @below {{reset driver is "x"}}
  firrtl.module @NonConstantAsyncReset_Port(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %x: !firrtl.uint<1>) {
    // expected-error @below {{register "r0" has an async reset, but its reset value "x" is not driven with a constant value through wires, nodes, or connects}}
    %r0 = firrtl.regreset %clock, %reset, %x : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "NonConstantAsyncReset_PrimOp" {
  firrtl.module @NonConstantAsyncReset_PrimOp(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // expected-note @+1 {{reset driver is here}}
    %c1_ui1 = firrtl.not %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    // expected-error @below {{register "r0" has an async reset, but its reset value is not driven with a constant value through wires, nodes, or connects}}
    %r0 = firrtl.regreset %clock, %reset, %c1_ui1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "NonConstantAsyncReset_Aggregate0" {
  // expected-note @below {{reset driver is "x"}}
  firrtl.module @NonConstantAsyncReset_Aggregate0(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %x : !firrtl.vector<uint<1>, 2>) {
    %value = firrtl.wire : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %value, %x : !firrtl.vector<uint<1>, 2>
    // expected-error @below {{register "r0" has an async reset, but its reset value "value" is not driven with a constant value through wires, nodes, or connects}}
    %r0 = firrtl.regreset %clock, %reset, %value : !firrtl.clock, !firrtl.asyncreset, !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
}

// -----

firrtl.circuit "NonConstantAsyncReset_Aggregate1" {
  // expected-note @below {{reset driver is "x[0].y"}}
  firrtl.module @NonConstantAsyncReset_Aggregate1(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %x : !firrtl.vector<bundle<y: uint<1>>, 1>) {

    // Aggregate wire used for the reset value.
    %value = firrtl.wire : !firrtl.vector<uint<1>, 2>

    // Connect a constant 0 to value[0].
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %value_0 = firrtl.subindex %value[0] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %value_0, %c0_ui1 : !firrtl.uint<1>

    // Connect a complex chain of operations leading to the port to value[1].
    %subindex = firrtl.subindex %x[0] : !firrtl.vector<bundle<y : uint<1>>, 1>
    %node = firrtl.node %subindex : !firrtl.bundle<y : uint<1>>
    %subfield = firrtl.subfield %node[y] : !firrtl.bundle<y : uint<1>>
    %value_1 = firrtl.subindex %value[1] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %value_1, %subfield : !firrtl.uint<1>

    // expected-error @below {{register "r0" has an async reset, but its reset value "value[1]" is not driven with a constant value through wires, nodes, or connects}}
    %r0 = firrtl.regreset %clock, %reset, %value : !firrtl.clock, !firrtl.asyncreset, !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
}
