// RUN: circt-opt %s --split-input-file --verify-diagnostics

// expected-error @+1 {{body contains non-pure operation}}
arc.define @Foo(%arg0: i1) {
  // expected-note @+1 {{first non-pure operation here:}}
  arc.state @Bar() clock %arg0 lat 1 : () -> ()
  arc.output
}
arc.define @Bar() {
  arc.output
}

// -----

hw.module @Foo() {
  // expected-error @+1 {{'arc.state' op with non-zero latency outside a clock domain requires a clock}}
  arc.state @Bar() lat 1 : () -> ()
}
arc.define @Bar() {
  arc.output
}

// -----

hw.module @Foo(%clock: i1) {
  // expected-error @+1 {{'arc.state' op with zero latency cannot have a clock}}
  arc.state @Bar() clock %clock lat 0 : () -> ()
}
arc.define @Bar() {
  arc.output
}

// -----

hw.module @Foo(%enable: i1) {
  // expected-error @+1 {{'arc.state' op with zero latency cannot have an enable}}
  arc.state @Bar() enable %enable lat 0 : () -> ()
}
arc.define @Bar() {
  arc.output
}

// -----

hw.module @Foo(%reset: i1) {
  // expected-error @+1 {{'arc.state' op with zero latency cannot have a reset}}
  arc.state @Bar() reset %reset lat 0 : () -> ()
}
arc.define @Bar() {
  arc.output
}

// -----

// expected-error @+1 {{body contains non-pure operation}}
arc.define @SupportRecursiveMemoryEffects(%arg0: i1, %arg1: i1) {
  // expected-note @+1 {{first non-pure operation here:}}
  scf.if %arg0 {
    arc.state @Bar() clock %arg1 lat 1 : () -> ()
  }
  arc.output
}
arc.define @Bar() {
  arc.output
}

// -----

// expected-error @below {{op must have exactly one argument}}
arc.model "MissingArg" {
^bb0:
}

// -----

// expected-error @below {{op must have exactly one argument}}
arc.model "TooManyArgs" {
^bb0(%arg0: !arc.storage, %arg1: !arc.storage):
}

// -----

// expected-error @below {{op argument must be of storage type}}
arc.model "WrongArgType" {
^bb0(%arg0: i32):
}

// -----

arc.define @Foo() {
  // expected-error @+1 {{`Bar` does not reference a valid `arc.define`}}
  arc.call @Bar() : () -> ()
  arc.output
}
func.func @Bar() {
  return
}

// -----

arc.define @Foo() {
  // expected-error @+1 {{incorrect number of operands: expected 1, but got 0}}
  arc.call @Bar() : () -> ()
  arc.output
}
arc.define @Bar(%arg0: i1) {
  arc.output
}

// -----

arc.define @Foo() {
  // expected-error @+1 {{incorrect number of results: expected 1, but got 0}}
  arc.call @Bar() : () -> ()
  arc.output
}
arc.define @Bar() -> i1 {
  %false = hw.constant false
  arc.output %false : i1
}

// -----

arc.define @Foo(%arg0: i1, %arg1: i32) {
  // expected-error @+3 {{operand type mismatch: operand #1}}
  // expected-note @+2 {{expected type: 'i42'}}
  // expected-note @+1 {{actual type: 'i32'}}
  arc.call @Bar(%arg0, %arg1) : (i1, i32) -> ()
  arc.output
}
arc.define @Bar(%arg0: i1, %arg1: i42) {
  arc.output
}

// -----

arc.define @Foo(%arg0: i1, %arg1: i32) {
  // expected-error @+3 {{result type mismatch: result #1}}
  // expected-note @+2 {{expected type: 'i42'}}
  // expected-note @+1 {{actual type: 'i32'}}
  %0, %1 = arc.call @Bar() : () -> (i1, i32)
  arc.output
}
arc.define @Bar() -> (i1, i42) {
  %false = hw.constant false
  %c0_i42 = hw.constant 0 : i42
  arc.output %false, %c0_i42 : i1, i42
}

// -----

arc.define @lut () -> () {
  // expected-error @+1 {{requires one result}}
  arc.lut () : () -> () {
    arc.output
  }
  arc.output
}

// -----

arc.define @lut () -> () {
  %0 = arc.lut () : () -> i32 {
    // expected-error @+1 {{incorrect number of outputs: expected 1, but got 0}}
    arc.output
  }
  arc.output
}

// -----

arc.define @lut () -> () {
  %0 = arc.lut () : () -> i32 {
    %1 = hw.constant 0 : i16
    // expected-error @+3 {{output type mismatch: output #0}}
    // expected-note @+2 {{expected type: 'i32'}}
    // expected-note @+1 {{actual type: 'i16'}}
    arc.output %1 : i16
  }
  arc.output
}

// -----

arc.define @lut (%arg0: i32, %arg1: i8) -> () {
  // expected-note @+1 {{required by region isolation constraints}}
  %1 = arc.lut (%arg1, %arg0) : (i8, i32) -> i32 {
    ^bb0(%arg2: i8, %arg3: i32):
      // expected-error @+1 {{using value defined outside the region}}
      arc.output %arg0 : i32
  }
  arc.output
}

// -----

arc.define @lutSideEffects () -> i32 {
  // expected-error @+1 {{no operations with side-effects allowed inside a LUT}}
  %0 = arc.lut () : () -> i32 {
    %true = hw.constant true
    // expected-note @+1 {{first operation with side-effects here}}
    %1 = arc.memory !arc.memory<20 x i32, i1>
    %2 = arc.memory_read_port %1[%true] : !arc.memory<20 x i32, i1>
    arc.output %2 : i32
  }
  arc.output %0 : i32
}

// -----

hw.module @clockDomainNumOutputs(%clk: i1) {
  %0 = arc.clock_domain () clock %clk : () -> (i32) {
  ^bb0:
    // expected-error @+1 {{incorrect number of outputs: expected 1, but got 0}}
    arc.output
  }
  hw.output
}

// -----

hw.module @clockDomainNumInputs(%clk: i1) {
  // expected-error @+1 {{incorrect number of inputs: expected 1, but got 0}}
  arc.clock_domain () clock %clk : () -> () {
  ^bb0(%arg0: i32):
    arc.output
  }
  hw.output
}

// -----

hw.module @clockDomainInputTypes(%clk: i1, %arg0: i16) {
  // expected-error @+3 {{input type mismatch: input #0}}
  // expected-note @+2 {{expected type: 'i32'}}
  // expected-note @+1 {{actual type: 'i16'}}
  arc.clock_domain (%arg0) clock %clk : (i16) -> () {
  ^bb0(%arg1: i32):
    arc.output
  }
  hw.output
}

// -----

hw.module @clockDomainOutputTypes(%clk: i1) {
  %0 = arc.clock_domain () clock %clk : () -> (i32) {
  ^bb0:
    %c0_i16 = hw.constant 0 : i16
    // expected-error @+3 {{output type mismatch: output #0}}
    // expected-note @+2 {{expected type: 'i32'}}
    // expected-note @+1 {{actual type: 'i16'}}
    arc.output %c0_i16 : i16
  }
  hw.output
}

// -----

hw.module @clockDomainIsolatedFromAbove(%clk: i1, %arg0: i32) {
  // expected-note @+1 {{required by region isolation constraints}}
  %0 = arc.clock_domain () clock %clk : () -> (i32) {
    // expected-error @+1 {{using value defined outside the region}}
    arc.output %arg0 : i32
  }
  hw.output
}

// -----

hw.module @stateOpInsideClockDomain(%clk: i1) {
  arc.clock_domain (%clk) clock %clk : (i1) -> () {
  ^bb0(%arg0: i1):
    // expected-error @+1 {{inside a clock domain cannot have a clock}}
    arc.state @dummyArc() clock %arg0 lat 1 : () -> ()
    arc.output
  }
  hw.output
}
arc.define @dummyArc() {
  arc.output
}

// -----

hw.module @memoryWritePortOpInsideClockDomain(%clk: i1) {
  arc.clock_domain (%clk) clock %clk : (i1) -> () {
  ^bb0(%arg0: i1):
    %mem = arc.memory <4 x i32, i32>
    %c0_i32 = hw.constant 0 : i32
    // expected-error @+1 {{inside a clock domain cannot have a clock}}
    arc.memory_write_port %mem, @identity(%c0_i32, %c0_i32, %arg0) clock %arg0 enable lat 1: !arc.memory<4 x i32, i32>, i32, i32, i1
    arc.output
  }
}
arc.define @identity(%addr: i32, %data: i32, %enable: i1) -> (i32, i32, i1) {
  arc.output %addr, %data, %enable : i32, i32, i1
}

// -----

hw.module @memoryWritePortOpOutsideClockDomain(%en: i1) {
  %mem = arc.memory <4 x i32, i32>
  %c0_i32 = hw.constant 0 : i32
  // expected-error @+1 {{outside a clock domain requires a clock}}
  arc.memory_write_port %mem, @identity(%c0_i32, %c0_i32, %en) lat 1 : !arc.memory<4 x i32, i32>, i32, i32, i1
}
arc.define @identity(%addr: i32, %data: i32, %enable: i1) -> (i32, i32, i1) {
  arc.output %addr, %data, %enable : i32, i32, i1
}

// -----

hw.module @memoryWritePortOpLatZero(%en: i1) {
  %mem = arc.memory <4 x i32, i32>
  %c0_i32 = hw.constant 0 : i32
  // expected-error @+1 {{latency must be at least 1}}
  arc.memory_write_port %mem, @identity(%c0_i32, %c0_i32, %en) lat 0 : !arc.memory<4 x i32, i32>, i32, i32, i1
}
arc.define @identity(%addr: i32, %data: i32, %enable: i1) -> (i32, i32, i1) {
  arc.output %addr, %data, %enable : i32, i32, i1
}

// -----

arc.define @outputOpVerifier () -> i32 {
  // expected-error @+1 {{incorrect number of outputs: expected 1, but got 0}}
  arc.output
}

// -----

arc.define @outputOpVerifier () -> i32 {
  %0 = hw.constant 0 : i16
  // expected-error @+3 {{output type mismatch: output #0}}
  // expected-note @+2 {{expected type: 'i32'}}
  // expected-note @+1 {{actual type: 'i16'}}
  arc.output %0 : i16
}
