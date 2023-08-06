// RUN: circt-opt %s -split-input-file -verify-diagnostics

firrtl.circuit "X" {

firrtl.module @X(in %b : !firrtl.unknowntype) {
  // expected-error @-1 {{unknown FIRRTL dialect type: "unknowntype"}}
}

}

// -----

firrtl.circuit "X" {

firrtl.module @X(in %b : !firrtl.uint<32>, in %d : !firrtl.uint<16>, in %out : !firrtl.uint) {
  // expected-error @+1 {{'firrtl.add' op expected 2 operands, but found 3}}
  %3 = "firrtl.add"(%b, %d, %out) : (!firrtl.uint<32>, !firrtl.uint<16>, !firrtl.uint) -> !firrtl.uint<32>
}

}

// -----

// expected-error @+1 {{'firrtl.circuit' op must contain one module that matches main name 'MyCircuit'}}
firrtl.circuit "MyCircuit" {

firrtl.module @X() {}

}

// -----


// expected-error @+1 {{'firrtl.module' op expects parent op 'firrtl.circuit'}}
firrtl.module @X() {}

// -----

// expected-error @+1 {{'firrtl.circuit' op must contain one module that matches main name 'Foo'}}
firrtl.circuit "Foo" {

firrtl.module @Bar() {}

}

// -----

// expected-error @+1 {{'firrtl.circuit' op must have a non-empty name}}
firrtl.circuit "" {
}

// -----

firrtl.circuit "foo" {
// expected-error @+1 {{requires 1 port directions}}
firrtl.module @foo(in %a : !firrtl.uint<1>) attributes {portDirections = 3 : i2} {}
}

// -----

firrtl.circuit "foo" {
// expected-error @+1 {{requires 1 port names}}
firrtl.module @foo(in %a : !firrtl.uint<1>) attributes {portNames=[]} {}
}

// -----

firrtl.circuit "foo" {
// expected-error @+1 {{port names should all be string attributes}}
firrtl.module @foo(in %a : !firrtl.uint<1>) attributes {portNames=[1 : i1]} {}
}

// -----

firrtl.circuit "foo" {
// expected-error @+1 {{op requires 1 port annotations}}
firrtl.module @foo(in %a : !firrtl.uint<1>) attributes {portAnnotations=[[], []]} {}
}

// -----

firrtl.circuit "foo" {
// expected-error @+1 {{annotations must be dictionaries or subannotations}}
firrtl.module @foo(in %a: !firrtl.uint<1> ["hello"]) {}
}

// -----

firrtl.circuit "foo" {
// expected-error @+1 {{requires one region}}
"firrtl.module"() ( { }, { })
   {sym_name = "foo", convention = #firrtl<convention internal>,
    portTypes = [!firrtl.uint], portDirections = 1 : i1,
    portNames = ["in0"], portAnnotations = [], portSyms = []} : () -> ()
}

// -----

firrtl.circuit "foo" {
// expected-error @+1 {{requires valid port locations}}
"firrtl.module"() ( {
  ^entry:
}) { sym_name = "foo", convention = #firrtl<convention internal>,
    portTypes = [!firrtl.uint], portDirections = 1 : i1,
    portNames = ["in0"], portAnnotations = [], portSyms = []} : () -> ()
}

// -----

firrtl.circuit "foo" {
// expected-error @+1 {{requires 1 port locations}}
"firrtl.module"() ( {
  ^entry:
}) {sym_name = "foo", convention = #firrtl<convention internal>,
    portTypes = [!firrtl.uint], portDirections = 1 : i1,
    portNames = ["in0"], portAnnotations = [], portSyms = [],
    portLocations = []} : () -> ()
}




// -----

firrtl.circuit "foo" {
// expected-error @+1 {{entry block must have 1 arguments to match module signature}}
"firrtl.module"() ( {
  ^entry:
}) {sym_name = "foo", convention = #firrtl<convention internal>,
    portTypes = [!firrtl.uint], portDirections = 1 : i1,
    portNames = ["in0"], portAnnotations = [], portSyms = [],
    portLocations = [loc("loc")]} : () -> ()
}

// -----

firrtl.circuit "foo" {
// expected-error @+1 {{block argument types should match signature types}}
"firrtl.module"() ( {
  ^entry(%a: i1):
}) {sym_name = "foo", convention = #firrtl<convention internal>,
    portTypes = [!firrtl.uint], portDirections = 1 : i1,
    portNames = ["in0"], portAnnotations = [], portSyms = [],
    portLocations = [loc("foo")]} : () -> ()
}

// -----

firrtl.circuit "Foo" {
firrtl.module @Foo() {
  // expected-error @+1 {{invalid kind of type specified}}
  firrtl.constant 100 : !firrtl.bundle<>
}
}

// -----

firrtl.circuit "Foo" {
firrtl.module @Foo() {
  // expected-error @+1 {{constant too large for result type}}
  firrtl.constant 100 : !firrtl.uint<4>
}
}

// -----

firrtl.circuit "Foo" {
firrtl.module @Foo() {
  // expected-error @+1 {{constant too large for result type}}
  firrtl.constant -100 : !firrtl.sint<4>
}
}

// -----

firrtl.circuit "Foo" {
firrtl.module @Foo() {
  // expected-error @+1 {{special constants can only be 0 or 1}}
  firrtl.specialconstant 2 : !firrtl.clock
}
}

// -----

firrtl.circuit "Foo" {
firrtl.module @Foo() {
  // expected-error @+1 {{special constants can only be 0 or 1}}
  firrtl.specialconstant 2 : !firrtl.reset
}
}

// -----

firrtl.circuit "Foo" {
firrtl.module @Foo() {
  // expected-error @+1 {{special constants can only be 0 or 1}}
  firrtl.specialconstant 2 : !firrtl.asyncreset
}
}

// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo(in %clk: !firrtl.clock, in %reset: !firrtl.uint<2>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    // expected-error @+1 {{'firrtl.regreset' op operand #1 must be Reset, but got '!firrtl.uint<2>'}}
    %a = firrtl.regreset %clk, %reset, %zero {name = "a"} : !firrtl.clock, !firrtl.uint<2>, !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo() {
  // expected-error @+1 {{'firrtl.mem' op attribute 'writeLatency' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 1}}
    %m = firrtl.mem Undefined {depth = 32 : i64, name = "m", portNames = ["rw"], readLatency = 0 : i32, writeLatency = 0 : i32} : !firrtl.bundle<>
  }
}

// -----

firrtl.circuit "Foo" {
  // expected-error @+1 {{'firrtl.extmodule' op attribute 'defname' with value "Bar" conflicts with the name of another module in the circuit}}
  firrtl.extmodule @Foo() attributes { defname = "Bar" }
  // expected-note @+1 {{previous module declared here}}
  firrtl.module @Bar() {}
  // Allow an extmodule to conflict with its own symbol name
  firrtl.extmodule @Baz() attributes { defname = "Baz" }

}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{previous extmodule definition occurred here}}
  firrtl.extmodule @Foo(in a : !firrtl.uint<1>) attributes { defname = "Foo" }
  // expected-error @+1 {{'firrtl.extmodule' op with 'defname' attribute "Foo" has 0 ports which is different from a previously defined extmodule with the same 'defname' which has 1 ports}}
  firrtl.extmodule @Bar() attributes { defname = "Foo" }
}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{previous extmodule definition occurred here}}
  firrtl.extmodule @Foo(in a : !firrtl.uint<1>) attributes { defname = "Foo" }
  // expected-error @+1 {{'firrtl.extmodule' op with 'defname' attribute "Foo" has a port with name "b" which does not match the name of the port in the same position of a previously defined extmodule with the same 'defname', expected port to have name "a"}}
  firrtl.extmodule @Foo_(in b : !firrtl.uint<1>) attributes { defname = "Foo" }
}

// -----

firrtl.circuit "Foo" {
  firrtl.extmodule @Foo<width: i32 = 2>(in a : !firrtl.uint<2>) attributes { defname = "Foo" }
  // expected-note @+1 {{previous extmodule definition occurred here}}
  firrtl.extmodule @Bar(in a : !firrtl.uint<1>) attributes { defname = "Foo" }
  // expected-error @+1 {{'firrtl.extmodule' op with 'defname' attribute "Foo" has a port with name "a" which has a different type '!firrtl.uint<2>' which does not match the type of the port in the same position of a previously defined extmodule with the same 'defname', expected port to have type '!firrtl.uint<1>'}}
  firrtl.extmodule @Baz(in a : !firrtl.uint<2>) attributes { defname = "Foo" }
}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{previous extmodule definition occurred here}}
  firrtl.extmodule @Foo(in a : !firrtl.uint<1>) attributes { defname = "Foo" }
  // expected-error @+1 {{'firrtl.extmodule' op with 'defname' attribute "Foo" has a port with name "a" which has a different type '!firrtl.sint<1>' which does not match the type of the port in the same position of a previously defined extmodule with the same 'defname', expected port to have type '!firrtl.uint<1>'}}
  firrtl.extmodule @Foo_(in a : !firrtl.sint<1>) attributes { defname = "Foo" }
}

// -----

firrtl.circuit "Foo" {

  // expected-note @+1 {{previous extmodule definition occurred here}}
  firrtl.extmodule @Foo<width: i32 = 2>(in a : !firrtl.uint<2>) attributes { defname = "Foo" }
  // expected-error @+1 {{'firrtl.extmodule' op with 'defname' attribute "Foo" has a port with name "a" which has a different type '!firrtl.sint' which does not match the type of the port in the same position of a previously defined extmodule with the same 'defname', expected port to have type '!firrtl.uint'}}
  firrtl.extmodule @Bar(in a : !firrtl.sint<1>) attributes { defname = "Foo" }

}

// -----

firrtl.circuit "Foo" {
  // expected-error @+1 {{has unknown extmodule parameter value 'width' = @Foo}}
  firrtl.extmodule @Foo<width: none = @Foo>(in a : !firrtl.uint<2>) attributes { defname = "Foo" }
}

// -----

firrtl.circuit "Foo" {
  firrtl.extmodule @Foo()
  // expected-error @+1 {{'firrtl.instance' op expects parent op to be one of 'firrtl.module, firrtl.group, firrtl.when, firrtl.match'}}
  firrtl.instance "" @Foo()
}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{containing module declared here}}
  firrtl.module @Foo() {
    // expected-error @+1 {{'firrtl.instance' op is a recursive instantiation of its containing module}}
    firrtl.instance "" @Foo()
  }
}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{original module declared here}}
  firrtl.module @Callee(in %arg0: !firrtl.uint<1>) { }
  firrtl.module @Foo() {
    // expected-error @+1 {{'firrtl.instance' op result type for "arg0" must be '!firrtl.uint<1>', but got '!firrtl.uint<2>'}}
    %a = firrtl.instance "" @Callee(in arg0: !firrtl.uint<2>)
  }
}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{original module declared here}}
  firrtl.module @Callee(in %arg0: !firrtl.uint<1> ) { }
  firrtl.module @Foo() {
    // expected-error @+1 {{'firrtl.instance' op has a wrong number of results; expected 1 but got 0}}
    firrtl.instance "" @Callee()
  }
}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{original module declared here}}
  firrtl.module @Callee(in %arg0: !firrtl.uint<1>, in %arg1: !firrtl.bundle<valid: uint<1>>) { }
  firrtl.module @Foo() {
    // expected-error @+1 {{'firrtl.instance' op result type for "arg1" must be '!firrtl.bundle<valid: uint<1>>', but got '!firrtl.bundle<valid: uint<2>>'}}
    %a:2 = firrtl.instance "" @Callee(in arg0: !firrtl.uint<1>, in arg1: !firrtl.bundle<valid: uint<2>>)
  }
}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{original module declared here}}
  firrtl.module @Callee(in %arg0: !firrtl.uint<1>, in %arg1: !firrtl.bundle<valid: uint<1>>) { }
  firrtl.module @Foo() {
    // expected-error @+1 {{'firrtl.instance' op name for port 1 must be "arg1", but got "xxx"}}
    %a:2 = firrtl.instance "" @Callee(in arg0: !firrtl.uint<1>, in xxx: !firrtl.bundle<valid: uint<1>>)
  }
}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{original module declared here}}
  firrtl.module @Callee(in %arg0: !firrtl.uint<1>, in %arg1: !firrtl.bundle<valid: uint<1>>) { }
  firrtl.module @Foo() {
    // expected-error @+1 {{'firrtl.instance' op direction for "arg1" must be "in", but got "out"}}
    %a:2 = firrtl.instance "" @Callee(in arg0: !firrtl.uint<1>, out arg1: !firrtl.bundle<valid: uint<1>>)
  }
}

// -----

firrtl.circuit "X" {

firrtl.module @X(in %a : !firrtl.uint<4>) {
  // expected-error @below {{failed to infer returned types}}
  // expected-error @+1 {{high must be equal or greater than low, but got high = 3, low = 4}}
  %0 = firrtl.bits %a 3 to 4 : (!firrtl.uint<4>) -> !firrtl.uint<2>
}

}

// -----

firrtl.circuit "X" {

firrtl.module @X(in %a : !firrtl.uint<4>) {
  // expected-error @below {{failed to infer returned types}}
  // expected-error @+1 {{high must be smaller than the width of input, but got high = 4, width = 4}}
  %0 = firrtl.bits %a 4 to 3 : (!firrtl.uint<4>) -> !firrtl.uint<2>
}

}

// -----

firrtl.circuit "X" {

firrtl.module @X(in %a : !firrtl.uint<4>) {
  // expected-error @below {{failed to infer returned types}}
  // expected-error @+1 {{'firrtl.bits' op inferred type(s) '!firrtl.uint<3>' are incompatible with return type(s) of operation '!firrtl.uint<2>'}}
  %0 = firrtl.bits %a 3 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<2>
}

}

// -----

firrtl.circuit "BadPort" {
  firrtl.module @BadPort(in %a : !firrtl.uint<1>) {
    // expected-error @+1 {{'firrtl.attach' op operand #0 must be analog type, but got '!firrtl.uint<1>'}}
    firrtl.attach %a, %a : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "BadAdd" {
  firrtl.module @BadAdd(in %a : !firrtl.uint<1>) {
    // expected-error @below {{failed to infer returned types}}
    // expected-error @+1 {{'firrtl.add' op inferred type(s) '!firrtl.uint<2>' are incompatible with return type(s) of operation '!firrtl.uint<1>'}}
    firrtl.add %a, %a : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "StructCast" {
  firrtl.module @StructCast() {
    %a = firrtl.wire : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
    // expected-error @+1 {{bundle and struct have different number of fields}}
    %b = firrtl.hwStructCast %a : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>) -> (!hw.struct<valid: i1, ready: i1>)
  }
}

// -----

firrtl.circuit "StructCast2" {
  firrtl.module @StructCast2() {
    %a = firrtl.wire : !firrtl.bundle<valid: uint<1>>
    // expected-error @+1 {{field names don't match 'valid', 'yovalid'}}
    %b = firrtl.hwStructCast %a : (!firrtl.bundle<valid: uint<1>>) -> (!hw.struct<yovalid: i1>)
  }
}

// -----

firrtl.circuit "StructCast3" {
  firrtl.module @StructCast3() {
    %a = firrtl.wire : !firrtl.bundle<valid: uint<1>>
    // expected-error @+1 {{size of field 'valid' don't match 1, 2}}
    %b = firrtl.hwStructCast %a : (!firrtl.bundle<valid: uint<1>>) -> (!hw.struct<valid: i2>)
  }
}

// -----

firrtl.circuit "OutOfOrder" {
  firrtl.module @OutOfOrder(in %a: !firrtl.uint<32>) {
    // expected-error @+1 {{operand #0 does not dominate this use}}
    %0 = firrtl.add %1, %1 : (!firrtl.uint<33>, !firrtl.uint<33>) -> !firrtl.uint<34>
    // expected-note @+1 {{operand defined here}}
    %1 = firrtl.add %a, %a : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<33>
  }
}

// -----

firrtl.circuit "CombMemInvalidReturnType" {
  firrtl.module @CombMemInvalidReturnType() {
    // expected-error @+1 {{'chirrtl.combmem' op result #0 must be a behavioral memory, but got '!firrtl.uint<1>'}}
    %mem = chirrtl.combmem : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "CombMemNonPassiveReturnType" {
  firrtl.module @CombMemNonPassiveReturnType() {
    // expected-error @+1 {{behavioral memory element type must be passive}}
    %mem = chirrtl.combmem : !chirrtl.cmemory<bundle<a flip : uint<1>>, 1>
  }
}

// -----

firrtl.circuit "CombMemPerFieldSym" {
  firrtl.module @CombMemPerFieldSym() {
    // expected-error @below {{op does not support per-field inner symbols}}
    %mem = chirrtl.combmem sym [<@x,1,public>] : !chirrtl.cmemory<bundle<a: uint<1>>, 1>
  }
}

// -----

firrtl.circuit "SeqMemInvalidReturnType" {
  firrtl.module @SeqMemInvalidReturnType() {
    // expected-error @+1 {{'chirrtl.seqmem' op result #0 must be a behavioral memory, but got '!firrtl.uint<1>'}}
    %mem = chirrtl.seqmem Undefined : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "SeqMemNonPassiveReturnType" {
  firrtl.module @SeqMemNonPassiveReturnType() {
    // expected-error @+1 {{behavioral memory element type must be passive}}
    %mem = chirrtl.seqmem Undefined : !chirrtl.cmemory<bundle<a flip : uint<1>>, 1>
  }
}

// -----

firrtl.circuit "SeqMemPerFieldSym" {
  firrtl.module @SeqMemPerFieldSym() {
    // expected-error @below {{op does not support per-field inner symbols}}
    %mem = chirrtl.seqmem sym [<@x,1,public>] Undefined : !chirrtl.cmemory<bundle<a: uint<1>>, 1>
  }
}

// -----

firrtl.circuit "SeqCombMemDupSym" {
  firrtl.module @SeqCombMemDupSym() {
    // expected-note @below {{see existing inner symbol definition here}}
    %smem = chirrtl.seqmem sym @x Undefined : !chirrtl.cmemory<bundle<a: uint<1>>, 1>
    // expected-error @below {{redefinition of inner symbol named 'x'}}
    %cmem = chirrtl.combmem sym @x : !chirrtl.cmemory<bundle<a: uint<1>>, 1>
  }
}

// -----

firrtl.circuit "MemoryPortInvalidReturnType" {
  firrtl.module @MemoryPortInvalidReturnType(in %sel : !firrtl.uint<8>, in %clock : !firrtl.clock) {
    %mem = chirrtl.combmem : !chirrtl.cmemory<uint<8>, 8>
    // expected-error @+1 {{'chirrtl.memoryport' op port should be used by a chirrtl.memoryport.access}}
    %memoryport_data, %memoryport_port = chirrtl.memoryport Infer %mem {name = "memoryport"} : (!chirrtl.cmemory<uint<8>, 8>) -> (!firrtl.uint<8>, !chirrtl.cmemoryport)
  }
}

// -----

firrtl.circuit "MemoryPortInvalidReturnType" {
  firrtl.module @MemoryPortInvalidReturnType(in %sel : !firrtl.uint<8>, in %clock : !firrtl.clock) {
    %mem = chirrtl.combmem : !chirrtl.cmemory<uint<8>, 8>
    // expected-error @below {{failed to infer returned types}}
    // expected-error @+1 {{'chirrtl.memoryport' op inferred type(s) '!firrtl.uint<8>', '!chirrtl.cmemoryport' are incompatible with return type(s) of operation '!firrtl.uint<9>', '!chirrtl.cmemoryport'}}
    %memoryport_data, %memoryport_port = chirrtl.memoryport Infer %mem {name = "memoryport"} : (!chirrtl.cmemory<uint<8>, 8>) -> (!firrtl.uint<9>, !chirrtl.cmemoryport)
    chirrtl.memoryport.access %memoryport_port[%sel], %clock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
  }
}

// -----

firrtl.circuit "MemoryInvalidmask" {
  firrtl.module @MemoryInvalidmask() {
    // expected-error @+1 {{'firrtl.mem' op the mask width cannot be greater than data width}}
    %memory_rw = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["rw"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, wdata: uint<8>, wmask: uint<9>>
  }
}
// -----

firrtl.circuit "MemoryNegativeReadLatency" {
  firrtl.module @MemoryNegativeReadLatency() {
    // expected-error @+1 {{'firrtl.mem' op attribute 'readLatency' failed to satisfy constraint}}
    %memory_r = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r", "rw", "w"], readLatency = -1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
}

// -----

firrtl.circuit "MemoryZeroWriteLatency" {
  firrtl.module @MemoryZeroWriteLatency() {
    // expected-error @+1 {{'firrtl.mem' op attribute 'writeLatency' failed to satisfy constraint}}
    %memory_r = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r", "rw", "w"], readLatency = 0 : i32, writeLatency = 0 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
}

// -----

firrtl.circuit "MemoryZeroDepth" {
  firrtl.module @MemoryZeroDepth() {
    // expected-error @+1 {{'firrtl.mem' op attribute 'depth' failed to satisfy constraint}}
    %memory_r = firrtl.mem Undefined {depth = 0 : i64, name = "memory", portNames = ["r", "rw", "w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
}

// -----

firrtl.circuit "MemoryBadPortType" {
  firrtl.module @MemoryBadPortType() {
    // expected-error @+1 {{'firrtl.mem' op has an invalid type on port "r"}}
    %memory_r = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "MemoryPortNamesCollide" {
  firrtl.module @MemoryPortNamesCollide() {
    // expected-error @+1 {{'firrtl.mem' op has non-unique port name "r"}}
    %memory_r, %memory_r_0 = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r", "r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
}

// -----

firrtl.circuit "MemoryUnexpectedNumberOfFields" {
  firrtl.module @MemoryUnexpectedNumberOfFields() {
    // expected-error @+1 {{'firrtl.mem' op has an invalid type on port "r"}}
    %memory_r = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<a: uint<1>>
  }
}

// -----

firrtl.circuit "MemoryMissingDataField" {
  firrtl.module @MemoryMissingDataField() {
    // expected-error @+1 {{'firrtl.mem' op has an invalid type on port}}
    %memory_r = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>>
  }
}

// -----

firrtl.circuit "MemoryMissingDataField2" {
  firrtl.module @MemoryMissingDataField2() {
    // expected-error @+1 {{'firrtl.mem' op has an invalid type on port}}
    %memory_rw = firrtl.mem Undefined {depth = 16 : i64, name = "memory2", portNames = ["rw"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, writedata: uint<8>, wmask: uint<1>>
  }
}

// -----

firrtl.circuit "MemoryDataNotPassive" {
  firrtl.module @MemoryDataNotPassive() {
    // expected-error @+1 {{'firrtl.mem' op has non-passive data type on port "r" (memory types must be passive)}}
    %memory_r = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a flip: uint<8>, b: uint<8>>>
  }
}

// -----

firrtl.circuit "MemoryDataContainsAnalog" {
  firrtl.module @MemoryDataContainsAnalog() {
    // expected-error @+1 {{'firrtl.mem' op has a data type that contains an analog type on port "r" (memory types cannot contain analog types)}}
    %memory_r = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: analog<8>>>
  }
}

// -----

firrtl.circuit "MemoryPortInvalidReadKind" {
  firrtl.module @MemoryPortInvalidReadKind() {
    // expected-error @+1 {{'firrtl.mem' op has an invalid type on port}}
    %memory_r= firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<BAD: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
}

// -----

firrtl.circuit "MemoryPortInvalidWriteKind" {
  firrtl.module @MemoryPortInvalidWriteKind() {
    // expected-error @+1 {{'firrtl.mem' op has an invalid type on port}}
    %memory_r= firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, BAD: uint<1>>
  }
}

// -----

firrtl.circuit "MemoryPortInvalidReadWriteKind" {
  firrtl.module @MemoryPortInvalidReadWriteKind() {
    // expected-error @+1 {{'firrtl.mem' op has an invalid type on port}}
    %memory_r= firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["rw"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, wdata: uint<8>, BAD: uint<1>>
  }
}

// -----

firrtl.circuit "MemoryPortsWithDifferentTypes" {
  firrtl.module @MemoryPortsWithDifferentTypes() {
    // expected-error @+1 {{'firrtl.mem' op port "r1" has a different type than port "r0" (expected '!firrtl.uint<8>', but got '!firrtl.sint<8>')}}
    %memory_r0, %memory_r1 = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r0", "r1"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>
  }
}

// -----

firrtl.circuit "SubfieldOpWithIntegerFieldIndex" {
  firrtl.module @SubfieldOpFieldError() {
    %w = firrtl.wire  : !firrtl.bundle<a: uint<2>, b: uint<2>>
    // expected-error @+1 {{'firrtl.subfield' expected valid keyword or string}}
    %w_a = firrtl.subfield %w[2] : !firrtl.bundle<a : uint<2>, b : uint<2>>
  }
}

// -----

firrtl.circuit "SubfieldOpFieldUnknown" {
  firrtl.module @SubfieldOpFieldError() {
    %w = firrtl.wire  : !firrtl.bundle<a: uint<2>, b: uint<2>>
    // expected-error @+1 {{'firrtl.subfield' unknown field c in bundle type '!firrtl.bundle<a: uint<2>, b: uint<2>>'}}
    %w_a = firrtl.subfield %w[c] : !firrtl.bundle<a : uint<2>, b : uint<2>>
  }
}

// -----

firrtl.circuit "SubfieldOpInputTypeMismatch" {
  firrtl.module @SubfieldOpFieldError() {
    %w = firrtl.wire : !firrtl.bundle<a: uint<2>, b: uint<2>>
    // expected-error @+2 {{use of value '%w' expects different type than prior uses}}
    // expected-note  @-2 {{prior use here}}
    %w_a = firrtl.subfield %w[a] : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "SubfieldOpNonBundleInputType" {
  firrtl.module @SubfieldOpFieldError() {
    %w = firrtl.wire : !firrtl.uint<1>
    // expected-error @+1 {{'firrtl.subfield' input must be bundle type, got '!firrtl.uint<1>'}}
    %w_a = firrtl.subfield %w[a] : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "BitCast1" {
  firrtl.module @BitCast1() {
    %a = firrtl.wire : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint>
    // expected-error @+1 {{bitwidth cannot be determined for input operand type}}
    %b = firrtl.bitcast %a : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint>) -> (!firrtl.uint<6>)
  }
}

// -----

firrtl.circuit "BitCast2" {
  firrtl.module @BitCast2() {
    %a = firrtl.wire : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>
    // expected-error @+1 {{the bitwidth of input (3) and result (6) don't match}}
    %b = firrtl.bitcast %a : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>) -> (!firrtl.uint<6>)
  }
}

// -----

firrtl.circuit "BitCast4" {
  firrtl.module @BitCast4() {
    %a = firrtl.wire : !firrtl.analog
    // expected-error @+1 {{bitwidth cannot be determined for input operand type}}
    %b = firrtl.bitcast %a : (!firrtl.analog) -> (!firrtl.uint<1>)
  }
}

// -----

firrtl.circuit "BitCast5" {
  firrtl.module @BitCast5() {
    %a = firrtl.wire : !firrtl.uint<3>
    // expected-error @below {{'firrtl.bitcast' op result #0 must be a passive base type (contain no flips), but got}}
    %b = firrtl.bitcast %a : (!firrtl.uint<3>) -> (!firrtl.bundle<valid flip: uint<1>, ready: uint<1>, data: uint<1>>)
  }
}

// -----

firrtl.circuit "NLAWithNestedReference" {
firrtl.module @NLAWithNestedReference() { }
// expected-error @below {{only one nested reference is allowed}}
hw.hierpath private @nla [@A::@B::@C]
}

// -----


firrtl.circuit "LowerToBind" {
 // expected-error @+1 {{the instance path cannot be empty}}
hw.hierpath private @NLA1 []
hw.hierpath private @NLA2 [@LowerToBind::@s1]
firrtl.module @InstanceLowerToBind() {}
firrtl.module @LowerToBind() {
  firrtl.instance foo sym @s1 {lowerToBind} @InstanceLowerToBind()
}
}

// -----

firrtl.circuit "NLATop" {

 // expected-error @+1 {{the instance path can only contain inner sym reference, only the leaf can refer to a module symbol}}
  hw.hierpath private @nla [@NLATop::@test, @Aardvark, @Zebra]
  hw.hierpath private @nla_1 [@NLATop::@test,@Aardvark::@test_1, @Zebra]
  firrtl.module @NLATop() {
    firrtl.instance test  sym @test @Aardvark()
    firrtl.instance test2 @Zebra()
  }

  firrtl.module @Aardvark() {
    firrtl.instance test sym @test @Zebra()
    firrtl.instance test1 sym @test_1 @Zebra()
  }

  firrtl.module @Zebra() {
  }
}

// -----

firrtl.circuit "NLATop1" {
  // expected-error @+1 {{instance path is incorrect. Expected module: "Aardvark" instead found: "Zebra"}}
  hw.hierpath private @nla [@NLATop1::@test, @Zebra::@test,@Aardvark::@test]
  hw.hierpath private @nla_1 [@NLATop1::@test,@Aardvark::@test_1, @Zebra]
  firrtl.module @NLATop1() {
    firrtl.instance test  sym @test @Aardvark()
    firrtl.instance test2 @Zebra()
  }

  firrtl.module @Aardvark() {
    firrtl.instance test sym @test @Zebra()
    firrtl.instance test1 sym @test_1 @Zebra()
  }

  firrtl.module @Zebra() {
    firrtl.instance test sym @test @Ext()
    firrtl.instance test1 sym @test_1 @Ext()
  }

  firrtl.module @Ext() {
  }
}

// -----

// This should not error out. Note that there is no symbol on the %bundle. This handles a special case, when the nonlocal is applied to a subfield.
firrtl.circuit "fallBackName" {

  hw.hierpath private @nla [@fallBackName::@test, @Aardvark::@test, @Zebra::@bundle]
  hw.hierpath private @nla_1 [@fallBackName::@test,@Aardvark::@test_1, @Zebra]
  firrtl.module @fallBackName() {
    firrtl.instance test  sym @test @Aardvark()
    firrtl.instance test2 @Zebra()
  }

  firrtl.module @Aardvark() {
    firrtl.instance test sym @test @Zebra()
    firrtl.instance test1 sym @test_1 @Zebra()
  }

  firrtl.module @Zebra(){
    %bundle = firrtl.wire  sym @bundle {annotations = [{circt.fieldID = 2 : i32, circt.nonlocal = @nla, class ="test"}]}: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
  }
}

// -----

firrtl.circuit "Foo"   {
  // expected-error @+1 {{operation with symbol: #hw.innerNameRef<@Bar::@b> was not found}}
  hw.hierpath private @nla_1 [@Foo::@bar, @Bar::@b]
  firrtl.module @Bar(in %a: !firrtl.uint<1>, out %b: !firrtl.bundle<baz: uint<1>, qux: uint<1>> [{circt.fieldID = 2 : i32, circt.nonlocal = @nla_1, three}], out %c: !firrtl.uint<1>) {
  }
  firrtl.module @Foo() {
    %bar_a, %bar_b, %bar_c = firrtl.instance bar sym @bar  @Bar(in a: !firrtl.uint<1> [{one}], out b: !firrtl.bundle<baz: uint<1>, qux: uint<1>> [{circt.fieldID = 1 : i32, two}], out c: !firrtl.uint<1> [{four}])
  }
}

// -----

firrtl.circuit "Top"   {
 // Legal nla would be:
//hw.hierpath private @nla [@Top::@mid, @Mid::@leaf, @Leaf::@w]
  // expected-error @+1 {{instance path is incorrect. Expected module: "Middle" instead found: "Leaf"}}
  hw.hierpath private @nla [@Top::@mid, @Leaf::@w]
  firrtl.module @Leaf() {
    %w = firrtl.wire sym @w  {annotations = [{circt.nonlocal = @nla, class = "fake1"}]} : !firrtl.uint<3>
  }
  firrtl.module @Middle() {
    firrtl.instance leaf sym @leaf  @Leaf()
  }
  firrtl.module @Top() {
    firrtl.instance mid sym @mid  @Middle()
  }
}

// -----

firrtl.circuit "Top" {
  firrtl.module @Top (in %in : !firrtl.uint) {
    %a = firrtl.wire : !firrtl.uint
    // expected-error @+1 {{op operand #0 must be a sized passive base type}}
    firrtl.strictconnect %a, %in : !firrtl.uint
  }
}

// -----

firrtl.circuit "AnalogRegister" {
  firrtl.module @AnalogRegister(in %clock: !firrtl.clock) {
    // expected-error @+1 {{'firrtl.reg' op result #0 must be a passive non-'const' base type that does not contain analog, but got '!firrtl.analog'}}
    %r = firrtl.reg %clock : !firrtl.clock, !firrtl.analog
  }
}

// -----

firrtl.circuit "AnalogVectorRegister" {
  firrtl.module @AnalogVectorRegister(in %clock: !firrtl.clock) {
    // expected-error @+1 {{'firrtl.reg' op result #0 must be a passive non-'const' base type that does not contain analog, but got '!firrtl.vector<analog, 2>'}}
    %r = firrtl.reg %clock : !firrtl.clock, !firrtl.vector<analog, 2>
  }
}

// -----

firrtl.circuit "MismatchedRegister" {
  firrtl.module @MismatchedRegister(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, out %z: !firrtl.vector<uint<1>, 1>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // expected-error @+1 {{type mismatch between register '!firrtl.vector<uint<1>, 1>' and reset value '!firrtl.uint<1>'}}
    %r = firrtl.regreset %clock, %reset, %c0_ui1  : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.vector<uint<1>, 1>
    firrtl.connect %z, %r : !firrtl.vector<uint<1>, 1>, !firrtl.vector<uint<1>, 1>
  }
}

// -----

firrtl.circuit "EnumOutOfRange" {
  firrtl.module @EnumSameCase(in %enum : !firrtl.enum<a : uint<8>>) {
    // expected-error @+1 {{the tag index 1 is out of the range of valid tags in '!firrtl.enum<a: uint<8>>'}}
    "firrtl.match"(%enum) ({
    ^bb0(%arg0: !firrtl.uint<8>):
    }) {tags = [1 : i32]} : (!firrtl.enum<a: uint<8>>) -> ()
  }
}
// -----

firrtl.circuit "EnumSameCase" {
  firrtl.module @EnumSameCase(in %enum : !firrtl.enum<a : uint<8>>) {
    // expected-error @+1 {{the tag "a" is matched more than once}}
    "firrtl.match"(%enum) ({
    ^bb0(%arg0: !firrtl.uint<8>):
    }, {
    ^bb0(%arg0: !firrtl.uint<8>):
    }) {tags = [0 : i32, 0 : i32]} : (!firrtl.enum<a: uint<8>>) -> ()
  }
}

// -----

firrtl.circuit "EnumNonExaustive" {
  firrtl.module @EnumNonExaustive(in %enum : !firrtl.enum<a : uint<8>>) {
    // expected-error @+1 {{missing case for tag "a"}}
    "firrtl.match"(%enum) {tags = []} : (!firrtl.enum<a: uint<8>>) -> ()
  }
}

// -----

// expected-error @+1 {{'firrtl.circuit' op main module 'private_main' must be public}}
firrtl.circuit "private_main" {
  firrtl.module private @private_main() {}
}

// -----

firrtl.circuit "InnerSymAttr" {
  firrtl.module @InnerSymAttr() {
    // expected-error @+1 {{cannot assign multiple symbol names to the field id:'2'}}
    %w3 = firrtl.wire sym [<@w3,2,public>,<@x2,2,private>,<@syh2,0,public>] : !firrtl.bundle<a: uint<1>, b: uint<1>, c: uint<1>, d: uint<1>>
  }
}

// -----

firrtl.circuit "InnerSymAttr2" {
  firrtl.module @InnerSymAttr2() {
    // expected-error @+1 {{cannot reuse symbol name:'w3'}}
    %w4 = firrtl.wire sym [<@w3,1,public>,<@w3,2,private>,<@syh2,0,public>] : !firrtl.bundle<a: uint<1>, b: uint<1>, c: uint<1>, d: uint<1>>
  }
}

// -----
firrtl.circuit "Parent" {
  firrtl.module @Child() {
    %w = firrtl.wire sym @w : !firrtl.uint<1>
  }
  firrtl.module @Parent() {
    // expected-error @below {{'firrtl.instance' op does not support per-field inner symbols}}
    firrtl.instance child sym [<@w3,1,public>,<@w3,2,private>,<@syh2,0,public>] @Child()
  }
}

// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo() {
  // expected-error @below {{'firrtl.mem' op does not support per-field inner symbols}}
    %m = firrtl.mem sym [<@x,1,public>,<@y,2,public>] Undefined {depth = 16 : i64, name = "ReadMemory", portNames = ["read0"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>
  }
}

// -----

firrtl.circuit "Foo"   {
  firrtl.module @Bar(in %a: !firrtl.uint<1>, out %b: !firrtl.bundle<baz: uint<1>, qux: uint<1>>, out %c: !firrtl.uint<1>) {
  }
  firrtl.module @Foo() {
    // expected-error @+1 {{'firrtl.instance' op does not support per-field inner symbols}}
    %bar_a, %bar_b, %bar_c = firrtl.instance bar sym [<@w3,1,public>,<@w3,2,private>,<@syh2,0,public>] @Bar(in a: !firrtl.uint<1> [{one}], out b: !firrtl.bundle<baz: uint<1>, qux: uint<1>> [{circt.fieldID = 1 : i32, two}], out c: !firrtl.uint<1> [{four}])
  }
}

// -----

firrtl.circuit "DupSyms" {
  firrtl.module @DupSyms() {
    // expected-note @+1 {{see existing inner symbol definition here}}
    %w1 = firrtl.wire sym @x : !firrtl.uint<2>
    // expected-error @+1 {{redefinition of inner symbol named 'x'}}
    %w2 = firrtl.wire sym @x : !firrtl.uint<2>
  }
}

// -----

firrtl.circuit "DupSymPort" {
  // expected-note @+1 {{see existing inner symbol definition here}}
  firrtl.module @DupSymPort(in %a : !firrtl.uint<1> sym @x) {
    // expected-error @+1 {{redefinition of inner symbol named 'x'}}
    %w1 = firrtl.wire sym @x : !firrtl.uint<2>
  }
}

// -----

firrtl.circuit "DupSymField" {
  firrtl.module @DupSymField() {
    // expected-note @+1 {{see existing inner symbol definition here}}
    %w1 = firrtl.wire sym @x : !firrtl.uint<2>
    // expected-error @+1 {{redefinition of inner symbol named 'x'}}
    %w3 = firrtl.wire sym [<@x,1,public>] : !firrtl.vector<uint<1>,1>
  }
}

// -----
// Node ops cannot have reference type

firrtl.circuit "NonRefNode" {
firrtl.module @NonRefNode(in %in1 : !firrtl.probe<uint<8>>) {
  // expected-error @+1 {{'firrtl.node' op operand #0 must be a passive base type (contain no flips), but got '!firrtl.probe<uint<8>>'}}
  %n1 = firrtl.node %in1 : !firrtl.probe<uint<8>>
  %a = firrtl.wire : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
}
}

// -----
// Registers cannot be reference type.

firrtl.circuit "NonRefRegister" {
  firrtl.module @NonRefRegister(in %clock: !firrtl.clock) {
    // expected-error @+1 {{'firrtl.reg' op result #0 must be a passive non-'const' base type that does not contain analog}}
    %r = firrtl.reg %clock : !firrtl.clock, !firrtl.probe<uint<8>>
  }
}

// -----
// Ref types must be ground

firrtl.circuit "RefBundle" {
  // expected-error @+1 {{reference base type must be passive}}
  firrtl.module @RefBundle(in %in1 : !firrtl.probe<bundle<valid flip : uint<1>>>) {
  }
}

// -----
// Ref types cannot be ref

firrtl.circuit "RefRef" {
  // expected-error @+1 {{expected base type, found '!firrtl.probe<uint<1>>'}}
  firrtl.module @RefRef(in %in1 : !firrtl.probe<probe<uint<1>>>) {
  }
}

// -----
// No ref in bundle

firrtl.circuit "RefField" {
  // expected-error @+1 {{expected base type, found '!firrtl.probe<uint<1>>'}}
  firrtl.module @RefField(in %in1 : !firrtl.bundle<r: probe<uint<1>>>) {
  }
}

// -----
// Invalid

firrtl.circuit "InvalidRef" {
  firrtl.module @InvalidRef() {
    // expected-error @+1 {{'firrtl.invalidvalue' op result #0 must be a base type, but got '!firrtl.probe<uint<1>>'}}
    %0 = firrtl.invalidvalue : !firrtl.probe<uint<1>>
  }
}

// -----
// Mux ref

firrtl.circuit "MuxRef" {
  firrtl.module @MuxRef(in %a: !firrtl.probe<uint<1>>, in %b: !firrtl.probe<uint<1>>,
                          in %cond: !firrtl.uint<1>) {
    // expected-error @+1 {{'firrtl.mux' op operand #1 must be a passive base type (contain no flips), but got '!firrtl.probe<uint<1>>'}}
    %a_or_b = firrtl.mux(%cond, %a, %b) : (!firrtl.uint<1>, !firrtl.probe<uint<1>>, !firrtl.probe<uint<1>>) -> !firrtl.probe<uint<1>>
  }
}

// -----
// Bitcast ref

firrtl.circuit "BitcastRef" {
  firrtl.module @BitcastRef(in %a: !firrtl.probe<uint<1>>) {
    // expected-error @+1 {{'firrtl.bitcast' op operand #0 must be a base type, but got '!firrtl.probe<uint<1>>}}
    %0 = firrtl.bitcast %a : (!firrtl.probe<uint<1>>) -> (!firrtl.probe<uint<1>>)
  }
}

// -----
// Cannot connect ref types

firrtl.circuit "Top" {
  firrtl.module @Foo (in %in: !firrtl.probe<uint<2>>) {}
  firrtl.module @Top (in %in: !firrtl.probe<uint<2>>) {
    %foo_in = firrtl.instance foo @Foo(in in: !firrtl.probe<uint<2>>)
    // expected-error @below {{must be a sized passive base type}}
    firrtl.strictconnect %foo_in, %in : !firrtl.probe<uint<2>>
  }
}

// -----
// Check flow semantics for ref.send

firrtl.circuit "Foo" {
  // expected-note @+1 {{destination was defined here}}
  firrtl.module @Foo(in  %_a: !firrtl.probe<uint<1>>) {
    %a = firrtl.wire : !firrtl.uint<1>
    %1 = firrtl.ref.send %a : !firrtl.uint<1>
    // expected-error @+1 {{connect has invalid flow: the destination expression "_a" has source flow, expected sink or duplex flow}}
    firrtl.ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
}

// -----
// Output reference port cannot be reused

firrtl.circuit "Bar" {
  firrtl.extmodule @Bar2(out _a: !firrtl.probe<uint<1>>)
  firrtl.module @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %x = firrtl.instance x @Bar2(out _a: !firrtl.probe<uint<1>>)
    %y = firrtl.instance y @Bar2(out _a: !firrtl.probe<uint<1>>)
    // expected-error @below {{destination reference cannot be reused by multiple operations, it can only capture a unique dataflow}}
    firrtl.ref.define %_a, %x : !firrtl.probe<uint<1>>
    firrtl.ref.define %_a, %y : !firrtl.probe<uint<1>>
  }
}

// -----
// Output reference port cannot be reused

firrtl.circuit "Bar" {
  firrtl.extmodule @Bar2(out _a: !firrtl.probe<uint<1>>)
  firrtl.module @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %x = firrtl.instance x @Bar2(out _a: !firrtl.probe<uint<1>>)
    %y = firrtl.wire : !firrtl.uint<1>
    // expected-error @below {{destination reference cannot be reused by multiple operations, it can only capture a unique dataflow}}
    firrtl.ref.define %_a, %x : !firrtl.probe<uint<1>>
    %1 = firrtl.ref.send %y : !firrtl.uint<1>
    firrtl.ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
}

// -----
// Output reference port cannot be reused

firrtl.circuit "Bar" {
  firrtl.module @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %x = firrtl.wire : !firrtl.uint<1>
    %y = firrtl.wire : !firrtl.uint<1>
    %1 = firrtl.ref.send %x : !firrtl.uint<1>
    %2 = firrtl.ref.send %y : !firrtl.uint<1>
    // expected-error @below {{destination reference cannot be reused by multiple operations, it can only capture a unique dataflow}}
    firrtl.ref.define %_a, %1 : !firrtl.probe<uint<1>>
    firrtl.ref.define %_a, %2 : !firrtl.probe<uint<1>>
  }
}

// -----
// Can't define into a ref.sub.

firrtl.circuit "NoDefineIntoRefSub" {
  firrtl.module @NoDefineIntoRefSub(out %r: !firrtl.probe<vector<uint<1>,2>>) {
    %sub = firrtl.ref.sub %r[1] : !firrtl.probe<vector<uint<1>,2>>
    %x = firrtl.wire : !firrtl.uint<1>
    %xref = firrtl.ref.send %x : !firrtl.uint<1>
    // expected-error @below {{destination reference cannot be a sub-element of a reference}}
    firrtl.ref.define %sub, %xref : !firrtl.probe<uint<1>>
  }
}

// -----
// Can't define into a ref.cast.

firrtl.circuit "NoDefineIntoRefCast" {
  firrtl.module @NoDefineIntoRefCast(out %r: !firrtl.probe<uint<1>>) {
    // expected-note @below {{the destination was defined here}}
    %dest_cast = firrtl.ref.cast %r : (!firrtl.probe<uint<1>>) -> !firrtl.probe<uint>
    %x = firrtl.wire : !firrtl.uint
    %xref = firrtl.ref.send %x : !firrtl.uint
    // expected-error @below {{has invalid flow: the destination expression has source flow, expected sink or duplex flow}}
    firrtl.ref.define %dest_cast, %xref : !firrtl.probe<uint>
  }
}

// -----
// Can't cast to gain width information.

firrtl.circuit "CastAwayRefWidth" {
  firrtl.module @CastAwayRefWidth(out %r: !firrtl.probe<uint<1>>) {
    %zero = firrtl.constant 0 : !firrtl.uint
    %xref = firrtl.ref.send %zero : !firrtl.uint
    // expected-error @below {{reference result must be compatible with reference input: recursively same or uninferred of same}}
    %source = firrtl.ref.cast %xref : (!firrtl.probe<uint>) -> !firrtl.probe<uint<1>>
    firrtl.ref.define %r, %source : !firrtl.probe<uint<1>>
  }
}

// -----
// Can't promote to rwprobe.

firrtl.circuit "CastPromoteToRWProbe" {
  firrtl.module @CastPromoteToRWProbe(out %r: !firrtl.rwprobe<uint>) {
    %zero = firrtl.constant 0 : !firrtl.uint
    %xref = firrtl.ref.send %zero : !firrtl.uint
    // expected-error @below {{reference result must be compatible with reference input: recursively same or uninferred of same}}
    %source = firrtl.ref.cast %xref : (!firrtl.probe<uint>) -> !firrtl.rwprobe<uint>
    firrtl.ref.define %r, %source : !firrtl.rwprobe<uint>
  }
}

// -----
// Can't add const-ness via ref.cast

firrtl.circuit "CastToMoreConst" {
  firrtl.module @CastToMoreConst(out %r: !firrtl.probe<const.uint<3>>) {
    %zero = firrtl.wire : !firrtl.uint<3>
    %zref = firrtl.ref.send %zero : !firrtl.uint<3>
    // expected-error @below {{reference result must be compatible with reference input: recursively same or uninferred of same}}
    %zconst_ref= firrtl.ref.cast %zref : (!firrtl.probe<uint<3>>) -> !firrtl.probe<const.uint<3>>
    firrtl.ref.define %r, %zconst_ref : !firrtl.probe<const.uint<3>>
  }
}

// -----
// Check that you can't drive a source.

firrtl.circuit "PropertyDriveSource" {
  // @expected-note @below {{the destination was defined here}}
  firrtl.module @PropertyDriveSource(in %in: !firrtl.string) {
    %0 = firrtl.string "hello"
    // expected-error @below {{connect has invalid flow: the destination expression "in" has source flow, expected sink or duplex flow}}
    firrtl.propassign %in, %0 : !firrtl.string
  }
}

// -----
// Check that you can't drive a sink more than once.

firrtl.circuit "PropertyDoubleDrive" {
  firrtl.module @PropertyDriveSource(out %out: !firrtl.string) {
    %0 = firrtl.string "hello"
    // expected-error @below {{destination property cannot be reused by multiple operations, it can only capture a unique dataflow}}
    firrtl.propassign %out, %0 : !firrtl.string
    firrtl.propassign %out, %0 : !firrtl.string
  }
}

// -----
// Check that you can't connect property types.

firrtl.circuit "PropertyConnect" {
  firrtl.module @PropertyConnect(out %out: !firrtl.string) {
    %0 = firrtl.string "hello"
    // expected-error @below {{must be a sized passive base type}}
    firrtl.strictconnect %out, %0 : !firrtl.string
  }
}

// -----
// Property aggregates can only contain properties.
// Check list.

firrtl.circuit "ListOfHW" {
  // expected-error @below {{expected property type, found '!firrtl.uint<2>'}}
  firrtl.module @MapOfHW(in %in: !firrtl.list<uint<2>>) {}
}

// -----
// Property aggregates can only contain properties.
// Check map.

firrtl.circuit "MapOfHW" {
  // expected-error @below {{expected property type, found '!firrtl.uint<4>'}}
  firrtl.module @MapOfHW(in %in: !firrtl.map<string,uint<4>>) {}
}

// -----
// Issue 4174-- handle duplicate module names.

firrtl.circuit "hi" {
    // expected-note @below {{see existing symbol definition here}}
    firrtl.module @hi() {}
    // expected-error @below {{redefinition of symbol named 'hi'}}
    firrtl.module @hi() {}
}

// -----

firrtl.circuit "AnalogDifferentWidths" {
  firrtl.module @AnalogDifferentWidths() {
    %a = firrtl.wire : !firrtl.analog<1>
    %b = firrtl.wire : !firrtl.analog<2>
    // expected-error @below {{not all known operand widths match}}
    firrtl.attach %a, %b : !firrtl.analog<1>, !firrtl.analog<2>
  }
}

// -----

firrtl.circuit "ForceableWithoutRefResult" {
  firrtl.module @ForceableWithoutRefResult() {
    // expected-error @below {{op must have ref result iff marked forceable}}
    %w = firrtl.wire forceable : !firrtl.uint<2>
  }
}

// -----

firrtl.circuit "RefResultButNotForceable" {
  firrtl.module @RefResultButNotForceable() {
    // expected-error @below {{op must have ref result iff marked forceable}}
    %w, %w_f = firrtl.wire : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
  }
}

// -----

firrtl.circuit "ForceableTypeMismatch" {
  firrtl.module @ForceableTypeMismatch() {
    // expected-error @below {{reference result of incorrect type, found}}
    %w, %w_f = firrtl.wire forceable : !firrtl.uint, !firrtl.rwprobe<uint<2>>
  }
}

// -----

// Check rwprobe<const T> is rejected.
firrtl.circuit "ForceableConstWire" {
  firrtl.module @ForceableConstWire() {
    // expected-error @below {{forceable reference base type cannot contain const}}
    %w, %w_f = firrtl.wire forceable : !firrtl.const.uint, !firrtl.rwprobe<const.uint>
  }
}

// -----

// Check forceable declarations of const-type w/o explicit ref type are rejected.
firrtl.circuit "ForceableConstNode" {
  firrtl.module @ForceableConstNode() {
    %w = firrtl.wire : !firrtl.const.uint
    // expected-error @below {{cannot force a node of type}}
    %n, %n_ref = firrtl.node %w forceable : !firrtl.const.uint
  }
}

// -----

// Check forceable declarations of const-type w/o explicit ref type are rejected.
firrtl.circuit "ForceableBundleConstNode" {
  firrtl.module @ForceableBundleConstNode() {
    %w = firrtl.wire : !firrtl.bundle<a: const.uint>
    // expected-error @below {{cannot force a node of type}}
    %n, %n_ref = firrtl.node %w forceable : !firrtl.bundle<a: const.uint>
  }
}

// -----

firrtl.circuit "RefForceProbe" {
  firrtl.module @RefForceProbe() {
    %a = firrtl.wire : !firrtl.uint<1>
    %1 = firrtl.ref.send %a : !firrtl.uint<1>
    // expected-note @above {{prior use here}}
    // expected-error @below {{use of value '%1' expects different type than prior uses: '!firrtl.rwprobe<uint<1>>' vs '!firrtl.probe<uint<1>>'}}
    firrtl.ref.force_initial %a, %1, %a : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "RefReleaseProbe" {
  firrtl.module @RefReleaseProbe() {
    %a = firrtl.wire : !firrtl.uint<1>
    %1 = firrtl.ref.send %a : !firrtl.uint<1>
    // expected-error @below {{op operand #1 must be rwprobe type, but got '!firrtl.probe<uint<1>>'}}
    firrtl.ref.release_initial %a, %1 : !firrtl.uint<1>, !firrtl.probe<uint<1>>
  }
}


// -----

firrtl.circuit "EnumCreateNoCase" {
firrtl.module @EnumCreateNoCase(in %in : !firrtl.uint<8>) {
  // expected-error @below {{unknown field SomeOther in enum type}}
  %some = firrtl.enumcreate SomeOther(%in) : (!firrtl.uint<8>) -> !firrtl.enum<None: uint<0>, Some: uint<8>>
}

// -----

firrtl.circuit "EnumCreateWrongType" {
  // expected-note @below {{prior use here}}
firrtl.module @EnumCreateWrongType(in %in : !firrtl.uint<7>) {
  // expected-error @below {{expects different type than prior uses}}
  %some = firrtl.enumcreate Some(%in) : (!firrtl.uint<8>) -> !firrtl.enum<None: uint<0>, Some: uint<8>>
}

// -----

firrtl.circuit "SubtagNoCase" {
firrtl.module @SubtagNoCase(in %in : !firrtl.enum<None: uint<0>, Some: uint<8>>) {
  // expected-error @below {{unknown field SomeOther in enum type}}
  %some = firrtl.subtag %in[SomeOther] : !firrtl.enum<None: uint<0>, Some: uint<8>>
}

// -----
// 'const' firrtl.reg is invalid

firrtl.circuit "ConstReg" {
firrtl.module @ConstReg(in %clock: !firrtl.clock) {
  // expected-error @+1 {{'firrtl.reg' op result #0 must be a passive non-'const' base type that does not contain analog, but got '!firrtl.const.uint<1>'}}
  %r = firrtl.reg %clock : !firrtl.clock, !firrtl.const.uint<1>
}
}

// -----
// 'const' firrtl.reg is invalid

firrtl.circuit "ConstBundleReg" {
firrtl.module @ConstBundleReg(in %clock: !firrtl.clock) {
  // expected-error @+1 {{'firrtl.reg' op result #0 must be a passive non-'const' base type that does not contain analog, but got '!firrtl.const.bundle<a: uint<1>>'}}
  %r = firrtl.reg %clock : !firrtl.clock, !firrtl.const.bundle<a: uint<1>>
}
}

// -----
// 'const' firrtl.reg is invalid

firrtl.circuit "ConstVectorReg" {
firrtl.module @ConstVectorReg(in %clock: !firrtl.clock) {
  // expected-error @+1 {{'firrtl.reg' op result #0 must be a passive non-'const' base type that does not contain analog, but got '!firrtl.const.vector<uint<1>, 3>'}}
  %r = firrtl.reg %clock : !firrtl.clock, !firrtl.const.vector<uint<1>, 3>
}
}

// -----
// 'const' firrtl.reg is invalid

firrtl.circuit "ConstEnumReg" {
firrtl.module @ConstEnumReg(in %clock: !firrtl.clock) {
  // expected-error @+1 {{'firrtl.reg' op result #0 must be a passive non-'const' base type that does not contain analog, but got '!firrtl.const.enum<a: uint<1>>'}}
  %r = firrtl.reg %clock : !firrtl.clock, !firrtl.const.enum<a: uint<1>>
}
}

// -----
// 'const' firrtl.regreset is invalid

firrtl.circuit "ConstRegReset" {
firrtl.module @ConstRegReset(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %resetVal: !firrtl.const.uint<1>) {
  // expected-error @+1 {{'firrtl.regreset' op result #0 must be a passive non-'const' base type that does not contain analog, but got '!firrtl.const.uint<1>'}}
  %r = firrtl.regreset %clock, %reset, %resetVal : !firrtl.clock, !firrtl.asyncreset, !firrtl.const.uint<1>, !firrtl.const.uint<1>
}
}

// -----
// 'const' firrtl.regreset is invalid

firrtl.circuit "ConstRegReset" {
firrtl.module @ConstRegReset(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %resetVal: !firrtl.const.uint<1>) {
  // expected-error @+1 {{'firrtl.regreset' op result #0 must be a passive non-'const' base type that does not contain analog, but got '!firrtl.const.uint<1>'}}
  %r = firrtl.regreset %clock, %reset, %resetVal : !firrtl.clock, !firrtl.asyncreset, !firrtl.const.uint<1>, !firrtl.const.uint<1>
}
}

// -----
// 'const' firrtl.regreset is invalid

firrtl.circuit "ConstBundleRegReset" {
firrtl.module @ConstBundleRegReset(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %resetVal: !firrtl.const.bundle<a: uint<1>>) {
  // expected-error @+1 {{'firrtl.regreset' op result #0 must be a passive non-'const' base type that does not contain analog, but got '!firrtl.const.bundle<a: uint<1>>'}}
  %r = firrtl.regreset %clock, %reset, %resetVal : !firrtl.clock, !firrtl.asyncreset, !firrtl.const.bundle<a: uint<1>>, !firrtl.const.bundle<a: uint<1>>
}
}

// -----
// 'const' firrtl.regreset is invalid

firrtl.circuit "ConstVectorRegReset" {
firrtl.module @ConstVectorRegReset(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %resetVal: !firrtl.const.vector<uint<1>, 3>) {
  // expected-error @+1 {{'firrtl.regreset' op result #0 must be a passive non-'const' base type that does not contain analog, but got '!firrtl.const.vector<uint<1>, 3>'}}
  %r = firrtl.regreset %clock, %reset, %resetVal : !firrtl.clock, !firrtl.asyncreset, !firrtl.const.vector<uint<1>, 3>, !firrtl.const.vector<uint<1>, 3>
}
}

// -----
// 'const' firrtl.regreset is invalid

firrtl.circuit "ConstEnumRegReset" {
firrtl.module @ConstEnumRegReset(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %resetVal: !firrtl.const.enum<a: uint<1>>) {
  // expected-error @+1 {{'firrtl.regreset' op result #0 must be a passive non-'const' base type that does not contain analog, but got '!firrtl.const.enum<a: uint<1>>'}}
  %r = firrtl.regreset %clock, %reset, %resetVal : !firrtl.clock, !firrtl.asyncreset, !firrtl.const.enum<a: uint<1>>, !firrtl.const.enum<a: uint<1>>
}
}

// -----
// nested 'const' firrtl.reg is invalid

firrtl.circuit "BundleNestedConstReg" {
firrtl.module @BundleNestedConstReg(in %clock: !firrtl.clock) {
  // expected-error @+1 {{'firrtl.reg' op result #0 must be a passive non-'const' base type that does not contain analog, but got '!firrtl.bundle<a: const.uint<1>>'}}
  %r = firrtl.reg %clock : !firrtl.clock, !firrtl.bundle<a: const.uint<1>>
}
}

// -----
// nested 'const' firrtl.reg is invalid

firrtl.circuit "VectorNestedConstReg" {
firrtl.module @VectorNestedConstReg(in %clock: !firrtl.clock) {
  // expected-error @+1 {{'firrtl.reg' op result #0 must be a passive non-'const' base type that does not contain analog, but got '!firrtl.vector<const.uint<1>, 3>'}}
  %r = firrtl.reg %clock : !firrtl.clock, !firrtl.vector<const.uint<1>, 3>
}
}

// -----
// nested 'const' firrtl.regreset is invalid

firrtl.circuit "BundleNestedConstRegReset" {
firrtl.module @BundleNestedConstRegReset(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %resetVal: !firrtl.const.bundle<a: uint<1>>) {
  // expected-error @+1 {{'firrtl.regreset' op result #0 must be a passive non-'const' base type that does not contain analog, but got '!firrtl.bundle<a: const.uint<1>>'}}
  %r = firrtl.regreset %clock, %reset, %resetVal : !firrtl.clock, !firrtl.asyncreset, !firrtl.const.bundle<a: uint<1>>, !firrtl.bundle<a: const.uint<1>>
}
}

// -----
// 'const' firrtl.regreset is invalid

firrtl.circuit "VectorNestedConstRegReset" {
firrtl.module @VectorNestedConstRegReset(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %resetVal: !firrtl.vector<const.uint<1>, 3>) {
  // expected-error @+1 {{'firrtl.regreset' op result #0 must be a passive non-'const' base type that does not contain analog, but got '!firrtl.vector<const.uint<1>, 3>'}}
  %r = firrtl.regreset %clock, %reset, %resetVal : !firrtl.clock, !firrtl.asyncreset, !firrtl.vector<const.uint<1>, 3>, !firrtl.vector<const.uint<1>, 3>
}
}

// -----
// 'const' firrtl.regreset is invalid

firrtl.circuit "EnumNestedConstRegReset" {
firrtl.module @EnumNestedConstRegReset(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %resetVal: !firrtl.const.enum<a: uint<1>>) {
  // expected-error @+1 {{'firrtl.regreset' op result #0 must be a passive non-'const' base type that does not contain analog, but got '!firrtl.const.enum<a: uint<1>>'}}
  %r = firrtl.regreset %clock, %reset, %resetVal : !firrtl.clock, !firrtl.asyncreset, !firrtl.const.enum<a: uint<1>>, !firrtl.const.enum<a: uint<1>>
}
}

// -----
// const hardware firrtl.string is invalid

firrtl.circuit "ConstHardwareString" {
// expected-error @+1 {{strings cannot be const}}
firrtl.module @ConstHardwareString(in %string: !firrtl.const.string) {}
}

// -----

// Constcast non-const to const
firrtl.circuit "ConstcastNonConstToConst" {
  firrtl.module @ConstcastNonConstToConst(in %a: !firrtl.uint<1>) {
    // expected-error @+1 {{'!firrtl.uint<1>' is not 'const'-castable to '!firrtl.const.uint<1>'}}
    %b = firrtl.constCast %a : (!firrtl.uint<1>) -> !firrtl.const.uint<1>
  }
}

// -----

// Constcast non-const to const-containing
firrtl.circuit "ConstcastNonConstToConst" {
  firrtl.module @ConstcastNonConstToConst(in %a: !firrtl.bundle<a: uint<1>>) {
    // expected-error @+1 {{'!firrtl.bundle<a: uint<1>>' is not 'const'-castable to '!firrtl.bundle<a: const.uint<1>>'}}
    %b = firrtl.constCast %a : (!firrtl.bundle<a: uint<1>>) -> !firrtl.bundle<a: const.uint<1>>
  }
}

// -----

// Constcast different types
firrtl.circuit "ConstcastDifferentTypes" {
  firrtl.module @ConstcastDifferentTypes(in %a: !firrtl.const.uint<1>) {
    // expected-error @+1 {{'!firrtl.const.uint<1>' is not 'const'-castable to '!firrtl.sint<1>'}}
    %b = firrtl.constCast %a : (!firrtl.const.uint<1>) -> !firrtl.sint<1>
  }
}

// -----

// Bitcast non-const to const
firrtl.circuit "BitcastNonConstToConst" {
  firrtl.module @BitcastNonConstToConst(in %a: !firrtl.uint<1>) {
    // expected-error @+1 {{cannot cast non-'const' input type '!firrtl.uint<1>' to 'const' result type '!firrtl.const.sint<1>'}}
    %b = firrtl.bitcast %a : (!firrtl.uint<1>) -> !firrtl.const.sint<1>
  }
}

// -----

// Bitcast non-const to const-containing
firrtl.circuit "BitcastNonConstToConstContaining" {
  firrtl.module @BitcastNonConstToConstContaining(in %a: !firrtl.bundle<a: uint<1>>) {
    // expected-error @+1 {{cannot cast non-'const' input type '!firrtl.bundle<a: uint<1>>' to 'const' result type '!firrtl.bundle<a: const.sint<1>>'}}
    %b = firrtl.bitcast %a : (!firrtl.bundle<a: uint<1>>) -> !firrtl.bundle<a: const.sint<1>>
  }
}

// -----

// Uninferred reset cast non-const to const
firrtl.circuit "UninferredWidthCastNonConstToConst" {
  firrtl.module @UninferredWidthCastNonConstToConst(in %a: !firrtl.reset) {
    // expected-error @+1 {{operand constness must match}}
    %b = firrtl.resetCast %a : (!firrtl.reset) -> !firrtl.const.asyncreset
  }
}

// -----

// Primitive ops with all 'const' operands must have a 'const' result type
firrtl.circuit "PrimOpConstOperandsNonConstResult" {
firrtl.module @PrimOpConstOperandsNonConstResult(in %a: !firrtl.const.uint<4>, in %b: !firrtl.const.uint<4>) {
  // expected-error @below {{failed to infer returned types}}
  // expected-error @+1 {{'firrtl.and' op inferred type(s) '!firrtl.const.uint<4>' are incompatible with return type(s) of operation '!firrtl.uint<4>'}}
  %0 = firrtl.and %a, %b : (!firrtl.const.uint<4>, !firrtl.const.uint<4>) -> !firrtl.uint<4>
}
}

// -----

// Primitive ops with mixed 'const' operands must have a non-'const' result type
firrtl.circuit "PrimOpMixedConstOperandsConstResult" {
firrtl.module @PrimOpMixedConstOperandsConstResult(in %a: !firrtl.const.uint<4>, in %b: !firrtl.uint<4>) {
  // expected-error @below {{failed to infer returned types}}
  // expected-error @+1 {{'firrtl.and' op inferred type(s) '!firrtl.uint<4>' are incompatible with return type(s) of operation '!firrtl.const.uint<4>'}}
  %0 = firrtl.and %a, %b : (!firrtl.const.uint<4>, !firrtl.uint<4>) -> !firrtl.const.uint<4>
}
}

// -----

// A 'const' bundle can only be created with 'const' operands
firrtl.circuit "ConstBundleCreateNonConstOperands" {
firrtl.module @ConstBundleCreateNonConstOperands(in %a: !firrtl.uint<1>) {
  // expected-error @+1 {{type of element doesn't match bundle for field "a"}}
  %0 = firrtl.bundlecreate %a : (!firrtl.uint<1>) -> !firrtl.const.bundle<a: uint<1>>
}
}

// -----

// A 'const' vector can only be created with 'const' operands
firrtl.circuit "ConstVectorCreateNonConstOperands" {
firrtl.module @ConstVectorCreateNonConstOperands(in %a: !firrtl.uint<1>) {
  // expected-error @+1 {{type of element doesn't match vector element}}
  %0 = firrtl.vectorcreate %a : (!firrtl.uint<1>) -> !firrtl.const.vector<uint<1>, 1>
}
}

// -----

// A 'const' enum can only be created with 'const' operands
firrtl.circuit "ConstEnumCreateNonConstOperands" {
firrtl.module @ConstEnumCreateNonConstOperands(in %a: !firrtl.uint<1>) {
  // expected-error @+1 {{type of element doesn't match enum element}}
  %0 = firrtl.enumcreate Some(%a) : (!firrtl.uint<1>) -> !firrtl.const.enum<None: uint<0>, Some: uint<1>>
}
}

// -----

// Enum types must be passive
firrtl.circuit "EnumNonPassive" {
  // expected-error @+1 {{enum field '"a"' not passive}}
  firrtl.module @EnumNonPassive(in %a : !firrtl.enum<a: bundle<a flip: uint<1>>>) {
  }
}

// -----

// Enum types must not contain analog
firrtl.circuit "EnumAnalog" {
  // expected-error @+1 {{enum field '"a"' contains analog}}
  firrtl.module @EnumAnalog(in %a : !firrtl.enum<a: analog<1>>) {
  }
}

// -----

// An enum that contains 'const' elements must be 'const'
firrtl.circuit "NonConstEnumConstElements" {
// expected-error @+1 {{enum with 'const' elements must be 'const'}}
firrtl.module @NonConstEnumConstElements(in %a: !firrtl.enum<None: uint<0>, Some: const.uint<1>>) {}
}

// -----
// No const with probes within.

firrtl.circuit "ConstOpenVector" {
  // expected-error @below {{vector cannot be const with references}}
  firrtl.extmodule @ConstOpenVector(out out : !firrtl.const.openvector<probe<uint<1>>, 2>)
}

// -----
// Elements must support FieldID's.

firrtl.circuit "OpenVectorNotFieldID" {
  // expected-error @below {{vector element type does not support fieldID's, type: '!firrtl.string'}}
  firrtl.extmodule @OpenVectorNotFieldID(out out : !firrtl.openvector<string, 2>)
}

// -----
// No const with probes within.

firrtl.circuit "ConstOpenBundle" {
  // expected-error @below {{'const' bundle cannot have references, but element "x" has type '!firrtl.probe<uint<1>>'}}
  firrtl.extmodule @ConstOpenBundle(out out : !firrtl.const.openbundle<x: probe<uint<1>>>)
}

// -----
// Elements must support FieldID's.

firrtl.circuit "OpenBundleNotFieldID" {
  // expected-error @below {{bundle element "a" has unsupported type that does not support fieldID's: '!firrtl.string'}}
  firrtl.extmodule @OpenBundleNotFieldID(out out : !firrtl.openbundle<a: string>)
}

// -----
// Strict connect between non-equivalent anonymous type operands.

firrtl.circuit "NonEquivalenctStrictConnect" {
  firrtl.module @NonEquivalenctStrictConnect(in %in: !firrtl.uint<1>, out %out: !firrtl.alias<foo, uint<2>>) {
    // expected-error @below {{op failed to verify that operands must be structurally equivalent}}
    firrtl.strictconnect %out, %in: !firrtl.alias<foo, uint<2>>, !firrtl.uint<1>
  }
}

// -----
// Classes cannot be the top module.

// expected-error @below {{'firrtl.circuit' op must have a non-class top module}}
firrtl.circuit "TopModuleIsClass" {
  firrtl.class @TopModuleIsClass() {}
}

// -----
// Classes cannot have hardware ports.

firrtl.circuit "ClassCannotHaveHardwarePorts" {
  firrtl.module @ClassCannotHaveHardwarePorts() {}
  // expected-error @below {{'firrtl.class' op ports on a class must be properties}}
  firrtl.class @ClassWithHardwarePort(in %in: !firrtl.uint<8>) {}
}

// -----
// Classes cannot hold hardware things.

firrtl.circuit "ClassCannotHaveWires" {
  firrtl.module @ClassCannotHaveWires() {}
  firrtl.class @ClassWithWire() {
    // expected-error @below {{'firrtl.wire' op expects parent op to be one of 'firrtl.module, firrtl.group, firrtl.when, firrtl.match'}}
    %w = firrtl.wire : !firrtl.uint<8>
  }
}

// -----

// A group definition, "@A::@B", is missing an outer nesting of a group
// definition with symbol "@A".
firrtl.circuit "GroupMissingNesting" {
  firrtl.declgroup @A bind {
    firrtl.declgroup @B bind {}
  }
  // expected-note @below {{illegal parent op defined here}}
  firrtl.module @GroupMissingNesting() {
    // expected-error @below {{'firrtl.group' op has a nested group symbol, but does not have a 'firrtl.group' op as a parent}}
    firrtl.group @A::@B {}
  }
}

// -----

// A group definition with a legal symbol, "@B", is illegaly nested under
// another group with a legal symbol, "@B".
firrtl.circuit "UnnestedGroup" {
  firrtl.declgroup @A bind {}
  firrtl.declgroup @B bind {}
  firrtl.module @UnnestedGroup() {
    // expected-note @below {{illegal parent op defined here}}
    firrtl.group @A {
      // expected-error @below {{'firrtl.group' op has an un-nested group symbol, but does not have a 'firrtl.module' op as a parent}}
      firrtl.group @B {}
    }
  }
}

// -----

// A group definition, "@B::@C", is nested under the wrong group, "@A".
firrtl.circuit "WrongGroupNesting" {
  firrtl.declgroup @A bind {}
  firrtl.declgroup @B bind {
    firrtl.declgroup @C bind {}
  }
  firrtl.module @WrongGroupNesting() {
    // expected-note @below {{illegal parent group defined here}}
    firrtl.group @A {
      // expected-error @below {{'firrtl.group' op is nested under an illegal group}}
      firrtl.group @B::@C {}
    }
  }
}

// -----

// A group captures a type which is not a FIRRTL base type.
firrtl.circuit "NonBaseTypeCapture" {
  firrtl.declgroup @A bind {}
  // expected-note @below {{operand is defined here}}
  firrtl.module @NonBaseTypeCapture(in %a: !firrtl.probe<uint<1>>) {
    // expected-error @below {{'firrtl.group' op captures an operand which is not a FIRRTL base type}}
    firrtl.group @A {
      // expected-note @below {{operand is used here}}
      %b = firrtl.ref.resolve %a : !firrtl.probe<uint<1>>
    }
  }
}

// -----

// A group captures a non-passive type.
firrtl.circuit "NonPassiveCapture" {
  firrtl.declgroup @A bind {}
  firrtl.module @NonPassiveCapture() {
    // expected-note @below {{operand is defined here}}
    %a = firrtl.wire : !firrtl.bundle<a flip: uint<1>>
    // expected-error @below {{'firrtl.group' op captures an operand which is not a passive type}}
    firrtl.group @A {
      %b = firrtl.wire : !firrtl.bundle<a flip: uint<1>>
      // expected-note @below {{operand is used here}}
      firrtl.connect %b, %a : !firrtl.bundle<a flip: uint<1>>, !firrtl.bundle<a flip: uint<1>>
    }
  }
}

// -----

// A group may not drive sinks outside the group.
firrtl.circuit "GroupDrivesSinksOutside" {
  firrtl.declgroup @A bind {}
  firrtl.module @GroupDrivesSinksOutside(in %cond : !firrtl.uint<1>) {
    %a = firrtl.wire : !firrtl.uint<1>
    // expected-note @below {{destination is defined here}}
    %b = firrtl.wire : !firrtl.bundle<c: uint<1>>
    // expected-note @below {{enclosing group is defined here}}
    firrtl.group @A {
      firrtl.when %cond : !firrtl.uint<1> {
        %b_c = firrtl.subfield %b[c] : !firrtl.bundle<c: uint<1>>
        // expected-error @below {{'firrtl.strictconnect' op connects to a destination which is defined outside its enclosing group}}
        firrtl.strictconnect %b_c, %a : !firrtl.uint<1>
      }
    }
  }
}

// -----

firrtl.circuit "RWProbeRemote" {
  firrtl.module @Other() {
    %w = firrtl.wire sym @x : !firrtl.uint<1>
  }
  firrtl.module @RWProbeRemote() {
    // expected-error @below {{op has non-local target}}
    %rw = firrtl.ref.rwprobe <@Other::@x> : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "RWProbeNonBase" {
  firrtl.module @RWProbeNonBase() {
    // expected-error @below {{cannot force type '!firrtl.string'}}
    %rw = firrtl.ref.rwprobe <@RWProbeTypes::@invalid> : !firrtl.string
  }
}

// -----

firrtl.circuit "RWProbeTypes" {
  firrtl.module @RWProbeTypes() {
    // expected-note @below {{target resolves here}}
    %w = firrtl.wire sym @x : !firrtl.sint<1>
    // expected-error @below {{op has type mismatch: target resolves to '!firrtl.sint<1>' instead of expected '!firrtl.uint<1>'}}
    %rw = firrtl.ref.rwprobe <@RWProbeTypes::@x> : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "RWProbeUninferred" {
  firrtl.module @RWProbeUninferred() {
    %w = firrtl.wire sym @x : !firrtl.uint
    // expected-error @below {{op attribute 'type' failed to satisfy constraint: type attribute of RWProbeTarget type (a FIRRTL base type with a known width and not abstract reset)}}
    %rw = firrtl.ref.rwprobe <@RWProbeUninferred::@x> : !firrtl.uint
  }
}

// -----

firrtl.circuit "RWProbeUninferredReset" {
  firrtl.module @RWProbeUninferredReset() {
    %w = firrtl.wire sym @x : !firrtl.bundle<a: reset>
    // expected-error @below {{op attribute 'type' failed to satisfy constraint: type attribute of RWProbeTarget type (a FIRRTL base type with a known width and not abstract reset)}}
    %rw = firrtl.ref.rwprobe <@RWProbeUninferred::@x> : !firrtl.bundle<a: reset>
  }
}

// -----

firrtl.circuit "RWProbeInstance" {
  firrtl.extmodule @Ext()
  firrtl.module @RWProbeInstance() {
    // expected-note @below {{target resolves here}}
    firrtl.instance inst sym @inst @Ext()
    // expected-error @below {{op has target that cannot be probed}}
    %rw = firrtl.ref.rwprobe <@RWProbeInstance::@inst> : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "MissingClassForObjectPortInModule" {
  // expected-error @below {{'firrtl.module' op target class 'Missing' not found}}
  firrtl.module @MissingClassForObjectPortInModule(out %o: !firrtl.class<@Missing()>) {}
}

// -----

firrtl.circuit "MissingClassForObjectPortInClass" {
  firrtl.module @MissingClassForObjectPortInClass() {}
  // expected-error @below {{'firrtl.class' op target class 'Missing' not found}}
  firrtl.class @MyClass(out %o: !firrtl.class<@Missing()>) {}
}

// -----

firrtl.circuit "MissingClassForObjectDeclaration" {
  firrtl.module @ObjectClassMissing() {
    // expected-error @below {{'firrtl.object' op target class 'Missing' not found}}
    %0 = firrtl.object @Missing()
  }
}

// -----

firrtl.circuit "ClassTypeWrongPortName" {
  firrtl.class @MyClass(out %str: !firrtl.string) {}
  // expected-error @below {{'firrtl.module' op port #0 has wrong name, got "xxx", expected "str"}}
  firrtl.module @ClassTypeWrongPortName(out %port: !firrtl.class<@MyClass(out xxx: !firrtl.string)>) {}
}

// -----

firrtl.circuit "ClassTypeWrongPortDir" {
  firrtl.class @MyClass(out %str: !firrtl.string) {}
    // expected-error @below {{'firrtl.module' op port "str" has wrong direction, got in, expected out}}
  firrtl.module @ClassTypeWrongPortDir(out %port: !firrtl.class<@MyClass(in str: !firrtl.string)>) {}
}

// -----

firrtl.circuit "ClassTypeWrongPortType" {
  firrtl.class @MyClass(out %str: !firrtl.string) {}
  // expected-error @below {{'firrtl.module' op port "str" has wrong type, got '!firrtl.integer', expected '!firrtl.string'}}
  firrtl.module @ClassTypeWrongPortType(out %port: !firrtl.class<@MyClass(out str: !firrtl.integer)>) {}
}

// -----

firrtl.circuit "ConstClassType" {
  firrtl.class @MyClass(out %str: !firrtl.string) {}
  // expected-error @below {{classes cannot be const}}
  firrtl.module @ConstClassType(out %port: !firrtl.const.class<@MyClass(in str: !firrtl.string)>) {}
}
