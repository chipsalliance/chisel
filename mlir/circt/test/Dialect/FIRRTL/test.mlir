// RUN: circt-opt %s | FileCheck %s

firrtl.circuit "MyModule" {

firrtl.module @mod() { }
firrtl.extmodule @extmod()
firrtl.memmodule @memmod () attributes {
  depth = 16 : ui64, dataWidth = 1 : ui32, extraPorts = [],
  maskBits = 0 : ui32, numReadPorts = 0 : ui32, numWritePorts = 0 : ui32,
  numReadWritePorts = 0 : ui32, readLatency = 0 : ui32,
  writeLatency = 1 : ui32}

// Constant op supports different return types.
firrtl.module @Constants() {
  // CHECK: %c0_ui0 = firrtl.constant 0 : !firrtl.uint<0>
  firrtl.constant 0 : !firrtl.uint<0>
  // CHECK: %c0_si0 = firrtl.constant 0 : !firrtl.sint<0>
  firrtl.constant 0 : !firrtl.sint<0>
  // CHECK: %c4_ui8 = firrtl.constant 4 : !firrtl.uint<8>
  firrtl.constant 4 : !firrtl.uint<8>
  // CHECK: %c-4_si16 = firrtl.constant -4 : !firrtl.sint<16>
  firrtl.constant -4 : !firrtl.sint<16>
  // CHECK: %c1_clock = firrtl.specialconstant 1 : !firrtl.clock
  firrtl.specialconstant 1 : !firrtl.clock
  // CHECK: %c1_reset = firrtl.specialconstant 1 : !firrtl.reset
  firrtl.specialconstant 1 : !firrtl.reset
  // CHECK: %c1_asyncreset = firrtl.specialconstant 1 : !firrtl.asyncreset
  firrtl.specialconstant 1 : !firrtl.asyncreset
  // CHECK: firrtl.constant 4 : !firrtl.uint<8> {name = "test"}
  firrtl.constant 4 : !firrtl.uint<8> {name = "test"}

  firrtl.aggregateconstant [1, 2, 3] : !firrtl.bundle<a: uint<8>, b: uint<5>, c: uint<4>>
  firrtl.aggregateconstant [1, 2, 3] : !firrtl.vector<uint<8>, 3>
  firrtl.aggregateconstant [[1, 2], [3, 4]] : !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>

}

//module MyModule :
//  input in: UInt<8>
//  output out: UInt<8>
//  out <= in
firrtl.module @MyModule(in %in : !firrtl.uint<8>,
                        out %out : !firrtl.uint<8>) {
  firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
}

// CHECK-LABEL: firrtl.module @MyModule(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>)
// CHECK-NEXT:    firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
// CHECK-NEXT:  }


//circuit Top :
//  module Top :
//    output out:UInt
//    input b:UInt<32>
//    input c:Analog<13>
//    input d:UInt<16>
//    out <= add(b,d)

firrtl.circuit "Top" {
  firrtl.module @Top(out %out: !firrtl.uint,
                     in %b: !firrtl.uint<32>,
                     in %c: !firrtl.analog<13>,
                     in %d: !firrtl.uint<16>) {
    %3 = firrtl.add %b, %d : (!firrtl.uint<32>, !firrtl.uint<16>) -> !firrtl.uint<33>

    %4 = firrtl.invalidvalue : !firrtl.analog<13>
    firrtl.attach %c, %4 : !firrtl.analog<13>, !firrtl.analog<13>
    %5 = firrtl.add %3, %d : (!firrtl.uint<33>, !firrtl.uint<16>) -> !firrtl.uint<34>

    firrtl.connect %out, %5 : !firrtl.uint, !firrtl.uint<34>
  }
}

// CHECK-LABEL: firrtl.circuit "Top" {
// CHECK-NEXT:    firrtl.module @Top(out %out: !firrtl.uint,
// CHECK:                            in %b: !firrtl.uint<32>, in %c: !firrtl.analog<13>, in %d: !firrtl.uint<16>) {
// CHECK-NEXT:      %0 = firrtl.add %b, %d : (!firrtl.uint<32>, !firrtl.uint<16>) -> !firrtl.uint<33>
// CHECK-NEXT:      %invalid_analog13 = firrtl.invalidvalue : !firrtl.analog<13>
// CHECK-NEXT:      firrtl.attach %c, %invalid_analog13 : !firrtl.analog<13>, !firrtl.analog<13>
// CHECK-NEXT:      %1 = firrtl.add %0, %d : (!firrtl.uint<33>, !firrtl.uint<16>) -> !firrtl.uint<34>
// CHECK-NEXT:      firrtl.connect %out, %1 : !firrtl.uint, !firrtl.uint<34>
// CHECK-NEXT:    }
// CHECK-NEXT:  }


// Test some hard cases of name handling.
firrtl.module @Mod2(in %in : !firrtl.uint<8>,
                    out %out : !firrtl.uint<8>) attributes {portNames = ["some_name", "out"]}{
  firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
}

// CHECK-LABEL: firrtl.module @Mod2(in %some_name: !firrtl.uint<8>,
// CHECK:                           out %out: !firrtl.uint<8>)
// CHECK-NEXT:    firrtl.connect %out, %some_name : !firrtl.uint<8>, !firrtl.uint<8>
// CHECK-NEXT:  }

// Check that quotes port names are paresable and printed with quote only if needed.
// CHECK: firrtl.extmodule @TrickyNames(in "777": !firrtl.uint, in abc: !firrtl.uint)
firrtl.extmodule @TrickyNames(in "777": !firrtl.uint, in "abc": !firrtl.uint)

// Modules may be completely empty.
// CHECK-LABEL: firrtl.module @no_ports() {
firrtl.module @no_ports() {
}

// stdIntCast can work with clock inputs/outputs too.
// CHECK-LABEL: @ClockCast
firrtl.module @ClockCast(in %clock: !firrtl.clock) {
  // CHECK: %0 = builtin.unrealized_conversion_cast %clock : !firrtl.clock to i1
  %0 = builtin.unrealized_conversion_cast %clock : !firrtl.clock to i1

  // CHECK: %1 = builtin.unrealized_conversion_cast %0 : i1 to !firrtl.clock
  %1 = builtin.unrealized_conversion_cast %0 : i1 to !firrtl.clock
}


// CHECK-LABEL: @TestDshRL
firrtl.module @TestDshRL(in %in1 : !firrtl.uint<2>, in %in2: !firrtl.uint<3>) {
  // CHECK: %0 = firrtl.dshl %in1, %in2 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<9>
  %0 = firrtl.dshl %in1, %in2 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<9>

  // CHECK: %1 = firrtl.dshr %in1, %in2 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<2>
  %1 = firrtl.dshr %in1, %in2 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<2>

  // CHECK: %2 = firrtl.dshlw %in1, %in2 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<2>
  %2 = firrtl.dshlw %in1, %in2 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<2>
}

// We allow implicit truncation of a register's reset value.
// CHECK-LABEL: @RegResetTruncation
firrtl.module @RegResetTruncation(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %value: !firrtl.bundle<a: uint<2>>, out %out: !firrtl.bundle<a: uint<1>>) {
  %r2 = firrtl.regreset %clock, %reset, %value  : !firrtl.clock, !firrtl.uint<1>, !firrtl.bundle<a: uint<2>>, !firrtl.bundle<a: uint<1>>
  firrtl.connect %out, %r2 : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
}

// CHECK-LABEL: @TestNodeName
firrtl.module @TestNodeName(in %in1 : !firrtl.uint<8>) {
  // CHECK: %n1 = firrtl.node %in1 : !firrtl.uint<8>
  %n1 = firrtl.node %in1 : !firrtl.uint<8>

  // CHECK: %n1_0 = firrtl.node %in1 {name = "n1"} : !firrtl.uint<8>
  %n2 = firrtl.node %in1 {name = "n1"} : !firrtl.uint<8>
}

// Basic test for NLA operations.
// CHECK: hw.hierpath private @nla [@Parent::@child, @Child]
hw.hierpath private @nla [@Parent::@child, @Child]
firrtl.module @Child() {
  %w = firrtl.wire sym @w : !firrtl.uint<1>
}
firrtl.module @Parent() {
  firrtl.instance child sym @child @Child()
}

// CHECK-LABEL: @VerbatimExpr
firrtl.module @VerbatimExpr() {
  // CHECK: %[[TMP:.+]] = firrtl.verbatim.expr "FOO" : () -> !firrtl.uint<42>
  // CHECK: %[[TMP2:.+]] = firrtl.verbatim.expr "$bits({{[{][{]0[}][}]}})"(%[[TMP]]) : (!firrtl.uint<42>) -> !firrtl.uint<32>
  // CHECK: firrtl.add %[[TMP]], %[[TMP2]] : (!firrtl.uint<42>, !firrtl.uint<32>) -> !firrtl.uint<43>
  %0 = firrtl.verbatim.expr "FOO" : () -> !firrtl.uint<42>
  %1 = firrtl.verbatim.expr "$bits({{0}})"(%0) : (!firrtl.uint<42>) -> !firrtl.uint<32>
  %2 = firrtl.add %0, %1 : (!firrtl.uint<42>, !firrtl.uint<32>) -> !firrtl.uint<43>
}

// CHECK-LABEL: @LowerToBind
// CHECK: firrtl.instance foo sym @s1 {lowerToBind} @InstanceLowerToBind()
firrtl.module @InstanceLowerToBind() {}
firrtl.module @LowerToBind() {
  firrtl.instance foo sym @s1 {lowerToBind} @InstanceLowerToBind()
}

// CHECK-LABEL: firrtl.module @InnerSymAttr
firrtl.module @InnerSymAttr() {
  %w = firrtl.wire sym [<@w,2,public>,<@x,1,private>,<@syh,4,public>] : !firrtl.bundle<a: uint<1>, b: uint<1>, c: uint<1>, d: uint<1>>
  // CHECK: %w = firrtl.wire sym [<@x,1,private>, <@w,2,public>, <@syh,4,public>]
  %w1 = firrtl.wire sym [<@w1,0,public>] : !firrtl.bundle<a: uint<1>, b: uint<1>, c: uint<1>, d: uint<1>>
  // CHECK: %w1 = firrtl.wire sym @w1
  %w2 = firrtl.wire sym [<@w2,0,private>] : !firrtl.bundle<a: uint<1>, b: uint<1>, c: uint<1>, d: uint<1>>
  // CHECK: %w2 = firrtl.wire sym [<@w2,0,private>]
  %w3, %w3_ref = firrtl.wire sym [<@w3,2,public>,<@x2,1,private>,<@syh2,0,public>] forceable : !firrtl.bundle<a: uint<1>, b: uint<1>, c: uint<1>, d: uint<1>>, !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>, c: uint<1>, d: uint<1>>>
  // CHECK: %w3, %w3_ref = firrtl.wire sym [<@syh2,0,public>, <@x2,1,private>, <@w3,2,public>]
}

// CHECK-LABEL: firrtl.module @EnumTest
firrtl.module @EnumTest(in %in : !firrtl.enum<a: uint<1>, b: uint<2>>,
                        out %out : !firrtl.uint<2>, out %tag : !firrtl.uint<1>) {
  %v = firrtl.subtag %in[b] : !firrtl.enum<a: uint<1>, b: uint<2>>
  // CHECK: = firrtl.subtag %in[b] : !firrtl.enum<a: uint<1>, b: uint<2>>

  %t = firrtl.tagextract %in : !firrtl.enum<a: uint<1>, b: uint<2>>
  // CHECK: = firrtl.tagextract %in : !firrtl.enum<a: uint<1>, b: uint<2>>

  firrtl.strictconnect %out, %v : !firrtl.uint<2>
  firrtl.strictconnect %tag, %t : !firrtl.uint<1>

  %p = firrtl.istag %in a : !firrtl.enum<a: uint<1>, b: uint<2>>
  // CHECK: = firrtl.istag %in a : !firrtl.enum<a: uint<1>, b: uint<2>>

  %c1_ui8 = firrtl.constant 1 : !firrtl.uint<8>
  %some = firrtl.enumcreate Some(%c1_ui8) : (!firrtl.uint<8>) -> !firrtl.enum<None: uint<0>, Some: uint<8>>
  // CHECK: = firrtl.enumcreate Some(%c1_ui8) : (!firrtl.uint<8>) -> !firrtl.enum<None: uint<0>, Some: uint<8>>

  firrtl.match %in : !firrtl.enum<a: uint<1>, b: uint<2>> {
    case a(%arg0) {
      %w = firrtl.wire : !firrtl.uint<1>
    }
    case b(%arg0) {
      %x = firrtl.wire : !firrtl.uint<1>
    }
  }
  // CHECK: firrtl.match %in : !firrtl.enum<a: uint<1>, b: uint<2>> {
  // CHECK:   case a(%arg0) {
  // CHECK:     %w = firrtl.wire : !firrtl.uint<1>
  // CHECK:   }
  // CHECK:   case b(%arg0) {
  // CHECK:     %x = firrtl.wire : !firrtl.uint<1>
  // CHECK:   }
  // CHECK: }

}

// CHECK-LABEL: OpenAggTest
// CHECK-SAME: !firrtl.openbundle<a: bundle<data: uint<1>>, b: openvector<openbundle<x: uint<2>, y: probe<uint<2>>>, 2>>
firrtl.module @OpenAggTest(in %in: !firrtl.openbundle<a: bundle<data: uint<1>>, b: openvector<openbundle<x: uint<2>, y: probe<uint<2>>>, 2>>) {
  %a = firrtl.opensubfield %in[a] : !firrtl.openbundle<a: bundle<data: uint<1>>, b: openvector<openbundle<x: uint<2>, y: probe<uint<2>>>, 2>>
  %data = firrtl.subfield %a[data] : !firrtl.bundle<data: uint<1>>
  %b = firrtl.opensubfield %in[b] : !firrtl.openbundle<a: bundle<data: uint<1>>, b: openvector<openbundle<x: uint<2>, y: probe<uint<2>>>, 2>>
  %b_0 = firrtl.opensubindex %b[0] : !firrtl.openvector<openbundle<x: uint<2>, y: probe<uint<2>>>, 2>
  %b_1 = firrtl.opensubindex %b[1] : !firrtl.openvector<openbundle<x: uint<2>, y: probe<uint<2>>>, 2>
  %b_0_y = firrtl.opensubfield %b_0[y] : !firrtl.openbundle<x : uint<2>, y: probe<uint<2>>>
}

// CHECK-LABEL: StringTest
// CHECK-SAME:  (in %in: !firrtl.string, out %out: !firrtl.string)
firrtl.module @StringTest(in %in: !firrtl.string, out %out: !firrtl.string) {
  firrtl.propassign %out, %in : !firrtl.string
  // CHECK: %0 = firrtl.string "hello"
  %0 = firrtl.string "hello"
}

// CHECK-LABEL: BigIntTest
// CHECK-SAME:  (in %in: !firrtl.integer, out %out: !firrtl.integer)
firrtl.module @BigIntTest(in %in: !firrtl.integer, out %out: !firrtl.integer) {
  firrtl.propassign %out, %in : !firrtl.integer

  // CHECK: %0 = firrtl.integer 4
  %0 = firrtl.integer 4
  // CHECK: %1 = firrtl.integer -4
  %1 = firrtl.integer -4
}

// CHECK-LABEL: ListTest
// CHECK-SAME:  (in %in: !firrtl.list<string>, out %out: !firrtl.list<string>)
firrtl.module @ListTest(in %in: !firrtl.list<string>, out %out: !firrtl.list<string>) {
  firrtl.propassign %out, %in : !firrtl.list<string>
}

// CHECK-LABEL: MapTest
// CHECK-SAME:  (in %in: !firrtl.map<integer, string>, out %out: !firrtl.map<integer, string>)
firrtl.module @MapTest(in %in: !firrtl.map<integer, string>, out %out: !firrtl.map<integer, string>) {
  firrtl.propassign %out, %in : !firrtl.map<integer, string>
}

// CHECK-LABEL: PropertyNestedTest
// CHECK-SAME:  (in %in: !firrtl.map<integer, list<map<string, integer>>>, out %out: !firrtl.map<integer, list<map<string, integer>>>)
firrtl.module @PropertyNestedTest(in %in: !firrtl.map<integer, list<map<string, integer>>>, out %out: !firrtl.map<integer, list<map<string, integer>>>) {
  firrtl.propassign %out, %in : !firrtl.map<integer, list<map<string, integer>>>
}

// CHECK-LABEL: firrtl.module @PathTest
// CHECK-SAME: (in %in: !firrtl.path, out %out: !firrtl.path)
firrtl.module @PathTest(in %in: !firrtl.path, out %out: !firrtl.path) {
  firrtl.propassign %out, %in : !firrtl.path
}

// CHECK-LABEL: TypeAlias
// CHECK-SAME: %in: !firrtl.alias<bar, uint<1>>
// CHECK-SAME: %const: !firrtl.const.alias<baz, const.uint<1>>
// CHECK-SAME: %r: !firrtl.openbundle<a: alias<baz, uint<1>>>
// CHECK-SAME: %out: !firrtl.alias<foo, uint<1>>
// CHECK-NEXT: firrtl.strictconnect %out, %in : !firrtl.alias<foo, uint<1>>, !firrtl.alias<bar, uint<1>>

firrtl.module @TypeAlias(in %in: !firrtl.alias<bar, uint<1>>,
                         in %const: !firrtl.const.alias<baz, const.uint<1>>,
                         out %r : !firrtl.openbundle<a: alias<baz, uint<1>>>,
                         out %out: !firrtl.alias<foo, uint<1>>) {
  firrtl.strictconnect %out, %in: !firrtl.alias<foo, uint<1>>, !firrtl.alias<bar, uint<1>>
}

}
