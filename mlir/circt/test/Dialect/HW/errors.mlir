// RUN: circt-opt %s -split-input-file -verify-diagnostics

func.func private @test_extract(%arg0: i4) {
  // expected-error @+1 {{'comb.extract' op from bit too large for input}}
  %a = comb.extract %arg0 from 6 : (i4) -> i3
}

// -----

func.func private @test_extract(%arg0: i4) {
  // expected-error @+1 {{'comb.extract' op from bit too large for input}}
  %b = comb.extract %arg0 from 2 : (i4) -> i3
}

// -----

func.func private @test_and() {
  // expected-error @+1 {{'comb.and' op expected 1 or more operands}}
  %b = comb.and : i111
}

// -----

hw.module @InnerSymVisibility() {
  // expected-error @+1 {{expected 'public', 'private', or 'nested'}}
  %wire = hw.wire %wire sym [<@x, 1, oops>] : i1
}

// -----

func.func private @notModule () {
  return
}

hw.module @A(%arg0: i1) {
  // expected-error @+1 {{symbol reference 'notModule' isn't a module}}
  hw.instance "foo" @notModule(a: %arg0: i1) -> ()
}

// -----

hw.module @A(%arg0: i1) {
  // expected-error @+1 {{Cannot find module definition 'doesNotExist'}}
  hw.instance "b1" @doesNotExist(a: %arg0: i1) -> ()
}

// -----

hw.generator.schema @S, "Test Schema", ["test"]
// expected-error @+1 {{Cannot find generator definition 'S2'}}
hw.module.generated @A, @S2(%arg0: i1) -> (a: i1) attributes { test = 1 }

// -----

hw.module @S() { }
// expected-error @+1 {{which is not a HWGeneratorSchemaOp}}
hw.module.generated @A, @S(%arg0: i1) -> (a: i1) attributes { test = 1 }


// -----

// expected-error @+1 {{'hw.output' op must have same number of operands as region results}}
hw.module @A() -> ("": i1) { }

// -----

// expected-error @+1 {{expected non-function type}}
func.func private @arrayDims(%a: !hw.array<3 x 4 x i5>) { }

// -----

// expected-error @+1 {{invalid element for hw.inout type}}
func.func private @invalidInout(%arg0: !hw.inout<tensor<*xf32>>) { }

// -----

hw.module @inout(%a: i42) {
  // expected-error @+1 {{'input' must be InOutType, but got 'i42'}}
  %aget = sv.read_inout %a: i42
}

// -----

hw.module @wire(%a: i42) {
  // expected-error @+1 {{'sv.wire' op result #0 must be InOutType, but got 'i42'}}
  %aget = sv.wire: i42
}

// -----

hw.module @struct(%a: i42) {
  // expected-error @+1 {{custom op 'hw.struct_create' expected !hw.struct type or alias}}
  %aget = hw.struct_create(%a) : i42
}

// -----

hw.module @struct(%a: !hw.struct<foo: i42>) {
  // expected-error @+1 {{custom op 'hw.struct_explode' invalid kind of type specified}}
  %aget = hw.struct_explode %a : i42
}

// -----

hw.module @struct(%a: !hw.struct<foo: i42>) {
  // expected-error @+1 {{custom op 'hw.struct_extract' invalid kind of type specified}}
  %aget = hw.struct_extract %a["foo"] : i42
}

// -----

hw.module @struct(%a: !hw.struct<foo: i42>) {
  // expected-error @+1 {{custom op 'hw.struct_extract' invalid field name specified}}
  %aget = hw.struct_extract %a["bar"] : !hw.struct<foo: i42>
}

// -----

hw.module @struct(%a: !hw.struct<foo: i42>, %b: i42) {
  // expected-error @+1 {{custom op 'hw.struct_inject' invalid kind of type specified}}
  %aget = hw.struct_inject %a["foo"], %b : i42
}

// -----

hw.module @struct(%a: !hw.struct<foo: i42>, %b: i42) {
  // expected-error @+1 {{custom op 'hw.struct_inject' invalid field name specified}}
  %aget = hw.struct_inject %a["bar"], %b : !hw.struct<foo: i42>
}

// -----

hw.module @union(%b: i42) {
  // expected-error @+1 {{custom op 'hw.union_create' cannot find union field 'bar'}}
  %u = hw.union_create "bar", %a : !hw.union<foo: i42>
}

// -----

// expected-note @+1 {{module declared here}}
hw.module @empty() -> () {
  hw.output
}

hw.module @test() -> () {
  // expected-error @+1 {{'hw.instance' op has a wrong number of results; expected 0 but got 3}}
  %0, %1, %3 = hw.instance "test" @empty() -> (a: i2, b: i2, c: i2)
  hw.output
}

// -----

// expected-note @+1 {{module declared here}}
hw.module @f() -> (a: i2) {
  %a = hw.constant 1 : i2
  hw.output %a : i2
}

hw.module @test() -> () {
  // expected-error @+1 {{'hw.instance' op result type #0 must be 'i2', but got 'i1'}}
  %0 = hw.instance "test" @f() -> (a: i1)
  hw.output
}

// -----

// expected-note @+1 {{module declared here}}
hw.module @empty() -> () {
  hw.output
}

hw.module @test(%a: i1) -> () {
  // expected-error @+1 {{'hw.instance' op has a wrong number of operands; expected 0 but got 1}}
  hw.instance "test" @empty(a: %a: i1) -> ()
  hw.output
}

// -----

// expected-note @+1 {{module declared here}}
hw.module @f(%a: i1) -> () {
  hw.output
}

hw.module @test(%a: i2) -> () {
  // expected-error @+1 {{'hw.instance' op operand type #0 must be 'i1', but got 'i2'}}
  hw.instance "test" @f(a: %a: i2) -> ()
  hw.output
}


// -----

// expected-note @+1 {{module declared here}}
hw.module @f(%a: i1) -> () {
  hw.output
}

hw.module @test(%a: i1) -> () {
  // expected-error @+1 {{'hw.instance' op input label #0 must be "a", but got "b"}}
  hw.instance "test" @f(b: %a: i1) -> ()
  hw.output
}

// -----

// expected-note @+1 {{module declared here}}
hw.module.extern @p<p1: i42 = 17, p2: i1>(%arg0: i8) -> (out: i8)

hw.module @Use(%a: i8) -> (xx: i8) {
  // expected-error @+1 {{op expected 2 parameters but had 1}}
  %r0 = hw.instance "inst1" @p<p1: i42 = 4>(arg0: %a: i8) -> (out: i8)
  hw.output %r0: i8
}

// -----

// expected-note @+1 {{module declared here}}
hw.module.extern @p<p1: i42 = 17, p2: i1>(%arg0: i8) -> (out: i8)

hw.module @Use(%a: i8) -> (xx: i8) {
  // expected-error @+1 {{op parameter #1 should have name "p2" but has name "p3"}}
  %r0 = hw.instance "inst1" @p<p1: i42 = 4, p3: i1 = 0>(arg0: %a: i8) -> (out: i8)
  hw.output %r0: i8
}

// -----

// expected-note @+1 {{module declared here}}
hw.module.extern @p<p1: i42 = 17, p2: i1>(%arg0: i8) -> (out: i8)

hw.module @Use(%a: i8) -> (xx: i8) {
  // expected-error @+1 {{op parameter "p2" should have type 'i1' but has type 'i2'}}
  %r0 = hw.instance "inst1" @p<p1: i42 = 4, p2: i2 = 0>(arg0: %a: i8) -> (out: i8)
  hw.output %r0: i8
}

// -----

hw.module.extern @p<p1: i42 = 17, p2: i1>(%arg0: i8) -> (out: i8)

hw.module @Use(%a: i8) -> (xx: i8) {
  // expected-error @+1 {{op parameter "p2" must have a value}}
  %r0 = hw.instance "inst1" @p<p1: i42 = 4, p2: i1>(arg0: %a: i8) -> (out: i8)
  hw.output %r0: i8
}

// -----
// Check attribute validity for parameters.

hw.module.extern @p<p: i42>() -> ()

// expected-note @+1 {{module declared here}}
hw.module @Use() {
  // expected-error @+1 {{op use of unknown parameter "FOO"}}
  hw.instance "inst1" @p<p: i42 = #hw.param.decl.ref<"FOO">>() -> ()
}

// -----
// Check attribute validity for parameters.

hw.module.extern @p<p: i42>() -> ()

// expected-note @+1 {{module declared here}}
hw.module @Use<xx: i41>() {
  // expected-error @+1 {{op parameter "xx" used with type 'i42'; should have type 'i41'}}
  hw.instance "inst1" @p<p: i42 = #hw.param.decl.ref<"xx">>() -> ()
}


// -----
// Check attribute validity for module parameters.

// expected-error @+1 {{op parameter "p" cannot be used as a default value for a parameter}}
hw.module.extern @p<p: i42 = #hw.param.decl.ref<"p">>() -> ()

// -----

// expected-note @+1 {{module declared here}}
hw.module @Use<xx: i41>() {
  // expected-error @+1 {{'hw.param.value' op parameter "xx" used with type 'i40'; should have type 'i41'}}
  %0 = hw.param.value i40 = #hw.param.decl.ref<"xx">
}

// -----

// expected-error @+1 {{parameter #hw.param.decl<"xx": i41> : i41 has the same name as a previous parameter}}
hw.module @Use<xx: i41, xx: i41>() {}

// -----

// expected-error @+1 {{parameter #hw.param.decl<"xx": i41 = 1> : i41 has the same name as a previous parameter}}
hw.module @Use<xx: i41, xx: i41 = 1>() {}

// -----

// expected-error @+1 {{parameter #hw.param.decl<"xx": none> has the same name as a previous parameter}}
hw.module @Use<xx: none, xx: none>() {}

// -----

module  {
// expected-error @+1 {{'inst_1' in module:'A' does not contain a reference to 'glbl_D_M1'}}
  hw.globalRef @glbl_D_M1 [#hw.innerNameRef<@A::@inst_1>]
  hw.module @C() -> () {
  }
  hw.module @A() -> () {
    hw.instance "h2" sym @inst_1 @C() -> () {circt.globalRef = []}
    hw.instance "h2" sym @inst_0 @C() -> () {circt.globalRef = [#hw.globalNameRef<@glbl_D_M1>]}
  }
}

// -----

module {
  hw.module @A(%a : !hw.int<41>) -> (out: !hw.int<42>) {
// expected-error @+1 {{'hw.instance' op operand type #0 must be 'i42', but got 'i41'}}
    %r0 = hw.instance "inst1" @parameters<p1: i42 = 42>(arg0: %a: !hw.int<41>) -> (out: !hw.int<42>)
    hw.output %r0: !hw.int<42>
  }
// expected-note @+1 {{module declared here}}
  hw.module.extern @parameters<p1: i42>(%arg0: !hw.int<#hw.param.decl.ref<"p1">>) -> (out: !hw.int<#hw.param.decl.ref<"p1">>)
}

// -----

module {
  hw.module @A(%a : !hw.int<42>) -> (out: !hw.int<41>) {
// expected-error @+1 {{'hw.instance' op result type #0 must be 'i42', but got 'i41'}}
    %r0 = hw.instance "inst1" @parameters<p1: i42 = 42>(arg0: %a: !hw.int<42>) -> (out: !hw.int<41>)
    hw.output %r0: !hw.int<41>
  }
// expected-note @+1 {{module declared here}}
  hw.module.extern @parameters<p1: i42>(%arg0: !hw.int<#hw.param.decl.ref<"p1">>) -> (out: !hw.int<#hw.param.decl.ref<"p1">>)
}

// -----

// expected-note @+1 {{module declared here}}
hw.module.extern @submodule () -> (out0: i32)

hw.module @wrongResultLabel() {
  // expected-error @+1 {{result label #0 must be "out0", but got "o"}}
  %inst0.out0 = hw.instance "inst0" @submodule () -> (o: i32)
}

// -----

// expected-note @+1 {{module declared here}}
hw.module.extern @submodule () -> (out0: i32)

hw.module @wrongNumberOfResultNames() {
  // expected-error @+1 {{has a wrong number of results port labels; expected 1 but got 0}}
  "hw.instance"() {instanceName="inst0", moduleName=@submodule, argNames=[], resultNames=[], parameters=[]} : () -> i32
}

// -----

// expected-note @+1 {{module declared here}}
hw.module.extern @submodule (%arg0: i32) -> ()

hw.module @wrongNumberOfInputNames(%arg0: i32) {
  // expected-error @+1 {{has a wrong number of input port names; expected 1 but got 0}}
  "hw.instance"(%arg0) {instanceName="inst0", moduleName=@submodule, argNames=[], resultNames=[], parameters=[]} : (i32) -> ()
}

// -----

// expected-error @+1 {{unsupported dimension kind in hw.array}}
hw.module @bab<param: i32, N: i32> ( %array2d: !hw.array<i3 x i4>) {}

// -----

hw.module @foo() {
  // expected-error @+1 {{enum value 'D' is not a member of enum type '!hw.enum<A, B, C>'}}
  %0 = hw.enum.constant D : !hw.enum<A, B, C>
  hw.output
}

// -----

hw.module @foo() {
  // expected-error @+1 {{return type '!hw.enum<A, B>' does not match attribute type #hw.enum.field<A, !hw.enum<A>>}}
  %0 = "hw.enum.constant"() {field = #hw.enum.field<A, !hw.enum<A>>} : () -> !hw.enum<A, B>
  hw.output
}

// -----

hw.module @foo() {
  %0 = hw.enum.constant A : !hw.enum<A>
  %1 = hw.enum.constant B : !hw.enum<B>
  // expected-error @+1 {{types do not match}}
  %2 = hw.enum.cmp %0, %1 : !hw.enum<A>, !hw.enum<B>
  hw.output
}
