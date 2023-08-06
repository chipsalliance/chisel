// RUN: circt-opt %s -split-input-file -verify-diagnostics

// expected-error @+1 {{incorrect number of function results (always has to be 0 or 1)}}
systemc.cpp.func private @return_i32_f32() -> (i32, f32)

// -----

systemc.cpp.func private @return_i32() -> i32

systemc.cpp.func @call() {
  // expected-error @+3 {{op result type mismatch at index 0}}
  // expected-note @+2 {{op result types: 'f32'}}
  // expected-note @+1 {{function result types: 'i32'}}
  %0 = systemc.cpp.call @return_i32() : () -> f32
  systemc.cpp.return
}

// -----

systemc.cpp.func @resulterror() -> i32 {
^bb42:
  systemc.cpp.return    // expected-error {{op has 0 operands, but enclosing function (@resulterror) returns 1}}
}

// -----

systemc.cpp.func @return_type_mismatch(%arg0: f32) -> i32 {
  systemc.cpp.return %arg0 : f32  // expected-error {{type of return operand 0 ('f32') doesn't match function result type ('i32') in function @return_type_mismatch}}
}

// -----

systemc.module @return_inside_module() {
  // expected-error@+1 {{'systemc.cpp.return' op expects parent op 'systemc.cpp.func'}}
  systemc.cpp.return
}

// -----

// expected-error@+1 {{expected non-function type}}
systemc.cpp.func @func_variadic(...)

// -----

systemc.cpp.func @redundant_signature(%a : i32) -> () {
^bb0(%b : i32):  // expected-error {{invalid block name in region with named arguments}}
  systemc.cpp.return
}

// -----

systemc.cpp.func @mixed_named_arguments(%a : i32,
                               f32) -> () {
    // expected-error @-1 {{expected SSA identifier}}
  systemc.cpp.return
}

// -----

systemc.cpp.func @mixed_named_arguments(f32,
                               %a : i32) -> () { // expected-error {{expected type instead of SSA identifier}}
  systemc.cpp.return
}

// -----

// expected-error @+1 {{@ identifier expected to start with letter or '_'}}
systemc.cpp.func @$invalid_function_name()

// -----

// expected-error @+1 {{arguments may only have dialect attributes}}
systemc.cpp.func private @invalid_func_arg_attr(i1 {non_dialect_attr = 10})

// -----

// expected-error @+1 {{results may only have dialect attributes}}
systemc.cpp.func private @invalid_func_result_attr() -> (i1 {non_dialect_attr = 10})

// -----

systemc.cpp.func @foo() {} // expected-error {{expected non-empty function body}}

// -----

func.func @bar () {
  return
}

systemc.cpp.func @foo() {
  // expected-error @+1 {{'bar' does not reference a valid function}}
  systemc.cpp.call @bar() : () -> ()
  systemc.cpp.return
}

// -----

systemc.cpp.func private @bar (i32)

systemc.cpp.func @foo() {
  // expected-error @+1 {{incorrect number of operands for callee}}
  systemc.cpp.call @bar() : () -> ()
  systemc.cpp.return
}

// -----

systemc.cpp.func private @bar (i32)

systemc.cpp.func @foo(%arg0: i8) {
  // expected-error @+1 {{operand type mismatch: expected operand type 'i32', but provided 'i8' for operand number 0}}
  systemc.cpp.call @bar(%arg0) : (i8) -> ()
  systemc.cpp.return
}

// -----

systemc.cpp.func private @bar ()

systemc.cpp.func @foo() {
  // expected-error @+1 {{incorrect number of results for callee}}
  %0 = systemc.cpp.call @bar() : () -> i32
  systemc.cpp.return
}

// -----

systemc.cpp.func private @bar () -> i32

systemc.cpp.func @foo() {
  // expected-error @+1 {{incorrect number of function results (always has to be 0 or 1)}}
  %0, %1 = systemc.cpp.call @bar() : () -> (i32, i32)
  systemc.cpp.return
}

// -----

systemc.cpp.func @foo(%arg0: () -> (i32, i32)) {
  // expected-error @+1 {{incorrect number of function results (always has to be 0 or 1)}}
  %0, %1 = systemc.cpp.call_indirect %arg0() : () -> (i32, i32)
  systemc.cpp.return
}

// -----

// expected-error @+1 {{'function_type' is an inferred attribute and should not be specified in the explicit attribute dictionary}}
systemc.cpp.func @foo() attributes {function_type=()->()} {
  systemc.cpp.return
}

// -----

// expected-error @+1 {{incorrect number of argument names}}
"systemc.cpp.func"() ({
  ^bb0(%arg0: i32):
    "systemc.cpp.return"() : () -> ()
}) {sym_name="foo", function_type=(i32)->(), argNames=[]} : () -> () 

// -----

// expected-error @+1 {{arg name must not be empty}}
"systemc.cpp.func"() ({
  ^bb0(%arg0: i32):
    "systemc.cpp.return"() : () -> ()
}) {sym_name="foo", function_type=(i32)->(), argNames=[""]} : () -> () 

// -----

// expected-note @+1 {{in function '@foo'}}
systemc.cpp.func @foo() {
  // expected-note @+1 {{'var0' first defined here}}
  %0 = "systemc.cpp.variable"() {name = "var0"} : () -> i32
  // expected-error @+1 {{redefines name 'var0'}}
  %1 = "systemc.cpp.variable"() {name = "var0"} : () -> i32
  systemc.cpp.return
}

// -----

// expected-note @+1 {{in function '@foo'}}
"systemc.cpp.func"() ({
  // expected-error @+2 {{redefines name 'a'}}
  // expected-note @+1 {{'a' first defined here}}
  ^bb0(%arg0: i32, %arg1: i32):
    "systemc.cpp.return"() : () -> ()
}) {sym_name="foo", function_type=(i32, i32)->(), argNames=["a", "a"]} : () -> () 

// -----

// expected-error @+1 {{entry block must have 2 arguments to match function signature}}
systemc.cpp.func @foo (i32, i32) {
  systemc.cpp.return
}

// -----

// expected-error @+1 {{incorrect number of argument names}}
systemc.cpp.func @foo (i32, i32) {
  ^bb0(%a: i32, %b: i32):
  systemc.cpp.return
}
