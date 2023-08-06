//RUN: circt-opt %s -split-input-file -verify-diagnostics

// Testing Objectives:
// * inst can only be used in entities
// * inst must always refer to a valid proc or entity (match symbol name, input and output operands)
// * syntax: no inputs and outputs, one input zero outputs, zero inputs one output, multiple inputs and outputs
// * check that number of inputs and number of outputs are verified separately

llhd.proc @empty_proc() -> () {
  llhd.halt
}

llhd.proc @fail() -> () {
  // expected-error @+1 {{expects parent op 'llhd.entity'}}
  llhd.inst "empty" @empty_proc() -> () : () -> ()
  llhd.halt
}

// -----

llhd.entity @operand_count_mismatch(%arg : !llhd.sig<i32>) -> () {}

llhd.entity @caller(%arg : !llhd.sig<i32>) -> () {
  // expected-error @+1 {{incorrect number of inputs for entity instantiation}}
  llhd.inst "mismatch" @operand_count_mismatch() -> (%arg) : () -> (!llhd.sig<i32>)
}

// -----

llhd.entity @caller() -> () {
  // expected-error @+1 {{does not reference a valid proc, entity, or hw.module}}
  llhd.inst "does_not_exist" @does_not_exist() -> () : () -> ()
}

// -----

llhd.entity @empty() -> () {}

llhd.entity @test_uniqueness() -> () {
  llhd.inst "inst" @empty() -> () : () -> ()
  // expected-error @+1 {{redefinition of instance named 'inst'!}}
  llhd.inst "inst" @empty() -> () : () -> ()
}

// -----

hw.module @module(%arg0: i2) -> () {}

llhd.entity @moduleTypeMismatch(%arg0: !llhd.sig<i3>) -> () {
  // expected-error @+1 {{input type mismatch}}
  llhd.inst "inst" @module(%arg0) -> () : (!llhd.sig<i3>) -> ()
}

// -----

hw.module @module() -> (arg0: i2) {
  %0 = hw.constant 0 : i2
  hw.output %0 : i2
}

llhd.entity @moduleTypeMismatch() -> (%arg0: !llhd.sig<i3>) {
  // expected-error @+1 {{output type mismatch}}
  llhd.inst "inst" @module() -> (%arg0) : () -> !llhd.sig<i3>
}
