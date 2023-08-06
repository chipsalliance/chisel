// RUN: circt-opt %s -split-input-file -verify-diagnostics

// Test missing initial state.

// expected-error @+1 {{'fsm.machine' op initial state 'IDLE' was not defined in the machine}}
fsm.machine @foo(%arg0: i1) -> i1 attributes {initialState = "IDLE"} {}

// -----

fsm.machine @foo(%arg0: i1) -> i1 attributes {initialState = "IDLE"} {
  %cnt = fsm.variable "cnt" {initValue = 0 : i16} : i16

  fsm.state @IDLE output  {
    %true = arith.constant true
// expected-error @+2 {{mismatch in number of types compared (1 != 2)}}
// expected-error @+1 {{'fsm.output' op operand types must match the machine output types}}
    fsm.output %true, %true : i1, i1
  } transitions {
    fsm.transition @IDLE
  }
}

// -----

fsm.machine @foo(%arg0: i1) -> i1 attributes {initialState = "IDLE"} {
  %cnt = fsm.variable "cnt" {initValue = 0 : i16} : i16

  // expected-error @+1 {{'fsm.state' op output block must have a single OutputOp terminator}}
  fsm.state @IDLE output  {
    %true = arith.constant true
  } transitions {
    fsm.transition @IDLE
  }
}


// -----

fsm.machine @foo(%arg0: i1) -> i1 attributes {initialState = "IDLE"} {
  %cnt = fsm.variable "cnt" {initValue = 0 : i16} : i16

  fsm.state @IDLE output  {
    %true = arith.constant true
    fsm.output %true : i1
  } transitions {
    // expected-error @+1 {{'fsm.transition' op guard region must terminate with a ReturnOp}}
    fsm.transition @IDLE guard {
      %true = arith.constant true
    }
  }
}

// -----

fsm.machine @foo(%arg0: i1) -> i1 attributes {initialState = "IDLE"} {
  %cnt = fsm.variable "cnt" {initValue = 0 : i16} : i16

  fsm.state @IDLE output  {
    %true = arith.constant true
    fsm.output %true : i1
  } transitions {
    fsm.transition @IDLE guard {
      // expected-note @+1 {{prior use here}}
      %c2 = arith.constant 2 : i2
      // expected-error @+1 {{use of value '%c2' expects different type than prior uses: 'i1' vs 'i2'}}
      fsm.return %c2
    }
  }
}

// -----

// expected-error @+1 {{'fsm.machine' op number of machine arguments (1) does not match the provided number of argument names (2)}}
fsm.machine @foo(%arg0: i1) -> i1 attributes {initialState = "IDLE", argNames = ["in0", "in1"]} {
  fsm.state @IDLE output {} transitions {}
}

// -----

// expected-error @+1 {{'fsm.machine' op number of machine results (1) does not match the provided number of result names (2)}}
fsm.machine @foo(%arg0: i1) -> i1 attributes {initialState = "IDLE", resNames = ["out0", "out1"]} {
  fsm.state @IDLE output {} transitions {}
}

// -----

fsm.machine @foo(%arg0: i1) -> () attributes {initialState = "A"} {
  %cnt = fsm.variable "cnt" {initValue = 0 : i16} : i16

  fsm.state @A output  {
    fsm.output
  } transitions {
    fsm.transition @A action {
      %c1 = hw.constant 1 : i16
      %add1 = comb.add %cnt, %c1 : i16
      fsm.update %cnt, %add1 : i16
      // expected-error@+1 {{'fsm.update' op multiple updates to the same variable within a single action region is disallowed}}
      fsm.update %cnt, %add1 : i16
    }
  }
}

// -----

fsm.machine @foo(%arg0: i1) -> (i1) attributes {initialState = "IDLE"} {
  // expected-error@+1 {{'fsm.state' op state must have a non-empty output region when the machine has results.}}
  fsm.state @IDLE output {}
}
