// RUN: circt-opt %s --split-input-file --verify-diagnostics

hw.module @entryBlockHasNoArguments () -> () {
  // expected-error @+1 {{region must not have any arguments}}
  interop.procedural.init cpp {
  ^bb0(%arg0: i32):
    interop.return
  }
} 

// -----

hw.module @entryBlockHasNumInputsPlusStatesArguments (%arg0: i1) -> () {
  // expected-error @+1 {{region must have the same number of arguments as inputs and states together, but got 3 arguments and 1 state plus input types}}
  %2 = interop.procedural.update cpp [%arg0]() : [i1]() -> i32 {
  ^bb0(%barg0: i1, %barg1: i32, %barg2: i32):
    interop.return %barg1 : i32
  }
}

// -----

hw.module @entryBlockArgumentTypesMatch (%arg0: i16, %arg1: i1) -> () {
  // expected-error @+1 {{region argument types must match state types}}
  %2 = interop.procedural.update cpp [%arg1](%arg0) : [i1](i16) -> i32 {
  ^bb0(%barg0: i1, %barg1: i32):
    interop.return %barg1 : i32
  }
}

// -----

hw.module @entryBlockArgumentTypesMatch (%arg0: i2) -> () {
  // expected-error @+1 {{region argument types must match state types}}
  %2 = interop.procedural.update cpp [%arg0]() : [i2]() -> i1 {
  ^bb0(%barg0: i1):
    interop.return %barg0 : i1
  }
}

// -----

hw.module @entryBlockHasNumInputsPlusStatesArguments (%arg0: i1) -> () {
  // expected-error @+1 {{region must have the same number of arguments as states, but got 2 arguments and 1 states}}
  interop.procedural.dealloc cpp %arg0 : i1 {
  ^bb0(%barg0: i1, %barg1: i32):
  }
}

// -----

hw.module @entryBlockArgumentTypesMatch (%arg0: i1) -> () {
  // expected-error @+1 {{region argument types must match state types}}
  interop.procedural.dealloc cpp %arg0 : i1 {
  ^bb0(%barg0: i2):
  }
}

// -----

hw.module @returnParentOp () -> () {
  // expected-error @+1 {{ expects parent op to be one of 'interop.procedural.init, interop.procedural.update'}}
  interop.return
}

// -----

hw.module @returnNumOperandsMatchParentsResults (%arg0: i32) -> () {
  interop.procedural.init cpp %arg0 : i32 {
    // expected-error @+1 {{has 0 operands, but enclosing interop operation requires 1 values}}
    interop.return
  }
}

// -----

hw.module @returnOperandsMustMatchParentsResultTypes (%arg0: i32) -> () {
  interop.procedural.init cpp %arg0 : i32 {
    %1 = hw.constant 0 : i8
    // expected-error @+1 {{type of return operand 0 ('i8') doesn't match required type ('i32')}}
    interop.return %1 : i8
  }
}
