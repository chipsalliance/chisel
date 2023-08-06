// RUN: circt-opt --split-input-file %s | circt-opt --split-input-file | FileCheck %s

// CHECK: fsm.machine @foo(%arg0: i1) attributes {initialState = "IDLE"} {
// CHECK:   fsm.state @IDLE
// CHECK: }

fsm.machine @foo(%arg0: i1) attributes {initialState = "IDLE"} {
  fsm.state @IDLE
}

// -----

// CHECK: fsm.machine @foo(%arg0: i1) -> i1 attributes {initialState = "IDLE"} {
// CHECK:   %cnt = fsm.variable "cnt" {initValue = 0 : i16} : i16
// CHECK:   fsm.state @IDLE output  {
// CHECK:     %true = arith.constant true
// CHECK:     fsm.output %true : i1
// CHECK:   } transitions  {
// CHECK:     fsm.transition @BUSY guard  {
// CHECK:       fsm.return %arg0
// CHECK:     } action {
// CHECK:       %c256_i16 = arith.constant 256 : i16
// CHECK:       fsm.update %cnt, %c256_i16 : i16
// CHECK:     }
// CHECK:   }
// CHECK:   fsm.state @BUSY output  {
// CHECK:     %false = arith.constant false
// CHECK:     fsm.output %false : i1
// CHECK:   } transitions  {
// CHECK:     fsm.transition @BUSY guard  {
// CHECK:       %c0_i16 = arith.constant 0 : i16
// CHECK:       %0 = arith.cmpi ne, %cnt, %c0_i16 : i16
// CHECK:       fsm.return %0
// CHECK:     } action {
// CHECK:       %c1_i16 = arith.constant 1 : i16
// CHECK:       %0 = arith.subi %cnt, %c1_i16 : i16
// CHECK:       fsm.update %cnt, %0 : i16
// CHECK:     }
// CHECK:     fsm.transition @IDLE guard  {
// CHECK:       %c0_i16 = arith.constant 0 : i16
// CHECK:       %0 = arith.cmpi eq, %cnt, %c0_i16 : i16
// CHECK:       fsm.return %0
// CHECK:     }
// CHECK:   }
// CHECK: }
// CHECK: hw.module @bar(%clk: i1, %rst_n: i1) {
// CHECK:   %true = hw.constant true
// CHECK:   %0 = fsm.hw_instance "foo_inst" @foo(%true), clock %clk, reset %rst_n : (i1) -> i1
// CHECK:   hw.output
// CHECK: }
// CHECK: func @qux() {
// CHECK:   %foo_inst = fsm.instance "foo_inst" @foo
// CHECK:   %true = arith.constant true
// CHECK:   %0 = fsm.trigger %foo_inst(%true) : (i1) -> i1
// CHECK:   %false = arith.constant false
// CHECK:   %1 = fsm.trigger %foo_inst(%false) : (i1) -> i1
// CHECK:   return
// CHECK: }

fsm.machine @foo(%arg0: i1) -> i1 attributes {initialState = "IDLE"} {
  %cnt = fsm.variable "cnt" {initValue = 0 : i16} : i16

  fsm.state @IDLE output  {
    %true = arith.constant true
    fsm.output %true : i1
  } transitions  {
    // Transit to BUSY when `arg0` is true.
    fsm.transition @BUSY guard  {
      fsm.return %arg0
    } action  {
      %c256_i16 = arith.constant 256 : i16
      fsm.update %cnt, %c256_i16 : i16
    }
  }

  fsm.state @BUSY output  {
    %false = arith.constant false
    fsm.output %false : i1
  } transitions  {
    // Transit to BUSY itself when `cnt` is not equal to zero. Meanwhile,
    // decrease `cnt` by one.
    fsm.transition @BUSY guard  {
      %c0_i16 = arith.constant 0 : i16
      %0 = arith.cmpi ne, %cnt, %c0_i16 : i16
      fsm.return %0
    } action  {
      %c1_i16 = arith.constant 1 : i16
      %0 = arith.subi %cnt, %c1_i16 : i16
      fsm.update %cnt, %0 : i16
    }
    // Transit back to IDLE when `cnt` is equal to zero.
    fsm.transition @IDLE guard  {
      %c0_i16 = arith.constant 0 : i16
      %0 = arith.cmpi eq, %cnt, %c0_i16 : i16
      fsm.return %0
    } action  {
    }
  }
}

// Hardware-style instantiation.
hw.module @bar(%clk: i1, %rst_n: i1) {
  %in = hw.constant true
  %out = fsm.hw_instance "foo_inst" @foo(%in), clock %clk, reset %rst_n : (i1) -> i1
}

// Software-style instantiation and triggering.
func.func @qux() {
  %foo_inst = fsm.instance "foo_inst" @foo
  %in0 = arith.constant true
  %out0 = fsm.trigger %foo_inst(%in0) : (i1) -> i1
  %in1 = arith.constant false
  %out1 = fsm.trigger %foo_inst(%in1) : (i1) -> i1
  return
}

// -----

// Optional guard and action regions

// CHECK:   fsm.machine @foo(%[[VAL_0:.*]]: i1) -> i1 attributes {initialState = "A"} {
// CHECK:           %[[VAL_1:.*]] = fsm.variable "cnt" {initValue = 0 : i16} : i16
// CHECK:           fsm.state @A output {
// CHECK:             fsm.output %[[VAL_0]] : i1
// CHECK:           } transitions {
// CHECK:             fsm.transition @A
// CHECK:           }
// CHECK:           fsm.state @B output {
// CHECK:             fsm.output %[[VAL_0]] : i1
// CHECK:           } transitions {
// CHECK:             fsm.transition @B
// CHECK:           }
// CHECK:           fsm.state @C output {
// CHECK:             fsm.output %[[VAL_0]] : i1
// CHECK:           } transitions {
// CHECK:             fsm.transition @C
// CHECK:           }
// CHECK:         }
fsm.machine @foo(%arg0: i1) -> i1 attributes {initialState = "A"} {
  %cnt = fsm.variable "cnt" {initValue = 0 : i16} : i16

  fsm.state @A output  {
    fsm.output %arg0 : i1
  } transitions {
    fsm.transition @A action  {
    }
  }

  fsm.state @B output  {
    fsm.output %arg0 : i1
  } transitions {
    fsm.transition @B guard {}
  }

  fsm.state @C output  {
    fsm.output %arg0 : i1
  } transitions {
    fsm.transition @C
  }
}
