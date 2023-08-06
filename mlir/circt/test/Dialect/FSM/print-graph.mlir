// RUN: circt-opt --fsm-print-graph %s 2>&1 | FileCheck %s

// CHECK: [[IDLE:[^ ]*]] [shape=record,label="{IDLE|fsm.output %true : i1}"];
// CHECK: [[IDLE]] -> [[BUSY:.*]][label="fsm.return %arg0"];
// CHECK: [[BUSY]] [shape=record,label="{BUSY|fsm.output %false : i1}"];
// CHECK: [[BUSY]] -> [[BUSY]][label="%0 = arith.cmpi ne, %cnt, %c0_i16 : i16\nfsm.return %0"];
// CHECK: [[BUSY]] -> [[IDLE]]
// CHECK-SAME{LITERAL}: [label="%0 = arith.cmpi eq, %cnt, %c0_i16 : i16\nfsm.return %0"];
// CHECK: variables [shape=record,label="Variables|%c1_i16 = arith.constant 1 : i16\n%c0_i16 = arith.constant 0 : i16\n%false = arith.constant false\n%c256_i16 = arith.constant 256 : i16\n%true = arith.constant true\n%cnt = fsm.variable \"cnt\" \{initValue = 0 : i16\} : i16"]}

fsm.machine @foo(%arg0: i1) -> i1 attributes {initialState = "IDLE"} {
  %c1_i16 = arith.constant 1 : i16
  %c0_i16 = arith.constant 0 : i16
  %false = arith.constant false
  %c256_i16 = arith.constant 256 : i16
  %true = arith.constant true
  %cnt = fsm.variable "cnt" {initValue = 0 : i16} : i16
  fsm.state @IDLE output {
    fsm.output %true : i1
  } transitions {
    fsm.transition @BUSY guard {
      fsm.return %arg0
    } action {
      fsm.update %cnt, %c256_i16 : i16
    }
  }
  fsm.state @BUSY output {
    fsm.output %false : i1
  } transitions {
    fsm.transition @BUSY guard {
      %0 = arith.cmpi ne, %cnt, %c0_i16 : i16
      fsm.return %0
    } action {
      %0 = arith.subi %cnt, %c1_i16 : i16
      fsm.update %cnt, %0 : i16
    }
    fsm.transition @IDLE guard {
      %0 = arith.cmpi eq, %cnt, %c0_i16 : i16
      fsm.return %0
    }
  }
}
