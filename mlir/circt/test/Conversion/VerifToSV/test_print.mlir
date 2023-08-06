// RUN: circt-opt --lower-verif-to-sv %s | FileCheck %s

// CHECK-LABEL:   hw.module @foo(
// CHECK-SAME:                   %[[VAL_0:.*]]: i1) {
// CHECK:           sv.always posedge %[[VAL_0]] {
// CHECK:             %[[VAL_1:.*]] = hw.constant true
// CHECK:             %[[VAL_2:.*]] = verif.format_verilog_string "Hi %[[VAL_3:.*]]\0A"(%[[VAL_1]]) : i1
// CHECK:             %[[VAL_4:.*]] = hw.constant -2147483647 : i32
// CHECK:             sv.fwrite %[[VAL_4]], "Hi %x\0A"(%[[VAL_1]]) : i1
// CHECK:           }
// CHECK:           hw.output
// CHECK:         }
hw.module @foo(%trigger : i1) {
  sv.always posedge %trigger {
    %true = hw.constant true
    %fstr = verif.format_verilog_string "Hi %x\0A" (%true) : i1
    verif.print %fstr
  }
}
