// RUN: circt-opt --lower-hw-to-sv --allow-unregistered-dialect %s | FileCheck %s

hw.module @foo(%trigger : i1, %in : i32) {
  // CHECK:       sv.always posedge %trigger {
  // CHECK-NEXT:    "some.user"(%in) : (i32) -> ()
  // CHECK-NEXT:  }
  hw.triggered posedge %trigger (%in) : i32 {
    ^bb0(%arg0 : i32):
      "some.user" (%arg0) : (i32) -> ()
  }
}
