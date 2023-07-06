// RUN: mill foo[%basename_t].compile | FileCheck %s

import chisel3._

class Fill2D extends RawModule {
  val u = VecInit.fil(2, 3)(0.U)
}
// CHECK: value fil is not a member of object chisel3.VecInit
// CHECK-NEXT: did you mean fill?

