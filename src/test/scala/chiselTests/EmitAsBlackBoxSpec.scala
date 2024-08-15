// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.stage._

class Leaf extends RawModule
class Branch extends RawModule {
  val l0 = Module(new Leaf)
  val l1 = Module(new Leaf)
}
class Root extends RawModule {
  val b0 = Module(new Branch)
  val b1 = Module(new Branch)
}
// TODO/Notes
// - In backport, clock/reset are not automatically assigned. I think this is fixed in 3.5
// - CircuitTarget for annotations on the definition are wrong - needs to be fixed.
class EmitAsBlackBoxSpec extends ChiselFunSpec with Utils {
  describe("(0): Emit as black box") {
    it("(0.a): selecting intermediate module creates black box, and omits children modules") {
      val (chirrtl, _) = getFirrtlAndAnnos(new Root, Seq(EmitAsExtModule("Branch_1"), PrintFullStackTraceAnnotation))
      val chirrtlSerialize = chirrtl.serialize
      chirrtlSerialize shouldNot include("module Leaf_2 :")
      chirrtlSerialize shouldNot include("module Leaf_3 :")
      chirrtlSerialize should include("extmodule Branch_1 :")
    }
  }
}