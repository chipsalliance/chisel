// SPDX-License-Identifier: Apache-2.0

package chisel3
package internal

import chisel3.internal.Context
import chiselTests.{ChiselFunSpec, Utils}

class ContextSpec extends ChiselFunSpec with Utils {
  describe("(0) Context") {
    it("(0.a): instantiateChild, checked with target") {
      val defLeaf = Context("Leaf")
      val defBranch = Context("Branch")
      defBranch.instantiateChild("l0", defLeaf)
      defBranch.instantiateChild("l1", defLeaf)
      val defRoot = Context("Root")
      defRoot.instantiateChild("b0", defBranch)
      defRoot.instantiateChild("b1", defBranch)

      assert(defRoot("b0").target == "Root/b0")
      assert(defRoot("b1").target == "Root/b1")
      assert(defRoot("b0")("l0").target == "Root/b0/l0")
      assert(defRoot("b0")("l1").target == "Root/b0/l1")
      assert(defRoot("b1")("l0").target == "Root/b1/l0")
      assert(defRoot("b1")("l1").target == "Root/b1/l1")
    }
    it("(0.b): copyTo") {
      val L = Context("Leaf")
      val B = Context("Branch")
      val B_l0 = B.instantiateChild("l0", L)
      val B_l1 = B.instantiateChild("l1", L)
      val R = Context("Root")
      val R_b0 = R.instantiateChild("b0", B)
      val R_b1 = R.instantiateChild("b1", B)
      val R_b0_l1 = R("b0")("l1")

      // Left subsumes right
      assert(B_l0.copyTo(R_b0) == R_b0("l0"))
      assert(B_l1.copyTo(R_b0_l1) == R_b0_l1)
      assert(B_l1.copyTo(R) == B_l1)

      // Right subsumes left
      assert(R_b0.copyTo(B_l0) == R_b0)
      assert(R_b0_l1.copyTo(B_l1) == R_b0_l1)
      assert(R.copyTo(B_l1) == R)

      // Left matches right
      assert(R_b0.copyTo(R_b0) == R_b0)
      assert(R_b0_l1.copyTo(R_b0_l1) == R_b0_l1)
      assert(R.copyTo(R_b0_l1) == R)
    }
  }
  describe("(1) Context used in modules and data") {
    it("Reports context in connection error") {
      class Top extends Module {
        val m0 = Module(new Middle)
        val m1 = Module(new Middle)
        m0.b0.in := m0.b1.out // Connect to and from data outside of local module's scope!
      }
      class Middle extends Module {
        val b0 = Module(new Bottom)
        val b1 = Module(new Bottom)
      }
      class Bottom extends Module {
        val in  = IO(Input(UInt(3.W)))
        val out = IO(Output(UInt(3.W)))
      }
      val e = intercept[chisel3.ChiselException] {
        circt.stage.ChiselStage.emitCHIRRTL(new Top, Array("--full-stacktrace"))
      }


      List(
        "Connection between sink (Bottom.in: IO[UInt<3>]) and source (Bottom_1.out: IO[UInt<3>]) failed",
        "Top/m0=Middle/b0=Bottom/in",    // Generated from Context.target
        "Top/m0=Middle/b1=Bottom_1/out", // Generated from Context.target
        "unavailable to current module",
        "Top"                            // Generated from Context.target
      ).forall(s => e.getMessage.contains(s))
    }
  }
}
