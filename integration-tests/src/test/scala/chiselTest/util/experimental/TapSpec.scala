// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.experimental

import circt.stage.ChiselStage
import chisel3._
import chisel3.util._
import chisel3.testers.BasicTester
import chisel3.util.experimental.BoringUtils._
import org.scalacheck._
import chiselTests.{ChiselFlatSpec, Utils}
import chisel3.probe._


class TapSpec extends ChiselFlatSpec with Utils {
  /** Circuit follows this layout:
    *
    *           UpDownTest
    *             /     \
    *       LeftUpper   RightUpper
    *          /           \
    *    LeftLower       RightLower
    *
    * Tap a signal in RightLower from LeftLower and check that it contains the
    * correct value.
    */
  "Up-and-down tap" should "work" in {
    class LeftLower(tapSignal: UInt) extends Module {
      val readTap = Wire(UInt(8.W))
      readTap := tapAndRead(tapSignal)
      chisel3.assert(readTap === 123.U)
      stop()
    }

    class LeftUpper(tapSignal: UInt) extends Module {
      val leftLower = Module(new LeftLower(tapSignal))
    }

    class RightLower extends Module {
      val tapMe = Wire(UInt(8.W))
      tapMe := 123.U
      dontTouch(tapMe)
    }

    class RightUpper extends Module {
      val rightLower = Module(new RightLower)
    }

    class UpDownTest extends BasicTester {
      val rightUpper = Module(new RightUpper)
      val leftUpper = Module(new LeftUpper(rightUpper.rightLower.tapMe))
    }

    assertTesterPasses(new UpDownTest)
  }

  "Writable taps with forces" should "work" in {
    class Grandchild extends Module {
      val tapMe = Wire(UInt(8.W))
      tapMe := 0.U
      dontTouch(tapMe)
    }

    class Child extends Module {
      val grandchild = Module(new Grandchild)
    }

    class WritableTapTest extends BasicTester {
      val grandChildTap = IO(RWProbe(UInt(8.W)))
      val child = Module(new Child)

      define(grandChildTap, rwTap(child.grandchild.tapMe))
      force(clock, true.B, grandChildTap, 123.U(8.W))
      forceInitial(grandChildTap, 123.U(8.W))

      // this test passes on compile and does not perform any checks
      // due to unexpected behavior with `force` and Verilator

      stop()
    }

    assertTesterPasses(new WritableTapTest)
  }
}
