// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.experimental

import circt.stage.ChiselStage
import chisel3._
import chisel3.util._
import chisel3.testers.BasicTester
import chisel3.util.experimental.BoringUtils._
import org.scalacheck._
import chiselTests.{ChiselFlatSpec, Utils}

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
class TapSpec extends ChiselFlatSpec with Utils {
  assertTesterPasses(new UpDownTest)
}
