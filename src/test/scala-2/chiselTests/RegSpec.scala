// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.reflect.DataMirror
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.util.{Counter, ShiftRegister, ShiftRegisters}
import circt.stage.ChiselStage
import org.scalacheck.Gen
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.propspec.AnyPropSpec
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

class RegSpec extends AnyFlatSpec with Matchers {
  "Reg" should "be of the same type and width as t" in {
    class RegOutTypeWidthTester extends Module {
      val reg = Reg(UInt(2.W))
      DataMirror.widthOf(reg) should be(2.W)
    }
    ChiselStage.emitCHIRRTL { new RegOutTypeWidthTester }
  }

  "RegNext" should "be of unknown width" in {
    class RegUnknownWidthTester extends Module {
      val reg1 = RegNext(2.U(3.W))
      DataMirror.widthOf(reg1).known should be(false)
      val reg2 = RegNext(2.U(3.W), 4.U)
      DataMirror.widthOf(reg2).known should be(false)
      val reg3 = RegNext(2.U(3.W), 4.U(5.W))
      DataMirror.widthOf(reg3).known should be(false)
    }
    ChiselStage.emitCHIRRTL { new RegUnknownWidthTester }
  }

  "RegInit" should "have width only if specified in the literal" in {
    class RegForcedWidthTester extends Module {
      val reg1 = RegInit(20.U)
      DataMirror.widthOf(reg1).known should be(false)
      val reg2 = RegInit(20.U(7.W))
      DataMirror.widthOf(reg2) should be(7.W)
    }
    ChiselStage.emitCHIRRTL { new RegForcedWidthTester }
  }
}

class ShiftTester(n: Int) extends Module {
  val (cntVal, _) = Counter(true.B, n + 2)
  val start = 23.U
  val sr = ShiftRegister(cntVal + start, n)
  when(cntVal === n.U) {
    assert(sr === start)
  }
  when(cntVal === (n + 1).U) {
    stop()
  }
}

class ShiftResetTester(n: Int) extends Module {
  val (cntVal, _) = Counter(true.B, 2)
  val start = 23.U
  val sr = ShiftRegister(cntVal + 23.U, n, 1.U, true.B)
  when(cntVal === 0.U) {
    assert(sr === (if (n == 0) cntVal + 23.U else 1.U))
  }
  when(cntVal === 1.U) {
    stop()
  }
}

class ShiftRegisterSpec extends AnyPropSpec with ScalaCheckPropertyChecks with ChiselSim {
  for (shift <- 0 to 4) {
    property(s"ShiftRegister (size $shift) should shift") {
      simulate { new ShiftTester(shift) }(RunUntilFinished(shift + 3))
    }
  }

  for (shift <- 0 to 4) {
    property(s"ShiftRegisters (size $shift) should reset all values inside") {
      simulate { new ShiftResetTester(shift) }(RunUntilFinished(3))
    }
  }
}

class ShiftsTester(n: Int) extends Module {
  val (cntVal, _) = Counter(true.B, n + 2)
  val start = 23.U
  val srs = ShiftRegisters(cntVal + start, n)
  when(cntVal === n.U) {
    srs.zipWithIndex.foreach { case (data, index) =>
      assert(data === (23 + n - 1 - index).U)
    }
  }
  when(cntVal === (n + 1).U) {
    stop()
  }
}

class ShiftRegistersSpec extends AnyPropSpec with ScalaCheckPropertyChecks with ChiselSim {
  for (shift <- 0 to 4) {
    property(s"ShiftRegisters (size $shift) should shift") {
      simulate { new ShiftsTester(shift) }(RunUntilFinished(shift + 3))
    }
  }
}
