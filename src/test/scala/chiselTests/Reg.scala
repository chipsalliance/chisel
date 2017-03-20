// See LICENSE for license details.

package chiselTests

import firrtl.ir.Input
import org.scalatest._
import org.scalatest.prop._
import chisel3._
import chisel3.util._
import chisel3.core.DataMirror
import chisel3.testers.BasicTester

class RegSpec extends ChiselFlatSpec {
  "Reg" should "be of the same type and width as t" in {
    class RegOutTypeWidthTester extends BasicTester {
      val reg = Reg(UInt(2.W))
      DataMirror.widthOf(reg) should be (2.W)
    }
    elaborate{ new RegOutTypeWidthTester }
  }

  "RegNext" should "be of unknown width" in {
    class RegUnknownWidthTester extends BasicTester {
      val reg1 = RegNext(2.U(3.W))
      DataMirror.widthOf(reg1).known should be (false)
      val reg2 = RegNext(2.U(3.W), 4.U)
      DataMirror.widthOf(reg2).known should be (false)
      val reg3 = RegNext(2.U(3.W), 4.U(5.W))
      DataMirror.widthOf(reg3).known should be (false)
    }
    elaborate { new RegUnknownWidthTester }
  }

  "RegInit" should "have width only if specified in the literal" in {
    class RegForcedWidthTester extends BasicTester {
      val reg1 = RegInit(20.U)
      DataMirror.widthOf(reg1).known should be (false)
      val reg2 = RegInit(20.U(7.W))
      DataMirror.widthOf(reg2) should be (7.W)
    }
    elaborate{ new RegForcedWidthTester }
  }
}

class ShiftTester(n: Int) extends BasicTester {
  val (cntVal, done) = Counter(true.B, n)
  val start = 23.U
  val sr = ShiftRegister(cntVal + start, n)
  when(done) {
    assert(sr === start)
    stop()
  }
}

class ShiftResetTester(n: Int) extends BasicTester {
  val (cntVal, done) = Counter(true.B, n-1)
  val start = 23.U
  val sr = ShiftRegister(cntVal + 23.U, n, 1.U, true.B)
  when(done) {
    assert(sr === 1.U)
    stop()
  }
}

class ShiftRegisterSpec extends ChiselPropSpec {
  property("ShiftRegister should shift") {
    forAll(smallPosInts) { (shift: Int) => assertTesterPasses{ new ShiftTester(shift) } }
  }

  property("ShiftRegister should reset all values inside") {
    forAll(smallPosInts) { (shift: Int) => assertTesterPasses{ new ShiftResetTester(shift) } }
  }
}
