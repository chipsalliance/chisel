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
  "A Reg" should "throw an exception if not given any parameters" in {
    a [Exception] should be thrownBy {
      val reg = Reg()
    }
  }

  "A Reg" should "be of the same type and width as outType, if specified" in {
    class RegOutTypeWidthTester extends BasicTester {
      val reg = Reg(t=UInt(2.W), next=Wire(UInt(3.W)), init=20.U)
      reg.getWidth should be (2)
    }
    elaborate{ new RegOutTypeWidthTester }
  }

  "A Reg" should "be of unknown width if outType is not specified and width is not forced" in {
    class RegUnknownWidthTester extends BasicTester {
      val reg1 = Reg(next=Wire(UInt(3.W)), init=20.U)
      reg1.isWidthKnown should be (false)
      DataMirror.widthOf(reg1).known should be (false)
      val reg2 = Reg(init=20.U)
      reg2.isWidthKnown should be (false)
      DataMirror.widthOf(reg2).known should be (false)
      val reg3 = Reg(next=Wire(UInt(3.W)), init=5.U)
      reg3.isWidthKnown should be (false)
      DataMirror.widthOf(reg3).known should be (false)
    }
    elaborate { new RegUnknownWidthTester }
  }

  "A Reg" should "be of width of init if outType and next are missing and init is a literal of forced width" in {
    class RegForcedWidthTester extends BasicTester {
      val reg2 = Reg(init=20.U(7.W))
      reg2.getWidth should be (7)
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
