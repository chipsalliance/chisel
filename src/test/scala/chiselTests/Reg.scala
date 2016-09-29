// See LICENSE for license details.

package chiselTests

import org.scalatest._
import chisel3._
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
      val reg = Reg(t=UInt(width=2), next=Wire(UInt(width=3)), init=UInt(20))
      reg.getWidth should be (2)
    }
    elaborate{ new RegOutTypeWidthTester }
  }

  "A Reg" should "be of unknown width if outType is not specified and width is not forced" in {
    class RegUnknownWidthTester extends BasicTester {
      val reg1 = Reg(next=Wire(UInt(width=3)), init=UInt(20))
      reg1.isWidthKnown should be (false)
      DataMirror.widthOf(reg1).known should be (false)
      val reg2 = Reg(init=UInt(20))
      reg2.isWidthKnown should be (false)
      DataMirror.widthOf(reg2).known should be (false)
      val reg3 = Reg(next=Wire(UInt(width=3)), init=UInt(5))
      reg3.isWidthKnown should be (false)
      DataMirror.widthOf(reg3).known should be (false)
    }
    elaborate { new RegUnknownWidthTester }
  }

  "A Reg" should "be of width of init if outType and next are missing and init is a literal of forced width" in {
    class RegForcedWidthTester extends BasicTester {
      val reg2 = Reg(init=UInt(20, width=7))
      reg2.getWidth should be (7)
    }
    elaborate{ new RegForcedWidthTester }
  }
}
