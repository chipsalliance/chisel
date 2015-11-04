// See LICENSE for license details.

package chiselTests

import org.scalatest._
import Chisel._
import Chisel.testers.BasicTester

class RegSpec extends ChiselFlatSpec {
  "A Reg" should "throw an exception if not given any parameters" in {
    a [ChiselException] should be thrownBy {
      val reg = Reg()
    }
  }

  "A Reg" should "be of the same type and width as outType, if specified" in {
    class RegOutTypeWidthTester extends BasicTester {
      val reg = Reg(t=UInt(width=2), next=UInt(width=3), init=UInt(20))
      reg.width.get should be (2)
    }
    elaborate{ new RegOutTypeWidthTester }
  }

  "A Reg" should "be of unknown width if outType is not specified and width is not forced" in {
    class RegUnknownWidthTester extends BasicTester {
      val reg1 = Reg(next=UInt(width=3), init=UInt(20))
      reg1.width.known should be (false)
      val reg2 = Reg(init=UInt(20))
      reg2.width.known should be (false)
      val reg3 = Reg(next=UInt(width=3), init=UInt(width=5))
      reg3.width.known should be (false)
    }
    elaborate { new RegUnknownWidthTester }
  }

  "A Reg" should "be of width of init if outType and next are missing and init is a literal of forced width" in {
    class RegForcedWidthTester extends BasicTester {
      val reg2 = Reg(init=UInt(20, width=7))
      reg2.width.get should be (7)
    }
    elaborate{ new RegForcedWidthTester }
  }
}
