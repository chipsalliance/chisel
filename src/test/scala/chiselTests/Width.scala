// See LICENSE for license details.

package chiselTests
import Chisel._
import org.scalatest._
import Chisel.testers.BasicTester

class ZeroWidthUInt extends Module {
  val io = new Bundle {
  }

  UInt(0, width = 0)
}

class WidthSpec extends ChiselPropSpec with Matchers {
  property("The literal 0 has a width of 0") {
    elaborate(new ZeroWidthUInt)
  }
}

