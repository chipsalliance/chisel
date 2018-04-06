// See LICENSE for license details.

package chisel3

import org.scalatest.{FreeSpec, Matchers}
import chisel3.tester._
import chisel3.tester.TestAdapters._
import org.scalatest.exceptions.TestFailedException

import scala.collection.mutable

class LowerBundle extends Bundle {
  val d = UInt(7.W)
  val e = Bool()
}

class TopBundle extends Bundle {
  val a = UInt(16.W)
  val b = UInt(32.W)
  val c = new LowerBundle
}

class GcdMock extends Module {
  val io = IO(new Bundle {
    val input = Input(new TopBundle)
    val output = Output(new TopBundle)
    val output2 = Output(new LowerBundle)
  })

  val x = new RecordLiteral(io.output2) {
    set(io.output2.d, 22.U)
    set(io.output2.e, true.B)
  }
  io.output := io.input
  io.output2 := DontCare // x
}


class BundleLiteralSpec extends FreeSpec with ChiselScalatestTester with Matchers {
  "a" in {
    test(new GcdMock) { c =>

      val lowerIn = new RecordLiteral(c.io.input.c) {
        set(c.io.input.c.d, 2.U)
        set(c.io.input.c.e, true.B)
      }

      val xx = new RecordLiteral(c.io.input) {
        set(c.io.input.a, 77.U)
        set(c.io.input.b, 33.U)
        set(c.io.input.c, lowerIn)
      }

      val yy = new RecordLiteral(c.io.input) {
        set(c.io.input.a, 77.U)
        set(c.io.input.b, 55.U)
      }

      c.io.input.poke(xx)

      c.clock.step(1)
      c.io.output.expect(xx)
      c.io.output.c.expect(lowerIn)
      c.io.output.c.d.expect(2.U)
      c.io.output.c.e.expect(true.B)

//      intercept[TestFailedException] {
//        c.io.output.expect(yy)
//      }
    }
    // println(s"x $x ${x.a.litValue()}")
  }

}
