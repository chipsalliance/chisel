package chisel3.tests

import org.scalatest._

import chisel3._
import chisel3.testers2._

class BasicTest extends FlatSpec with ImplicitPokeTester {
  "Testers2" should "run" in {
    test(new Module {
      val io = IO(new Bundle {
        val out = UInt(8.W)
      })
      out := 42.U
    }) { c =>
      c.io.out.check(42.U)
    }
  }
}
