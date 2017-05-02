// See LICENSE for license details.

package chiselTests

import chisel3._
import org.scalatest._
import org.scalatest.matchers._
import org.scalatest.prop._
import chisel3.testers.BasicTester

class DirectionHaver extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(32.W))
    val out = Output(UInt(32.W))
  })
}

class GoodDirection extends DirectionHaver {
  io.out := 0.U
}

class BadDirection extends DirectionHaver {
  io.in := 0.U
}

class DirectionSpec extends ChiselPropSpec with Matchers {

  //TODO: In Chisel3 these are actually FIRRTL errors. Remove from tests?

  property("Outputs should be assignable") {
    elaborate(new GoodDirection)
  }

  property("Inputs should not be assignable") {
    a[Exception] should be thrownBy {
     elaborate(new BadDirection)
    }
  }
}
