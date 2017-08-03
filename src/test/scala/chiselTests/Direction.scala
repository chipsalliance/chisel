// See LICENSE for license details.

package chiselTests

import chisel3._
import org.scalatest._
import org.scalatest.matchers._
import org.scalatest.prop._
import chisel3.testers.BasicTester

class DirectionedBundle extends Bundle {
  val in = Input(UInt(32.W))
  val out = Output(UInt(32.W))
}

class DirectionHaver extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(32.W))
    val out = Output(UInt(32.W))
    val inBundle = Input(new DirectionedBundle)  // should override elements
    val outBundle = Output(new DirectionedBundle)  // should override elements
  })
}

class GoodDirection extends DirectionHaver {
  io.out := 0.U
  io.outBundle.in := 0.U
  io.outBundle.out := 0.U
}

class BadDirection extends DirectionHaver {
  io.in := 0.U
}

class BadSubDirection extends DirectionHaver {
  io.inBundle.out := 0.U
}

class TopDirectionOutput extends Module {
  val io = IO(Output(new DirectionedBundle))
  io.in := 42.U
  io.out := 117.U
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
    a[Exception] should be thrownBy {
     elaborate(new BadSubDirection)
    }
  }

  property("Top-level forced outputs should be assignable") {
    elaborate(new TopDirectionOutput)
  }
}
