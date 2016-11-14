// See LICENSE for license details.

package chiselTests

import chisel3._
import org.scalatest._

class DirectionHaver extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt.width(32))
    val out = Output(UInt.width(32))
  })
}

class GoodDirection extends DirectionHaver {
  io.out := UInt.Lit(0)
}

class BadDirection extends DirectionHaver {
  io.in := UInt.Lit(0)
}

class DeprecatedDirection extends Module {
  import chisel3.core.ExplicitCompileOptions.Strict
  val io = IO(new Bundle {
    val in = UInt(INPUT, 32)
    val out = UInt(OUTPUT, 32)
  })
}

class DirectionSpec extends ChiselPropSpec with ShouldMatchers {

  //TODO: In Chisel3 these are actually FIRRTL errors. Remove from tests?

  property("Outputs should be assignable") {
    elaborate(new GoodDirection)
  }

  property("Inputs should not be assignable") {
    a[Exception] should be thrownBy {
     elaborate(new BadDirection)
    }
  }

  property("Chisel2 directions should be deprecated") {
    val output = new java.io.ByteArrayOutputStream()
    Console.withOut(output) {
      elaborate(new DeprecatedDirection)
    }
    println(output)
    assert(output.toString.contains("Direction(UInt(...)) should be used over UInt(DIRECTION, ...)"))
  }
}
