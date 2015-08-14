package chiselTests

import Chisel._
import org.scalatest._
import org.scalatest.prop._
import Chisel.testers.BasicTester

class DirectionHaver extends Module {
  val io = new Bundle {
    val in = UInt(INPUT, 32)
    val out = UInt(OUTPUT, 32)
  }
}

class GoodDirection extends DirectionHaver {
  io.out := UInt(0)
}

class BadDirection extends DirectionHaver {
  io.in := UInt(0)
}

class DirectionSpec extends ChiselPropSpec {

  //TODO: In Chisel3 these are actually FIRRTL errors. Remove from tests?

  property("Outputs should be assignable") {
    elaborate(new GoodDirection)
  }

  property("Inputs should not be assignable") {
    elaborate(new BadDirection)
  } 

}
