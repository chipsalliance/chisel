// See LICENSE for license details.

package chiselTests

import chisel3._
import org.scalatest._
import org.scalatest.prop._
import chisel3.testers.BasicTester

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

class DirectionSpec extends ChiselPropSpec {

  //TODO: In Chisel3 these are actually FIRRTL errors. Remove from tests?

  property("Outputs should be assignable") {
    elaborate(new GoodDirection)
  }

  property("Inputs should not be assignable") {
    var excepts: Boolean = false
    try elaborate(new BadDirection)
    catch {
      case e: Exception => {
        excepts = true
      }
      // Should except so this is okay
      // Ideally, would throw and catch more precise exception
    }
    assert(excepts, "Bad connection should have thrown exception!")
  }
}
