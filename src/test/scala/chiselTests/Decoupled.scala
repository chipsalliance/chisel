package chiselTests

/**
 * Created by chick on 1/4/16.
 */
package chiselTests

import Chisel._
import org.scalatest._
import org.scalatest.prop._
import Chisel.testers.BasicTester

class ReadCmd extends Bundle {
  val addr = UInt(width = 32)
}

class ReaderExample extends Module {
  val io = new Bundle {
    val in = new DeqIO(new ReadCmd)
  }
}

class ReaderExampleTester extends BasicTester {
  val c = new ReaderExample
  stop()
}

class DecoupledSpec extends ChiselPropSpec {
  property("Creating module that uses DeqIO should not cause infinite recursion") {
    assert(execute{ new ReaderExampleTester })
  }
}
