// See LICENSE for license details.

package chiselTests

import org.scalatest._
import Chisel._
import Chisel.testers.BasicTester

class SinglePrintfTester() extends BasicTester {
  printf("done=%x", io.done)
  io.done := Bool(true)
  io.error := Bool(false)
}

class MultiPrintfTester() extends BasicTester {
  printf("done=%x error=%x", io.done, io.error)
  io.done := Bool(true)
  io.error := Bool(false)
}

class PrintfSpec extends ChiselFlatSpec {
  "A printf with a single argument" should "run" in {
    assert(execute{ new SinglePrintfTester })
  }
  "A printf with multiple arguments" should "run" in {
    assert(execute{ new MultiPrintfTester })
  }
}
