// See LICENSE for license details.

package chiselTests

import org.scalatest._
import Chisel._
import Chisel.testers.BasicTester

class SinglePrintfTester() extends BasicTester {
  val x = UInt(254)
  printf("x=%x", x)
  stop()
}

class MultiPrintfTester() extends BasicTester {
  val x = UInt(254)
  val y = UInt(255)
  printf("x=%x y=%x", x, y)
  stop()
}

class PrintfSpec extends ChiselFlatSpec {
  "A printf with a single argument" should "run" in {
    assert(execute{ new SinglePrintfTester })
  }
  "A printf with multiple arguments" should "run" in {
    assert(execute{ new MultiPrintfTester })
  }
}
