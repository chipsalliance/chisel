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

class ASCIIPrintfTester() extends BasicTester {
  // TODO: use this more aggressive test once FIRRTL issue #59 is resolved.
  //printf((1 to 127).map(_ toChar) mkString)
  printf("\"\'\\\n")
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
    assertTesterPasses { new SinglePrintfTester }
  }
  "A printf with multiple arguments" should "run" in {
    assertTesterPasses { new MultiPrintfTester }
  }
  "A printf with ASCII characters 1-127" should "run" in {
    assertTesterPasses { new ASCIIPrintfTester }
  }
}
