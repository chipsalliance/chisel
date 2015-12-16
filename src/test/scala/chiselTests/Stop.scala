// See LICENSE for license details.

package chiselTests

import org.scalatest._
import Chisel._
import Chisel.testers.BasicTester

class StopTester() extends BasicTester {
  stop()
}

class StopSpec extends ChiselFlatSpec {
  "stop()" should "stop and succeed the testbench" in {
    assertTesterPasses { new StopTester }
  }
}
