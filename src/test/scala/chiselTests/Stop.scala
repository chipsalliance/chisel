// See LICENSE for license details.

package chiselTests

import org.scalatest._
import chisel3._
import chisel3.testers.BasicTester

class StopTester() extends BasicTester {
  stop()
}

class StopSpec extends ChiselFlatSpec {
  "stop()" should "stop and succeed the testbench" in {
    assertTesterPasses { new StopTester }
  }
}
