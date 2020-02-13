// See README.md for license details.

package chiselTests

import chisel3._
import chisel3.experimental._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers

class ScalaIntervalSimulatorSpec extends AnyFreeSpec with Matchers {
  "clip tests" - {
    "Should work for closed ranges" in {
      val sim = ScalaIntervalSimulator(range"[2,4]")
      sim.clip(BigDecimal(1.0)) should be (2.0)
      sim.clip(BigDecimal(2.0)) should be (2.0)
      sim.clip(BigDecimal(3.0)) should be (3.0)
      sim.clip(BigDecimal(4.0)) should be (4.0)
      sim.clip(BigDecimal(5.0)) should be (4.0)
    }
    "Should work for closed ranges with binary point" in {
      val sim = ScalaIntervalSimulator(range"[2,6].2")
      sim.clip(BigDecimal(1.75)) should be (2.0)
      sim.clip(BigDecimal(2.0))  should be (2.0)
      sim.clip(BigDecimal(2.25)) should be (2.25)
      sim.clip(BigDecimal(2.5))  should be (2.5)
      sim.clip(BigDecimal(5.75)) should be (5.75)
      sim.clip(BigDecimal(6.0))  should be (6.0)
      sim.clip(BigDecimal(6.25)) should be (6.0)
      sim.clip(BigDecimal(6.5))  should be (6.0)
      sim.clip(BigDecimal(8.5))  should be (6.0)
    }
    "Should work for open ranges" in {
      val sim = ScalaIntervalSimulator(range"(2,4)")
      sim.clip(BigDecimal(1.0)) should be (3.0)
      sim.clip(BigDecimal(2.0)) should be (3.0)
      sim.clip(BigDecimal(3.0)) should be (3.0)
      sim.clip(BigDecimal(4.0)) should be (3.0)
      sim.clip(BigDecimal(5.0)) should be (3.0)
    }
    "Should work for open ranges with binary point" in {
      val sim = ScalaIntervalSimulator(range"(2,6).2")
      sim.clip(BigDecimal(1.75)) should be (2.25)
      sim.clip(BigDecimal(2.0))  should be (2.25)
      sim.clip(BigDecimal(2.25)) should be (2.25)
      sim.clip(BigDecimal(2.5))  should be (2.5)
      sim.clip(BigDecimal(5.75)) should be (5.75)
      sim.clip(BigDecimal(6.0))  should be (5.75)
      sim.clip(BigDecimal(6.25)) should be (5.75)
      sim.clip(BigDecimal(6.5))  should be (5.75)
      sim.clip(BigDecimal(8.5))  should be (5.75)
    }
  }
  "wrap tests" - {
    "Should work for closed ranges" in {
      val sim = ScalaIntervalSimulator(range"[2,6]")
      sim.wrap(BigDecimal(1.0)) should be (6.0)
      sim.wrap(BigDecimal(2.0)) should be (2.0)
      sim.wrap(BigDecimal(3.0)) should be (3.0)
      sim.wrap(BigDecimal(4.0)) should be (4.0)
      sim.wrap(BigDecimal(5.0)) should be (5.0)
      sim.wrap(BigDecimal(6.0)) should be (6.0)
      sim.wrap(BigDecimal(7.0)) should be (2.0)
    }
    "Should work for closed ranges with binary point" in {
      val sim = ScalaIntervalSimulator(range"[2,6].2")
      sim.wrap(BigDecimal(1.75)) should be (6.0)
      sim.wrap(BigDecimal(2.0))  should be (2.0)
      sim.wrap(BigDecimal(2.25)) should be (2.25)
      sim.wrap(BigDecimal(2.5))  should be (2.5)
      sim.wrap(BigDecimal(5.75)) should be (5.75)
      sim.wrap(BigDecimal(6.0))  should be (6.0)
      sim.wrap(BigDecimal(6.25))  should be (2.0)
      sim.wrap(BigDecimal(6.5))  should be (2.25)
    }
    "Should work for open ranges" in {
      val sim = ScalaIntervalSimulator(range"(2,6)")
      sim.wrap(BigDecimal(1.0)) should be (4.0)
      sim.wrap(BigDecimal(2.0)) should be (5.0)
      sim.wrap(BigDecimal(3.0)) should be (3.0)
      sim.wrap(BigDecimal(4.0)) should be (4.0)
      sim.wrap(BigDecimal(5.0)) should be (5.0)
      sim.wrap(BigDecimal(6.0)) should be (3.0)
      sim.wrap(BigDecimal(7.0)) should be (4.0)
    }
    "Should work for open ranges with binary point" in {
      val sim = ScalaIntervalSimulator(range"(2,6).2")
      sim.wrap(BigDecimal(1.75)) should be (5.5)
      sim.wrap(BigDecimal(2.0)) should be (5.75)
      sim.wrap(BigDecimal(2.25)) should be (2.25)
      sim.wrap(BigDecimal(2.5)) should be (2.5)
      sim.wrap(BigDecimal(5.75)) should be (5.75)
      sim.wrap(BigDecimal(6.0)) should be (2.25)
      sim.wrap(BigDecimal(6.25)) should be (2.5)
      sim.wrap(BigDecimal(7.0)) should be (3.25)
    }
  }
}
