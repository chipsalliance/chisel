// SPDX-License-Identifier: Apache-2.0

package chiselTests.simulator.scalatest

import chisel3._
import chisel3.ltl.AssertProperty
import chisel3.ltl.Sequence.BoolSequence
import chisel3.simulator.HasSimulator
import chisel3.simulator.scalatest.{ChiselSim, Cli, HasCliOptions}
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.testing.scalatest.HasConfigMap
import chisel3.util.Counter
import org.scalatest.TestSuite
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class HasCliOptionsSpec extends AnyFlatSpec with Matchers with HasCliOptions with Cli.Simulator {

  override protected def defaultCliSimulator: Option[HasSimulator] = None

  behavior of "Cli.Simulator"

  it should "should be capable of forcing a user to specify a simulator" in {

    intercept[IllegalArgumentException] {
      implicitly[HasSimulator]
    }.getMessage should include("a simulator must be provided to this test using '-Dsimulator=<simulator-name>'")

  }

}

class HasCliOptionsSpecVerilatorTemporalLayers extends AnyFlatSpec with ChiselSim with Cli.Simulator {

  behavior of "Simulator CLI"

  it should "disable temporal layers automatically for Verilator" in {

    class Foo extends Module {
      val a = IO(Input(Bool()))
      layer.block(layers.Verification.Assert.Temporal) {
        AssertProperty(a.eventually)
      }
      when (Counter(true.B, 2)._2) {
        stop()
      }
    }

    simulate(new Foo) { foo =>
      foo.a.poke(true.B)
      RunUntilFinished(4)
    }

  }

}
