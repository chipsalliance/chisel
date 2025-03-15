// SPDX-License-Identifier: Apache-2.0

package chiselTests.simulator.scalatest

import chisel3.simulator.HasSimulator
import chisel3.simulator.scalatest.{Cli, HasCliOptions}
import chisel3.testing.scalatest.HasConfigMap
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
