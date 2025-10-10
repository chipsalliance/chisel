// SPDX-License-Identifier: Apache-2.0

package chiselTests.simulator

import chisel3.simulator.{HasSimulator, Simulator}
import chisel3.testing.HasTestingDirectory
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

class HasSimulatorSpec extends AnyFunSpec with Matchers {

  /** Return a field from the simulator so that we can check its type. */
  def foo(implicit a: HasSimulator): String = a.getSimulator.tag

  describe("HasSimulator") {

    it("should use Verilator as a low-priority default") {

      foo should be("verilator")

    }

    it("should allow overriding via an implicit val") {

      implicit val bar: HasSimulator = new HasSimulator {
        override def getSimulator(implicit testingDirectory: HasTestingDirectory): Simulator[svsim.verilator.Backend] =
          new Simulator[svsim.verilator.Backend] {
            override val backend = svsim.verilator.Backend.initializeFromProcessEnvironment()
            override val tag = "still-verilator"
            override val commonCompilationSettings = svsim.CommonCompilationSettings()
            override val backendSpecificCompilationSettings = svsim.verilator.Backend.CompilationSettings.default
            override val workspacePath = ""
          }
      }

      foo should be("still-verilator")

    }

  }

}
