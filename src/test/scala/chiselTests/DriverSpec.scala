// See LICENSE for license details.

package chiselTests

import java.io.File

import chisel3._
import org.scalatest.{Matchers, FreeSpec}

class DriverSpec extends FreeSpec with Matchers {
  "Use Driver.execute to implement custom toolchains" - {
    "Driver.execute can" - {
      "elaborate circuit in memory" in {
        val executionResult = Driver.execute(() => new GCD, Array.empty[String])
        executionResult.circuit.name should be ("GCD")
        executionResult.emittedString should include ("module GCD")
        println(s"executing result is $executionResult")
      }
      "elaborate circuit to a file" in {
        val dirName = "./test_run_dir"
        Driver.execute(() => new GCD, Array("--target-dir", dirName))
        new File(s"$dirName/GCD.fir").exists() should be (true)
      }
    }
  }
}
