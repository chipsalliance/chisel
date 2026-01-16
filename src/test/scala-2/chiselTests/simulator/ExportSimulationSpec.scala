package chiselTests.simulator

import chisel3._
import chisel3.simulator._
import chisel3.testing.HasTestingDirectory
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import java.io.File
import java.nio.file.Paths

class ExportSimulationSpec extends AnyFunSpec with Matchers with ChiselSim {

  describe("exportSimulation") {
    it("should generate FIRRTL, testbench, and ninja files") {
      implicit val testingDirectory: HasTestingDirectory = new HasTestingDirectory {
        override def getDirectory = Paths.get("test_run_dir/export_simulation_test")
      }

      // Clean up any previous test artifacts
      val workspaceDir = new File("test_run_dir/export_simulation_test")
      if (workspaceDir.exists()) {
        def deleteRecursively(file: File): Unit = {
          if (file.isDirectory) {
            file.listFiles().foreach(deleteRecursively)
          }
          file.delete()
        }
        deleteRecursively(workspaceDir)
      }

      // Export the simulation
      val exported = exportSimulation(new GCD(), "chiselTests.simulator.GCDTestMain")

      // Verify the exported files exist
      new File(exported.workspacePath).exists() shouldBe true
      new File(exported.firFilePath).exists() shouldBe true
      new File(exported.ninjaFilePath).exists() shouldBe true

      // Verify the generated sources exist
      val generatedSourcesPath = s"${exported.workspacePath}/generated-sources"
      new File(s"$generatedSourcesPath/testbench.sv").exists() shouldBe true
      new File(s"$generatedSourcesPath/simulation-driver.cpp").exists() shouldBe true
      new File(s"$generatedSourcesPath/c-dpi-bridge.cpp").exists() shouldBe true

      // Verify the module info JSON exists
      val supportArtifactsPath = s"${exported.workspacePath}/support-artifacts"
      new File(s"$supportArtifactsPath/module-info.json").exists() shouldBe true

      // Verify the ninja file contains expected targets
      val ninjaContent = scala.io.Source.fromFile(exported.ninjaFilePath).mkString
      ninjaContent should include("rule firtool")
      ninjaContent should include("rule verilator")
      ninjaContent should include("build verilog:")
      ninjaContent should include("build verilate:")
      // Note: individual test targets (test1, test2, etc.) are only generated when
      // test descriptions are provided via ChiselSimSuite
    }
  }
}

