// SPDX-License-Identifier: Apache-2.0

package chiselTests.testing.scalatest

import chisel3._
import chisel3.simulator.SimulatorAPI
import chisel3.testing.scalatest.TestingDirectory
import java.nio.file.FileSystems
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import scala.reflect.io.Directory

class TestingDirectorySpec extends AnyFunSpec with Matchers with SimulatorAPI with TestingDirectory {

  private class Foo extends Module {
    stop()
  }

  /** Check that the directory structure and the files contained within make sense
    * for a Chiselsim/svsim build.
    */
  private def checkDirectoryStructure[A](dir: String, subDirs: String*): Unit = {

    val directory = Directory(
      FileSystems.getDefault
        .getPath(
          dir,
          subDirs: _*
        )
        .toFile
    )
    directory.deleteRecursively()

    simulate(new Foo) { _ => }

    val allFiles = directory.deepFiles.toSeq.map(_.toString).toSet
    for (
      file <- Seq(
        directory.toFile.toString + "/workdir-verilator/Makefile",
        directory.toFile.toString + "/primary-sources/Foo.sv"
      )
    ) {
      info(s"found expected file: '$file'")
      allFiles should contain(file)
    }

  }

  describe("A test suite mixing in WithTestingDirectory") {

    it("should generate a directory structure derived from the suite and test name") {
      checkDirectoryStructure(
        "build",
        "chiselsim",
        "TestingDirectorySpec",
        "A-test-suite-mixing-in-WithTestingDirectory",
        "should-generate-a-directory-structure-derived-from-the-suite-and-test-name"
      )
    }

    it("should generate another directory, too") {
      checkDirectoryStructure(
        "build",
        "chiselsim",
        "TestingDirectorySpec",
        "A-test-suite-mixing-in-WithTestingDirectory",
        "should-generate-another-directory--too"
      )
    }

    it("should mangle emojis, e.g., 🚀") {
      checkDirectoryStructure(
        "build",
        "chiselsim",
        "TestingDirectorySpec",
        "A-test-suite-mixing-in-WithTestingDirectory",
        "should-mangle-emojis--e.g.---"
      )
    }

    it("should mangle CJK characters, e.g., 好猫咪") {
      checkDirectoryStructure(
        "build",
        "chiselsim",
        "TestingDirectorySpec",
        "A-test-suite-mixing-in-WithTestingDirectory",
        "should-mangle-CJK-characters--e.g.-----"
      )
    }

  }

}
