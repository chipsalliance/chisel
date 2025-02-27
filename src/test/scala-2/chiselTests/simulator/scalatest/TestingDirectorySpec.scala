// SPDX-License-Identifier: Apache-2.0

package chiselTests.simulator.scalatest

import chisel3._
import chisel3.testing.scalatest.TestingDirectory
import chisel3.simulator.DefaultSimulator._
import java.nio.file.FileSystems
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import scala.reflect.io.Directory

class TestingDirectorySpec extends AnyFunSpec with Matchers with TestingDirectory {

  /** Check that the directory structure and the files contained within make sense
    * for a Chiselsim/svsim build.
    */
  private def checkDirectoryStructure[A](dir: String, subDirs: String*)(thunk: => A): Unit = {

    val directory = Directory(
      FileSystems.getDefault
        .getPath(
          dir,
          subDirs: _*
        )
        .toFile
    )
    directory.deleteRecursively()

    thunk

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

  private class Foo extends Module {
    stop()
  }

  describe("A test suite mixing in WithTestingDirectory") {

    it("should generate a directory structure derived from the suite and test name") {
      checkDirectoryStructure(
        "build",
        "WithTestingDirectorySpec",
        "A-test-suite-mixing-in-WithTestingDirectory",
        "should-generate-a-directory-structure-derived-from-the-suite-and-test-name"
      ) {
        simulate(new Foo()) { _ => }
      }
    }

    it("should generate another directory, too") {
      checkDirectoryStructure(
        "build",
        "WithTestingDirectorySpec",
        "A-test-suite-mixing-in-WithTestingDirectory",
        "should-generate-another-directory,-too"
      ) {
        simulate(new Foo()) { _ => }
      }
    }

    it("should handle emojis, e.g., 🚀") {
      checkDirectoryStructure(
        "build",
        "WithTestingDirectorySpec",
        "A-test-suite-mixing-in-WithTestingDirectory",
        "should-handle-emojis,-e.g.,-🚀"
      ) {
        simulate(new Foo()) { _ => }
      }
    }

    it("should handle CJK characters, e.g., 好猫咪") {
      checkDirectoryStructure(
        "build",
        "WithTestingDirectorySpec",
        "A-test-suite-mixing-in-WithTestingDirectory",
        "should-handle-CJK-characters,-e.g.,-好猫咪"
      ) {
        simulate(new Foo()) { _ => }
      }
    }

  }

}
