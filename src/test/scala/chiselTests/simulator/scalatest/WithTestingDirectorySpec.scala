// SPDX-License-Identifier: Apache-2.0

package chiselTests.simulator.scalatest

import chisel3._
import chisel3.simulator.scalatest.WithTestingDirectory
import chisel3.simulator.DefaultSimulator._
import java.nio.file.FileSystems
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import scala.reflect.io.Directory

class WithTestingDirectorySpec extends AnyFunSpec with Matchers with WithTestingDirectory {

  class Foo extends Module {
    stop()
  }

  describe("A test suite mixing in WithTestingDirectory") {
    it("should generate a directory structure derived from the suite and test name") {

      val directory = Directory(
        FileSystems
          .getDefault()
          .getPath(
            "test_run_dir",
            "WithTestingDirectorySpec",
            "A_test_suite_mixing_in_WithTestingDirectory",
            "should_generate_a_directory_structure_derived_from_the_suite_and_test_name"
          )
          .toFile()
      )
      directory.deleteRecursively()

      simulate(new Foo()) { _ => }

      info(s"found expected directory: '$directory'")
      assert(directory.exists)
      assert(directory.isDirectory)

      val allFiles = directory.deepFiles.toSeq.map(_.toString).toSet
      for (
        file <- Seq(
          directory.toFile.toString + "/workdir-default/Makefile",
          directory.toFile.toString + "/primary-sources/Foo.sv"
        )
      ) {
        info(s"found expected file: '$file'")
        allFiles should contain(file)
      }
    }
    it("should generate another directory, too") {
      val directory = Directory(
        FileSystems
          .getDefault()
          .getPath(
            "test_run_dir",
            "WithTestingDirectorySpec",
            "A_test_suite_mixing_in_WithTestingDirectory",
            "should_generate_another_directory_too"
          )
          .toFile()
      )
      directory.deleteRecursively()

      simulate(new Foo()) { _ => }

      info(s"found expected directory: '$directory'")
      assert(directory.exists)
      assert(directory.isDirectory)
    }
  }

}
