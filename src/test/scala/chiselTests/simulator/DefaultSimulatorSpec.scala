// SPDX-License-Identifier: Apache-2.0

package chiselTests.simulator

import chisel3._
import chisel3.simulator.HasTestingDirectory
import chisel3.simulator.DefaultSimulator._
import java.nio.file.FileSystems
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import scala.reflect.io.Directory

class DefaultSimulatorSpec extends AnyFunSpec with Matchers {
  class Foo extends Module {
    stop()
  }

  describe("DefaultSimulator") {
    it("writes files to a permanent directory on disk") {

      /** An implementation that always writes to the subdirectory "test_run_dir/<class-name>/foo/" */
      implicit val fooDirectory = new HasTestingDirectory {
        override def getDirectory(testClassName: String) =
          FileSystems.getDefault().getPath("test_run_dir", testClassName, "foo")
      }

      val directory = Directory(FileSystems.getDefault().getPath("test_run_dir", "DefaultSimulator", "foo").toFile())
      directory.deleteRecursively()

      simulate(new Foo()) { _ => }

      info(s"found expected directory: '$directory'")
      assert(directory.exists)
      assert(directory.isDirectory)

      val allFiles = directory.deepFiles.toSeq.map(_.toString).toSet
      for (
        file <- Seq(
          "test_run_dir/DefaultSimulator/foo/workdir-default/Makefile",
          "test_run_dir/DefaultSimulator/foo/primary-sources/Foo.sv"
        )
      ) {
        info(s"found expected file: '$file'")
        allFiles should contain(file)
      }
    }
  }
}
