// SPDX-License-Identifier: Apache-2.0

package circtTests.stage

import circt.stage.CIRCTStage

import firrtl.stage.{FirrtlFileAnnotation, OutputFileAnnotation}

import java.io.File
import java.io.{PrintWriter, Writer}

import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

import scala.io.Source

class CIRCTStageSpec extends AnyFunSpec with Matchers {

  private def writeFile(file: File, string: String): Unit = {
    val writer = {
      file.getParentFile.mkdirs()
      new PrintWriter(file)
    }
    writer.write(string)
    writer.close()
  }

  describe("CIRCTStage") {

    it("should compile a FIRRTL file to Verilog") {

      val input =
        """|circuit Foo:
           |  module Foo:
           |    input a: UInt<1>
           |    output b: UInt<1>
           |    b <= not(a)
           |""".stripMargin

      val targetDir = new File("test_run_dir/CIRCTStage")
      val inputFile = new File(targetDir, "Foo.fir")

      writeFile(inputFile, input)

      val outputFile = new File(targetDir, "Foo.sv")
      outputFile.delete()

      val stage = new CIRCTStage

      stage.execute(
        Array(
          "--target",
          "systemverilog",
          "--target-dir",
          targetDir.toString,
          "--input-file",
          inputFile.toString
        ),
        Seq.empty
      )

      info(s"output file '$outputFile' was created")
      outputFile should exist

      info(s"file looks like Verilog")
      Source.fromFile(outputFile).getLines.mkString should include("endmodule")

    }

  }

}
