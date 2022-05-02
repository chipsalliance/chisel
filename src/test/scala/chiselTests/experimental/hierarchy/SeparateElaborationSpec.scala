// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental.hierarchy

import chiselTests.ChiselFunSpec
import chisel3._
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage, DesignAnnotation}
import chisel3.experimental.hierarchy.{Definition, Instance}
import chisel3.experimental.hierarchy.core.ImportedDefinitionAnnotation
import firrtl.options.TargetDirAnnotation

import java.nio.file.Paths
import scala.io.Source

class SeparateElaborationSpec extends ChiselFunSpec with Utils {
  import Examples._

  /** Elaborates [[AddOne]] and returns its [[Definition]]. */
  def getAddOneDefinition(testDir: String): Definition[AddOne] = {
    val dutAnnos = (new ChiselStage).run(
      Seq(
        ChiselGeneratorAnnotation(() => new AddOne),
        TargetDirAnnotation(testDir)
      )
    )

    // Grab DUT definition to pass into testbench
    val designDefs = dutAnnos.flatMap { a =>
      a match {
        case a: DesignAnnotation[_] =>
          Some(a.design.asInstanceOf[AddOne].toDefinition)
        case _ => None
      }
    }
    require(designDefs.length == 1, s"Exactly one DesignAnnotation should exist, but found: $designDefs.")
    designDefs.head
  }

  describe("(0): Name conflicts") {
    it("(0.a): should not occur between a Module and an Instance of a previously elaborated Definition.") {
      val testDir = createTestDirectory(this.getClass.getSimpleName).toString

      val dutDef = getAddOneDefinition(testDir)

      class Testbench(defn: Definition[AddOne]) extends Module {
        val mod = Module(new AddOne)
        val inst = Instance(defn)

        // Tie inputs to a value so ChiselStage does not complain
        mod.in := 0.U
        inst.in := 0.U
        dontTouch(mod.out)
      }

      (new ChiselStage).run(
        Seq(
          ChiselGeneratorAnnotation(() => new Testbench(dutDef)),
          TargetDirAnnotation(testDir),
          ImportedDefinitionAnnotation(dutDef)
        )
      )

      val tb_rtl = Source.fromFile(s"$testDir/Testbench.v").getLines.mkString
      tb_rtl should include("module AddOne_1(")
      tb_rtl should include("AddOne_1 mod (")
      (tb_rtl should not).include("module AddOne(")
      tb_rtl should include("AddOne inst (")
    }

    it(
      "(0.b): should not occur between an Instance of a Definition and an Instance of a previously elaborated Definition."
    ) {
      val testDir = createTestDirectory(this.getClass.getSimpleName).toString

      val dutDef = getAddOneDefinition(testDir)

      class Testbench(defn: Definition[AddOne]) extends Module {
        val inst0 = Instance(Definition(new AddOne))
        val inst1 = Instance(defn)

        // Tie inputs to a value so ChiselStage does not complain
        inst0.in := 0.U
        inst1.in := 0.U
        dontTouch(inst0.out)
      }

      (new ChiselStage).run(
        Seq(
          ChiselGeneratorAnnotation(() => new Testbench(dutDef)),
          TargetDirAnnotation(testDir),
          ImportedDefinitionAnnotation(dutDef)
        )
      )

      val tb_rtl = Source.fromFile(s"$testDir/Testbench.v").getLines.mkString
      tb_rtl should include("module AddOne_1(")
      tb_rtl should include("AddOne_1 inst0 (")
      (tb_rtl should not).include("module AddOne(")
      tb_rtl should include("AddOne inst1 (")
    }
  }

  describe("(1): Repeat Module definitions") {
    it("(1.a): should not occur when elaborating multiple Instances separately from its Definition.") {
      val testDir = createTestDirectory(this.getClass.getSimpleName).toString

      val dutDef = getAddOneDefinition(testDir)

      class Testbench(defn: Definition[AddOne]) extends Module {
        val inst0 = Instance(defn)
        val inst1 = Instance(defn)

        inst0.in := 0.U
        inst1.in := 0.U
      }

      // If there is a repeat module definition, FIRRTL emission will fail
      (new ChiselStage).emitFirrtl(
        gen = new Testbench(dutDef),
        args = Array("-td", testDir, "--full-stacktrace"),
        annotations = Seq(ImportedDefinitionAnnotation(dutDef))
      )
    }
  }
}
