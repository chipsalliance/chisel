// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental.hierarchy

import chiselTests.ChiselFunSpec
import chisel3._
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage, DesignAnnotation}
import chisel3.internal.ImportedDefinitionAnnotation
import chisel3.experimental.hierarchy.{Definition, Instance}
import firrtl.options.TargetDirAnnotation

import java.nio.file.Paths
import scala.io.Source

class SeparateElaborationSpec extends ChiselFunSpec with Utils {
  import Examples._

  describe("(0): Elaborating an Instance separately from its Definition") {
    it("should result in an instantiation in FIRRTL without a module declaration.") {
      val testDir = createTestDirectory(this.getClass.getSimpleName).toString
      val dutAnnos = (new ChiselStage).run(Seq(
        ChiselGeneratorAnnotation(() => new AddOne),
        TargetDirAnnotation(testDir),
      ))

      // Grab DUT definition to pass into testbench
      val designAnnos = dutAnnos.flatMap { a =>
        a match {
          case a: DesignAnnotation[_] =>
            Some(a.design.asInstanceOf[AddOne].toDefinition)
          case _ => None
        }
      }
      require(designAnnos.length == 1, s"Exactly one DesignAnnotation should exist, but found: $designAnnos.")
      val dutDef = designAnnos.head

      class Testbench(defn: Definition[AddOne]) extends Module {
        // Make sure names do not conflict
        val mod = Module(new AddOne)
        val inst = Instance(defn)

        // Tie inputs to a value so ChiselStage does not complain
        mod.in := 0.U
        inst.in := 0.U
      }

      (new ChiselStage).run(Seq(
        ChiselGeneratorAnnotation(() => new Testbench(dutDef)),
        TargetDirAnnotation(testDir),
        ImportedDefinitionAnnotation(dutDef)
      ))

      // Check that the output RTL has only a module instantiation and no
      // module declaration.
      val tb_rtl = Source.fromFile(s"$testDir/Testbench.v").getLines.mkString
      tb_rtl should include("AddOne inst (")
      tb_rtl should not include("module AddOne")
    }
  }

  describe("(1): Elaborating an Instance and Definition together") {
    it("should not result in a repeat definition of the module.") {
      val testDir = createTestDirectory(this.getClass.getSimpleName).toString

      class Top extends Module {
        val inst = Instance(Definition(new AddOne))
        inst.in := 0.U
      }

      // If there is a repeat module definition, FIRRTL emission will fail
      (new ChiselStage).emitFirrtl(
        gen = new Top,
        args = Array("-td", testDir),
      )
    }
  }

  describe("(2): Elaborating multiple Instances separately from its Definition") {
    it ("should not result in a repeat definition of the module.") {
      val testDir = createTestDirectory(this.getClass.getSimpleName).toString
      val dutAnnos = (new ChiselStage).run(Seq(
        ChiselGeneratorAnnotation(() => new AddOne),
        TargetDirAnnotation(testDir),
      ))

      // Grab DUT definition to pass into testbench
      val designAnnos = dutAnnos.flatMap { a =>
        a match {
          case a: DesignAnnotation[_] =>
            Some(a.design.asInstanceOf[AddOne].toDefinition)
          case _ => None
        }
      }
      require(designAnnos.length == 1, s"Exactly one DesignAnnotation should exist, but found: $designAnnos.")
      val dutDef = designAnnos.head

      class Testbench(defn: Definition[AddOne]) extends Module {
        val inst0 = Instance(defn)
        val inst1 = Instance(defn)

        inst0.in := 0.U
        inst1.in := 0.U
      }

      // If there is a repeat module definition, FIRRTL emission will fail
      val firrtl = (new ChiselStage).emitFirrtl(
        gen = new Testbench(dutDef),
        args = Array("-td", testDir, "--full-stacktrace"),
        annotations = Seq(ImportedDefinitionAnnotation(dutDef))
      )
    }
  }
}