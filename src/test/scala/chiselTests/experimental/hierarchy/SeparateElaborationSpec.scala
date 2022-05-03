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

  describe("(2): Multiple imported Definitions") {
    it(
      "(2.a): should work if a list of imported Definitions is passed between Stages."
    ) {
      val testDir = createTestDirectory(this.getClass.getSimpleName).toString

      val dutAnnos0 = (new ChiselStage).run(
        Seq(
          ChiselGeneratorAnnotation(() => new AddOneParameterized(4)),
          TargetDirAnnotation(s"$testDir/dutDef0")
        )
      )

      // Grab DUT definition to pass into testbench
      val designDefs0 = dutAnnos0.flatMap { a =>
        a match {
          case a: DesignAnnotation[_] =>
            Some(a.design.asInstanceOf[AddOneParameterized].toDefinition)
          case _ => None
        }
      }
      require(designDefs0.length == 1, s"Exactly one DesignAnnotation should exist, but found: $designDefs0.")
      val dutDef0 = designDefs0.head

      val dutAnnos1 = (new ChiselStage).run(
        Seq(
          ChiselGeneratorAnnotation(() => new AddOneParameterized(8)),
          TargetDirAnnotation(s"$testDir/dutDef1"),
          // pass in previously elaborated Definitions
          ImportedDefinitionAnnotation(dutDef0)
        )
      )

      // Grab DUT definition to pass into testbench
      val designDefs1 = dutAnnos1.flatMap { a =>
        a match {
          case a: DesignAnnotation[_] =>
            Some(a.design.asInstanceOf[AddOneParameterized].toDefinition)
          case _ => None
        }
      }
      require(designDefs1.length == 1, s"Exactly one DesignAnnotation should exist, but found: $designDefs1.")
      val dutDef1 = designDefs1.head

      class Testbench(defn0: Definition[AddOneParameterized], defn1: Definition[AddOneParameterized]) extends Module {
        val inst0 = Instance(defn0)
        val inst1 = Instance(defn1)

        // Tie inputs to a value so ChiselStage does not complain
        inst0.in := 0.U
        inst1.in := 0.U
      }

      (new ChiselStage).run(
        Seq(
          ChiselGeneratorAnnotation(() => new Testbench(dutDef0, dutDef1)),
          TargetDirAnnotation(testDir),
          ImportedDefinitionAnnotation(dutDef0),
          ImportedDefinitionAnnotation(dutDef1)
        )
      )

      val dutDef0_rtl = Source.fromFile(s"$testDir/dutDef0/AddOneParameterized.v").getLines.mkString
      dutDef0_rtl should include("module AddOneParameterized(")
      val dutDef1_rtl = Source.fromFile(s"$testDir/dutDef1/AddOneParameterized_1.v").getLines.mkString
      dutDef1_rtl should include("module AddOneParameterized_1(")

      val tb_rtl = Source.fromFile(s"$testDir/Testbench.v").getLines.mkString
      tb_rtl should include("AddOneParameterized inst0 (")
      tb_rtl should include("AddOneParameterized_1 inst1 (")
      (tb_rtl should not).include("module AddOneParameterized(")
      (tb_rtl should not).include("module AddOneParameterized_1(")
    }

    it(
      "(2.b): should throw an exception if information is not passed between Stages."
    ) {
      val testDir = createTestDirectory(this.getClass.getSimpleName).toString

      val dutAnnos0 = (new ChiselStage).run(
        Seq(
          ChiselGeneratorAnnotation(() => new AddOneParameterized(4)),
          TargetDirAnnotation(s"$testDir/dutDef0")
        )
      )

      // Grab DUT definition to pass into testbench
      val designDefs0 = dutAnnos0.flatMap { a =>
        a match {
          case a: DesignAnnotation[_] =>
            Some(a.design.asInstanceOf[AddOneParameterized].toDefinition)
          case _ => None
        }
      }
      require(designDefs0.length == 1, s"Exactly one DesignAnnotation should exist, but found: $designDefs0.")
      val dutDef0 = designDefs0.head

      val dutAnnos1 = (new ChiselStage).run(
        Seq(
          ChiselGeneratorAnnotation(() => new AddOneParameterized(8)),
          TargetDirAnnotation(s"$testDir/dutDef1"),
        )
      )

      // Grab DUT definition to pass into testbench
      val designDefs1 = dutAnnos1.flatMap { a =>
        a match {
          case a: DesignAnnotation[_] =>
            Some(a.design.asInstanceOf[AddOneParameterized].toDefinition)
          case _ => None
        }
      }
      require(designDefs1.length == 1, s"Exactly one DesignAnnotation should exist, but found: $designDefs1.")
      val dutDef1 = designDefs1.head

      class Testbench(defn0: Definition[AddOneParameterized], defn1: Definition[AddOneParameterized]) extends Module {
        val inst0 = Instance(defn0)
        val inst1 = Instance(defn1)

        // Tie inputs to a value so ChiselStage does not complain
        inst0.in := 0.U
        inst1.in := 0.U
      }

      // TODO assertThrows()
      (new ChiselStage).run(
        Seq(
          ChiselGeneratorAnnotation(() => new Testbench(dutDef0, dutDef1)),
          TargetDirAnnotation(testDir),
          ImportedDefinitionAnnotation(dutDef0),
          ImportedDefinitionAnnotation(dutDef1)
        )
      )

      // Because these elaborations have no knowledge of each other, they create
      // modules of the same name
      val dutDef0_rtl = Source.fromFile(s"$testDir/dutDef0/AddOneParameterized.v").getLines.mkString
      dutDef0_rtl should include("module AddOneParameterized(")
      val dutDef1_rtl = Source.fromFile(s"$testDir/dutDef1/AddOneParameterized_1.v").getLines.mkString
      dutDef1_rtl should include("module AddOneParameterized(")
    }
  }

}
