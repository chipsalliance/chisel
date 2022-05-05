// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental.hierarchy

import chiselTests.ChiselFunSpec
import chisel3._
import chisel3.experimental.BaseModule
import chisel3.stage.{ChiselCircuitAnnotation, ChiselGeneratorAnnotation, ChiselStage, DesignAnnotation}
import chisel3.experimental.hierarchy.{Definition, Instance}
import chisel3.experimental.hierarchy.core.ImportDefinitionAnnotation
import firrtl.AnnotationSeq
import firrtl.options.TargetDirAnnotation

import java.nio.file.Paths
import scala.io.Source

class SeparateElaborationSpec extends ChiselFunSpec with Utils {
  import Examples._

  /** Return a [[DesignAnnotation]] from a list of annotations. */
  private def getDesignAnnotation[T <: RawModule](annos: AnnotationSeq): DesignAnnotation[T] = {
    val designAnnos = annos.flatMap { a =>
      a match {
        case a: DesignAnnotation[T] => Some(a)
        case _ => None
      }
    }
    require(designAnnos.length == 1, s"Exactly one DesignAnnotation should exist, but found: $designAnnos.")
    designAnnos.head
  }

  /** Elaborates [[AddOne]] and returns its [[Definition]]. */
  private def getAddOneDefinition(testDir: String): Definition[AddOne] = {
    val dutAnnos = (new ChiselStage).run(
      Seq(
        ChiselGeneratorAnnotation(() => new AddOne),
        TargetDirAnnotation(testDir)
      )
    )

    // Grab DUT definition to pass into testbench
    getDesignAnnotation(dutAnnos).design.asInstanceOf[AddOne].toDefinition
  }

  /** Return [[Definition]]s of all modules in a circuit. */
  private def allModulesToImportedDefs(annos: AnnotationSeq): Seq[ImportDefinitionAnnotation[_]] = {
    annos.flatMap { a =>
      a match {
        case a: ChiselCircuitAnnotation =>
          a.circuit.components.map { c => ImportDefinitionAnnotation(c.id.toDefinition) }
        case _ => Seq.empty
      }
    }
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
          ImportDefinitionAnnotation(dutDef)
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
          ImportDefinitionAnnotation(dutDef)
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
        annotations = Seq(ImportDefinitionAnnotation(dutDef))
      )
    }
  }

  describe("(2): Multiple imported Definitions of modules without submodules") {
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
      val dutDef0 = getDesignAnnotation(dutAnnos0).design.asInstanceOf[AddOneParameterized].toDefinition

      val dutAnnos1 = (new ChiselStage).run(
        Seq(
          ChiselGeneratorAnnotation(() => new AddOneParameterized(8)),
          TargetDirAnnotation(s"$testDir/dutDef1"),
          // pass in previously elaborated Definitions
          ImportDefinitionAnnotation(dutDef0)
        )
      )
      val dutDef1 = getDesignAnnotation(dutAnnos1).design.asInstanceOf[AddOneParameterized].toDefinition

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
          ImportDefinitionAnnotation(dutDef0),
          ImportDefinitionAnnotation(dutDef1)
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
      val dutDef0 = getDesignAnnotation(dutAnnos0).design.asInstanceOf[AddOneParameterized].toDefinition

      val dutAnnos1 = (new ChiselStage).run(
        Seq(
          ChiselGeneratorAnnotation(() => new AddOneParameterized(8)),
          TargetDirAnnotation(s"$testDir/dutDef1")
        )
      )
      val dutDef1 = getDesignAnnotation(dutAnnos1).design.asInstanceOf[AddOneParameterized].toDefinition

      class Testbench(defn0: Definition[AddOneParameterized], defn1: Definition[AddOneParameterized]) extends Module {
        val inst0 = Instance(defn0)
        val inst1 = Instance(defn1)

        // Tie inputs to a value so ChiselStage does not complain
        inst0.in := 0.U
        inst1.in := 0.U
      }

      // Because these elaborations have no knowledge of each other, they create
      // modules of the same name
      val dutDef0_rtl = Source.fromFile(s"$testDir/dutDef0/AddOneParameterized.v").getLines.mkString
      dutDef0_rtl should include("module AddOneParameterized(")
      val dutDef1_rtl = Source.fromFile(s"$testDir/dutDef1/AddOneParameterized.v").getLines.mkString
      dutDef1_rtl should include("module AddOneParameterized(")

      val errMsg = intercept[ChiselException] {
        (new ChiselStage).run(
          Seq(
            ChiselGeneratorAnnotation(() => new Testbench(dutDef0, dutDef1)),
            TargetDirAnnotation(testDir),
            ImportDefinitionAnnotation(dutDef0),
            ImportDefinitionAnnotation(dutDef1)
          )
        )
      }
      errMsg.getMessage should include(
        "Expected distinct imported Definition names but found duplicates for: AddOneParameterized"
      )
    }
  }

  describe("(3): Multiple imported Definitions of modules with submodules") {
    it(
      "(3.a): should work if a list of imported Definitions for all modules is passed between Stages."
    ) {
      val testDir = createTestDirectory(this.getClass.getSimpleName).toString

      val dutAnnos0 = (new ChiselStage).run(
        Seq(
          ChiselGeneratorAnnotation(() => new AddTwoMixedModules),
          TargetDirAnnotation(s"$testDir/dutDef0")
        )
      )
      val dutDef0 = getDesignAnnotation(dutAnnos0).design.asInstanceOf[AddTwoMixedModules].toDefinition
      val importDefinitionAnnos0 = allModulesToImportedDefs(dutAnnos0)

      val dutAnnos1 = (new ChiselStage).run(
        Seq(
          ChiselGeneratorAnnotation(() => new AddTwoMixedModules),
          TargetDirAnnotation(s"$testDir/dutDef1")
        ) ++ importDefinitionAnnos0
      )
      val dutDef1 = getDesignAnnotation(dutAnnos1).design.asInstanceOf[AddTwoMixedModules].toDefinition
      val importDefinitionAnnos1 = allModulesToImportedDefs(dutAnnos1)

      class Testbench(defn0: Definition[AddTwoMixedModules], defn1: Definition[AddTwoMixedModules]) extends Module {
        val inst0 = Instance(defn0)
        val inst1 = Instance(defn1)

        // Tie inputs to a value so ChiselStage does not complain
        inst0.in := 0.U
        inst1.in := 0.U
      }

      val dutDef0_rtl = Source.fromFile(s"$testDir/dutDef0/AddTwoMixedModules.v").getLines.mkString
      dutDef0_rtl should include("module AddOne(")
      dutDef0_rtl should include("module AddTwoMixedModules(")
      val dutDef1_rtl = Source.fromFile(s"$testDir/dutDef1/AddTwoMixedModules_1.v").getLines.mkString
      dutDef1_rtl should include("module AddOne_2(")
      dutDef1_rtl should include("module AddTwoMixedModules_1(")

      (new ChiselStage).run(
        Seq(
          ChiselGeneratorAnnotation(() => new Testbench(dutDef0, dutDef1)),
          TargetDirAnnotation(testDir)
        ) ++ importDefinitionAnnos0 ++ importDefinitionAnnos1
      )

      val tb_rtl = Source.fromFile(s"$testDir/Testbench.v").getLines.mkString
      tb_rtl should include("AddTwoMixedModules inst0 (")
      tb_rtl should include("AddTwoMixedModules_1 inst1 (")
      (tb_rtl should not).include("module AddTwoMixedModules(")
      (tb_rtl should not).include("module AddTwoMixedModules_1(")
    }
  }

  it(
    "(3.b): should throw an exception if submodules are not passed between Definition elaborations."
  ) {
    val testDir = createTestDirectory(this.getClass.getSimpleName).toString

    val dutAnnos0 = (new ChiselStage).run(
      Seq(
        ChiselGeneratorAnnotation(() => new AddTwoMixedModules),
        TargetDirAnnotation(s"$testDir/dutDef0")
      )
    )
    val dutDef0 = getDesignAnnotation(dutAnnos0).design.asInstanceOf[AddTwoMixedModules].toDefinition
    val importDefinitionAnnos0 = allModulesToImportedDefs(dutAnnos0)

    val dutAnnos1 = (new ChiselStage).run(
      Seq(
        ChiselGeneratorAnnotation(() => new AddTwoMixedModules),
        ImportDefinitionAnnotation(dutDef0),
        TargetDirAnnotation(s"$testDir/dutDef1")
      )
    )
    val dutDef1 = getDesignAnnotation(dutAnnos1).design.asInstanceOf[AddTwoMixedModules].toDefinition
    val importDefinitionAnnos1 = allModulesToImportedDefs(dutAnnos1)

    class Testbench(defn0: Definition[AddTwoMixedModules], defn1: Definition[AddTwoMixedModules]) extends Module {
      val inst0 = Instance(defn0)
      val inst1 = Instance(defn1)

      // Tie inputs to a value so ChiselStage does not complain
      inst0.in := 0.U
      inst1.in := 0.U
    }

    val dutDef0_rtl = Source.fromFile(s"$testDir/dutDef0/AddTwoMixedModules.v").getLines.mkString
    dutDef0_rtl should include("module AddOne(")
    dutDef0_rtl should include("module AddTwoMixedModules(")
    val dutDef1_rtl = Source.fromFile(s"$testDir/dutDef1/AddTwoMixedModules_1.v").getLines.mkString
    dutDef1_rtl should include("module AddOne(")
    dutDef1_rtl should include("module AddTwoMixedModules_1(")

    val errMsg = intercept[ChiselException] {
      (new ChiselStage).run(
        Seq(
          ChiselGeneratorAnnotation(() => new Testbench(dutDef0, dutDef1)),
          TargetDirAnnotation(testDir)
        ) ++ importDefinitionAnnos0 ++ importDefinitionAnnos1
      )
    }
    errMsg.getMessage should include(
      "Expected distinct imported Definition names but found duplicates for: AddOne"
    )
  }

}
