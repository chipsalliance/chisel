// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental.hierarchy

import chiselTests.ChiselFunSpec
import chisel3._
import chisel3.experimental.BaseModule
import chisel3.stage.{ChiselCircuitAnnotation, ChiselGeneratorAnnotation, DesignAnnotation}
import chisel3.experimental.hierarchy.{Definition, Instance}
import chisel3.experimental.hierarchy.core.ImportDefinitionAnnotation
import circt.stage.{CIRCTTarget, CIRCTTargetAnnotation, ChiselStage}
import firrtl.AnnotationSeq
import firrtl.util.BackendCompilationUtilities.createTestDirectory

import java.nio.file.Paths
import scala.annotation.nowarn
import scala.io.Source

class SeparateElaborationSpec extends ChiselFunSpec with Utils {
  import Examples._

  /** Return a [[DesignAnnotation]] from a list of annotations. */
  @nowarn("msg=is unchecked since it is eliminated by erasure")
  private def getDesignAnnotation[T <: RawModule](annos: AnnotationSeq): DesignAnnotation[T] = {
    val designAnnos = annos.flatMap { a =>
      a match {
        case a: DesignAnnotation[T] => Some(a) //TODO: cleanup, T is necessary to make type of designAnnos right
        case _ => None
      }
    }
    require(designAnnos.length == 1, s"Exactly one DesignAnnotation should exist, but found: $designAnnos.")
    designAnnos.head
  }

  /** Elaborates [[AddOne]] and returns its [[Definition]]. */
  private def getAddOneDefinition(testDir: String): Definition[AddOne] = {
    val dutAnnos = (new ChiselStage).execute(
      Array("--target-dir", testDir, "--target", "systemverilog"),
      Seq(
        ChiselGeneratorAnnotation(() => new AddOne)
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

      (new ChiselStage).execute(
        Array("--target-dir", testDir, "--target", "systemverilog"),
        Seq(
          ChiselGeneratorAnnotation(() => new Testbench(dutDef)),
          ImportDefinitionAnnotation(dutDef)
        )
      )

      val tb_rtl = Source.fromFile(s"$testDir/Testbench.sv").getLines().mkString
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

      (new ChiselStage).execute(
        Array("--target-dir", testDir, "--target", "systemverilog"),
        Seq(
          ChiselGeneratorAnnotation(() => new Testbench(dutDef)),
          ImportDefinitionAnnotation(dutDef)
        )
      )

      val tb_rtl = Source.fromFile(s"$testDir/Testbench.sv").getLines().mkString
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
      (new ChiselStage).execute(
        args = Array("-td", testDir, "--full-stacktrace", "--target", "chirrtl"),
        annotations = Seq(ChiselGeneratorAnnotation(() => new Testbench(dutDef)), ImportDefinitionAnnotation(dutDef))
      )
    }
  }

  describe("(2): Multiple imported Definitions of modules without submodules") {
    it(
      "(2.a): should work if a list of imported Definitions is passed between Stages."
    ) {
      val testDir = createTestDirectory(this.getClass.getSimpleName).toString

      val dutAnnos0 = (new ChiselStage).execute(
        Array("--target-dir", s"$testDir/dutDef0", "--target", "systemverilog"),
        Seq(
          ChiselGeneratorAnnotation(() => new AddOneParameterized(4))
        )
      )
      val dutDef0 = getDesignAnnotation(dutAnnos0).design.asInstanceOf[AddOneParameterized].toDefinition

      val dutAnnos1 = (new ChiselStage).execute(
        Array("--target-dir", s"$testDir/dutDef1", "--target", "systemverilog"),
        Seq(
          ChiselGeneratorAnnotation(() => new AddOneParameterized(8)),
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

      (new ChiselStage).execute(
        Array("--target-dir", testDir, "--target", "systemverilog", "--split-verilog"),
        Seq(
          ChiselGeneratorAnnotation(() => new Testbench(dutDef0, dutDef1)),
          ImportDefinitionAnnotation(dutDef0),
          ImportDefinitionAnnotation(dutDef1)
        )
      )

      val dutDef0_rtl = Source.fromFile(s"$testDir/dutDef0/AddOneParameterized.sv").getLines().mkString
      dutDef0_rtl should include("module AddOneParameterized(")
      val dutDef1_rtl = Source.fromFile(s"$testDir/dutDef1/AddOneParameterized_1.sv").getLines().mkString
      dutDef1_rtl should include("module AddOneParameterized_1(")

      val tb_rtl = Source.fromFile(s"$testDir/Testbench.sv").getLines().mkString
      tb_rtl should include("AddOneParameterized inst0 (")
      tb_rtl should include("AddOneParameterized_1 inst1 (")
      (tb_rtl should not).include("module AddOneParameterized(")
      (tb_rtl should not).include("module AddOneParameterized_1(")
    }

    it(
      "(2.b): should throw an exception if information is not passed between Stages."
    ) {
      val testDir = createTestDirectory(this.getClass.getSimpleName).toString

      val dutAnnos0 = (new ChiselStage).execute(
        Array("--target-dir", s"$testDir/dutDef0", "--target", "systemverilog"),
        Seq(
          ChiselGeneratorAnnotation(() => new AddOneParameterized(4))
        )
      )
      val dutDef0 = getDesignAnnotation(dutAnnos0).design.asInstanceOf[AddOneParameterized].toDefinition

      val dutAnnos1 = (new ChiselStage).execute(
        Array("--target-dir", s"$testDir/dutDef1", "--target", "systemverilog"),
        Seq(
          ChiselGeneratorAnnotation(() => new AddOneParameterized(8))
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
      val dutDef0_rtl = Source.fromFile(s"$testDir/dutDef0/AddOneParameterized.sv").getLines().mkString
      dutDef0_rtl should include("module AddOneParameterized(")
      val dutDef1_rtl = Source.fromFile(s"$testDir/dutDef1/AddOneParameterized.sv").getLines().mkString
      dutDef1_rtl should include("module AddOneParameterized(")

      val errMsg = intercept[ChiselException] {
        (new ChiselStage).execute(
          Array("--target-dir", testDir, "--target", "systemverilog"),
          Seq(
            ChiselGeneratorAnnotation(() => new Testbench(dutDef0, dutDef1)),
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

      val dutAnnos0 = (new ChiselStage).execute(
        Array("--target-dir", s"$testDir/dutDef0", "--target", "systemverilog"),
        Seq(
          ChiselGeneratorAnnotation(() => new AddTwoMixedModules)
        )
      )
      val dutDef0 = getDesignAnnotation(dutAnnos0).design.asInstanceOf[AddTwoMixedModules].toDefinition
      val importDefinitionAnnos0 = allModulesToImportedDefs(dutAnnos0)

      val dutAnnos1 = (new ChiselStage).execute(
        Array("--target-dir", s"$testDir/dutDef1", "--target", "systemverilog"),
        Seq(
          ChiselGeneratorAnnotation(() => new AddTwoMixedModules)
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

      val dutDef0_rtl = Source.fromFile(s"$testDir/dutDef0/AddTwoMixedModules.sv").getLines().mkString
      dutDef0_rtl should include("module AddOne(")
      dutDef0_rtl should include("module AddTwoMixedModules(")
      val dutDef1_rtl = Source.fromFile(s"$testDir/dutDef1/AddTwoMixedModules_1.sv").getLines().mkString
      dutDef1_rtl should include("module AddOne_2(")
      dutDef1_rtl should include("module AddTwoMixedModules_1(")

      (new ChiselStage).execute(
        Array("--target-dir", testDir, "--target", "systemverilog"),
        Seq(
          ChiselGeneratorAnnotation(() => new Testbench(dutDef0, dutDef1))
        ) ++ importDefinitionAnnos0 ++ importDefinitionAnnos1
      )

      val tb_rtl = Source.fromFile(s"$testDir/Testbench.sv").getLines().mkString
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

    val dutAnnos0 = (new ChiselStage).execute(
      Array("--target-dir", s"$testDir/dutDef0", "--target", "systemverilog"),
      Seq(
        ChiselGeneratorAnnotation(() => new AddTwoMixedModules)
      )
    )
    val dutDef0 = getDesignAnnotation(dutAnnos0).design.asInstanceOf[AddTwoMixedModules].toDefinition
    val importDefinitionAnnos0 = allModulesToImportedDefs(dutAnnos0)

    val dutAnnos1 = (new ChiselStage).execute(
      Array("--target-dir", s"$testDir/dutDef1", "--target", "systemverilog"),
      Seq(
        ChiselGeneratorAnnotation(() => new AddTwoMixedModules),
        ImportDefinitionAnnotation(dutDef0)
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

    val dutDef0_rtl = Source.fromFile(s"$testDir/dutDef0/AddTwoMixedModules.sv").getLines().mkString
    dutDef0_rtl should include("module AddOne(")
    dutDef0_rtl should include("module AddTwoMixedModules(")
    val dutDef1_rtl = Source.fromFile(s"$testDir/dutDef1/AddTwoMixedModules_1.sv").getLines().mkString
    dutDef1_rtl should include("module AddOne(")
    dutDef1_rtl should include("module AddTwoMixedModules_1(")

    val errMsg = intercept[ChiselException] {
      (new ChiselStage).execute(
        Array("--target-dir", testDir, "--target", "systemverilog"),
        Seq(
          ChiselGeneratorAnnotation(() => new Testbench(dutDef0, dutDef1))
        ) ++ importDefinitionAnnos0 ++ importDefinitionAnnos1
      )
    }
    errMsg.getMessage should include(
      "Expected distinct imported Definition names but found duplicates for: AddOne"
    )
  }

  describe("(4): With ExtMod Names") {
    it("(4.a): should pick correct ExtMod names when passed") {
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

      (new ChiselStage).execute(
        Array("--target-dir", testDir, "--target", "systemverilog"),
        Seq(
          ChiselGeneratorAnnotation(() => new Testbench(dutDef)),
          ImportDefinitionAnnotation(dutDef, Some("CustomPrefix_AddOne_CustomSuffix"))
        )
      )

      val tb_rtl = Source.fromFile(s"$testDir/Testbench.sv").getLines().mkString

      tb_rtl should include("module AddOne_1(")
      tb_rtl should include("AddOne_1 mod (")
      (tb_rtl should not).include("module AddOne(")
      tb_rtl should include("CustomPrefix_AddOne_CustomSuffix inst (")
    }
  }

  it(
    "(4.b): should work if a list of imported Definitions is passed between Stages with ExtModName."
  ) {
    val testDir = createTestDirectory(this.getClass.getSimpleName).toString

    val dutAnnos0 = (new ChiselStage).execute(
      Array("--target-dir", s"$testDir/dutDef0", "--target", "systemverilog"),
      Seq(
        ChiselGeneratorAnnotation(() => new AddOneParameterized(4))
      )
    )
    val dutDef0 = getDesignAnnotation(dutAnnos0).design.asInstanceOf[AddOneParameterized].toDefinition

    val dutAnnos1 = (new ChiselStage).execute(
      Array("--target-dir", s"$testDir/dutDef1", "--target", "systemverilog"),
      Seq(
        ChiselGeneratorAnnotation(() => new AddOneParameterized(8)),
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

    (new ChiselStage).execute(
      Array("--target-dir", testDir, "--target", "systemverilog"),
      Seq(
        ChiselGeneratorAnnotation(() => new Testbench(dutDef0, dutDef1)),
        ImportDefinitionAnnotation(dutDef0, Some("Inst1_Prefix_AddOnePramaterized_Inst1_Suffix")),
        ImportDefinitionAnnotation(dutDef1, Some("Inst2_Prefix_AddOnePrameterized_1_Inst2_Suffix"))
      )
    )

    val dutDef0_rtl = Source.fromFile(s"$testDir/dutDef0/AddOneParameterized.sv").getLines().mkString
    dutDef0_rtl should include("module AddOneParameterized(")
    val dutDef1_rtl = Source.fromFile(s"$testDir/dutDef1/AddOneParameterized_1.sv").getLines().mkString
    dutDef1_rtl should include("module AddOneParameterized_1(")

    val tb_rtl = Source.fromFile(s"$testDir/Testbench.sv").getLines().mkString
    tb_rtl should include("Inst1_Prefix_AddOnePramaterized_Inst1_Suffix inst0 (")
    tb_rtl should include("Inst2_Prefix_AddOnePrameterized_1_Inst2_Suffix inst1 (")
    (tb_rtl should not).include("module AddOneParameterized(")
    (tb_rtl should not).include("module AddOneParameterized_1(")
  }

  it(
    "(4.c): should throw an exception  if a list of imported Definitions is passed between Stages with same ExtModName."
  ) {
    val testDir = createTestDirectory(this.getClass.getSimpleName).toString

    val dutAnnos0 = (new ChiselStage).execute(
      Array("--target-dir", s"$testDir/dutDef0", "--target", "systemverilog"),
      Seq(
        ChiselGeneratorAnnotation(() => new AddOneParameterized(4))
      )
    )
    val importDefinitionAnnos0 = allModulesToImportedDefs(dutAnnos0)
    val dutDef0 = getDesignAnnotation(dutAnnos0).design.asInstanceOf[AddOneParameterized].toDefinition

    val dutAnnos1 = (new ChiselStage).execute(
      Array("--target-dir", s"$testDir/dutDef1", "--target", "systemverilog"),
      Seq(
        ChiselGeneratorAnnotation(() => new AddOneParameterized(8)),
        // pass in previously elaborated Definitions
        ImportDefinitionAnnotation(dutDef0)
      )
    )
    val importDefinitionAnnos1 = allModulesToImportedDefs(dutAnnos1)
    val dutDef1 = getDesignAnnotation(dutAnnos1).design.asInstanceOf[AddOneParameterized].toDefinition

    class Testbench(defn0: Definition[AddOneParameterized], defn1: Definition[AddOneParameterized]) extends Module {
      val inst0 = Instance(defn0)
      val inst1 = Instance(defn1)

      // Tie inputs to a value so ChiselStage does not complain
      inst0.in := 0.U
      inst1.in := 0.U
    }

    val dutDef0_rtl = Source.fromFile(s"$testDir/dutDef0/AddOneParameterized.sv").getLines().mkString
    dutDef0_rtl should include("module AddOneParameterized(")
    val dutDef1_rtl = Source.fromFile(s"$testDir/dutDef1/AddOneParameterized_1.sv").getLines().mkString
    dutDef1_rtl should include("module AddOneParameterized_1(")

    val errMsg = intercept[ChiselException] {
      (new ChiselStage).execute(
        Array("--target-dir", testDir, "--target", "systemverilog"),
        Seq(
          ChiselGeneratorAnnotation(() => new Testbench(dutDef0, dutDef1)),
          ImportDefinitionAnnotation(dutDef0, Some("Inst1_Prefix_AddOnePrameterized_Inst1_Suffix")),
          ImportDefinitionAnnotation(dutDef1, Some("Inst1_Prefix_AddOnePrameterized_Inst1_Suffix"))
        )
      )
    }
    errMsg.getMessage should include(
      "Expected distinct overrideDef names but found duplicates for: Inst1_Prefix_AddOnePrameterized_Inst1_Suffix"
    )
  }
}
