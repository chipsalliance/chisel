// SPDX-License-Identifier: Apache-2.0

package firrtlTests.transforms

import firrtl.annotations._
import firrtl.transforms._
import firrtl._
import firrtl.FileUtils
import firrtl.testutils.LowTransformSpec
import firrtl.stage._
import firrtl.options.Dependency
import java.io.File
import firrtl.util.BackendCompilationUtilities.createTestDirectory
import _root_.logger._

class ChangeBlackBoxTargetDir extends Transform with DependencyAPIMigration {
  override def prerequisites =
    Dependency[firrtl.passes.memlib.ReplSeqMem] +: Forms.LowFormOptimized
  override def optionalPrerequisiteOf = Forms.BackendEmitters
  override def invalidates(a: Transform): Boolean = false

  def execute(state: CircuitState): CircuitState = {
    val annosx = state.annotations.map {
      case BlackBoxTargetDirAnno(dir) => BlackBoxTargetDirAnno(s"$dir/subdir")
      case other                      => other
    }
    state.copy(annotations = annosx)
  }
}

class BlacklBoxSourceHelperTransformSpec extends LowTransformSpec {
  def transform: Transform = new BlackBoxSourceHelper

  private val moduleName = ModuleName("Top", CircuitName("Top"))
  private val bbTarget = CircuitTarget("Top").module("AdderExtModule")
  private val input = """
                        |circuit Top :
                        |
                        |  extmodule AdderExtModule :
                        |    input foo : UInt<16>
                        |    output bar : UInt<16>
                        |
                        |    defname = BBFAdd
                        |
                        |  module Top :
                        |   input x : UInt<16>
                        |   output y : UInt<16>
                        |
                        |   inst a1 of AdderExtModule
                        |   a1.foo <= x
                        |   y <= a1.bar
                      """.stripMargin
  private val output = """
                         |circuit Top :
                         |
                         |  extmodule AdderExtModule :
                         |    input foo : UInt<16>
                         |    output bar : UInt<16>
                         |
                         |    defname = BBFAdd
                         |
                         |  module Top :
                         |   input x : UInt<16>
                         |   output y : UInt<16>
                         |
                         |   inst a1 of AdderExtModule
                         |   y <= a1.bar
                         |   a1.foo <= x
                      """.stripMargin

  "annotated external modules with absolute path" should "appear in output directory" in {

    val absPath = new java.io.File("src/test/resources/blackboxes/AdderExtModule.v").getCanonicalPath
    val annos = Seq(
      BlackBoxTargetDirAnno("test_run_dir"),
      BlackBoxPathAnno(moduleName, absPath)
    )

    execute(input, output, annos)

    val module = new java.io.File("test_run_dir/AdderExtModule.v")
    val fileList = new java.io.File(s"test_run_dir/${BlackBoxSourceHelper.defaultFileListName}")

    module.exists should be(true)
    fileList.exists should be(true)

    module.delete()
    fileList.delete()
  }

  "annotated external modules with relative path" should "appear in output directory" in {

    val annos = Seq(
      BlackBoxTargetDirAnno("test_run_dir"),
      BlackBoxPathAnno(moduleName, "src/test/resources/blackboxes/AdderExtModule.v")
    )

    execute(input, output, annos)

    val module = new java.io.File("test_run_dir/AdderExtModule.v")
    val fileList = new java.io.File(s"test_run_dir/${BlackBoxSourceHelper.defaultFileListName}")

    module.exists should be(true)
    fileList.exists should be(true)

    module.delete()
    fileList.delete()
  }

  "annotated external modules" should "appear in output directory" in {

    val annos = Seq(
      BlackBoxTargetDirAnno("test_run_dir"),
      BlackBoxResourceAnno(moduleName, "/blackboxes/AdderExtModule.v")
    )

    execute(input, output, annos)

    new java.io.File("test_run_dir/AdderExtModule.v").exists should be(true)
    new java.io.File(s"test_run_dir/${BlackBoxSourceHelper.defaultFileListName}").exists should be(true)
  }

  "verilog header files" should "be available but not mentioned in the file list" in {
    // Issue #917 - We don't want to list Verilog header files ("*.vh") in our file list.
    // We don't actually verify that the generated verilog code works,
    //  we just ensure that the correct files end up where they're expected.

    // We're taking the liberty of recycling the above code with a few minor edits to change the external module name.
    val sourceModuleName = "AdderExtModule"
    val replacedModuleName = "ParameterizedViaHeaderAdderExtModule"
    val pInput = input.replaceAll(sourceModuleName, replacedModuleName)
    val pOutput = output.replaceAll(sourceModuleName, replacedModuleName)

    // We'll copy the following resources to the test_run_dir via BlackBoxResourceAnno's
    val resourceNames = Seq("ParameterizedViaHeaderAdderExtModule.v", "VerilogHeaderFile.vh")

    val annos = Seq(BlackBoxTargetDirAnno("test_run_dir")) ++ resourceNames.map { n =>
      BlackBoxResourceAnno(moduleName, "/blackboxes/" + n)
    }

    execute(pInput, pOutput, annos)

    // Our resource files should exist in the test_run_dir,
    for (n <- resourceNames)
      new java.io.File("test_run_dir/" + n).exists should be(true)

    //  but our file list should not include the verilog header file.
    val fileListFile = new java.io.File(s"test_run_dir/${BlackBoxSourceHelper.defaultFileListName}")
    fileListFile.exists should be(true)
    val fileList = FileUtils.getText(fileListFile)
    fileList.contains("ParameterizedViaHeaderAdderExtModule.v") should be(true)
    fileList.contains("VerilogHeaderFile.vh") should be(false)
  }

  behavior.of("BlackBox resources that do not exist")

  it should "provide a useful error message for BlackBoxResourceAnno" in {
    val annos = Seq(BlackBoxTargetDirAnno("test_run_dir"), BlackBoxResourceAnno(moduleName, "/blackboxes/IDontExist.v"))

    (the[BlackBoxNotFoundException] thrownBy { execute(input, "", annos) }).getMessage should include(
      "Did you misspell it?"
    )
  }

  it should "provide a useful error message for BlackBoxPathAnno" in {
    val absPath = new java.io.File("src/test/resources/blackboxes/IDontExist.v").getCanonicalPath
    val annos = Seq(BlackBoxTargetDirAnno("test_run_dir"), BlackBoxPathAnno(moduleName, absPath))

    (the[BlackBoxNotFoundException] thrownBy { execute(input, "", annos) }).getMessage should include(
      "Did you misspell it?"
    )
  }

  "BlackBoxSourceHelper" should "not run until late" in {
    val name = "BlackBoxSourceHelper_late"

    val dir = createTestDirectory(name)
    val subdir = new File(dir, "subdir")

    val filename = "BFFAdd.v"

    val annos = List(
      FirrtlSourceAnnotation(input),
      RunFirrtlTransformAnnotation(new ChangeBlackBoxTargetDir),
      BlackBoxTargetDirAnno(dir.toString),
      BlackBoxInlineAnno(bbTarget, filename, "<contentx>"),
      EmitCircuitAnnotation(classOf[VerilogEmitter]),
      LogLevelAnnotation(LogLevel.Debug)
    )

    (new FirrtlPhase).transform(annos)

    new File(subdir, filename) should exist
    new File(dir, filename) shouldNot exist
  }

  "verilog file list" should "end with a newline" in {
    val annos = Seq(
      BlackBoxTargetDirAnno("test_run_dir"),
      BlackBoxResourceAnno(moduleName, "/blackboxes/AdderExtModule.v")
    )

    execute(input, output, annos)

    val filename = os.pwd / "test_run_dir" / BlackBoxSourceHelper.defaultFileListName
    assert(os.exists(filename), "verilog file list should exist")
    val content = os.read(filename)
    assert(content.contains("AdderExtModule.v"))
    assert(content.last == '\n', "Some simulators like Icarus Verilog seem to expect a trailing new line character")
  }
}
