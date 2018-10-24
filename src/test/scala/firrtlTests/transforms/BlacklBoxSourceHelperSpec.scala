// See LICENSE for license details.

package firrtlTests.transforms

import firrtl.annotations.{Annotation, CircuitName, ModuleName}
import firrtl.transforms._
import firrtl.{FIRRTLException, Transform, VerilogCompiler, VerilogEmitter}
import firrtlTests.{HighTransformSpec, LowTransformSpec}
import org.scalacheck.Test.Failed
import org.scalatest.{FreeSpec, Matchers, Succeeded}


class BlacklBoxSourceHelperTransformSpec extends LowTransformSpec {
   def transform: Transform = new BlackBoxSourceHelper

  private val moduleName = ModuleName("Top", CircuitName("Top"))
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
    val fileList = new java.io.File(s"test_run_dir/${BlackBoxSourceHelper.fileListName}")

    module.exists should be (true)
    fileList.exists should be (true)

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
    val fileList = new java.io.File(s"test_run_dir/${BlackBoxSourceHelper.fileListName}")

    module.exists should be (true)
    fileList.exists should be (true)

    module.delete()
    fileList.delete()
  }

  "annotated external modules" should "appear in output directory" in {

    val annos = Seq(
      BlackBoxTargetDirAnno("test_run_dir"),
      BlackBoxResourceAnno(moduleName, "/blackboxes/AdderExtModule.v")
    )

    execute(input, output, annos)

    new java.io.File("test_run_dir/AdderExtModule.v").exists should be (true)
    new java.io.File(s"test_run_dir/${BlackBoxSourceHelper.fileListName}").exists should be (true)
  }

  "verilog compiler" should "have BlackBoxSourceHelper transform" in {
    val verilogCompiler = new VerilogEmitter
    verilogCompiler.transforms.map { x => x.getClass } should contain (classOf[BlackBoxSourceHelper])
  }

  behavior of "BlackBox resources that do not exist"

  it should "provide a useful error message for BlackBoxResourceAnno" in {
    val annos = Seq( BlackBoxTargetDirAnno("test_run_dir"),
                     BlackBoxResourceAnno(moduleName, "/blackboxes/IDontExist.v") )

    (the [BlackBoxNotFoundException] thrownBy { execute(input, "", annos) })
      .getMessage should include ("Did you misspell it?")
  }

  it should "provide a useful error message for BlackBoxPathAnno" in {
    val absPath = new java.io.File("src/test/resources/blackboxes/IDontExist.v").getCanonicalPath
    val annos = Seq( BlackBoxTargetDirAnno("test_run_dir"),
                     BlackBoxPathAnno(moduleName, absPath) )

    (the [BlackBoxNotFoundException] thrownBy { execute(input, "", annos) })
      .getMessage should include ("Did you misspell it?")
  }

}
