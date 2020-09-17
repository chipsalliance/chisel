// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import java.io.{File, FileWriter}

import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers
import firrtl.passes.InlineAnnotation
import firrtl.passes.memlib.{InferReadWriteAnnotation, ReplSeqMemAnnotation}
import firrtl.transforms.BlackBoxTargetDirAnno
import firrtl._
import firrtl.FileUtils
import firrtl.annotations._
import firrtl.util.BackendCompilationUtilities
import firrtl.testutils.FirrtlFlatSpec

import scala.util.Success
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers

class ExceptingTransform extends Transform {
  def inputForm = HighForm
  def outputForm = HighForm
  def execute(state: CircuitState): CircuitState = {
    throw new ExceptingTransform.CustomException("I ran!")
  }
}
object ExceptingTransform {
  case class CustomException(msg: String) extends Exception
}

//noinspection ScalaStyle
class DriverSpec extends AnyFreeSpec with Matchers with BackendCompilationUtilities {
  val outputDir = createTestDirectory("DriverSpec")
  "CommonOptions are some simple options available across the chisel3 ecosystem" - {
    "CommonOption provide an scopt implementation of an OptionParser" - {
      "Options can be set from an Array[String] as is passed into a main" - {
        "With no arguments default values come out" in {
          val optionsManager = new ExecutionOptionsManager("test")
          optionsManager.parse(Array.empty[String]) should be(true)

          val commonOptions = optionsManager.commonOptions
          commonOptions.topName should be("")
          commonOptions.targetDirName should be(".")
        }
        "top name and target can be set" in {
          val optionsManager = new ExecutionOptionsManager("test")
          optionsManager.parse(Array("--top-name", "dog", "--target-dir", "a/b/c")) should be(true)
          val commonOptions = optionsManager.commonOptions

          commonOptions.topName should be("dog")
          commonOptions.targetDirName should be("a/b/c")

          optionsManager.getBuildFileName(".fir") should be("a/b/c/dog.fir")
          optionsManager.getBuildFileName("fir") should be("a/b/c/dog.fir")
        }
      }
      "CommonOptions can create a directory" in {
        val top_dir = new java.io.File(outputDir, "a")
        val dir = new java.io.File(top_dir, "b/c")
        if (dir.exists()) {
          dir.delete()
        }
        val optionsManager = new ExecutionOptionsManager("test")
        val targetPath = dir.getPath
        optionsManager.parse(Array("--top-name", "dog", "--target-dir", targetPath)) should be(true)
        val commonOptions = optionsManager.commonOptions

        commonOptions.topName should be("dog")
        commonOptions.targetDirName should be(targetPath)

        optionsManager.makeTargetDir() should be(true)
        val tdir = new java.io.File(targetPath)
        tdir.exists() should be(true)
        FileUtils.deleteDirectoryHierarchy(top_dir) should be(true)
      }
    }
    "options include by default a list of strings that are returned in commonOptions.programArgs" in {
      val optionsManager = new ExecutionOptionsManager("test")

      optionsManager.parse(Array("--top-name", "dog", "fox", "tardigrade", "stomatopod")) should be(true)
      info(s"programArgs ${optionsManager.commonOptions.programArgs}")
      optionsManager.commonOptions.programArgs.length should be(3)
      optionsManager.commonOptions.programArgs should be("fox" :: "tardigrade" :: "stomatopod" :: Nil)

      optionsManager.commonOptions = CommonOptions()
      optionsManager.parse(Array("dog", "stomatopod")) should be(true)
      info(s"programArgs ${optionsManager.commonOptions.programArgs}")
      optionsManager.commonOptions.programArgs.length should be(2)
      optionsManager.commonOptions.programArgs should be("dog" :: "stomatopod" :: Nil)

      optionsManager.commonOptions = CommonOptions()
      optionsManager.parse(Array("fox", "--top-name", "dog", "tardigrade", "stomatopod")) should be(true)
      info(s"programArgs ${optionsManager.commonOptions.programArgs}")
      optionsManager.commonOptions.programArgs.length should be(3)
      optionsManager.commonOptions.programArgs should be("fox" :: "tardigrade" :: "stomatopod" :: Nil)

    }
  }
  "FirrtlOptions holds option information for the firrtl compiler" - {
    "It includes a CommonOptions" in {
      val optionsManager = new ExecutionOptionsManager("test")
      optionsManager.commonOptions.targetDirName should be(".")
    }
    "It provides input and output file names based on target" in {
      val optionsManager = new ExecutionOptionsManager("test") with HasFirrtlOptions

      optionsManager.parse(Array("--top-name", "cat")) should be(true)

      val firrtlOptions = optionsManager.firrtlOptions
      val inputFileName = optionsManager.getBuildFileName("fir", firrtlOptions.inputFileNameOverride)
      inputFileName should be("./cat.fir")
      val outputFileName = firrtlOptions.getTargetFile(optionsManager)
      outputFileName should be("./cat.v")
    }
    "input and output file names can be overridden, overrides do not use targetDir" in {
      val optionsManager = new ExecutionOptionsManager("test") with HasFirrtlOptions

      optionsManager.parse(
        Array("--top-name", "cat", "-i", "./bob.fir", "-o", "carol.v")
      ) should be(true)

      val firrtlOptions = optionsManager.firrtlOptions
      val inputFileName = optionsManager.getBuildFileName("fir", firrtlOptions.inputFileNameOverride)
      inputFileName should be("./bob.fir")
      val outputFileName = firrtlOptions.getTargetFile(optionsManager)
      outputFileName should be("carol.v")
    }
    val input = """
                  |circuit Top :
                  |  module Top :
                  |    input x : UInt<8>
                  |    output y : UInt<8>
                  |    y <= x""".stripMargin
    val circuit = Parser.parse(input.split("\n").toIterator)
    "firrtl source can be provided directly" in {
      val manager = new ExecutionOptionsManager("test") with HasFirrtlOptions {
        commonOptions = CommonOptions(topName = "Top")
        firrtlOptions = FirrtlExecutionOptions(firrtlSource = Some(input))
      }
      assert(firrtl.Driver.getCircuit(manager).isSuccess)
    }
    "firrtl Circuits can be provided directly" in {
      val manager = new ExecutionOptionsManager("test") with HasFirrtlOptions {
        commonOptions = CommonOptions(topName = "Top")
        firrtlOptions = FirrtlExecutionOptions(firrtlCircuit = Some(circuit))
      }
      firrtl.Driver.getCircuit(manager) shouldBe Success(circuit)
    }
    "Only one of inputFileNameOverride, firrtlSource, and firrtlCircuit can be used at a time" in {
      val manager1 = new ExecutionOptionsManager("test") with HasFirrtlOptions {
        commonOptions = CommonOptions(topName = "Top")
        firrtlOptions = FirrtlExecutionOptions(firrtlCircuit = Some(circuit), firrtlSource = Some(input))
      }
      val manager2 = new ExecutionOptionsManager("test") with HasFirrtlOptions {
        commonOptions = CommonOptions(topName = "Top")
        firrtlOptions = FirrtlExecutionOptions(inputFileNameOverride = "hi", firrtlSource = Some(input))
      }
      val manager3 = new ExecutionOptionsManager("test") with HasFirrtlOptions {
        commonOptions = CommonOptions(topName = "Top")
        firrtlOptions = FirrtlExecutionOptions(inputFileNameOverride = "hi", firrtlCircuit = Some(circuit))
      }
      assert(firrtl.Driver.getCircuit(manager1).isFailure)
      assert(firrtl.Driver.getCircuit(manager2).isFailure)
      assert(firrtl.Driver.getCircuit(manager3).isFailure)
    }
    "various annotations can be created from command line, currently:" - {
      "inline annotation" in {
        val optionsManager = new ExecutionOptionsManager("test") with HasFirrtlOptions

        optionsManager.parse(
          Array("--inline", "module,module.submodule,module.submodule.instance")
        ) should be(true)

        val firrtlOptions = optionsManager.firrtlOptions
        firrtlOptions.annotations.length should be(3)
        firrtlOptions.annotations.foreach(_ shouldBe an[InlineAnnotation])
      }
      "infer-rw annotation" in {
        val optionsManager = new ExecutionOptionsManager("test") with HasFirrtlOptions

        optionsManager.parse(
          Array("--infer-rw")
        ) should be(true)

        val firrtlOptions = optionsManager.firrtlOptions
        firrtlOptions.annotations.length should be(1)
        firrtlOptions.annotations.head should be(InferReadWriteAnnotation)
      }
      "repl-seq-mem annotation" in {
        val optionsManager = new ExecutionOptionsManager("test") with HasFirrtlOptions

        optionsManager.parse(
          Array("--repl-seq-mem", "-c:circuit1:-i:infile1:-o:outfile1")
        ) should be(true)

        val firrtlOptions = optionsManager.firrtlOptions
        firrtlOptions.annotations.length should be(1)
        firrtlOptions.annotations.head should matchPattern {
          case ReplSeqMemAnnotation("infile1", "outfile1") =>
        }
      }
    }
  }

  "Annotations can be read from multiple files" in {
    val filename = "SampleAnnotations.anno.json"
    val optionsManager = new ExecutionOptionsManager("test") with HasFirrtlOptions {
      commonOptions = commonOptions.copy(topName = "a.fir")
      firrtlOptions = firrtlOptions.copy(
        annotationFileNames = List.fill(2)(filename) // just read the safe file twice
      )
    }
    val annotationsTestFile = new File(optionsManager.commonOptions.targetDirName, filename)
    copyResourceToFile(s"/annotations/$filename", annotationsTestFile)
    optionsManager.firrtlOptions.annotations.length should be(0)
    val annos = Driver.getAnnotations(optionsManager)
    annos.length should be(21) // 18 from files plus 3 general purpose
    annos.count(_.isInstanceOf[InlineAnnotation]) should be(18)
    annotationsTestFile.delete()
  }

  "Annotations can be created from the command line and read from a file at the same time" in {
    val optionsManager = new ExecutionOptionsManager("test") with HasFirrtlOptions
    val targetDir = new File(optionsManager.commonOptions.targetDirName)
    val annoFile = new File(targetDir, "annotations.anno")

    optionsManager.parse(
      Array("--infer-rw", "-faf", annoFile.toString)
    ) should be(true)

    copyResourceToFile("/annotations/SampleAnnotations.anno.json", annoFile)

    val firrtlOptions = optionsManager.firrtlOptions
    firrtlOptions.annotations.length should be(1) // infer-rw

    val anns = Driver.getAnnotations(optionsManager)
    anns should contain(BlackBoxTargetDirAnno(".")) // built in to getAnnotations
    anns should contain(InferReadWriteAnnotation) // --infer-rw
    anns.collect { case a: InlineAnnotation => a }.length should be(9) // annotations file

    annoFile.delete()
  }

  "Circuits are emitted on properly" - {
    val input =
      """|circuit Top :
         |  module Top :
         |    output foo : UInt<32>
         |    inst c of Child
         |    inst e of External
         |    foo <= tail(add(c.foo, e.foo), 1)
         |  module Child :
         |    output foo : UInt<32>
         |    inst e of External
         |    foo <= e.foo
         |  extmodule External :
         |    output foo : UInt<32>
      """.stripMargin

    "To a single file with file extension depending on the compiler by default" in {
      Seq(
        "none" -> "./Foo.fir",
        "low" -> "./Foo.lo.fir",
        "high" -> "./Foo.hi.fir",
        "middle" -> "./Foo.mid.fir",
        "verilog" -> "./Foo.v",
        "mverilog" -> "./Foo.v",
        "sverilog" -> "./Foo.sv"
      ).foreach {
        case (compilerName, expectedOutputFileName) =>
          info(s"$compilerName -> $expectedOutputFileName")
          val manager = new ExecutionOptionsManager("test") with HasFirrtlOptions {
            commonOptions = CommonOptions(topName = "Foo")
            firrtlOptions = FirrtlExecutionOptions(firrtlSource = Some(input), compilerName = compilerName)
          }

          firrtl.Driver.execute(manager) match {
            case success: FirrtlExecutionSuccess =>
              success.emitted.size should not be (0)
              success.circuitState.annotations.length should be > (0)
            case a: FirrtlExecutionFailure =>
              fail(s"Got a FirrtlExecutionFailure! Expected FirrtlExecutionSuccess. Full message:\n${a.message}")
          }

          val file = new File(expectedOutputFileName)
          file.exists() should be(true)
          file.delete()
      }
    }
    "To a single file per module if OneFilePerModule is specified" in {
      Seq(
        "none" -> Seq("./Top.fir", "./Child.fir"),
        "low" -> Seq("./Top.lo.fir", "./Child.lo.fir"),
        "high" -> Seq("./Top.hi.fir", "./Child.hi.fir"),
        "middle" -> Seq("./Top.mid.fir", "./Child.mid.fir"),
        "verilog" -> Seq("./Top.v", "./Child.v"),
        "mverilog" -> Seq("./Top.v", "./Child.v"),
        "sverilog" -> Seq("./Top.sv", "./Child.sv")
      ).foreach {
        case (compilerName, expectedOutputFileNames) =>
          info(s"$compilerName -> $expectedOutputFileNames")
          val manager = new ExecutionOptionsManager("test") with HasFirrtlOptions {
            firrtlOptions = FirrtlExecutionOptions(
              firrtlSource = Some(input),
              compilerName = compilerName,
              emitOneFilePerModule = true
            )
          }

          firrtl.Driver.execute(manager) match {
            case success: FirrtlExecutionSuccess =>
              success.emitted.size should not be (0)
              success.circuitState.annotations.length should be > (0)
            case failure: FirrtlExecutionFailure =>
              fail(s"Got a FirrtlExecutionFailure! Expected FirrtlExecutionSuccess. Full message:\n${failure.message}")
          }

          for (name <- expectedOutputFileNames) {
            val file = new File(name)
            file.exists() should be(true)
            file.delete()
          }
      }
    }
  }

  "The Driver is sensitive to the file extension of input files" - {
    val design = "GCDTester"
    val outputDir = createTestDirectory("DriverFileExtensionSensitivity")
    val verilogFromFir = new File(outputDir, s"$design.fromfir.v")
    val verilogFromPb = new File(outputDir, s"$design.frompb.v")
    val commonArgs = Array("-X", "verilog", "--info-mode", "use")
    ".fir means FIRRTL file" in {
      val inFile = new File(getClass.getResource(s"/integration/$design.fir").getFile)
      val args = Array("-i", inFile.getAbsolutePath, "-o", verilogFromFir.getAbsolutePath) ++ commonArgs
      Driver.execute(args)
    }
    ".pb means ProtoBuf file" in {
      val inFile = new File(getClass.getResource(s"/integration/$design.pb").getFile)
      val args = Array("-i", inFile.getAbsolutePath, "-o", verilogFromPb.getAbsolutePath) ++ commonArgs
      Driver.execute(args)
    }
    "Both paths do the same thing" in {
      val s1 = FileUtils.getText(verilogFromFir)
      val s2 = FileUtils.getText(verilogFromPb)
      s1 should equal(s2)
    }
  }

  "Directory deleter is handy for cleaning up after tests" - {
    "for example making a directory tree, and deleting it looks like" in {
      FileUtils.makeDirectory("dog/fox/wolf")
      val dir = new File("dog/fox/wolf")
      dir.exists() should be(true)
      dir.isDirectory should be(true)

      FileUtils.deleteDirectoryHierarchy("wolf") should be(false)
      FileUtils.deleteDirectoryHierarchy("dog") should be(true)
      dir.exists() should be(false)
    }
  }
}

class VcdSuppressionSpec extends FirrtlFlatSpec {
  "Default option" should "generate a vcd" in {
    val prefix = "ZeroPortMem"

    def testIfVcdCreated(suppress: Boolean): Unit = {
      val optionsManager = new ExecutionOptionsManager("test") with HasFirrtlOptions

      val testDir = compileFirrtlTest(prefix, "/features", Seq.empty, Seq.empty)
      val harness = new File(testDir, s"top.cpp")
      copyResourceToFile(cppHarnessResourceName, harness)

      verilogToCpp(prefix, testDir, Seq.empty, harness, suppress) #&&
        cppToExe(prefix, testDir) ! loggingProcessLogger

      assert(executeExpectingSuccess(prefix, testDir))

      val vcdFile = new File(s"$testDir/dump.vcd")
      vcdFile.exists() should be(!suppress)
    }

    testIfVcdCreated(suppress = false)
    testIfVcdCreated(suppress = true)
  }
}
