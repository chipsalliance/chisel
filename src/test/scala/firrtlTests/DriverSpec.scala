// See LICENSE for license details.

package firrtlTests

import java.io.{File, FileNotFoundException, FileOutputStream}
import org.scalatest.{FreeSpec, Matchers}

import firrtl.passes.InlineInstances
import firrtl.passes.memlib.{InferReadWrite, ReplSeqMem}
import firrtl.transforms.BlackBoxSourceHelper
import firrtl._
import firrtl.util.BackendCompilationUtilities

class DriverSpec extends FreeSpec with Matchers with BackendCompilationUtilities {
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
        var dir = new java.io.File("a/b/c")
        if(dir.exists()) {
          dir.delete()
        }
        val optionsManager = new ExecutionOptionsManager("test")
        optionsManager.parse(Array("--top-name", "dog", "--target-dir", "a/b/c")) should be (true)
        val commonOptions = optionsManager.commonOptions

        commonOptions.topName should be ("dog")
        commonOptions.targetDirName should be ("a/b/c")

        optionsManager.makeTargetDir() should be (true)
        dir = new java.io.File("a/b/c")
        dir.exists() should be (true)
        FileUtils.deleteDirectoryHierarchy("a") should be (true)
      }
    }
  }
  "FirrtlOptions holds option information for the firrtl compiler" - {
    "It includes a CommonOptions" in {
      val optionsManager = new ExecutionOptionsManager("test")
      optionsManager.commonOptions.targetDirName should be (".")
    }
    "It provides input and output file names based on target" in {
      val optionsManager = new ExecutionOptionsManager("test") with HasFirrtlOptions

      optionsManager.parse(Array("--top-name", "cat")) should be (true)

      val firrtlOptions = optionsManager.firrtlOptions
      val inputFileName = optionsManager.getBuildFileName("fir", firrtlOptions.inputFileNameOverride)
      inputFileName should be ("./cat.fir")
      val outputFileName = firrtlOptions.getTargetFile(optionsManager)
      outputFileName should be ("./cat.v")
    }
    "input and output file names can be overridden, overrides do not use targetDir" in {
      val optionsManager = new ExecutionOptionsManager("test") with HasFirrtlOptions

      optionsManager.parse(
        Array("--top-name", "cat", "-i", "./bob.fir", "-o", "carol.v")
      ) should be (true)

      val firrtlOptions = optionsManager.firrtlOptions
      val inputFileName = optionsManager.getBuildFileName("fir", firrtlOptions.inputFileNameOverride)
      inputFileName should be ("./bob.fir")
      val outputFileName = firrtlOptions.getTargetFile(optionsManager)
      outputFileName should be ("carol.v")
    }
    "various annotations can be created from command line, currently:" - {
      "inline annotation" in {
        val optionsManager = new ExecutionOptionsManager("test") with HasFirrtlOptions

        optionsManager.parse(
          Array("--inline", "module,module.submodule,module.submodule.instance")
        ) should be (true)

        val firrtlOptions = optionsManager.firrtlOptions
        firrtlOptions.annotations.length should be (3)
        firrtlOptions.annotations.foreach { annotation =>
          annotation.transform shouldBe classOf[InlineInstances]
        }
      }
      "infer-rw annotation" in {
        val optionsManager = new ExecutionOptionsManager("test") with HasFirrtlOptions

        optionsManager.parse(
          Array("--infer-rw", "circuit")
        ) should be (true)

        val firrtlOptions = optionsManager.firrtlOptions
        firrtlOptions.annotations.length should be (1)
        firrtlOptions.annotations.foreach { annotation =>
          annotation.transform shouldBe classOf[InferReadWrite]
        }
      }
      "repl-seq-mem annotation" in {
        val optionsManager = new ExecutionOptionsManager("test") with HasFirrtlOptions

        optionsManager.parse(
          Array("--repl-seq-mem", "-c:circuit1:-i:infile1:-o:outfile1")
        ) should be (true)

        val firrtlOptions = optionsManager.firrtlOptions
        firrtlOptions.annotations.length should be (1)
        firrtlOptions.annotations.foreach { annotation =>
          annotation.transform shouldBe classOf[ReplSeqMem]
        }
      }
    }
  }

  "Annotations can be read from a file" in {
    val optionsManager = new ExecutionOptionsManager("test") with HasFirrtlOptions {
      commonOptions = commonOptions.copy(topName = "a.fir")
      firrtlOptions = firrtlOptions.copy(
        annotationFileNameOverride = "SampleAnnotations"
      )
    }
    val annotationsTestFile =  new File(optionsManager.commonOptions.targetDirName, optionsManager.firrtlOptions.annotationFileNameOverride + ".anno")
    copyResourceToFile("/annotations/SampleAnnotations.anno", annotationsTestFile)
    optionsManager.firrtlOptions.annotations.length should be (0)
    Driver.loadAnnotations(optionsManager)
    optionsManager.firrtlOptions.annotations.length should be (10) // 9 from circuit plus 1 for targetDir

    optionsManager.firrtlOptions.annotations.head.transformClass should be ("firrtl.passes.InlineInstances")
    annotationsTestFile.delete()
  }

  "Annotations can be created from the command line and read from a file at the same time" in {
    val optionsManager = new ExecutionOptionsManager("test") with HasFirrtlOptions
    val targetDir = new File(optionsManager.commonOptions.targetDirName)
    val annoFile = new File(targetDir, "annotations.anno")

    optionsManager.parse(
      Array("--infer-rw", "circuit", "-faf", annoFile.toString, "-ffaaf")
    ) should be (true)

    copyResourceToFile("/annotations/SampleAnnotations.anno", annoFile)

    val firrtlOptions = optionsManager.firrtlOptions
    firrtlOptions.annotations.length should be (1) // infer-rw

    Driver.loadAnnotations(optionsManager)

    val anns = optionsManager.firrtlOptions.annotations.groupBy(_.transform)
    anns(classOf[BlackBoxSourceHelper]).length should be (1) // built in to loadAnnotations
    anns(classOf[InferReadWrite]).length should be (1) // --infer-rw
    anns(classOf[InlineInstances]).length should be (9) // annotations file

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
        "low" -> "./Top.lo.fir",
        "high" -> "./Top.hi.fir",
        "middle" -> "./Top.mid.fir",
        "verilog" -> "./Top.v"
      ).foreach { case (compilerName, expectedOutputFileName) =>
        val manager = new ExecutionOptionsManager("test") with HasFirrtlOptions {
          commonOptions = CommonOptions(topName = "Top")
          firrtlOptions = FirrtlExecutionOptions(firrtlSource = Some(input), compilerName = compilerName)
        }

        firrtl.Driver.execute(manager)

        val file = new File(expectedOutputFileName)
        file.exists() should be (true)
        file.delete()
      }
    }
    "To a single file per module if OneFilePerModule is specified" in {
      Seq(
        "low" -> Seq("./Top.lo.fir", "./Child.lo.fir"),
        "high" -> Seq("./Top.hi.fir", "./Child.hi.fir"),
        "middle" -> Seq("./Top.mid.fir", "./Child.mid.fir"),
        "verilog" -> Seq("./Top.v", "./Child.v")
      ).foreach { case (compilerName, expectedOutputFileNames) =>
        println(s"$compilerName -> $expectedOutputFileNames")
        val manager = new ExecutionOptionsManager("test") with HasFirrtlOptions {
          commonOptions = CommonOptions(topName = "Top")
          firrtlOptions = FirrtlExecutionOptions(firrtlSource = Some(input),
                                                 compilerName = compilerName,
                                                 emitOneFilePerModule = true)
        }

        firrtl.Driver.execute(manager)

        for (name <- expectedOutputFileNames) {
          val file = new File(name)
          file.exists() should be (true)
          file.delete()
        }
      }
    }
  }


  "Directory deleter is handy for cleaning up after tests" - {
    "for example making a directory tree, and deleting it looks like" in {
      FileUtils.makeDirectory("dog/fox/wolf")
      val dir = new File("dog/fox/wolf")
      dir.exists() should be (true)
      dir.isDirectory should be (true)

      FileUtils.deleteDirectoryHierarchy("wolf") should be (false)
      FileUtils.deleteDirectoryHierarchy("dog") should be (true)
      dir.exists() should be (false)
    }
  }
}
