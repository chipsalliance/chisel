// SPDX-License-Identifier: Apache-2.0

package firrtlTests.annotationTests

import firrtl.FileUtils
import firrtl.annotations._
import firrtl.passes.memlib.ReplSeqMemAnnotation
import firrtl.testutils.FirrtlFlatSpec
import firrtl.transforms.BlackBoxInlineAnno
import logger.Logger
import logger.Logger.OutputCaptor

import java.io.{File, PrintWriter}

class UnrecognizedAnnotationSpec extends FirrtlFlatSpec {
  behavior.of("unrecognized annotations can be carried through serialization and deserialization")

  it should "preserve unknown annotations when allowed" in {
    val annotations =
      JsonProtocol.deserialize(UnrecognizedAnnotationTextGenerator.jsonText(includeAllowUnrecognizedAnnotations = true))

    annotations.exists(_.isInstanceOf[BlackBoxInlineAnno]) should be(true)
    annotations.count(_.isInstanceOf[UnrecognizedAnnotation]) should be(2)
    annotations.exists(_.isInstanceOf[ReplSeqMemAnnotation]) should be(false)

    val jsonOutputText = JsonProtocol.serialize(annotations)

    jsonOutputText should include(""""class":"firrtl.transforms.BlackBoxInlineAnno"""")
    jsonOutputText should include(""""class":"freechips.rocketchip.util.RegFieldDescMappingAnnotation"""")
    jsonOutputText should include(""""class":"freechips.rocketchip.util.SRAMAnnotation"""")
  }

  it should "throw an error when unknown annotations are present but AllowUnrecognizedAnnotation is not" in {

    // Default log level is error, which the JSON parsing uses here
    Logger.makeScope(Seq()) {
      val captor = new OutputCaptor
      Logger.setOutput(captor.printStream)

      val parsingError = intercept[UnrecogizedAnnotationsException] {
        JsonProtocol.deserialize(
          UnrecognizedAnnotationTextGenerator.jsonText(includeAllowUnrecognizedAnnotations = false)
        )
      }
      parsingError.getMessage should include("RegFieldDescMappingAnnotation")

      val output = captor.getOutputAsString
      output should include("Annotation parsing found unrecognized annotations")
      output should include(
        "This error can be ignored with an AllowUnrecognizedAnnotationsAnnotation or command line flag --allow-unrecognized-annotations"
      )
      output should include(
        "freechips.rocketchip.util.RegFieldDescMappingAnnotation"
      )
      output should include(
        "freechips.rocketchip.util.SRAMAnnotation"
      )
    }
  }

  // Following test will operate on an annotation JSON file with two unrecognized annotations in it
  //

  case class TestFileNames(
    allowUnrecognized:     Boolean,
    inputAnnotations:      String,
    outputAnnotations:     String,
    outputAnnotationsFull: String,
    firrtlSource:          String,
    firrtlOutput:          String)

  def setupFiles(addAllowUnrecognizedFlag: Boolean, addAllowUnrecognizedAnno: Boolean): TestFileNames = {
    val dirName = (addAllowUnrecognizedFlag, addAllowUnrecognizedAnno) match {
      case (false, false) => s"test_run_dir/unrecognized_annotation_fail"
      case (true, false)  => s"test_run_dir/unrecognized_annotation_flag"
      case (false, true)  => s"test_run_dir/unrecognized_annotation_anno"
      case (true, true)   => s"test_run_dir/unrecognized_annotation_flag_and_anno"
    }
    val dir = new File(dirName)
    dir.mkdirs()

    val fileNames = TestFileNames(
      allowUnrecognized = addAllowUnrecognizedFlag,
      inputAnnotations = s"$dirName/input_annotations.json",
      outputAnnotations = s"$dirName/output_annotations",
      outputAnnotationsFull = s"$dirName/output_annotations.anno.json",
      firrtlSource = s"$dirName/trivial.fir",
      firrtlOutput = s"$dirName/trivial_out"
    )

    def writeText(fileName: String, text: String): Unit = {
      val writer = new PrintWriter(fileName)
      writer.write(text)
      writer.close()
    }

    writeText(
      fileNames.inputAnnotations,
      UnrecognizedAnnotationTextGenerator.jsonText(includeAllowUnrecognizedAnnotations = addAllowUnrecognizedAnno)
    )
    writeText(
      fileNames.firrtlSource,
      s"""
         |circuit Trivial :
         |  module Trivial :
         |    input clock : Clock
         |    input reset : UInt<1>
         |""".stripMargin
    )
    fileNames
  }

  /* construct an array of command line strings, based on file names and  */
  def makeCommandLineArgs(fileNames: TestFileNames): Array[String] = {

    (if (fileNames.allowUnrecognized) {
       Array("--allow-unrecognized-annotations")
     } else {
       Array.empty[String]
     }) ++
      Array(
        "--annotation-file",
        fileNames.inputAnnotations,
        "-i",
        fileNames.firrtlSource,
        "-X",
        "high",
        "-o",
        fileNames.firrtlOutput,
        "--output-annotation-file",
        fileNames.outputAnnotations
      )
  }
}

object UnrecognizedAnnotationTextGenerator {

  def jsonText(includeAllowUnrecognizedAnnotations: Boolean): String = {
    val serializedAllowUnrecognized = if (includeAllowUnrecognizedAnnotations) {
      """
        |  {
        |    "class": "firrtl.stage.AllowUnrecognizedAnnotations$"
        |  },""".stripMargin
    } else {
      ""
    }

    s"""|[$serializedAllowUnrecognized
        |  {
        |    "class": "firrtl.transforms.BlackBoxInlineAnno",
        |    "target": "TestHarness.plusarg_reader_27",
        |    "name": "plusarg_reader.v",
        |    "text": "License text"
        |  },
        |  {
        |    "class": "freechips.rocketchip.util.RegFieldDescMappingAnnotation",
        |  },
        |  {
        |    "class": "freechips.rocketchip.util.SRAMAnnotation",
        |  }
        |]
        |""".stripMargin
  }
}
