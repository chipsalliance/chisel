// SPDX-License-Identifier: Apache-2.0

package firrtlTests.options.phases

import java.io.File

import firrtl.AnnotationSeq
import firrtl.annotations.NoTargetAnnotation
import firrtl.options.{
  BufferedCustomFileEmission,
  CustomFileEmission,
  InputAnnotationFileAnnotation,
  OutputAnnotationFileAnnotation,
  Phase,
  PhaseException,
  StageOption,
  StageOptions,
  TargetDirAnnotation
}
import firrtl.options.Viewer.view
import firrtl.options.phases.{GetIncludes, WriteOutputAnnotations}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class WriteOutputAnnotationsSpec extends AnyFlatSpec with Matchers with firrtl.testutils.Utils {

  val dir = "test_run_dir/WriteOutputAnnotationSpec"

  /** Check if the annotations contained by a [[File]] and the same, and in the same order, as a reference
    * [[AnnotationSeq]]. This uses [[GetIncludes]] as that already knows how to read annotation files.
    * @param f a file to read
    * @param a the expected annotations
    */
  private def fileContainsAnnotations(f: File, a: AnnotationSeq): Unit = {
    info(s"output file '$f' exists")
    f should (exist)

    info(s"reading '$f' works")
    val read = (new GetIncludes)
      .transform(Seq(InputAnnotationFileAnnotation(f.toString)))

    info(s"annotations in file are expected size")
    read.size should be(a.size)

    read
      .zip(a)
      .foreach {
        case (read, expected) =>
          info(s"$read matches")
          read should be(expected)
      }

    f.delete()
  }

  class Fixture { val phase: Phase = new WriteOutputAnnotations }

  behavior.of(classOf[WriteOutputAnnotations].toString)

  it should "write annotations to a file (excluding StageOptions)" in new Fixture {
    val file = new File(dir + "/should-write-annotations-to-a-file.anno.json")
    val annotations = Seq(
      OutputAnnotationFileAnnotation(file.toString),
      WriteOutputAnnotationsSpec.FooAnnotation,
      WriteOutputAnnotationsSpec.BarAnnotation(0),
      WriteOutputAnnotationsSpec.BarAnnotation(1)
    )
    val expected = annotations.filter {
      case _: StageOption => false
      case _ => true
    }
    val out = phase.transform(annotations)

    info("annotations should be as expected")
    out.toSeq should be(annotations)

    fileContainsAnnotations(file, expected)
  }

  it should "write CustomFileEmission annotations" in new Fixture {
    val file = new File("write-CustomFileEmission-annotations.anno.json")
    val annotations = Seq(
      TargetDirAnnotation(dir),
      OutputAnnotationFileAnnotation(file.toString),
      WriteOutputAnnotationsSpec.Custom("hello!")
    )
    val serializedFileName = view[StageOptions](annotations).getBuildFileName("Custom", Some(".Emission"))
    val expected = annotations.flatMap {
      case _: WriteOutputAnnotationsSpec.Custom => Some(WriteOutputAnnotationsSpec.Replacement(serializedFileName))
      case _: StageOption                       => None
      case a => Some(a)
    }

    val out = phase.transform(annotations)

    info("annotations are unmodified")
    out.toSeq should be(annotations)

    fileContainsAnnotations(new File(dir, file.toString), expected)

    info(s"file '$serializedFileName' exists")
    new File(serializedFileName) should (exist)
  }

  it should "support CustomFileEmission to binary files" in new Fixture {
    val file = new File("write-CustomFileEmission-binary-files.anno.json")
    val data = Array[Byte](0x0a, 0xa0.toByte)
    val annotations = Seq(
      TargetDirAnnotation(dir),
      OutputAnnotationFileAnnotation(file.toString),
      WriteOutputAnnotationsSpec.Binary(data)
    )

    val serializedFileName = view[StageOptions](annotations).getBuildFileName("Binary", Some(".Emission"))
    val out = phase.transform(annotations)

    info(s"file '$serializedFileName' exists")
    new File(serializedFileName) should (exist)

    info(s"file '$serializedFileName' is correct")
    val inputStream = new java.io.FileInputStream(serializedFileName)
    val result = new Array[Byte](2)
    inputStream.read(result)
    result should equal(data)
  }

  it should "write BufferedCustomFileEmission annotations" in new Fixture {
    val file = new File("write-CustomFileEmission-annotations.anno.json")
    val data = List("hi", "bye", "yo")
    val annotations = Seq(
      TargetDirAnnotation(dir),
      OutputAnnotationFileAnnotation(file.toString),
      WriteOutputAnnotationsSpec.Buffered(data)
    )
    val serializedFileName = view[StageOptions](annotations).getBuildFileName("Buffered", Some(".Emission"))
    val out = phase.transform(annotations)

    info(s"file '$serializedFileName' exists")
    new File(serializedFileName) should (exist)

    info(s"file '$serializedFileName' is correct")
    val result = scala.io.Source.fromFile(serializedFileName).mkString
    result should equal(data.mkString)
  }

  it should "error if multiple annotations try to write to the same file" in new Fixture {
    val file = new File("write-CustomFileEmission-annotations-error.anno.json")
    val annotations = Seq(
      TargetDirAnnotation(dir),
      OutputAnnotationFileAnnotation(file.toString),
      WriteOutputAnnotationsSpec.Custom("foo"),
      WriteOutputAnnotationsSpec.Custom("bar")
    )
    intercept[PhaseException] {
      phase.transform(annotations)
    }.getMessage should startWith("Multiple CustomFileEmission annotations")
  }

}

private object WriteOutputAnnotationsSpec {

  case object FooAnnotation extends NoTargetAnnotation

  case class BarAnnotation(x: Int) extends NoTargetAnnotation

  case class Custom(value: String) extends NoTargetAnnotation with CustomFileEmission {

    override protected def baseFileName(a: AnnotationSeq): String = "Custom"

    override protected def suffix: Option[String] = Some(".Emission")

    override def getBytes: Iterable[Byte] = value.getBytes

    override def replacements(file: File): AnnotationSeq = Seq(Replacement(file.toString))

  }

  case class Binary(value: Array[Byte]) extends NoTargetAnnotation with CustomFileEmission {

    override protected def baseFileName(a: AnnotationSeq): String = "Binary"

    override protected def suffix: Option[String] = Some(".Emission")

    override def getBytes: Iterable[Byte] = value
  }

  case class Replacement(file: String) extends NoTargetAnnotation

  case class Buffered(content: List[String]) extends NoTargetAnnotation with BufferedCustomFileEmission {

    override protected def baseFileName(a: AnnotationSeq): String = "Buffered"

    override protected def suffix: Option[String] = Some(".Emission")

    override def getBytesBuffered: Iterable[Array[Byte]] = content.view.map(_.getBytes)
  }
}
