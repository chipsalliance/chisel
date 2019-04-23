// See LICENSE for license details.

package firrtlTests.options.phases

import org.scalatest.{FlatSpec, Matchers}

import java.io.{File, PrintWriter}

import firrtl.AnnotationSeq
import firrtl.annotations.{Annotation, AnnotationFileNotFoundException, JsonProtocol,
  NoTargetAnnotation}
import firrtl.options.phases.GetIncludes
import firrtl.options.{InputAnnotationFileAnnotation, Phase}
import firrtl.util.BackendCompilationUtilities

case object A extends NoTargetAnnotation
case object B extends NoTargetAnnotation
case object C extends NoTargetAnnotation
case object D extends NoTargetAnnotation
case object E extends NoTargetAnnotation

class GetIncludesSpec extends FlatSpec with Matchers with BackendCompilationUtilities with firrtlTests.Utils {

  val dir = new File("test_run_dir/GetIncludesSpec")
  dir.mkdirs()

  def ref(filename: String): InputAnnotationFileAnnotation = InputAnnotationFileAnnotation(s"$dir/$filename.anno.json")

  def checkAnnos(a: AnnotationSeq, b: AnnotationSeq): Unit = {
    info("read the expected number of annotations")
    a.size should be (b.size)

    info("annotations match exact order")
    a.zip(b).foreach{ case (ax, bx) => ax should be (bx) }
  }

  val files = Seq(
    new File(dir + "/a.anno.json") -> Seq(A, ref("b")),
    new File(dir + "/b.anno.json") -> Seq(B, ref("c"), ref("a")),
    new File(dir + "/c.anno.json") -> Seq(C, ref("d"), ref("e")),
    new File(dir + "/d.anno.json") -> Seq(D),
    new File(dir + "/e.anno.json") -> Seq(E)
  )

  files.foreach{ case (file, annotations) =>
    val pw = new PrintWriter(file)
    pw.write(JsonProtocol.serialize(annotations))
    pw.close()
  }

  class Fixture { val phase: Phase = new GetIncludes }

  behavior of classOf[GetIncludes].toString

  it should "throw an exception if the annotation file doesn't exit" in new Fixture {
    intercept[AnnotationFileNotFoundException]{ phase.transform(Seq(ref("f"))) }
      .getMessage should startWith("Annotation file")
  }

  it should "read annotations from a file" in new Fixture {
    val e = ref("e")
    val in = Seq(e)
    val expect = Seq(E)
    val out = phase.transform(in)

    checkAnnos(out, expect)
  }

  it should "read annotations from multiple files, but not reading duplicates" in new Fixture {
    val Seq(d, e) = Seq("d", "e").map(ref)
    val in = Seq(d, e, e, d)
    val expect = Seq(D, E)
    val (stdout, _, out) = grabStdOutErr { phase.transform(in) }

    checkAnnos(out, expect)

    Seq("d", "e").foreach{ x =>
      info(s"a warning about '$x.anno.json' was printed")
      stdout should include (s"Warning: Annotation file ($dir/$x.anno.json) already included!")
    }
  }

  it should "handle recursive references gracefully, but show a warning" in new Fixture {
    val Seq(a, b, c, d, e) = Seq("a", "b", "c", "d", "e").map(ref)
    val in = Seq(a)
    val expect = Seq(A, B, C, D, E)
    val (stdout, _, out) = grabStdOutErr { phase.transform(in) }

    checkAnnos(out, expect)

    info("a warning about 'a.anno.json' was printed")
    stdout should include (s"Warning: Annotation file ($dir/a.anno.json)")
  }

}
