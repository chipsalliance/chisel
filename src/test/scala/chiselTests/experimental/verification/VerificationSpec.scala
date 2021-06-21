// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental.verification

import chisel3._
import chisel3.experimental.{ChiselAnnotation, verification => formal}
import chisel3.stage.ChiselStage
import chiselTests.ChiselPropSpec
import firrtl.annotations.{ReferenceTarget, SingleTargetAnnotation}

import java.io.File
import org.scalatest.matchers.should.Matchers

class SimpleTest extends Module {
  val io = IO(new Bundle{
    val in = Input(UInt(8.W))
    val out = Output(UInt(8.W))
  })
  io.out := io.in
  formal.cover(io.in === 3.U)
  when (io.in === 3.U) {
    formal.assume(io.in =/= 2.U)
    formal.assert(io.out === io.in)
  }
}

/** The following test marks verification components with dummy annotations. */
object VerificationAnnotationTest {
  /** Dummy verification annotation.
    * @param target target of component to be annotated
    */
  case class VerifAnnotation(target: ReferenceTarget) extends SingleTargetAnnotation[ReferenceTarget] {
    def duplicate(n: ReferenceTarget): VerifAnnotation = this.copy(target = n)
  }

  object VerifAnnotation {
    /** Create annotation for a given verification component.
      * @param c component to be annotated
      */
    def annotate(c: experimental.BaseSim): Unit = {
      chisel3.experimental.annotate(new ChiselAnnotation {
        def toFirrtl: VerifAnnotation = VerifAnnotation(c.toTarget)
      })
    }
  }

  /** Circuit that contains and annotates verification components. */
  class AnnotationTest extends Module {
    val io = IO(new Bundle{
      val in = Input(UInt(8.W))
      val out = Output(UInt(8.W))
    })
    io.out := io.in
    val cov = formal.cover(io.in === 3.U)
    VerifAnnotation.annotate(cov)
    val assm = formal.assume(io.in =/= 2.U)
    VerifAnnotation.annotate(assm)
    val asst = formal.assert(io.out === io.in)
    VerifAnnotation.annotate(asst)
  }
}

class VerificationSpec extends ChiselPropSpec with Matchers {

  def assertContains(s: Seq[String], x: String): Unit = {
    val containsLine = s.map(_.contains(x)).reduce(_ || _)
    assert(containsLine, s"\n  $x\nwas not found in`\n  ${s.mkString("\n  ")}``")
  }

  property("basic equality check should work") {
    val fir = ChiselStage.emitChirrtl(new SimpleTest)
    val lines = fir.split("\n").map(_.trim)

    // reset guard around the verification statement
    assertContains(lines, "when _T_2 : @[VerificationSpec.scala")
    assertContains(lines, "cover(clock, _T, UInt<1>(\"h1\"), \"\") @[VerificationSpec.scala")

    assertContains(lines, "when _T_6 : @[VerificationSpec.scala")
    assertContains(lines, "assume(clock, _T_4, UInt<1>(\"h1\"), \"\") @[VerificationSpec.scala")

    assertContains(lines, "when _T_9 : @[VerificationSpec.scala")
    assertContains(lines, "assert(clock, _T_7, UInt<1>(\"h1\"), \"\") @[VerificationSpec.scala")
  }

  property("annotation of verification constructs should work") {
    val testDir = new File("test_run_dir", "VerificationAnnotationTests")
    // delete contents from past runs
    testDir.listFiles.foreach(f => f.delete())
    (new ChiselStage).emitSystemVerilog(
      gen = new VerificationAnnotationTest.AnnotationTest,
      args = Array ("-td", testDir.getPath)
    )

    // read in annotation file
    val annoFile = new File(testDir, "AnnotationTest.anno.json")
    annoFile.exists()
    val annoLines = scala.io.Source.fromFile(annoFile).getLines.toList

    // check for expected verification annotations
    exactly(3, annoLines) should include ("chiselTests.experimental.verification.VerificationAnnotationTest$VerifAnnotation")
    exactly(1, annoLines) should include ("~AnnotationTest|AnnotationTest>asst")
    exactly(1, annoLines) should include ("~AnnotationTest|AnnotationTest>assm")
    exactly(1, annoLines) should include ("~AnnotationTest|AnnotationTest>cov")
  }

}
