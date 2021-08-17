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
    assertContains(lines, "cover(clock, _T, UInt<1>(\"h1\"), \"\")")

    assertContains(lines, "when _T_6 : @[VerificationSpec.scala")
    assertContains(lines, "assume(clock, _T_4, UInt<1>(\"h1\"), \"\")")

    assertContains(lines, "when _T_9 : @[VerificationSpec.scala")
    assertContains(lines, "assert(clock, _T_7, UInt<1>(\"h1\"), \"\")")
  }

  property("annotation of verification constructs should work") {
    /** Circuit that contains and annotates verification nodes. */
    class AnnotationTest extends Module {
      val io = IO(new Bundle{
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
      io.out := io.in
      val cov = formal.cover(io.in === 3.U)
      val assm = formal.assume(io.in =/= 2.U)
      val asst = formal.assert(io.out === io.in)
      VerifAnnotation.annotate(cov)
      VerifAnnotation.annotate(assm)
      VerifAnnotation.annotate(asst)
    }

    // compile circuit
    val testDir = new File("test_run_dir", "VerificationAnnotationTests")
    (new ChiselStage).emitSystemVerilog(
      gen = new AnnotationTest,
      args = Array("-td", testDir.getPath)
    )

    // read in annotation file
    val annoFile = new File(testDir, "AnnotationTest.anno.json")
    annoFile should exist
    val annoLines = scala.io.Source.fromFile(annoFile).getLines.toList

    // check for expected verification annotations
    exactly(3, annoLines) should include ("chiselTests.experimental.verification.VerifAnnotation")
    exactly(1, annoLines) should include ("~AnnotationTest|AnnotationTest>asst")
    exactly(1, annoLines) should include ("~AnnotationTest|AnnotationTest>assm")
    exactly(1, annoLines) should include ("~AnnotationTest|AnnotationTest>cov")

    // read in FIRRTL file
    val firFile = new File(testDir, "AnnotationTest.fir")
    firFile should exist
    val firLines = scala.io.Source.fromFile(firFile).getLines.toList

    // check that verification components have expected names
    exactly(1, firLines) should include ("cover(clock, _T, UInt<1>(1), \"\") : cov")
    exactly(1, firLines) should include ("assume(clock, _T_3, UInt<1>(1), \"\") : assm")
    exactly(1, firLines) should include ("assert(clock, _T_6, UInt<1>(1), \"\") : asst")
  }

  property("annotation of verification constructs with suggested name should work") {
    /** Circuit that annotates a renamed verification nodes. */
    class AnnotationRenameTest extends Module {
      val io = IO(new Bundle{
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
      io.out := io.in

      val goodbye = formal.assert(io.in === 1.U)
      goodbye.suggestName("hello")
      VerifAnnotation.annotate(goodbye)

      VerifAnnotation.annotate(formal.assume(io.in =/= 2.U).suggestName("howdy"))
    }

    // compile circuit
    val testDir = new File("test_run_dir", "VerificationAnnotationRenameTests")
    (new ChiselStage).emitSystemVerilog(
      gen = new AnnotationRenameTest,
      args = Array("-td", testDir.getPath)
    )

    // read in annotation file
    val annoFile = new File(testDir, "AnnotationRenameTest.anno.json")
    annoFile should exist
    val annoLines = scala.io.Source.fromFile(annoFile).getLines.toList

    // check for expected verification annotations
    exactly(2, annoLines) should include ("chiselTests.experimental.verification.VerifAnnotation")
    exactly(1, annoLines) should include ("~AnnotationRenameTest|AnnotationRenameTest>hello")
    exactly(1, annoLines) should include ("~AnnotationRenameTest|AnnotationRenameTest>howdy")

    // read in FIRRTL file
    val firFile = new File(testDir, "AnnotationRenameTest.fir")
    firFile should exist
    val firLines = scala.io.Source.fromFile(firFile).getLines.toList

    // check that verification components have expected names
    exactly(1, firLines) should include ("assert(clock, _T, UInt<1>(1), \"\") : hello")
    exactly(1, firLines) should include ("assume(clock, _T_3, UInt<1>(1), \"\") : howdy")
  }
}
