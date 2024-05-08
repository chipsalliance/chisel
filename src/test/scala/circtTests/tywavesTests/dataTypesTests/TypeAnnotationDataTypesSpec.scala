package circtTests.tywavesTests.dataTypesTests

import chisel3._
import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.tywaves.ClassParam
import circt.stage.ChiselStage
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.exceptions.TestFailedException

class TypeAnnotationDataTypesSpec extends AnyFunSpec with Matchers with chiselTests.Utils {
  import circtTests.tywavesTests.TywavesAnnotationCircuits.DataTypesCircuits._
  import circtTests.tywavesTests.TywavesAnnotationCircuits.{BindingChoice, PortBinding, RegBinding, WireBinding}
  import circtTests.tywavesTests.TestUtils._

  def typeTests(args: Array[String], targetDir: os.Path, b: BindingChoice): Unit = {
    def addClockReset(nameTop: String, nameInst: Option[String] = None) = if (b == RegBinding)
      Seq(
        (createExpected(s"~$nameTop\\|${nameInst.getOrElse(nameTop)}>clock", "Clock", "IO"), 1),
        (createExpected(s"~$nameTop\\|${nameInst.getOrElse(nameTop)}>reset", "Reset", "IO"), 1)
      )
    else Seq.empty
    it("should annotate ground types") {
      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitGroundTypes(b))))
      val string = os.read(targetDir / "TopCircuitGroundTypes.fir")
      val analog =
        if (b != RegBinding)
          Seq((createExpected("~TopCircuitGroundTypes\\|TopCircuitGroundTypes>analog", "Analog<1>", b.toString), 1))
        else Seq.empty
      val expectedMatches = Seq(
        (createExpected("~TopCircuitGroundTypes\\|TopCircuitGroundTypes>uint", "UInt<8>", b.toString), 1),
        (createExpected("~TopCircuitGroundTypes\\|TopCircuitGroundTypes>sint", "SInt<8>", b.toString), 1),
        (createExpected("~TopCircuitGroundTypes\\|TopCircuitGroundTypes>bool", "Bool", b.toString), 1),
        // TODO: there's no distinction between Bits and UInt since internally Bits.apply() returns UInt
        (createExpected("~TopCircuitGroundTypes\\|TopCircuitGroundTypes>bits", "UInt<8>", b.toString), 1)
      ) ++ addClockReset("TopCircuitGroundTypes") ++ analog
      checkAnno(expectedMatches, string)
    }

    it("should annotate bundles") {
      new ChiselStage(true).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitBundles(b))))
      val string = os.read(targetDir / "TopCircuitBundles.fir")

      val expectedMatches = Seq(
        (createExpected("~TopCircuitBundles\\|TopCircuitBundles>a", "AnonymousBundle", b.toString), 1),
        (createExpected("~TopCircuitBundles\\|TopCircuitBundles>b", "MyEmptyBundle", b.toString), 1),
        (createExpected("~TopCircuitBundles\\|TopCircuitBundles>c", "MyBundle", b.toString), 1),
        (createExpected("~TopCircuitBundles\\|TopCircuitBundles>c.a", "UInt<8>", b.toString), 1),
        (createExpected("~TopCircuitBundles\\|TopCircuitBundles>c.b", "SInt<8>", b.toString), 1),
        (createExpected("~TopCircuitBundles\\|TopCircuitBundles>c.c", "Bool", b.toString), 1)
      ) ++ addClockReset("TopCircuitBundles")
      checkAnno(expectedMatches, string)
    }

    it("should annotate nested bundles") {
      new ChiselStage(true)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitBundlesNested(b))))
      val string = os.read(targetDir / "TopCircuitBundlesNested.fir")
      val expectedMatches = Seq(
        (createExpected("~TopCircuitBundlesNested\\|TopCircuitBundlesNested>a", "MyNestedBundle", b.toString), 1),
        (createExpected("~TopCircuitBundlesNested\\|TopCircuitBundlesNested>a.a", "Bool", b.toString), 1),
        (createExpected("~TopCircuitBundlesNested\\|TopCircuitBundlesNested>a.b", "MyBundle", b.toString), 1),
        (createExpected("~TopCircuitBundlesNested\\|TopCircuitBundlesNested>a.b.a", "UInt<8>", b.toString), 1),
        (createExpected("~TopCircuitBundlesNested\\|TopCircuitBundlesNested>a.b.b", "SInt<8>", b.toString), 1),
        (createExpected("~TopCircuitBundlesNested\\|TopCircuitBundlesNested>a.b.c", "Bool", b.toString), 1),
        (createExpected("~TopCircuitBundlesNested\\|TopCircuitBundlesNested>a.c", "MyBundle", b.toString), 1),
        (createExpected("~TopCircuitBundlesNested\\|TopCircuitBundlesNested>a.c.a", "UInt<8>", b.toString), 1),
        (createExpected("~TopCircuitBundlesNested\\|TopCircuitBundlesNested>a.c.b", "SInt<8>", b.toString), 1),
        (createExpected("~TopCircuitBundlesNested\\|TopCircuitBundlesNested>a.c.c", "Bool", b.toString), 1)
      ) ++ addClockReset("TopCircuitBundlesNested")
      checkAnno(expectedMatches, string)
    }

    it("should annotate vecs") {
      new ChiselStage(true).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitVecs(b))))
      val string = os.read(targetDir / "TopCircuitVecs.fir")
      // format: off
      val expectedMatches = Seq(
        (createExpected("~TopCircuitVecs\\|TopCircuitVecs>a", "SInt<23>\\[5\\]", b.toString,
          params = Some(Seq(ClassParam("gen", "=> T", None), ClassParam("length", "Int", Some("5"))))), 1),
        (createExpected("~TopCircuitVecs\\|TopCircuitVecs>a\\[0\\]", "SInt<23>", b.toString), 1),
        (createExpected("~TopCircuitVecs\\|TopCircuitVecs>b", "SInt<23>\\[3\\]\\[5\\]", b.toString,
          params = Some(Seq(ClassParam("gen", "=> T", None), ClassParam("length", "Int", Some("5"))))), 1),
        (createExpected("~TopCircuitVecs\\|TopCircuitVecs>b\\[0\\]", "SInt<23>\\[3\\]", b.toString,
          params = Some(Seq(ClassParam("gen", "=> T", None), ClassParam("length", "Int", Some("3"))))), 1),
        (createExpected("~TopCircuitVecs\\|TopCircuitVecs>b\\[0\\]\\[0\\]", "SInt<23>", b.toString), 1),
        (createExpected("~TopCircuitVecs\\|TopCircuitVecs>c", "AnonymousBundle\\[5\\]", b.toString,
          params = Some(Seq(ClassParam("gen", "=> T", None), ClassParam("length", "Int", Some("5"))))), 1),
        (createExpected("~TopCircuitVecs\\|TopCircuitVecs>c\\[0\\]", "AnonymousBundle", b.toString), 1),
        (createExpected("~TopCircuitVecs\\|TopCircuitVecs>c\\[0\\].x", "UInt<8>", b.toString), 1),

        (createExpected("~TopCircuitVecs\\|TopCircuitVecs>d", "MixedVec", b.toString,
          // TODO: check for MixedVec what the value of the parameter should be
          params = Some(Seq(ClassParam("eltsIn", "Seq\\[T\\]", Some("Seq\\[T\\]\\(unsafeArray(.*?)\\)"))))), 1),
        (createExpected("~TopCircuitVecs\\|TopCircuitVecs>d.0", "UInt<3>", b.toString), 1),
        (createExpected("~TopCircuitVecs\\|TopCircuitVecs>d.1", "SInt<10>", b.toString), 1)
      ) ++ addClockReset("TopCircuitVecs")
      // format: on
      checkAnno(expectedMatches, string)
    }

    it("should annotate bundle with vec") {
      new ChiselStage(true).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitBundleWithVec(b))))
      val string = os.read(targetDir / "TopCircuitBundleWithVec.fir")
      // format: off
      val expectedMatches = Seq(
        (createExpected("~TopCircuitBundleWithVec\\|TopCircuitBundleWithVec>a", "AnonymousBundle", b.toString), 1),
        (createExpected("~TopCircuitBundleWithVec\\|TopCircuitBundleWithVec>a.vec", "UInt<8>\\[5\\]", b.toString,
          params = Some(Seq(ClassParam("gen", "=> T", None), ClassParam("length", "Int", Some("5"))))), 1),
        (createExpected("~TopCircuitBundleWithVec\\|TopCircuitBundleWithVec>a.vec\\[0\\]", "UInt<8>", b.toString), 1)
      ) ++ addClockReset("TopCircuitBundleWithVec")
      // format: on
      checkAnno(expectedMatches, string)
    }

    it("should annotate submodule types") {
      new ChiselStage(true)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitTypeInSubmodule(b))))
      val string = os.read(targetDir / "TopCircuitTypeInSubmodule.fir")
      val analog =
        if (b != RegBinding)
          Seq((createExpected("~TopCircuitTypeInSubmodule\\|TopCircuitGroundTypes>analog", "Analog<1>", b.toString), 1))
        else Seq.empty
      val expectedMatches = Seq(
        (createExpected("~TopCircuitTypeInSubmodule\\|TopCircuitGroundTypes>uint", "UInt<8>", b.toString), 1),
        (createExpected("~TopCircuitTypeInSubmodule\\|TopCircuitGroundTypes>sint", "SInt<8>", b.toString), 1),
        (createExpected("~TopCircuitTypeInSubmodule\\|TopCircuitGroundTypes>bool", "Bool", b.toString), 1),
        (createExpected("~TopCircuitTypeInSubmodule\\|TopCircuitGroundTypes>bits", "UInt<8>", b.toString), 1),
        (""""target":"~TopCircuitTypeInSubmodule\|TopCircuitGroundTypes",\s+"typeName":"TopCircuitGroundTypes"""", 1)
      ) ++ addClockReset("TopCircuitTypeInSubmodule", Some("TopCircuitGroundTypes")) ++ analog
      checkAnno(expectedMatches, string)
    }
  }

  describe("Port Annotations") {
    val targetDir = os.pwd / "test_run_dir" / "TywavesAnnotationSpec" / "Port Annotations"
    val args: Array[String] = Array("--target", "chirrtl", "--target-dir", targetDir.toString)

    it("should fail") {
      class TopCircuitFail extends Module
      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitFail)))
      val string = os.read(targetDir / "TopCircuitFail.fir")
      val expectedMatches = Seq(
        (createExpected("~TopCircuitFail\\|TopCircuitFail>clock", "Clock", "IO"), 1)
        //        (createExpected("~TopCircuitFail\\|TopCircuitFail>reset", "Bool", "IO"), 1)
      )
      assertThrows[TestFailedException] { checkAnno(expectedMatches, string) }
    }

    it("should annotate clock and reset") {
      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitClockReset)))
      val string = os.read(targetDir / "TopCircuitClockReset.fir")
      val expectedMatches = Seq(
        (createExpected("~TopCircuitClockReset\\|TopCircuitClockReset>clock", "Clock", "IO"), 1),
        (createExpected("~TopCircuitClockReset\\|TopCircuitClockReset>syncReset", "Bool", "IO"), 1),
        (createExpected("~TopCircuitClockReset\\|TopCircuitClockReset>reset", "Reset", "IO"), 1),
        (createExpected("~TopCircuitClockReset\\|TopCircuitClockReset>asyncReset", "AsyncReset", "IO"), 1)
      )
      checkAnno(expectedMatches, string)
    }

    it("should annotate implicit clock and reset") {
      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitImplicitClockReset)))
      val string = os.read(targetDir / "TopCircuitImplicitClockReset.fir")
      val expectedMatches = Seq(
        (createExpected("~TopCircuitImplicitClockReset\\|TopCircuitImplicitClockReset>clock", "Clock", "IO"), 1),
        (createExpected("~TopCircuitImplicitClockReset\\|TopCircuitImplicitClockReset>reset", "Bool", "IO"), 1)
      )
      checkAnno(expectedMatches, string)

    }

    typeTests(args, targetDir, PortBinding)
  }

  describe("Wire Annotations") {
    val targetDir = os.pwd / "test_run_dir" / "TywavesAnnotationSpec" / "Wire Annotations"
    val args: Array[String] = Array("--target", "chirrtl", "--target-dir", targetDir.toString)
    typeTests(args, targetDir, WireBinding)
  }

  describe("Reg Annotations") {
    val targetDir = os.pwd / "test_run_dir" / "TywavesAnnotationSpec" / "Reg Annotations"
    val args: Array[String] = Array("--target", "chirrtl", "--target-dir", targetDir.toString)
    typeTests(args, targetDir, RegBinding)
  }

}
