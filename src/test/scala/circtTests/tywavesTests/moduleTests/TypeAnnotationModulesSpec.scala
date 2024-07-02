package circtTests.tywavesTests.moduleTests

import chisel3.stage.ChiselGeneratorAnnotation
import chisel3._
import circt.stage.ChiselStage
import org.scalatest.AppendedClues.convertToClueful
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

class TypeAnnotationModulesSpec extends AnyFunSpec with Matchers with chiselTests.Utils {

  import circtTests.tywavesTests.TywavesAnnotationCircuits.ModuleCircuits._
  import circtTests.tywavesTests.TestUtils._

  describe("Module Annotations") {
    val targetDir = os.pwd / "test_run_dir" / "TywavesAnnotationSpec" / "Module Annotations"
    val args: Array[String] = Array("--target", "chirrtl", "--target-dir", targetDir.toString)

    it("should annotate a circuit") {
      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuit)))
      os.read(targetDir / "TopCircuit.fir") should include("TywavesAnnotation").and(
        include("\"target\":\"~TopCircuit|TopCircuit\"").and(
          include("\"typeName\":\"TopCircuit\"")
        )
      )
    }

    it("should annotate a module in a circuit") {
      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitSubModule)))
      os.read(targetDir / "TopCircuitSubModule.fir") should include("TywavesAnnotation").and(
        include("\"target\":\"~TopCircuitSubModule|MyModule\"").and(include("\"typeName\":\"MyModule\""))
      )
    }

    it("should annotate a module in a circuit multiple times") {
      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitMultiModule)))
      // The file should include it
      val string = os.read(targetDir / "TopCircuitMultiModule.fir")
      countSubstringOccurrences(string, "\"class\":\"chisel3.tywaves.TywavesAnnotation\"") should be(5)
      countSubstringOccurrences(string, "\"typeName\":\"MyModule\"") should be(4)
    }

    it("should annotate black boxes") {

      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitBlackBox)))
      val string = os.read(targetDir / "TopCircuitBlackBox.fir")
      countSubstringOccurrences(string, "\"class\":\"chisel3.tywaves.TywavesAnnotation\"") should be(5)
      countSubstringOccurrences(string, "\"typeName\":\"MyBlackBox") should be(4)
    }

    it("should annotate intrinsic modules") {
      import chisel3.experimental.IntrinsicModule

      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitIntrinsic)))
      val string = os.read(targetDir / "TopCircuitIntrinsic.fir")
      countSubstringOccurrences(string, "\"class\":\"chisel3.tywaves.TywavesAnnotation\"") should be(13)
      countSubstringOccurrences(string, "\"typeName\":\"ExampleIntrinsicModule") should be(4)
    }

    it("should annotate classes modules") {
      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitClasses)))
      val string = os.read(targetDir / "TopCircuitClasses.fir")
      countSubstringOccurrences(string, "\"typeName\":\"CSRModule\"") should be(4)
      countSubstringOccurrences(
        string,
        "\"target\":\"~TopCircuitClasses\\|CSRDescription\",\\s*\"typeName\":\"CSRDescription"
      ) should be(1)
      countSubstringOccurrences(string, "\"class\":\"chisel3.tywaves.TywavesAnnotation\"") should be(6)
    }

    it("should annotate parametric module") {
      class MyModule[T <: Data](gen: T) extends RawModule

      class TopCircuitParametric extends RawModule {
        val myModule1: MyModule[UInt] = Module(new MyModule(UInt(8.W)))
        val myModule2: MyModule[SInt] = Module(new MyModule(SInt(8.W)))
        val myModule3: MyModule[Bool] = Module(new MyModule(Bool()))
      }

      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitParametric)))
      val string = os.read(targetDir / "TopCircuitParametric.fir")

      // TODO: Fix this test for parametric module
//      val expectedMatches = Seq(
//        ("\"typeName\":\"MyModule\\[UInt<8>\\]\"", 1),
//        ("\"typeName\":\"MyModule\\[SInt<8>\\]\"", 1),
//        ("\"typeName\":\"MyModule\\[Bool\\]\"", 1)
//      )
//      checkAnno(expectedMatches, string)
    }
  }

}
