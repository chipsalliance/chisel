package circtTests.tywavesTests

import chisel3._
import chisel3.experimental.Analog
import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.util.MixedVec
import circt.stage.ChiselStage
import org.scalatest.AppendedClues.convertToClueful
import org.scalatest.exceptions.TestFailedException
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

object TywavesAnnotationSpec {
  import chisel3._

  // A user defined type from Bundle: FooBundle
  class FooBundle extends Bundle {
    val a: UInt = Input(UInt(8.W))
    val b: SInt = Input(SInt(8.W))
    val c: Bool = Input(Bool())
  }

  // A user defined type as nested bundle: BazBundle
  class BarBundle extends Bundle {
    val a: Bool = Input(Bool())
    val b: FooBundle = new FooBundle
    val c: FooBundle = Flipped(new FooBundle)
  }

  // A class
  class Foo extends RawModule {
    // A user defined type
    val a: FooBundle = IO(Flipped(new FooBundle))
    // An anonymous bundle
    val b = IO(new Bundle {
      val a: UInt = Input(UInt(8.W))
      val b: SInt = Input(SInt(8.W))
      val c: Bool = Input(Bool())
    })

    // A nested bundle
    val c: BarBundle = IO(new BarBundle)

    a <> b
    c.b <> c.c
  }

  // A class with a submodule in it
  class Bar extends RawModule {
    val a:   BarBundle = IO(new BarBundle)
    val foo: Foo = Module(new Foo)
    val bar: Foo = Module(new Foo)

    a <> foo.c
    a <> bar.c
  }

}

class TywavesAnnotationSpec extends AnyFunSpec with Matchers with chiselTests.Utils {
  private val baseDir = os.pwd / "test_run_dir" / this.getClass.getSimpleName

  def countSubstringOccurrences(mainString: String, subString: String): Int = {
    val pattern = subString.r
    pattern.findAllMatchIn(mainString).length
  }

  def createExpected(target: String, typeName: String, binding: String = ""): String = {
    val realTypeName = binding match {
      case "" => typeName
      case _  => s"$binding\\[$typeName\\]"
    }
    s"""\"target\":\"$target\",\\s+\"typeName\":\"$realTypeName\"""".stripMargin
  }

  def checkAnno(expectedMatches: Seq[(String, Int)], refString: String): Unit = {
    def totalAnnoCheck(n: Int): (String, Int) = (""""class":"chisel3.tywaves.TywavesAnnotation"""", n + 1)

    (expectedMatches :+ totalAnnoCheck(expectedMatches.map(_._2).sum)).foreach {
      case (pattern, count) =>
        (countSubstringOccurrences(refString, pattern) should be(count)).withClue(s"Pattern: $pattern")
    }
  }

  describe("Module Annotations") {
    val targetDir = os.pwd / "test_run_dir" / "TywavesAnnotationSpec" / "Module Annotations"
    val args: Array[String] = Array("--target", "chirrtl", "--target-dir", targetDir.toString)

    it("should annotate a circuit") {
      class TopCircuit extends RawModule

      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuit)))
      os.read(targetDir / "TopCircuit.fir") should include("TywavesAnnotation").and(
        include("\"target\":\"~TopCircuit|TopCircuit\"").and(
          include("\"typeName\":\"TopCircuit\"")
        )
      )
    }

    it("should annotate a module in a circuit") {
      class MyModule extends RawModule
      class TopCircuitSubModule extends RawModule { val mod = Module(new MyModule) }

      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitSubModule)))
      os.read(targetDir / "TopCircuitSubModule.fir") should include("TywavesAnnotation").and(
        include("\"target\":\"~TopCircuitSubModule|MyModule\"").and(include("\"typeName\":\"MyModule\""))
      )
    }

    it("should annotate a module in a circuit multiple times") {
      class MyModule extends RawModule
      class TopCircuitMultiModule extends RawModule {
        // 4 times MyModule in total
        val mod1 = Module(new MyModule)
        val mod2 = Module(new MyModule)
        val mods = Seq.fill(2)(Module(new MyModule))
      }

      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitMultiModule)))
      // The file should include it
      val string = os.read(targetDir / "TopCircuitMultiModule.fir")
      countSubstringOccurrences(string, "\"class\":\"chisel3.tywaves.TywavesAnnotation\"") should be(5)
      countSubstringOccurrences(string, "\"typeName\":\"MyModule\"") should be(4)
    }

    it("should annotate variants of Parametrized module in a circuit") {
      class MyModule(n: Int) extends RawModule
      class TopCircuitWithParams extends RawModule {
        val mod1: MyModule = Module(new MyModule(8))
        val mod2: MyModule = Module(new MyModule(16))
        val mod3: MyModule = Module(new MyModule(32))
      }
      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitWithParams)))
      // The file should include it
      val string = os.read(targetDir / "TopCircuitWithParams.fir")
      countSubstringOccurrences(string, "\"class\":\"chisel3.tywaves.TywavesAnnotation\"") should be(4)
      countSubstringOccurrences(string, "\"typeName\":\"MyModule") should be(3)
      // This now fails!!! TODO: it is something that I want to have, PARAMETERIZED MODULES
      countSubstringOccurrences(string, "\"typeName\":\"MyModule(8)\"") should be(1)
      countSubstringOccurrences(string, "\"typeName\":\"MyModule(16)\"") should be(1)
      countSubstringOccurrences(string, "\"typeName\":\"MyModule(32)\"") should be(1)
    }

    it("should annotate black boxes") {
      class TopCircuitBlackBox extends RawModule {
        class MyBlackBox extends BlackBox(Map("PARAM1" -> "TRUE", "PARAM2" -> "DEFAULT")) { val io = IO(new Bundle {}) }
        val myBlackBox1:  MyBlackBox = Module(new MyBlackBox)
        val myBlackBox2:  MyBlackBox = Module(new MyBlackBox)
        val myBlackBoxes: Seq[MyBlackBox] = Seq.fill(2)(Module(new MyBlackBox))
      }
      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitBlackBox)))
      val string = os.read(targetDir / "TopCircuitBlackBox.fir")
      countSubstringOccurrences(string, "\"class\":\"chisel3.tywaves.TywavesAnnotation\"") should be(5)
      countSubstringOccurrences(string, "\"typeName\":\"MyBlackBox") should be(4)
    }

    it("should annotate intrinsic modules") {
      import chisel3.experimental.IntrinsicModule
      class ExampleIntrinsicModule(str: String) extends IntrinsicModule("OtherIntrinsic", Map("STRING" -> str)) {}
      class TopCircuitIntrinsic extends RawModule {
        val myIntrinsicModule1: ExampleIntrinsicModule = Module(new ExampleIntrinsicModule("Hello"))
        val myIntrinsicModule2: ExampleIntrinsicModule = Module(new ExampleIntrinsicModule("World"))
        val myIntrinsicModules: Seq[ExampleIntrinsicModule] = Seq.fill(2)(Module(new ExampleIntrinsicModule("Hello")))
      }
      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitIntrinsic)))
      val string = os.read(targetDir / "TopCircuitIntrinsic.fir")
      countSubstringOccurrences(string, "\"class\":\"chisel3.tywaves.TywavesAnnotation\"") should be(5)
      countSubstringOccurrences(string, "\"typeName\":\"ExampleIntrinsicModule") should be(4)
    }

    it("should annotate classes modules") {
      import chisel3.properties._
      import chisel3.experimental.hierarchy.{instantiable, public, Definition, Instance}

      // An abstract description of a CSR, represented as a Class.
      @instantiable
      class CSRDescription extends Class
      class CSRModule(csrDescDef: Definition[CSRDescription]) extends RawModule {
        val csrDescription = Instance(csrDescDef)
      }
      class TopCircuitClasses extends RawModule {
        val csrDescDef = Definition(new CSRDescription)
        val csrModule1: CSRModule = Module(new CSRModule(csrDescDef))
        val csrModule2: CSRModule = Module(new CSRModule(csrDescDef))
        val csrModules: Seq[CSRModule] = Seq.fill(2)(Module(new CSRModule(csrDescDef)))
      }
      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitClasses)))
      val string = os.read(targetDir / "TopCircuitClasses.fir")
      countSubstringOccurrences(string, "\"typeName\":\"CSRModule\"") should be(4)
      // TODO: this fails, I don't know why it has twice the same class in the components
      countSubstringOccurrences(string, "\"typeName\":\"CSRDescription") should be(1)
      countSubstringOccurrences(string, "\"class\":\"chisel3.tywaves.TywavesAnnotation\"") should be(6)
    }

    it("should annotate parametric module") {
      class MyModule[T <: Data](gen: T) extends RawModule {
        val a: T = IO(Input(gen))
      }

      class TopCircuitParametric extends RawModule {
        val myModule1: MyModule[UInt] = Module(new MyModule(UInt(8.W)))
        val myModule2: MyModule[SInt] = Module(new MyModule(SInt(8.W)))
        val myModule3: MyModule[Bool] = Module(new MyModule(Bool()))
        val (x, y, z) = (IO(Input(UInt(8.W))), IO(Input(SInt(8.W))), IO(Input(Bool())))
        x <> myModule1.a
        y <> myModule2.a
        z <> myModule3.a
      }

      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitParametric)))
      val string = os.read(targetDir / "TopCircuitParametric.fir")

      val expectedMatches = Seq(
        (createExpected("~TopCircuitParametric\\|MyModule>a", "UInt<8>", "IO"), 1),
        (createExpected("~TopCircuitParametric\\|MyModule_1>a", "SInt<8>", "IO"), 1),
        (createExpected("~TopCircuitParametric\\|MyModule_2>a", "Bool", "IO"), 1),
        (createExpected("~TopCircuitParametric\\|TopCircuitParametric>x", "UInt<8>", "IO"), 1),
        (createExpected("~TopCircuitParametric\\|TopCircuitParametric>y", "SInt<8>", "IO"), 1),
        (createExpected("~TopCircuitParametric\\|TopCircuitParametric>z", "Bool", "IO"), 1),
        // TODO: the goal is to have MyModule[UInt<8>], MyModule[SInt<8>], MyModule[Bool]
        ("\"typeName\":\"MyModule\"", 3)
      )
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
      class TopCircuitClockReset extends RawModule {
        val clock: Clock = IO(Input(Clock()))
        // Types od reset https://www.chisel-lang.org/docs/explanations/reset
        val syncReset:  Bool = IO(Input(Bool()))
        val reset:      Reset = IO(Input(Reset()))
        val asyncReset: AsyncReset = IO(Input(AsyncReset()))
      }
      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitClockReset)))
      val string = os.read(targetDir / "TopCircuitClockReset.fir")
      val expectedMatches = Seq(
        (createExpected("~TopCircuitClockReset\\|TopCircuitClockReset>clock", "Clock", "IO"), 1),
        (createExpected("~TopCircuitClockReset\\|TopCircuitClockReset>syncReset", "Bool", "IO"), 1),
        // TODO: check if this should be Reset or ResetType (the actual type is ResetType)
        (createExpected("~TopCircuitClockReset\\|TopCircuitClockReset>reset", "Reset", "IO"), 1),
        (createExpected("~TopCircuitClockReset\\|TopCircuitClockReset>asyncReset", "AsyncReset", "IO"), 1)
      )
      checkAnno(expectedMatches, string)
    }

    it("should annotate implicit clock and reset") {
      class TopCircuitImplicitClockReset extends Module
      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitImplicitClockReset)))
      val string = os.read(targetDir / "TopCircuitImplicitClockReset.fir")
      val expectedMatches = Seq(
        (createExpected("~TopCircuitImplicitClockReset\\|TopCircuitImplicitClockReset>clock", "Clock", "IO"), 1),
        (createExpected("~TopCircuitImplicitClockReset\\|TopCircuitImplicitClockReset>reset", "Bool", "IO"), 1)
      )
      checkAnno(expectedMatches, string)

    }

    it("should annotate ground type ports") {
      // TODO: add the other ground types
      class TopCircuitGroundPorts extends RawModule {
        val uint:   UInt = IO(Input(UInt(8.W)))
        val sint:   SInt = IO(Input(SInt(8.W)))
        val bool:   Bool = IO(Input(Bool()))
        val analog: Analog = IO(Analog(1.W))
//        val fixedPoint: FixedPoint TODO: does fixed point still exist?
//        val interval: Interval = IO(Input(Interval())) TODO: does interval still exist?
        val bits: UInt = IO(Input(Bits(8.W)))
      }

      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitGroundPorts)))
      val string = os.read(targetDir / "TopCircuitGroundPorts.fir")
      val expectedMatches = Seq(
        (createExpected("~TopCircuitGroundPorts\\|TopCircuitGroundPorts>uint", "UInt<8>", "IO"), 1),
        (createExpected("~TopCircuitGroundPorts\\|TopCircuitGroundPorts>sint", "SInt<8>", "IO"), 1),
        (createExpected("~TopCircuitGroundPorts\\|TopCircuitGroundPorts>bool", "Bool", "IO"), 1),
        (createExpected("~TopCircuitGroundPorts\\|TopCircuitGroundPorts>analog", "Analog<1>", "IO"), 1),
        // TODO: there's no distinction between Bits and UInt since internally Bits.apply() returns UInt
        (createExpected("~TopCircuitGroundPorts\\|TopCircuitGroundPorts>bits", "UInt<8>", "IO"), 1)
      )
      checkAnno(expectedMatches, string)

    }

    it("should annotate bundles") {
      class TopCircuitBundles extends RawModule {
        class MyEmptyBundle extends Bundle

        class MyBundle extends Bundle {
          val a: UInt = Input(UInt(8.W))
          val b: SInt = Input(SInt(8.W))
          val c: Bool = Input(Bool())
        }

        val a: Bundle = IO(Input(new Bundle {}))
        val b: MyEmptyBundle = IO(Input(new MyEmptyBundle))
        val c: MyBundle = IO(Input(new MyBundle))
      }

      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitBundles)))
      val string = os.read(targetDir / "TopCircuitBundles.fir")

      val expectedMatches = Seq(
        (createExpected("~TopCircuitBundles\\|TopCircuitBundles>a", "AnonymousBundle", "IO"), 1),
        (createExpected("~TopCircuitBundles\\|TopCircuitBundles>b", "MyEmptyBundle", "IO"), 1),
        (createExpected("~TopCircuitBundles\\|TopCircuitBundles>c", "MyBundle", "IO"), 1),
        (createExpected("~TopCircuitBundles\\|TopCircuitBundles>c.a", "UInt<8>", "IO"), 1),
        (createExpected("~TopCircuitBundles\\|TopCircuitBundles>c.b", "SInt<8>", "IO"), 1),
        (createExpected("~TopCircuitBundles\\|TopCircuitBundles>c.c", "Bool", "IO"), 1)
      )
      checkAnno(expectedMatches, string)
    }

    it("should annotate nested bundles") {
      class TopCircuitBundlesNested extends RawModule {
        class MyBundle extends Bundle {
          val a: UInt = Input(UInt(8.W))
          val b: SInt = Input(SInt(8.W))
          val c: Bool = Input(Bool())
        }

        class MyNestedBundle extends Bundle {
          val a: Bool = Input(Bool())
          val b: MyBundle = new MyBundle
          val c: MyBundle = Flipped(new MyBundle)
        }

        val a: MyNestedBundle = IO(Input(new MyNestedBundle))
      }
      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitBundlesNested)))
      val string = os.read(targetDir / "TopCircuitBundlesNested.fir")
      val expectedMatches = Seq(
        (createExpected("~TopCircuitBundlesNested\\|TopCircuitBundlesNested>a", "MyNestedBundle", "IO"), 1),
        (createExpected("~TopCircuitBundlesNested\\|TopCircuitBundlesNested>a.a", "Bool", "IO"), 1),
        (createExpected("~TopCircuitBundlesNested\\|TopCircuitBundlesNested>a.b", "MyBundle", "IO"), 1),
        (createExpected("~TopCircuitBundlesNested\\|TopCircuitBundlesNested>a.b.a", "UInt<8>", "IO"), 1),
        (createExpected("~TopCircuitBundlesNested\\|TopCircuitBundlesNested>a.b.b", "SInt<8>", "IO"), 1),
        (createExpected("~TopCircuitBundlesNested\\|TopCircuitBundlesNested>a.b.c", "Bool", "IO"), 1),
        (createExpected("~TopCircuitBundlesNested\\|TopCircuitBundlesNested>a.c", "MyBundle", "IO"), 1),
        (createExpected("~TopCircuitBundlesNested\\|TopCircuitBundlesNested>a.c.a", "UInt<8>", "IO"), 1),
        (createExpected("~TopCircuitBundlesNested\\|TopCircuitBundlesNested>a.c.b", "SInt<8>", "IO"), 1),
        (createExpected("~TopCircuitBundlesNested\\|TopCircuitBundlesNested>a.c.c", "Bool", "IO"), 1)
      )
      checkAnno(expectedMatches, string)
    }

    it("should annotate vecs") {
      class TopCircuitVecs extends RawModule {
        val a: Vec[SInt] = IO(Input(Vec(5, SInt(23.W))))
        val b: Vec[Vec[SInt]] = IO(Input(Vec(5, Vec(3, SInt(23.W)))))
        val c = IO(Input(Vec(5, new Bundle { val x: UInt = UInt(8.W) })))
        val d = {
          IO(
            Input(MixedVec(UInt(3.W), SInt(10.W)))
          ) // TODO: check if this should have a better representation, now its type is represented as other Records
        }
        print(d.toString())
      }

      (new ChiselStage(true)).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitVecs)))
      val string = os.read(targetDir / "TopCircuitVecs.fir")

      val expectedMatches = Seq(
        (createExpected("~TopCircuitVecs\\|TopCircuitVecs>a", "SInt<23>\\[5\\]", "IO"), 1),
        (createExpected("~TopCircuitVecs\\|TopCircuitVecs>a\\[0\\]", "SInt<23>", "IO"), 1),
        (createExpected("~TopCircuitVecs\\|TopCircuitVecs>b", "SInt<23>\\[3\\]\\[5\\]", "IO"), 1),
        (createExpected("~TopCircuitVecs\\|TopCircuitVecs>b\\[0\\]", "SInt<23>\\[3\\]", "IO"), 1),
        (createExpected("~TopCircuitVecs\\|TopCircuitVecs>b\\[0\\]\\[0\\]", "SInt<23>", "IO"), 1),
        (createExpected("~TopCircuitVecs\\|TopCircuitVecs>c", "AnonymousBundle\\[5\\]", "IO"), 1),
        (createExpected("~TopCircuitVecs\\|TopCircuitVecs>c\\[0\\]", "AnonymousBundle", "IO"), 1),
        (createExpected("~TopCircuitVecs\\|TopCircuitVecs>c\\[0\\].x", "UInt<8>", "IO"), 1),
        // TODO: finish this test
        (createExpected("~TopCircuitVecs\\|TopCircuitVecs>d", "MixedVec", "IO"), 1),
        (createExpected("~TopCircuitVecs\\|TopCircuitVecs>d.0", "UInt<3>", "IO"), 1),
        (createExpected("~TopCircuitVecs\\|TopCircuitVecs>d.1", "SInt<10>", "IO"), 1)
      )
      checkAnno(expectedMatches, string)
    }

    it("should annotate bundle with vec") {
      class TopCircuitBundleWithVec extends RawModule {
        val a = IO(Input(new Bundle {
          val vec = Vec(5, UInt(8.W))
        }))
      }
      new ChiselStage(true).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitBundleWithVec)))
      val string = os.read(targetDir / "TopCircuitBundleWithVec.fir")
      val expectedMatches = Seq(
        (createExpected("~TopCircuitBundleWithVec\\|TopCircuitBundleWithVec>a", "AnonymousBundle", "IO"), 1),
        (createExpected("~TopCircuitBundleWithVec\\|TopCircuitBundleWithVec>a.vec", "UInt<8>\\[5\\]", "IO"), 1),
        (createExpected("~TopCircuitBundleWithVec\\|TopCircuitBundleWithVec>a.vec\\[0\\]", "UInt<8>", "IO"), 1)
      )
      checkAnno(expectedMatches, string)
    }
  }

}
