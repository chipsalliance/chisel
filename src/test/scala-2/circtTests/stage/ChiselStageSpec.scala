// SPDX-License-Identifier: Apache-2.0

package circtTests.stage

import chisel3.stage.{ChiselGeneratorAnnotation, CircuitSerializationAnnotation}
import chisel3.experimental.{ExtModule, SourceLine}
import chisel3.util.HasExtModuleInline

import circt.stage.{ChiselStage, FirtoolOption, PreserveAggregate}

import _root_.logger.LogLevel
import firrtl.EmittedVerilogCircuitAnnotation
import firrtl.stage.FirrtlCircuitAnnotation

import java.io.File

import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

object ChiselStageSpec {

  import chisel3._
  import chisel3.probe.{define, Probe, ProbeValue}
  import chisel3.layer.{block, Layer, LayerConfig}

  class FooBundle extends Bundle {
    val a = Input(Bool())
    val b = Output(Bool())
  }

  class Foo(hasDontTouch: Boolean = false) extends RawModule {
    val a = IO(new FooBundle)
    val b = IO(Flipped(new FooBundle))
    if (hasDontTouch) {
      dontTouch(a)
    }
    b <> a
  }

  class Bar extends RawModule {
    val sel = IO(Input(UInt(3.W)))
    val in = IO(Input(Vec(8, UInt(8.W))))
    val out = IO(Output(UInt(8.W)))
    out := in(sel)
  }

  class BazBundle extends Bundle {
    val a = Input(UInt(3.W))
    val b = Input(UInt(4.W))
  }

  class Baz extends RawModule {
    val in = IO(Input(new BazBundle))
    val out = IO(Output(new BazBundle))
    out := in
  }

  class Qux extends RawModule {
    val a = IO(Input(Bool()))
    val b = IO(Output(Bool()))
    b := a
  }

  class Quz extends RawModule {
    val a = IO(Input(Bool()))
    val b = IO(Output(Bool()))
    val qux = Module(new Qux)
    qux.a := a
    b := qux.b
  }

  class Corge extends ExtModule with HasExtModuleInline {
    val a = IO(Input(Bool()))
    val b = IO(Output(Bool()))
    setInline(
      "Corge.v",
      """|module Corge(
         |  input a,
         |  output b
         |);
         |  assign b = a;
         |endmodule
         |""".stripMargin
    )
  }

  class Gralp extends RawModule {
    val a = IO(Input(Bool()))
    val b = IO(Output(Bool()))
    val corge = Module(new Corge)
    corge.a := a
    val quz = Module(new Quz)
    quz.a := corge.b
    b := quz.b
  }

  import firrtl.annotations.NoTargetAnnotation
  import firrtl.options.Unserializable
  case object DummyAnnotation extends NoTargetAnnotation with Unserializable

  class HasUnserializableAnnotation extends RawModule {
    val a = IO(Input(Bool()))
    val b = IO(Output(Bool()))
    b := a
    chisel3.experimental.annotate()(Seq(DummyAnnotation))
  }

  class UserAssertionModule extends RawModule {
    assert(false, "User had an assertion")
  }

  class UserExceptionModule extends RawModule {
    throw new Exception("User threw an exception")
  }

  class UserExceptionNoStackTrace extends RawModule {
    throw new Exception("Something bad happened") with scala.util.control.NoStackTrace
  }

  class RecoverableError extends RawModule {
    3.U >> -1
  }

  class RecoverableErrorFakeSourceInfo extends RawModule {
    implicit val info = SourceLine("Foo", 3, 10)
    3.U >> -1
  }

  class ErrorCaughtByFirtool extends RawModule {
    implicit val info = SourceLine("Foo", 3, 10)
    val w = Wire(UInt(8.W))
  }

  object A extends Layer(LayerConfig.Extract())

  object B extends Layer(LayerConfig.Extract())

  class LayerRemappingTest extends RawModule {
    val out = IO(Output(Probe(Bool(), A)))
    block(A) {
      val a = Wire(Bool())
      define(out, ProbeValue(a))
    }
  }
}

/** A fixture used that exercises features of the Trace API.
  */
class TraceSpec {

  import chisel3._
  import chisel3.experimental.Trace
  import chisel3.util.experimental.InlineInstance

  /** A mutable Chisel reference to an internal wire inside Bar. This is done to enable later use of the Trace API to find this wire. */
  var id: Option[Bool] = None

  /** A submodule that will be inlined into the parent, Foo. */
  class Bar extends RawModule with InlineInstance {

    /** The wire that we want to trace. */
    val a = WireDefault(false.B)
    id = Some(a)
    dontTouch(a)
    Trace.traceName(a)
  }

  /** The top module. */
  class Foo extends RawModule {
    val bar = Module(new Bar)
  }

}

class ChiselStageSpec extends AnyFunSpec with Matchers with chiselTests.LogUtils {

  private val baseDir = os.pwd / "test_run_dir" / this.getClass.getSimpleName

  describe("ChiselStage") {

    it("should elaborate a Chisel module and emit specification FIRRTL (CHIRRTL)") {

      val targetDir = new File("test_run_dir/ChiselStageSpec")

      val args: Array[String] = Array(
        "--target",
        "chirrtl",
        "--target-dir",
        targetDir.toString
      )

      val expectedOutput = new File(targetDir, "Foo.fir")
      expectedOutput.delete()

      (new ChiselStage).execute(args, Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.Foo)))

      info(s"'$expectedOutput' exists")
      expectedOutput should (exist)

    }

    it("should compile a Chisel module to FIRRTL dialect") {

      val targetDir = new File("test_run_dir/ChiselStageSpec")

      val args: Array[String] = Array(
        "--target",
        "firrtl",
        "--target-dir",
        targetDir.toString
      )

      val expectedOutput = new File(targetDir, "Foo.fir.mlir")
      expectedOutput.delete()

      (new ChiselStage).execute(args, Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.Foo)))

      info(s"'$expectedOutput' exists")
      expectedOutput should (exist)

    }

    it("should compile a Chisel module to RTL dialect") {

      val targetDir = new File("test_run_dir/ChiselStageSpec")

      val args: Array[String] = Array(
        "--target",
        "hw",
        "--target-dir",
        targetDir.toString
      )

      val expectedOutput = new File(targetDir, "Foo.hw.mlir")
      expectedOutput.delete()

      (new ChiselStage).execute(args, Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.Foo)))

      info(s"'$expectedOutput' exists")
      expectedOutput should (exist)

    }

    it("should compile a Chisel module to SystemVerilog") {

      val targetDir = new File("test_run_dir/ChiselStageSpec")

      val args: Array[String] = Array(
        "--target",
        "systemverilog",
        "--target-dir",
        targetDir.toString
      )

      val expectedOutput = new File(targetDir, "Foo.sv")
      expectedOutput.delete()

      (new ChiselStage)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.Foo)))

      info(s"'$expectedOutput' exists")
      expectedOutput should (exist)

    }

    it("should optionally emit .fir when compiling to SystemVerilog") {

      val targetDir = new File("test_run_dir/ChiselStageSpec")

      val args: Array[String] = Array(
        "--target",
        "systemverilog",
        "--target-dir",
        targetDir.toString,
        "--dump-fir"
      )

      val expectedSV = new File(targetDir, "Foo.sv")
      expectedSV.delete()

      val expectedFir = new File(targetDir, "Foo.fir")
      expectedFir.delete()

      (new ChiselStage)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.Foo)))

      info(s"'$expectedSV' exists")
      expectedSV should (exist)
      info(s"'$expectedFir' exists")
      expectedFir should (exist)

    }

    it("should support custom firtool options") {
      val targetDir = new File("test_run_dir/ChiselStageSpec")

      val args: Array[String] = Array(
        "--target",
        "systemverilog",
        "--target-dir",
        targetDir.toString
      )

      info(s"output contains a case statement using --lowering-options=disallowPackedArrays")
      (new ChiselStage)
        .execute(
          args,
          Seq(
            ChiselGeneratorAnnotation(() => new ChiselStageSpec.Bar),
            FirtoolOption("--lowering-options=disallowPackedArrays")
          )
        )
        .collectFirst { case EmittedVerilogCircuitAnnotation(a) =>
          a
        }
        .get
        .value should include("case")
    }

    it("should support aggregate preservation mode") {
      val targetDir = new File("test_run_dir/ChiselStageSpec")

      val args: Array[String] = Array(
        "--target",
        "systemverilog",
        "--target-dir",
        targetDir.toString
      )

      info(s"output contains a verilog struct using preserve-aggregate option")
      (new ChiselStage)
        .execute(
          args,
          Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.Baz), PreserveAggregate(PreserveAggregate.All))
        )
        .collectFirst { case EmittedVerilogCircuitAnnotation(a) =>
          a
        }
        .get
        .value should include("struct")
    }

    it("should support split Verilog output") {
      val targetDir = new File("test_run_dir/ChiselStageSpec")

      val args: Array[String] = Array(
        "--target",
        "systemverilog",
        "--target-dir",
        targetDir.toString,
        "--split-verilog"
      )

      val expectedOutputs = Seq(new File(targetDir, "Qux.sv"), new File(targetDir, "Qux.sv"))
      expectedOutputs.foreach(_.delete)

      info("output contains multiple Verilog files")
      (new ChiselStage)
        .execute(
          args,
          Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.Quz), PreserveAggregate(PreserveAggregate.All))
        )

      expectedOutputs.foreach { file =>
        info(s"'$file' exists")
        file should exist
      }
    }

    it("should emit Annotations inline in emitted CHIRRTL") {
      val targetDir = os.pwd / "ChiselStageSpec" / "should-inline-Annotations-in-emitted-CHIRRTL"

      val args: Array[String] = Array(
        "--target",
        "chirrtl",
        "--target-dir",
        targetDir.toString
      )

      (new ChiselStage)
        .execute(
          args,
          Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.Foo(hasDontTouch = true)))
        )

      info("output file included an Annotation")
      os.read(targetDir / "Foo.fir") should include("firrtl.transforms.DontTouchAnnotation")
    }

    it("should NOT emit Unserializable Annotations inline in emitted CHIRRTL") {
      val targetDir = baseDir / "should-not-inline-Unserializable-Annotations-in-emitted-CHIRRTL"

      val args: Array[String] = Array(
        "--target",
        "chirrtl",
        "--target-dir",
        targetDir.toString
      )

      (new ChiselStage)
        .execute(
          args,
          Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.HasUnserializableAnnotation))
        )

      os.read(targetDir / "HasUnserializableAnnotation.fir") shouldNot include("DummyAnnotation")
    }

    it("should forward firtool-resolver logging under log-level debug") {
      val targetDir = baseDir / "should-forward-firtool-resolver-logging-under-log-level-debug"

      val args: Array[String] = Array(
        "--target",
        "systemverilog",
        "--target-dir",
        targetDir.toString
      )
      val annos = Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.Foo))
      val msg = "Checking CHISEL_FIRTOOL_PATH for firtool"
      // By default it should NOT show anything
      val (log1, _) = grabLog {
        (new ChiselStage).execute(args, annos)
      }
      log1 shouldNot include(msg)

      val (log2, _) = grabLog {
        (new ChiselStage).execute(args ++ Array("--log-level", "debug"), annos)
      }
      log2 should include(msg)
    }

    it("should support --log-level") {
      def args(level: String): Array[String] = Array("--target", "chirrtl", "--log-level", level)
      val annos = Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.Foo))
      val (log1, _) = grabLog {
        (new ChiselStage).execute(args("info"), annos)
      }
      log1 should include("Done elaborating.")

      val (log2, _) = grabLog {
        (new ChiselStage).execute(args("warn"), annos)
      }
      log2 shouldNot include("Done elaborating.")
    }

    it("should support firtool options") {
      val targetDir = new File("test_run_dir/ChiselStageSpec")

      val args: Array[String] = Array(
        "--target",
        "systemverilog",
        "--target-dir",
        targetDir.toString,
        "--firtool-option=-strip-debug-info"
      )

      ((new ChiselStage)
        .execute(
          args,
          Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.Bar))
        )
        .collectFirst { case EmittedVerilogCircuitAnnotation(a) =>
          a
        }
        .get
        .value should not).include("ChiselStageSpec.scala")
    }
  }

  describe("ChiselStage exception handling") {

    it("should truncate a user exception") {
      info("The user's java.lang.AssertionError was thrown")
      val exception = intercept[java.lang.AssertionError] {
        (new ChiselStage)
          .execute(
            Array("--target", "chirrtl"),
            Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.UserAssertionModule))
          )
      }

      info(s""" -  Exception was a ${exception.getClass.getName}""")

      val message = exception.getMessage
      info("The exception includes the user's message")
      message should include("User had an assertion")

      val stackTrace = exception.getStackTrace
      info("The stack trace is trimmed at the top")
      stackTrace.head.toString should include("...")

      info("The stack trace is trimmed at the bottom")
      stackTrace.last.toString should include("...")

      info("The stack trace include information about running --full-stacktrace")
      stackTrace.last.toString should include("--full-stacktrace")
    }

    it("should NOT truncate the top of the stack trace if it points to user code") {
      info("The user's Exception was thrown")
      val exception = intercept[Exception] {
        (new ChiselStage)
          .execute(
            Array("--target", "chirrtl"),
            Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.UserExceptionModule))
          )
      }

      val message = exception.getMessage
      info("The exception includes the user's message")
      message should include("User threw an exception")

      val stackTrace = exception.getStackTrace
      info("The stack trace is NOT trimmed at the top")
      stackTrace.head.toString shouldNot include("...")

      info("The stack trace is trimmed at the bottom")
      stackTrace.last.toString should include("...")
    }

    it("""should not truncate a user exception with "--full-stacktrace"""") {
      info("The user's java.lang.AssertionError was thrown")
      val exception = intercept[java.lang.AssertionError] {
        (new ChiselStage).execute(
          Array("--target", "chirrtl", "--full-stacktrace"),
          Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.UserAssertionModule))
        )
      }

      info(s""" -  Exception was a ${exception.getClass.getName}""")

      val message = exception.getMessage
      info("The exception includes the user's message")
      message should include("User had an assertion")

      info("The stack trace is not trimmed")
      exception.getStackTrace.mkString("\n") should include("java")
    }

    it("should NOT add a stack trace to an exception with no stack trace") {
      val exception = intercept[java.lang.Exception] {
        (new ChiselStage)
          .execute(
            Array("--target", "chirrtl"),
            Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.UserExceptionNoStackTrace))
          )
      }

      val message = exception.getMessage
      info("The exception includes the user's message")
      message should include("Something bad happened")

      info("The exception should not contain a stack trace")
      exception.getStackTrace should be(Array())
    }

    it("should NOT include a stack trace for recoverable errors") {
      val exception = intercept[java.lang.Exception] {
        (new ChiselStage)
          .execute(
            Array("--target", "chirrtl"),
            Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.RecoverableError))
          )
      }

      val message = exception.getMessage
      info("The exception includes the standard error message")
      message should include("Fatal errors during hardware elaboration. Look above for error list.")

      info("The exception should not contain a stack trace")
      exception.getStackTrace should be(Array())
    }

    it("should include a stack trace for recoverable errors with '--throw-on-first-error'") {
      val exception = intercept[java.lang.Exception] {
        (new ChiselStage).execute(
          Array("--target", "chirrtl", "--throw-on-first-error"),
          Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.RecoverableError))
        )
      }

      val stackTrace = exception.getStackTrace.mkString("\n")
      info("The exception should contain a truncated stack trace")
      stackTrace shouldNot include("java")

      info("The stack trace include information about running --full-stacktrace")
      stackTrace should include("--full-stacktrace")
    }

    it(
      "include an untruncated stack trace for recoverable errors when given both '--throw-on-first-error' and '--full-stacktrace'"
    ) {
      val exception = intercept[java.lang.Exception] {
        val args = Array("--throw-on-first-error", "--full-stacktrace")
        (new ChiselStage).execute(
          Array("--target", "chirrtl", "--throw-on-first-error", "--full-stacktrace"),
          Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.RecoverableError))
        )
      }

      val stackTrace = exception.getStackTrace.mkString("\n")
      info("The exception should contain a truncated stack trace")
      stackTrace should include("java")
    }

    it("should include source line and a caret for recoverable errors") {
      val (stdout, stderr, _) = grabStdOutErr {
        intercept[java.lang.Exception] {
          (new ChiselStage)
            .execute(
              Array("--target", "chirrtl"),
              Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.RecoverableError))
            )
        }
      }

      val lines = stdout.split("\n")
      // Fuzzy includes aren't ideal but there is ANSI color in these strings that is hard to match
      lines(0) should include(
        "src/test/scala-2/circtTests/stage/ChiselStageSpec.scala 121:9: Negative shift amounts are illegal (got -1)"
      )
      lines(1) should include("    3.U >> -1")
      lines(2) should include("        ^")
    }

    it("should NOT include source line and caret with an incorrect --source-root") {
      val (stdout, stderr, _) = grabStdOutErr {
        intercept[java.lang.Exception] {
          (new ChiselStage)
            .execute(
              Array("--target", "chirrtl", "--source-root", ".github"),
              Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.RecoverableError))
            )
        }
      }

      val lines = stdout.split("\n")
      // Fuzzy includes aren't ideal but there is ANSI color in these strings that is hard to match
      lines.size should equal(2)
      lines(0) should include(
        "src/test/scala-2/circtTests/stage/ChiselStageSpec.scala 121:9: Negative shift amounts are illegal (got -1)"
      )
      (lines(1) should not).include("3.U >> -1")
    }

    it("should include source line and a caret for recoverable errors with multiple --source-roots") {
      val (stdout, stderr, _) = grabStdOutErr {
        intercept[java.lang.Exception] {
          (new ChiselStage)
            .execute(
              Array(
                "--target",
                "chirrtl",
                "--source-root",
                ".",
                "--source-root",
                "src/test/resources/chisel3/sourceroot1"
              ),
              Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.RecoverableErrorFakeSourceInfo))
            )
        }
      }

      val lines = stdout.split("\n")
      // Fuzzy includes aren't ideal but there is ANSI color in these strings that is hard to match
      lines(0) should include("Foo 3:10: Negative shift amounts are illegal (got -1)")
      lines(1) should include("I am the file in sourceroot1")
      lines(2) should include("         ^")
    }

    it("should include source line and a caret picking the first --source-root if there is ambiguity") {
      val (stdout, stderr, _) = grabStdOutErr {
        intercept[java.lang.Exception] {
          (new ChiselStage)
            .execute(
              Array(
                "--target",
                "chirrtl",
                "--source-root",
                "src/test/resources/chisel3/sourceroot2",
                "--source-root",
                "src/test/resources/chisel3/sourceroot1"
              ),
              Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.RecoverableErrorFakeSourceInfo))
            )
        }
      }

      val lines = stdout.split("\n")
      // Fuzzy includes aren't ideal but there is ANSI color in these strings that is hard to match
      lines(0) should include("Foo 3:10: Negative shift amounts are illegal (got -1)")
      lines(1) should include("I am the file in sourceroot2")
      lines(2) should include("         ^")
    }

    it("should propagate --source-root as --include-dir to firtool") {
      val e = intercept[java.lang.Exception] {
        (new ChiselStage)
          .execute(
            Array("--target", "systemverilog", "--source-root", "src/test/resources/chisel3/sourceroot1"),
            Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.ErrorCaughtByFirtool))
          )
      }

      val lines = e.getMessage.split("\n")
      val idx = lines.indexWhere(_.contains("not fully initialized"))
      lines(idx) should include(
        "src/test/resources/chisel3/sourceroot1/Foo:3:10: error: sink \"w\" not fully initialized"
      )
      lines(idx + 1) should equal("I am the file in sourceroot1")
      lines(idx + 2) should equal("         ^")
    }

    it("should report the firtool version against which Chisel was published in error messages") {
      val e = intercept[java.lang.Exception] {
        ChiselStage.emitSystemVerilog(new ChiselStageSpec.ErrorCaughtByFirtool)
      }
      val version = chisel3.BuildInfo.firtoolVersion.getOrElse("<unknown>")
      e.getMessage should include(s"firtool version $version")
    }

    it("should properly report Builder.errors even if there is a later Exception") {
      val (log, _) = grabLog {
        intercept[java.lang.Exception] {
          import chisel3._
          ChiselStage.emitCHIRRTL(new Module {
            val in = IO(Input(UInt(8.W)))
            val y = in >> -1
            require(false) // This should not suppress reporting the negative shift
          })
        }
      }
      log should include("Negative shift amounts are illegal (got -1)")
    }
  }

  describe("ChiselStage custom transform support") {

    it("should work with InlineInstance") {

      import chisel3._
      import chisel3.util.experimental.InlineInstance

      trait SimpleIO { this: RawModule =>
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))
      }

      class Bar extends RawModule with SimpleIO with InlineInstance {
        b := ~a
      }

      class Foo extends RawModule with SimpleIO {
        val bar = Module(new Bar)
        bar.a := a
        b := bar.b
      }

      val targetDir = new File("test_run_dir/InlineInstance")

      val args: Array[String] = Array(
        "--target",
        "systemverilog",
        "--target-dir",
        targetDir.toString
      )

      ((new ChiselStage)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new Foo)))
        .collectFirst { case EmittedVerilogCircuitAnnotation(a) =>
          a
        }
        .get
        .value should not).include("module Bar")
    }

    it("should work with FlattenInstance") {

      import chisel3._
      import chisel3.util.experimental.FlattenInstance

      trait SimpleIO { this: RawModule =>
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))
      }

      class Baz extends RawModule with SimpleIO {
        b := ~a
      }

      class Bar extends RawModule with SimpleIO {
        val baz = Module(new Baz)
        baz.a := a
        b := baz.b
      }

      class Foo extends RawModule with SimpleIO {
        val bar = Module(new Bar with FlattenInstance)
        bar.a := a
        b := bar.b
      }

      val targetDir = new File("test_run_dir/FlattenInstance")

      val args: Array[String] = Array(
        "--target",
        "systemverilog",
        "--target-dir",
        targetDir.toString
      )

      val verilog = (new ChiselStage)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new Foo)))
        .collectFirst { case EmittedVerilogCircuitAnnotation(a) =>
          a
        }
        .get
        .value

      (verilog should not).include("module Baz")
      (verilog should not).include("module Bar")

    }

    it("should work with the Trace API for unified output") {

      import chisel3.experimental.Trace

      val fixture = new TraceSpec

      val targetDir = new File("test_run_dir/TraceAPIUnified")

      val args: Array[String] = Array(
        "--target",
        "systemverilog",
        "--target-dir",
        targetDir.toString
      )

      val annos = (new ChiselStage).execute(
        args,
        Seq(
          ChiselGeneratorAnnotation(() => new fixture.Foo)
        )
      )

      val finalTargets = Trace.finalTarget(annos)(fixture.id.get)
      info("there is one final target")
      finalTargets should have size (1)

      val expectedTarget = firrtl.annotations.ModuleTarget("Foo").ref("bar_a")
      info(s"the final target is $expectedTarget")
      finalTargets.head should be(expectedTarget)

    }

    it("should work with the Trace API for split  verilog output") {

      import chisel3.experimental.Trace

      val fixture = new TraceSpec

      val targetDir = new File("test_run_dir/TraceAPISplit")

      val args: Array[String] = Array(
        "--target",
        "systemverilog",
        "--target-dir",
        targetDir.toString,
        "--split-verilog"
      )

      val annos = (new ChiselStage).execute(
        args,
        Seq(
          ChiselGeneratorAnnotation(() => new fixture.Foo)
        )
      )

      val finalTargets = Trace.finalTarget(annos)(fixture.id.get)
      info("there is one final target")
      finalTargets should have size (1)

      val expectedTarget = firrtl.annotations.ModuleTarget("Foo").ref("bar_a")
      info(s"the final target is $expectedTarget")
      finalTargets.head should be(expectedTarget)

    }

  }

  describe("ChiselStage DontTouchAnnotation support") {

    it("should block removal of wires and nodes") {

      import chisel3._

      class Foo extends RawModule {
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))

        val w = WireDefault(a)
        dontTouch(w)

        val n = ~w
        dontTouch(n)

        b := n
      }

      val targetDir = new File("test_run_dir/DontTouch")

      val args: Array[String] = Array(
        "--target",
        "systemverilog",
        "--target-dir",
        targetDir.toString
      )

      val verilog = (new ChiselStage)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new Foo)))
        .collectFirst { case EmittedVerilogCircuitAnnotation(a) =>
          a
        }
        .get
        .value

      verilog should include("wire w")
      verilog should include("wire n")

    }

  }

  describe("ChiselStage forceName support") {

    it("should work when forcing a module name") {

      import chisel3._
      import chisel3.util.experimental.{forceName, InlineInstance}

      trait SimpleIO { this: RawModule =>
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))
      }

      class Baz extends RawModule with SimpleIO {
        b := ~a
      }

      class Bar extends RawModule with SimpleIO with InlineInstance {
        val baz = Module(new Baz)
        baz.a := a
        b := baz.b
      }

      class Foo extends RawModule with SimpleIO {
        val bar = Module(new Bar)
        bar.a := a
        b := bar.b

        forceName(bar.baz)
      }

      val targetDir = new File("test_run_dir/ForceName")

      val args: Array[String] = Array(
        "--target",
        "systemverilog",
        "--target-dir",
        targetDir.toString
      )

      val verilog: String = (new ChiselStage)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new Foo)))
        .collectFirst { case EmittedVerilogCircuitAnnotation(a) =>
          a
        }
        .get
        .value

      verilog should include("module Baz")

    }

  }

  describe("ChiselStage dedup behavior") {

    it("should be on by default and work") {

      import chisel3._

      class Baz extends RawModule {
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))

        // Chisel groups modules for deduping using their
        // desiredNames. By setting the desiredName here, we can make
        // sure that these RawModules get put in the same
        // deduplication group
        override def desiredName = "Bar"

        b := a
      }

      class Bar extends RawModule {
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))

        override def desiredName = "Bar"

        b := a
      }

      class Foo extends RawModule {
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))

        val bar = Module(new Bar)
        val baz = Module(new Baz)
        bar.a := a
        baz.a := a
        b := bar.b ^ baz.b
      }

      val targetDir = new File("test_run_dir/Dedup")

      val args: Array[String] = Array(
        "--target",
        "systemverilog",
        "--target-dir",
        targetDir.toString
      )

      val verilog: String = (new ChiselStage)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new Foo)))
        .collectFirst { case EmittedVerilogCircuitAnnotation(a) =>
          a
        }
        .get
        .value

      verilog should include("module Bar")
      (verilog should not).include("module Baz")

    }

    it("should respect the doNotDedup API") {

      import chisel3._
      import chisel3.experimental.doNotDedup

      class Baz extends RawModule {
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))

        b := a
      }

      class Bar extends RawModule {
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))

        b := a
      }

      class Foo extends RawModule {
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))

        val bar = Module(new Bar)
        val baz = Module(new Baz)
        bar.a := a
        baz.a := a
        b := bar.b ^ baz.b
        doNotDedup(baz)
      }

      val targetDir = new File("test_run_dir/Dedup")

      val args: Array[String] = Array(
        "--target",
        "systemverilog",
        "--target-dir",
        targetDir.toString
      )

      val verilog: String = (new ChiselStage)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new Foo)))
        .collectFirst { case EmittedVerilogCircuitAnnotation(a) =>
          a
        }
        .get
        .value

      verilog should include("module Bar")
      verilog should include("module Baz")

    }

  }

  describe("ChiselStage$") {

    it("should convert a module to FIRRTL IR") {

      ChiselStage.elaborate(new ChiselStageSpec.Foo).name should be("Foo")

    }

    it("should emit specification FIRRTL (CHIRRTL) with the correct FIRRTL spec version") {

      val text = ChiselStage.emitCHIRRTL(new ChiselStageSpec.Foo(hasDontTouch = true))
      info("found a version string")
      text should include("FIRRTL version 6.0.0")
      info("found an Annotation")
      text should include("firrtl.transforms.DontTouchAnnotation")
      info("found a circuit")
      text should include("circuit Foo")

    }

    it("should emit CHIRRTL files") {
      val targetDir = new File("test_run_dir/ChiselStageSpec/emitCHIRRTLFile")

      val args: Array[String] = Array(
        "--target-dir",
        targetDir.toString
      )

      // Should we be returning the CircuitSerializationAnnotation? It's consistent with emitSystemVerilogFile to do so.
      ChiselStage
        .emitCHIRRTLFile(
          new ChiselStageSpec.Bar,
          args
        )
        .collectFirst { case a: CircuitSerializationAnnotation =>
          a.filename
        }
        .get should be("Bar")

      val expectedOutput = new File(targetDir, "Bar.fir")
      expectedOutput should (exist)
      info(s"'$expectedOutput' exists")
    }

    it("should emit CHIRRTL files with custom names") {
      val targetDir = new File("test_run_dir/ChiselStageSpec/emitCHIRRTLFile")
      val fileBaseName = "Foo"

      val args: Array[String] = Array(
        "--target-dir",
        targetDir.toString,
        "--chisel-output-file",
        fileBaseName
      )

      ChiselStage
        .emitCHIRRTLFile(
          new ChiselStageSpec.Bar,
          args
        )
        .collectFirst { case a: CircuitSerializationAnnotation =>
          a.filename
        }
        .get should be(fileBaseName)

      val expectedOutput = new File(targetDir, s"${fileBaseName}.fir")
      expectedOutput should (exist)
      info(s"'$expectedOutput' exists")
    }

    it("should emit FIRRTL dialect") {

      ChiselStage.emitFIRRTLDialect(new ChiselStageSpec.Foo) should include(" firrtl.module")

    }

    it("should emit HW dialect") {

      ChiselStage.emitHWDialect(new ChiselStageSpec.Foo) should include(" hw.module")

    }

    it("should emit SystemVerilog to string") {

      ChiselStage.emitSystemVerilog(new ChiselStageSpec.Foo) should include("endmodule")

    }
    it("should emit SystemVerilog to string with firtool options") {

      val sv = ChiselStage
        .emitSystemVerilog(
          new ChiselStageSpec.Foo,
          firtoolOpts = Array("--strip-debug-info")
        )
      (sv should not).include("// <stdin>:")

    }
    it("should emit SystemVerilog to string with chisel arguments and firtool options") {

      val sv = ChiselStage.emitSystemVerilog(
        new ChiselStageSpec.Foo,
        Array("--full-stacktrace"),
        Array("--strip-debug-info")
      )
      sv should include("Generated by CIRCT")

    }

    it("emitSystemVerilogFile should write to multiple files by default") {
      val targetDir = new File("test_run_dir/ChiselStageSpec/generated")

      val args: Array[String] = Array(
        "--target-dir",
        targetDir.toString
      )

      info(s"output contains a case statement using --lowering-options=disallowPackedArrays")
      ChiselStage
        .emitSystemVerilogFile(
          new ChiselStageSpec.Gralp,
          args
        )

      Seq("Corge.v", "Gralp.sv", "Qux.sv", "Qux.sv").map(new File(targetDir, _)).foreach { file =>
        info(s"'$file' exists")
        file should (exist)
      }
    }

    it("should emit btor2 to string") {
      import chisel3._
      import chisel3.ltl._

      class Counter extends Module {
        val count = RegInit(0.U(32.W))
        when(count === 42.U) { count := 0.U }.otherwise { count := count + 1.U }

        AssertProperty((count <= 42.U))
      }

      val btor2 = ChiselStage.emitBtor2(
        new Counter,
        firtoolOpts = Array("-enable-layers=Verification,Verification.Assert,Verification.Assume,Verification.Cover")
      )
      btor2 should include("""|1 sort bitvec 1
                              |2 input 1 reset
                              |3 sort bitvec 32
                              |4 state 3 count
                              |5 constd 1 0
                              |6 state 1 hbr
                              |7 init 1 6 5
                              |8 constd 3 43
                              |9 constd 3 1
                              |10 constd 3 42
                              |11 constd 3 0
                              |12 eq 1 4 10
                              |13 add 3 4 9
                              |14 ite 3 12 11 13
                              |15 ult 1 4 8
                              |16 constd 1 -1
                              |17 or 1 2 6
                              |18 xor 1 2 16
                              |19 and 1 6 18
                              |20 implies 1 19 15
                              |21 not 1 20
                              |22 bad 21
                              |23 ite 3 2 11 14
                              |24 next 3 4 23
                              |25 next 1 6 17""".stripMargin)
    }

    it("""should error if give a "--target-directory" option""") {

      val exception = intercept[firrtl.options.OptionsException] {
        ChiselStage.emitCHIRRTL(new ChiselStageSpec.Foo, Array("--target-directory")) should include("circuit Foo")
      }

      val message = exception.getMessage
      info("""The exception includes "Unknown option"""")
      message should include("Unknown option --target-directory")

    }

    it("should forward firtool-resolver logging under log-level debug") {
      // By default it should NOT show anything
      val (log1, _) = grabLog {
        ChiselStage.emitSystemVerilog(new ChiselStageSpec.Foo)
      }
      log1 shouldNot include("Checking CHISEL_FIRTOOL_PATH for firtool")

      // circt.stage.ChiselStage does not currently accept --log-level so we have to use testing
      // APIs to set the level
      val (log2, _) = grabLog {
        ChiselStage.emitSystemVerilog(new ChiselStageSpec.Foo, args = Array("--log-level", "debug"))
      }
      log2 should include("Checking CHISEL_FIRTOOL_PATH for firtool")
    }

    it("should support --log-level") {
      val (log1, _) = grabLog {
        ChiselStage.emitCHIRRTL(new ChiselStageSpec.Foo, Array("--log-level", "info"))
      }
      log1 should include("Done elaborating.")

      val (log2, _) = grabLog {
        ChiselStage.emitCHIRRTL(new ChiselStageSpec.Foo, Array("--log-level", "warn"))
      }
      log2 shouldNot include("Done elaborating.")
    }

    it("should not emit logger annotations") {
      import chisel3.RawModule
      (ChiselStage.emitCHIRRTL(new RawModule {}) should not).include("LogLevelAnnotation")
      (ChiselStage.emitCHIRRTL(new RawModule {}, Array("-ll", "info")) should not).include("LogLevelAnnotation")
    }

    it("should allow remapping one layer to another") {
      val chirrtl = ChiselStage.emitCHIRRTL(
        new ChiselStageSpec.LayerRemappingTest,
        Array("--remap-layer", "circtTests.stage.ChiselStageSpec$A$,circtTests.stage.ChiselStageSpec$B$")
      )
      chirrtl should include("output out : Probe<UInt<1>, B>")
      chirrtl should include("layer B")
      chirrtl should include("layerblock B")
    }

  }

  describe("ChiselStage$ exception handling") {

    it("should truncate a user exception") {
      info("The user's java.lang.AssertionError was thrown")
      val exception = intercept[java.lang.AssertionError] {
        ChiselStage.emitCHIRRTL(new ChiselStageSpec.UserAssertionModule)
      }

      val message = exception.getMessage
      info("The exception includes the user's message")
      message should include("User had an assertion")

      val stackTrace = exception.getStackTrace
      info("The stack trace is trimmed at the top")
      stackTrace.head.toString should include("...")

      info("The stack trace is trimmed at the bottom")
      stackTrace.last.toString should include("...")

      info("The stack trace include information about running --full-stacktrace")
      stackTrace.last.toString should include("--full-stacktrace")
    }

    it("should NOT truncate the top of the stack trace if it points to user code") {
      info("The user's Exception was thrown")
      val exception = intercept[Exception] {
        ChiselStage.emitCHIRRTL(new ChiselStageSpec.UserExceptionModule)
      }

      val message = exception.getMessage
      info("The exception includes the user's message")
      message should include("User threw an exception")

      val stackTrace = exception.getStackTrace
      info("The stack trace is NOT trimmed at the top")
      stackTrace.head.toString shouldNot include("...")

      info("The stack trace is trimmed at the bottom")
      stackTrace.last.toString should include("...")
    }

    it("should NOT add a stack trace to an exception with no stack trace") {
      val exception = intercept[java.lang.Exception] {
        ChiselStage.emitCHIRRTL(new ChiselStageSpec.UserExceptionNoStackTrace)
      }

      val message = exception.getMessage
      info("The exception includes the user's message")
      message should include("Something bad happened")

      info("The exception should not contain a stack trace")
      exception.getStackTrace should be(Array())
    }

    it("should NOT include a stack trace for recoverable errors") {
      val exception = intercept[java.lang.Exception] {
        ChiselStage.emitCHIRRTL(new ChiselStageSpec.RecoverableError)
      }

      val message = exception.getMessage
      info("The exception includes the standard error message")
      message should include("Fatal errors during hardware elaboration. Look above for error list.")

      info("The exception should not contain a stack trace")
      exception.getStackTrace should be(Array())
    }

    it("should report a specific error if firtool is not found on the PATH") {
      val exception = intercept[Exception] {
        ChiselStage.emitSystemVerilog(new ChiselStageSpec.Foo, Array("--firtool-binary-path", "potato"))
      }

      info("The exception includes a useful error message")
      val message = exception.getMessage
      message should include("Cannot run program \"potato\"")
      message should include("Chisel requires firtool, the MLIR-based FIRRTL Compiler (MFC), to generate Verilog.")

      info("The exception should not contain a stack trace")
      exception.getStackTrace should be(Array())
    }

  }
}
