package chiselTests

import chisel3._
import chisel3.experimental.hierarchy._
import chisel3.experimental.inlinetest._
import chisel3.testers._
import chisel3.properties.Property
import chisel3.testing.scalatest.FileCheck
import chisel3.simulator.scalatest.InlineTests
import chisel3.simulator.ChiselSim
import chisel3.util.{is, switch, Decoupled}
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import circt.stage.ChiselStage.emitCHIRRTL

// Here is a testharness that expects some sort of interface on its DUT, e.g. a probe
// socket to which to attach a monitor.
class TestHarnessWithMonitorSocket[M <: RawModule with HasMonitorSocket](test: TestParameters[M])
    extends TestHarness[M](test) {
  val monitor = Module(new ProtocolMonitor(dut.monProbe.cloneType))
  monitor.io :#= probe.read(dut.monProbe)
}

object TestHarnessWithMonitorSocket {
  implicit def testharnessGenerator[M <: RawModule with HasMonitorSocket] =
    TestHarnessGenerator[M](new TestHarnessWithMonitorSocket(_))
}

@instantiable
trait HasMonitorSocket { this: RawModule =>
  protected def makeProbe(bundle: ProtocolBundle): ProtocolBundle = {
    val monProbe = IO(probe.Probe(chiselTypeOf(bundle)))
    probe.define(monProbe, probe.ProbeValue(bundle))
    monProbe
  }
  @public val monProbe: ProtocolBundle
}

class ProtocolBundle(width: Int) extends Bundle {
  val in = Flipped(UInt(width.W))
  val out = UInt(width.W)
}

class ProtocolMonitor(bundleType: ProtocolBundle) extends Module {
  val io = IO(Input(bundleType))
  assert(io.in === io.out, "in === out")
}

@instantiable
trait HasProtocolInterface extends HasTests { this: RawModule =>
  @public val io: ProtocolBundle

  test("check1") { dut =>
    ProtocolChecks.check(1)(dut)
    TestBehavior.RunForCycles(10)
  }
}

object ProtocolChecks {
  def check(v: Int)(instance: Instance[RawModule with HasProtocolInterface]) = {
    instance.io.in := v.U
    assert(instance.io.out === v.U)
  }
}

trait HasTestsProperty { this: RawModule with HasTests =>
  def enableTestsProperty: Boolean

  val testNames = Option.when(enableTestsProperty) {
    IO(Output(Property[Seq[String]]()))
  }

  atModuleBodyEnd {
    testNames.foreach { testNames =>
      testNames := Property(this.getTests.map(_.testName))
    }
  }
}

@instantiable
class ModuleWithTests(
  ioWidth:                          Int = 32,
  override val resetType:           Module.ResetType.Type = Module.ResetType.Synchronous,
  override val elaborateTests:      Boolean = true,
  override val enableTestsProperty: Boolean = false
) extends Module
    with HasMonitorSocket
    with HasTests
    with HasProtocolInterface
    with HasTestsProperty {
  @public val io = IO(new ProtocolBundle(ioWidth))

  override val monProbe = makeProbe(io)

  io.out := io.in

  test("passing") { instance =>
    instance.io.in := 3.U(ioWidth.W)
    assert(instance.io.out === 3.U)
    TestBehavior.RunForCycles(10)
  }

  test("failing") { instance =>
    instance.io.in := 5.U(ioWidth.W)
    TestBehavior.FinishWhen(true.B, success = instance.io.out =/= 5.U)
  }

  test("assertion.unexpected") { instance =>
    instance.io.in := 5.U(ioWidth.W)
    chisel3.assert(instance.io.out =/= 5.U, "assertion fired in ModuleWithTests")
    TestBehavior.RunForCycles(10)
  }

  test("assertion.expected") { instance =>
    instance.io.in := 5.U(ioWidth.W)
    val message = "assertion fired in ModuleWithTests"
    chisel3.assert(instance.io.out =/= 5.U, message)
    TestBehavior.ExpectAssertion.contains(message)
  }

  {
    import TestHarnessWithMonitorSocket._
    test("with_monitor") { instance =>
      instance.io.in := 5.U(ioWidth.W)
      assert(instance.io.out =/= 0.U)
      TestBehavior.RunForCycles(10)
    }
  }

  test("check2") { dut =>
    ProtocolChecks.check(2)(dut)
    TestBehavior.RunForCycles(10)
  }
}

@instantiable
class RawModuleWithTests(ioWidth: Int = 32) extends RawModule with HasTests {
  @public val io = IO(new ProtocolBundle(ioWidth))
  io.out := io.in
  test("passing") { instance =>
    instance.io.in := 3.U(ioWidth.W)
    assert(instance.io.out === 3.U)
    TestBehavior.RunForCycles(10)
  }
}

class InlineTestSpec extends AnyFlatSpec with FileCheck with ChiselSim {
  private def makeArgs(moduleGlobs: Seq[String], testGlobs: Seq[String]): Array[String] =
    (
      moduleGlobs.map { glob => s"--include-tests-module=$glob" } ++
        testGlobs.map { glob => s"--include-tests-name=$glob" }
    ).toArray

  private def makeArgs(moduleGlob: String, testGlob: String): Array[String] =
    makeArgs(Seq(moduleGlob), Seq(testGlob))

  private val argsElaborateAllTests: Array[String] = makeArgs(Seq("*"), Seq("*"))

  it should "generate a public module for each test" in {
    emitCHIRRTL(new ModuleWithTests, args = argsElaborateAllTests).fileCheck()(
      """
      | CHECK:      module ModuleWithTests
      | CHECK:        output monProbe : Probe<{ in : UInt<32>, out : UInt<32>}>
      |
      | CHECK:      public module test_ModuleWithTests_check1
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input reset
      | CHECK:        inst dut of ModuleWithTests
      |
      | CHECK:      public module test_ModuleWithTests_passing
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input reset
      | CHECK-NEXT:   output finish : UInt<1>
      | CHECK-NEXT:   output success : UInt<1>
      | CHECK:        inst dut of ModuleWithTests
      |
      | CHECK:      public module test_ModuleWithTests_failing
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input reset
      | CHECK-NEXT:   output finish : UInt<1>
      | CHECK-NEXT:   output success : UInt<1>
      | CHECK:        inst dut of ModuleWithTests
      |
      | CHECK:      public module test_ModuleWithTests_with_monitor
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input reset
      | CHECK-NEXT:   output finish : UInt<1>
      | CHECK-NEXT:   output success : UInt<1>
      | CHECK:        inst dut of ModuleWithTests
      | CHECK:        inst monitor of ProtocolMonitor
      | CHECK-NEXT:   connect monitor.clock, clock
      | CHECK-NEXT:   connect monitor.reset, reset
      | CHECK-NEXT:   connect monitor.io.out, read(dut.monProbe).out
      | CHECK-NEXT:   connect monitor.io.in, read(dut.monProbe).in
      |
      | CHECK:      public module test_ModuleWithTests_check2
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input reset
      | CHECK:        inst dut of ModuleWithTests
      """
    )
  }

  it should "not elaborate tests without flag" in {
    emitCHIRRTL(new ModuleWithTests).fileCheck()(
      """
      | CHECK:      module ModuleWithTests
      | CHECK:        output monProbe : Probe<{ in : UInt<32>, out : UInt<32>}>
      |
      | CHECK-NOT:  module test_ModuleWithTests_passing
      | CHECK-NOT:  module test_ModuleWithTests_failing
      | CHECK-NOT:  module test_ModuleWithTests_with_monitor
      """
    )
  }

  it should "only elaborate tests whose name matches the test name glob" in {
    emitCHIRRTL(new ModuleWithTests, makeArgs(moduleGlob = "*", testGlob = "passing")).fileCheck()(
      """
      | CHECK:      module ModuleWithTests
      | CHECK:        output monProbe : Probe<{ in : UInt<32>, out : UInt<32>}>
      |
      | CHECK:      module test_ModuleWithTests_passing
      | CHECK-NOT:  module test_ModuleWithTests_failing
      | CHECK-NOT:  module test_ModuleWithTests_with_monitor
      """
    )
  }

  it should "elaborate tests whose name matches the test name glob when module glob is omitted" in {
    emitCHIRRTL(new ModuleWithTests, makeArgs(moduleGlobs = Nil, testGlobs = Seq("passing"))).fileCheck()(
      """
      | CHECK:      module ModuleWithTests
      | CHECK:        output monProbe : Probe<{ in : UInt<32>, out : UInt<32>}>
      |
      | CHECK:      module test_ModuleWithTests_passing
      | CHECK-NOT:  module test_ModuleWithTests_failing
      | CHECK-NOT:  module test_ModuleWithTests_with_monitor
      """
    )
  }

  it should "elaborate all tests when module glob is provided but test name glob is omitted" in {
    emitCHIRRTL(new ModuleWithTests, makeArgs(moduleGlobs = Seq("*WithTests"), testGlobs = Nil)).fileCheck()(
      """
      | CHECK:      module ModuleWithTests
      | CHECK:        output monProbe : Probe<{ in : UInt<32>, out : UInt<32>}>
      |
      | CHECK:      module test_ModuleWithTests_passing
      | CHECK:      module test_ModuleWithTests_failing
      | CHECK:      module test_ModuleWithTests_with_monitor
      """
    )
  }

  it should "only elaborate tests whose name matches the test name glob with multiple globs" in {
    emitCHIRRTL(new ModuleWithTests, makeArgs(moduleGlobs = Seq("*"), testGlobs = Seq("passing", "with_*")))
      .fileCheck()(
        """
      | CHECK:      module ModuleWithTests
      | CHECK:        output monProbe : Probe<{ in : UInt<32>, out : UInt<32>}>
      |
      | CHECK:      module test_ModuleWithTests_passing
      | CHECK-NOT:  module test_ModuleWithTests_failing
      | CHECK:      module test_ModuleWithTests_with_monitor
      """
      )
  }

  it should "only elaborate tests whose name and module match their globs" in {
    emitCHIRRTL(new ModuleWithTests, makeArgs(moduleGlobs = Seq("*WithTests"), testGlobs = Seq("passing", "with_*")))
      .fileCheck()(
        """
      | CHECK:      module ModuleWithTests
      | CHECK:        output monProbe : Probe<{ in : UInt<32>, out : UInt<32>}>
      |
      | CHECK:      module test_ModuleWithTests_passing
      | CHECK-NOT:  module test_ModuleWithTests_failing
      | CHECK:      module test_ModuleWithTests_with_monitor
      """
      )
  }

  it should "not elaborate tests whose module does not match the glob" in {
    emitCHIRRTL(new ModuleWithTests, makeArgs(moduleGlob = "*WithoutTests", testGlob = "*")).fileCheck()(
      """
      | CHECK:      module ModuleWithTests
      | CHECK:        output monProbe : Probe<{ in : UInt<32>, out : UInt<32>}>
      |
      | CHECK-NOT:  module test_ModuleWithTests_passing
      | CHECK-NOT:  module test_ModuleWithTests_failing
      | CHECK-NOT:  module test_ModuleWithTests_with_monitor
      """
    )
  }

  it should "not elaborate tests with HasTests.elaborateTests set to false" in {
    emitCHIRRTL(new ModuleWithTests(elaborateTests = false), argsElaborateAllTests).fileCheck()(
      """
      | CHECK:      module ModuleWithTests
      | CHECK:        output monProbe : Probe<{ in : UInt<32>, out : UInt<32>}>
      |
      | CHECK-NOT:  module test_ModuleWithTests_passing
      | CHECK-NOT:  module test_ModuleWithTests_failing
      | CHECK-NOT:  module test_ModuleWithTests_with_monitor
      """
    )
  }

  it should "compile to verilog" in {
    ChiselStage
      .emitSystemVerilog(new ModuleWithTests, args = argsElaborateAllTests)
      .fileCheck()(
        """
      | CHECK: module ModuleWithTests
      | CHECK: module test_ModuleWithTests_check1
      | CHECK: module test_ModuleWithTests_passing
      | CHECK: module test_ModuleWithTests_failing
      | CHECK: module test_ModuleWithTests_with_monitor
      | CHECK: module test_ModuleWithTests_check2
      """
      )
  }

  it should "support iterating over registered tests to capture metadata" in {
    ChiselStage
      .emitCHIRRTL(
        new ModuleWithTests(enableTestsProperty = true),
        args = makeArgs(Seq("*"), Seq("passing", "failing"))
      )
      .fileCheck()(
        """
        | CHECK: module ModuleWithTests
        | CHECK:   output testNames : List<String>
        | CHECK:   propassign testNames, List<String>(String("passing"), String("failing"))
        | CHECK: module test_ModuleWithTests_passing
        | CHECK: module test_ModuleWithTests_failing
        """
      )
  }

  it should "emit the correct reset types" in {
    def fileCheckString(resetType: String) =
      s"""
      | CHECK:      module ModuleWithTests
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input reset : ${resetType}
      |
      | CHECK:      public module test_ModuleWithTests_check1
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input reset : ${resetType}
      |
      | CHECK:      public module test_ModuleWithTests_passing
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input reset : ${resetType}
      | CHECK-NEXT:   output finish : UInt<1>
      | CHECK-NEXT:   output success : UInt<1>
      |
      | CHECK:      public module test_ModuleWithTests_failing
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input reset : ${resetType}
      | CHECK-NEXT:   output finish : UInt<1>
      | CHECK-NEXT:   output success : UInt<1>
      |
      | CHECK:      public module test_ModuleWithTests_with_monitor
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input reset : ${resetType}
      |
      | CHECK:      public module test_ModuleWithTests_check2
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input reset : ${resetType}
      | CHECK-NEXT:   output finish : UInt<1>
      | CHECK-NEXT:   output success : UInt<1>
      """

    emitCHIRRTL(new ModuleWithTests(resetType = Module.ResetType.Synchronous), args = argsElaborateAllTests)
      .fileCheck()(
        fileCheckString("UInt<1>")
      )
    emitCHIRRTL(new ModuleWithTests(resetType = Module.ResetType.Asynchronous), args = argsElaborateAllTests)
      .fileCheck()(
        fileCheckString("AsyncReset")
      )
    emitCHIRRTL(new ModuleWithTests(resetType = Module.ResetType.Default), args = argsElaborateAllTests).fileCheck()(
      fileCheckString("UInt<1>")
    )

    emitCHIRRTL(new RawModuleWithTests(), args = argsElaborateAllTests).fileCheck()(
      """
      | CHECK:      module RawModuleWithTests
      | CHECK-NEXT:   output io
      |
      | CHECK:      public module test_RawModuleWithTests_passing
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input reset : UInt<1>
      | CHECK-NEXT:   output finish : UInt<1>
      | CHECK-NEXT:   output success : UInt<1>
      """
    )
  }

  def assertPass(result: TestResult.Type): Unit = result match {
    case other: TestResult.Failure =>
      fail(s"Test unexpectedly failed: ${other}")
    case TestResult.Success => ()
  }

  def assertFail(expectedMessage: String)(result: TestResult.Type): Unit = result match {
    case TestResult.Success =>
      fail("Test unexpectedly passed")
    case TestResult.Failure(actualMessage) if !actualMessage.contains(expectedMessage) =>
      fail(s"'${actualMessage}' does not match '${expectedMessage}'")
    case _ => ()
  }

  it should "simulate and pass if finish asserted with success=1" in {
    val results = simulateTests(
      new ModuleWithTests,
      tests = TestChoice.Name("passing"),
      timeout = 100
    )
    assert(results.size == 1, "Expected exactly one test result")
    assertPass(results.head.result)
  }

  it should "simulate and fail if finish asserted with success=0" in {
    val results = simulateTests(
      new ModuleWithTests,
      tests = TestChoice.Name("failing"),
      timeout = 100
    )
    assert(results.size == 1, "Expected exactly one test result")
    assertFail("test signaled failure")(results.head.result)
  }

  it should "simulate and fail early if assertion raised" in {
    val results = simulateTests(
      new ModuleWithTests,
      tests = TestChoice.Name("assertion.unexpected"),
      timeout = 100
    )
    assert(results.size == 1, "Expected exactly one test result")
    assertFail("assertion fired")(results.head.result)
  }

  it should "simulate and pass if expected assertion fired" in {
    val results = simulateTests(
      new ModuleWithTests,
      tests = TestChoice.Name("assertion.expected"),
      timeout = 100
    )
    assert(results.size == 1, "Expected exactly one test result")
    assertPass(results.head.result)
  }

  it should "run multiple passing simulations" in {
    val results = simulateTests(
      new ModuleWithTests,
      tests = TestChoice.Names(Seq("passing", "with_monitor")),
      timeout = 100
    )
    assert(results.size == 2, "Expected exactly two test results")
    results.foreach { simTest =>
      assertPass(simTest.result)
    }
  }

  it should "run one passing and one failing simulation" in {
    val results = simulateTests(
      new ModuleWithTests,
      tests = TestChoice.Names(Seq("passing", "failing")),
      timeout = 100
    )
    assert(results.size == 2, "Expected exactly two test results")
    val signalPass = results.find(_.testName == "passing").get
    val signalFail = results.find(_.testName == "failing").get
    assertPass(signalPass.result)
    assertFail("test signaled failure")(signalFail.result)
  }
}

case class AluParameters(operandWidth: Int, widenResult: Boolean) {
  def resultWidth = if (widenResult) operandWidth + 1 else operandWidth
  def name: String = s"ALU_${operandWidth}" + (if (widenResult) "w" else "")
}

object AluOp extends ChiselEnum {
  val Add = Value("b000".U)
  val Sub = Value("b001".U)
  val And = Value("b010".U)
  val Or = Value("b011".U)
  val Xor = Value("b100".U)
}

// Separate request bundle for ALU inputs
class AluReq(params: AluParameters) extends Bundle {
  val op1 = UInt(params.operandWidth.W)
  val op2 = UInt(params.operandWidth.W)
  val opcode = AluOp()
}

// Separate response bundle for ALU output
class AluResp(params: AluParameters) extends Bundle {
  val result = UInt(params.resultWidth.W)
}

class AluIO(params: AluParameters) extends Bundle {
  val req = Flipped(Decoupled(new AluReq(params)))
  val resp = Decoupled(new AluResp(params))
}

object Alu {
  object Assertions {
    sealed abstract class Type(val message: String) {
      def assert(cond: Bool) = chisel3.assert(cond, message)
    }

    case object AdditionOverflow extends Type("addition overflow detected")
    case object SubtractionOverflow extends Type("subtraction overflow detected")
  }
}

@instantiable
class Alu(params: AluParameters) extends Module with HasTests {
  @public val io = IO(new AluIO(params))

  // Register request inputs when valid transaction occurs
  val reqReg = RegInit(0.U.asTypeOf(new AluReq(params)))
  val busy = RegInit(false.B)

  // Default values
  io.req.ready := !busy
  io.resp.valid := busy
  io.resp.bits.result := 0.U

  // Capture input when there's a valid request and we're ready
  when(io.req.fire) {
    reqReg := io.req.bits
    busy := true.B
  }

  // Clear busy when response is accepted
  when(io.resp.fire) {
    busy := false.B
  }

  // ALU logic using registered inputs
  val result = Wire(UInt(params.resultWidth.W))
  result := 0.U

  switch(reqReg.opcode) {
    is(AluOp.Add) {
      val sum = reqReg.op1 +& reqReg.op2
      result := sum

      if (!params.widenResult) {
        val assertion = Alu.Assertions.AdditionOverflow
        assertion.assert(!sum(params.operandWidth))
        test("sanity.add.zero") { alu =>
          alu.io.req.valid := true.B
          alu.io.req.bits.op1 := 0.U
          alu.io.req.bits.op2 := 0.U
          alu.io.req.bits.opcode := AluOp.Add
          alu.io.resp.ready := true.B
          TestBehavior.FinishWhen(
            finish = alu.io.resp.fire,
            success = alu.io.resp.valid && alu.io.resp.bits.result === 0.U
          )
        }
      }
    }
    is(AluOp.Sub) {
      val diff = reqReg.op1 -& reqReg.op2
      result := diff
      if (!params.widenResult) {
        val assertion = Alu.Assertions.SubtractionOverflow
        assertion.assert(!diff(params.operandWidth))
      }
    }
    is(AluOp.And) {
      result := reqReg.op1 & reqReg.op2
    }
    is(AluOp.Or) {
      result := reqReg.op1 | reqReg.op2
    }
    is(AluOp.Xor) {
      result := reqReg.op1 ^ reqReg.op2
    }
  }

  io.resp.bits.result := result

  test("sanity.and.zero") { alu =>
    alu.io.req.valid := true.B
    alu.io.req.bits.op1 := "b1010".U
    alu.io.req.bits.op2 := 0.U
    alu.io.req.bits.opcode := AluOp.And
    alu.io.resp.ready := true.B
    TestBehavior.FinishWhen(
      finish = alu.io.resp.fire,
      success = alu.io.resp.valid && alu.io.resp.bits.result === 0.U
    )
  }

  test("sanity.or.identity") { alu =>
    val value = "b1010".U
    alu.io.req.valid := true.B
    alu.io.req.bits.op1 := value
    alu.io.req.bits.op2 := 0.U
    alu.io.req.bits.opcode := AluOp.Or
    alu.io.resp.ready := true.B
    TestBehavior.FinishWhen(
      finish = alu.io.resp.fire,
      success = alu.io.resp.valid && alu.io.resp.bits.result === value
    )
  }

  test("sanity.xor.self") { alu =>
    alu.io.req.valid := true.B
    alu.io.req.bits.op1 := "b1010".U
    alu.io.req.bits.op2 := "b1010".U
    alu.io.req.bits.opcode := AluOp.Xor
    alu.io.resp.ready := true.B
    TestBehavior.FinishWhen(
      finish = alu.io.resp.fire,
      success = alu.io.resp.valid && alu.io.resp.bits.result === 0.U
    )
  }
}

class AluSpec extends InlineTests {
  val configs = for {
    width <- Seq(4, 8, 16)
    widenResult <- Seq(false, true)
  } yield AluParameters(width, widenResult)

  val alu4W = AluParameters(4, false)
  runInlineTests(alu4W.name)(new Alu(alu4W))(TestChoice.Glob("*overflow*"))

  describe(s"sanity test: ${configs.map(_.name).mkString(", ")}") {
    configs.foreach { params =>
      runInlineTests(params.name)(new Alu(params))(TestChoice.Glob("sanity.*"))
    }
  }

  runInlineTests(alu4W.name)(new Alu(alu4W))(TestChoice.All)
}
