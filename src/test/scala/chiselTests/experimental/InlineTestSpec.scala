package chiselTests

import chisel3.{assert => _, _}
import chisel3.experimental.hierarchy._
import chisel3.experimental.inlinetest._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.testers._
import chisel3.properties.Property
import chisel3.testing.scalatest.FileCheck
import chisel3.util.{Counter, Enum}
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import circt.stage.ChiselStage.{emitCHIRRTL, emitSystemVerilog}

// Here is a testharness that expects some sort of interface on its DUT, e.g. a probe
// socket to which to attach a monitor.
class TestHarnessWithMonitorSocket[M <: RawModule with HasMonitorSocket](test: TestParameters[M, Unit])
    extends TestHarness[M, Unit](test) {
  val monitor = Module(new ProtocolMonitor(dut.monProbe.cloneType))
  monitor.io :#= probe.read(dut.monProbe)
}

object TestHarnessWithMonitorSocket {
  implicit def generator[M <: RawModule with HasMonitorSocket] =
    TestHarnessGenerator[M, Unit](new TestHarnessWithMonitorSocket(_))
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
  chisel3.assert(io.in === io.out, "in === out")
}

@instantiable
trait HasProtocolInterface extends HasTests { this: RawModule =>
  @public val io: ProtocolBundle

  test("check1")(ProtocolChecks.check(1))
}

object ProtocolChecks {
  def check(v: Int)(instance: Instance[RawModule with HasProtocolInterface]) = {
    instance.io.in := v.U
    chisel3.assert(instance.io.out === v.U): Unit
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

  test("foo") { instance =>
    instance.io.in := 3.U(ioWidth.W)
    chisel3.assert(instance.io.out === 3.U): Unit
  }

  test("bar") { instance =>
    instance.io.in := 5.U(ioWidth.W)
    chisel3.assert(instance.io.out =/= 0.U): Unit
  }

  test("with_result") { instance =>
    val result = Wire(new TestResultBundle)
    val timer = RegInit(0.U)
    timer := timer + 1.U
    instance.io.in := 5.U(ioWidth.W)
    val outValid = instance.io.out =/= 0.U
    when(outValid) {
      result.success := 0.U
      result.finish := timer > 1000.U
    }.otherwise {
      result.success := 1.U
      result.finish := true.B
    }
    result
  }

  {
    import TestHarnessWithMonitorSocket._
    test("with_monitor") { instance =>
      instance.io.in := 5.U(ioWidth.W)
      chisel3.assert(instance.io.out =/= 0.U): Unit
    }
  }

  test("check2")(ProtocolChecks.check(2))

  test("signal_pass") { instance =>
    val counter = Counter(16)
    counter.inc()
    instance.io.in := counter.value
    val result = Wire(new TestResultBundle())
    result.success := true.B
    result.finish := counter.value === 15.U
    result
  }

  test("signal_pass_2") { instance =>
    val counter = Counter(16)
    counter.inc()
    instance.io.in := counter.value
    val result = Wire(new TestResultBundle())
    result.success := true.B
    result.finish := counter.value === 15.U
    result
  }

  test("signal_fail") { instance =>
    val counter = Counter(16)
    counter.inc()
    instance.io.in := counter.value
    val result = Wire(new TestResultBundle())
    result.success := false.B
    result.finish := counter.value === 15.U
    result
  }

  test("timeout") { instance =>
    val counter = Counter(16)
    counter.inc()
    instance.io.in := counter.value
  }

  test("assertion") { instance =>
    val counter = Counter(16)
    counter.inc()
    instance.io.in := counter.value
    chisel3.assert(instance.io.out < 15.U, "counter hit max"): Unit
  }
}

@instantiable
class RawModuleWithTests(ioWidth: Int = 32) extends RawModule with HasTests {
  @public val io = IO(new ProtocolBundle(ioWidth))
  io.out := io.in
  test("foo") { instance =>
    instance.io.in := 3.U(ioWidth.W)
    chisel3.assert(instance.io.out === 3.U): Unit
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
      | CHECK:      public module test_ModuleWithTests_foo
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input reset
      | CHECK-NEXT:   output finish : UInt<1>
      | CHECK-NEXT:   output success : UInt<1>
      | CHECK:        inst dut of ModuleWithTests
      |
      | CHECK:      public module test_ModuleWithTests_bar
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input reset
      | CHECK-NEXT:   output finish : UInt<1>
      | CHECK-NEXT:   output success : UInt<1>
      | CHECK:        inst dut of ModuleWithTests
      |
      | CHECK:      public module test_ModuleWithTests_with_result
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
      | CHECK-NOT:  module test_ModuleWithTests_foo
      | CHECK-NOT:  module test_ModuleWithTests_bar
      | CHECK-NOT:  module test_ModuleWithTests_with_result
      | CHECK-NOT:  module test_ModuleWithTests_with_monitor
      """
    )
  }

  it should "only elaborate tests whose name matches the test name glob" in {
    emitCHIRRTL(new ModuleWithTests, makeArgs(moduleGlob = "*", testGlob = "foo")).fileCheck()(
      """
      | CHECK:      module ModuleWithTests
      | CHECK:        output monProbe : Probe<{ in : UInt<32>, out : UInt<32>}>
      |
      | CHECK:      module test_ModuleWithTests_foo
      | CHECK-NOT:  module test_ModuleWithTests_bar
      | CHECK-NOT:  module test_ModuleWithTests_with_result
      | CHECK-NOT:  module test_ModuleWithTests_with_monitor
      """
    )
  }

  it should "elaborate tests whose name matches the test name glob when module glob is omitted" in {
    emitCHIRRTL(new ModuleWithTests, makeArgs(moduleGlobs = Nil, testGlobs = Seq("foo"))).fileCheck()(
      """
      | CHECK:      module ModuleWithTests
      | CHECK:        output monProbe : Probe<{ in : UInt<32>, out : UInt<32>}>
      |
      | CHECK:      module test_ModuleWithTests_foo
      | CHECK-NOT:  module test_ModuleWithTests_bar
      | CHECK-NOT:  module test_ModuleWithTests_with_result
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
      | CHECK:      module test_ModuleWithTests_foo
      | CHECK:      module test_ModuleWithTests_bar
      | CHECK:      module test_ModuleWithTests_with_result
      | CHECK:      module test_ModuleWithTests_with_monitor
      """
    )
  }

  it should "only elaborate tests whose name matches the test name glob with multiple globs" in {
    emitCHIRRTL(new ModuleWithTests, makeArgs(moduleGlobs = Seq("*"), testGlobs = Seq("foo", "with_*"))).fileCheck()(
      """
      | CHECK:      module ModuleWithTests
      | CHECK:        output monProbe : Probe<{ in : UInt<32>, out : UInt<32>}>
      |
      | CHECK:      module test_ModuleWithTests_foo
      | CHECK-NOT:  module test_ModuleWithTests_bar
      | CHECK:      module test_ModuleWithTests_with_result
      | CHECK:      module test_ModuleWithTests_with_monitor
      """
    )
  }

  it should "only elaborate tests whose name and module match their globs" in {
    emitCHIRRTL(new ModuleWithTests, makeArgs(moduleGlobs = Seq("*WithTests"), testGlobs = Seq("foo", "with_*")))
      .fileCheck()(
        """
      | CHECK:      module ModuleWithTests
      | CHECK:        output monProbe : Probe<{ in : UInt<32>, out : UInt<32>}>
      |
      | CHECK:      module test_ModuleWithTests_foo
      | CHECK-NOT:  module test_ModuleWithTests_bar
      | CHECK:      module test_ModuleWithTests_with_result
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
      | CHECK-NOT:  module test_ModuleWithTests_foo
      | CHECK-NOT:  module test_ModuleWithTests_bar
      | CHECK-NOT:  module test_ModuleWithTests_with_result
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
      | CHECK-NOT:  module test_ModuleWithTests_foo
      | CHECK-NOT:  module test_ModuleWithTests_bar
      | CHECK-NOT:  module test_ModuleWithTests_with_result
      | CHECK-NOT:  module test_ModuleWithTests_with_monitor
      """
    )
  }

  it should "compile to verilog" in {
    emitSystemVerilog(new ModuleWithTests, args = argsElaborateAllTests)
      .fileCheck()(
        """
      | CHECK: module ModuleWithTests
      | CHECK: module test_ModuleWithTests_check1
      | CHECK: module test_ModuleWithTests_foo
      | CHECK: module test_ModuleWithTests_bar
      | CHECK: module test_ModuleWithTests_with_result
      | CHECK: module test_ModuleWithTests_with_monitor
      | CHECK: module test_ModuleWithTests_check2
      """
      )
  }

  it should "support iterating over registered tests to capture metadata" in {
    ChiselStage
      .emitCHIRRTL(new ModuleWithTests(enableTestsProperty = true), args = makeArgs(Seq("*"), Seq("foo", "bar")))
      .fileCheck()(
        """
        | CHECK: module ModuleWithTests
        | CHECK:   output testNames : List<String>
        | CHECK:   propassign testNames, List<String>(String("foo"), String("bar"))
        | CHECK: module test_ModuleWithTests_foo
        | CHECK: module test_ModuleWithTests_bar
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
      | CHECK:      public module test_ModuleWithTests_foo
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input reset : ${resetType}
      | CHECK-NEXT:   output finish : UInt<1>
      | CHECK-NEXT:   output success : UInt<1>
      |
      | CHECK:      public module test_ModuleWithTests_bar
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input reset : ${resetType}
      | CHECK-NEXT:   output finish : UInt<1>
      | CHECK-NEXT:   output success : UInt<1>
      |
      | CHECK:      public module test_ModuleWithTests_with_result
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
      | CHECK:      public module test_RawModuleWithTests_foo
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input reset : UInt<1>
      | CHECK-NEXT:   output finish : UInt<1>
      | CHECK-NEXT:   output success : UInt<1>
      """
    )
  }

  def assertPass(result: TestResult.Type): Unit = result match {
    case TestResult.Success => ()
    case result: TestResult.Failure => fail(s"unexpected failure: ${result}")
  }

  def assertFail(result: TestResult.Type): Unit = result match {
    case TestResult.Success => fail("Test unexpectedly passed")
    case TestResult.SignaledFailure => () // expected failure
    case other: TestResult.Failure => fail(s"wrong type of failure: ${other}")
  }

  def assertTimeout(timeout: Int)(result: TestResult.Type): Unit = result match {
    case TestResult.Success => fail("Test unexpectedly passed")
    case TestResult.Timeout(msg) if msg.contains(s"after ${timeout} timesteps") => ()
    case other: TestResult.Failure => fail(s"wrong type of failure: ${other}")
  }

  def assertAssertion(message: String)(result: TestResult.Type): Unit = result match {
    case TestResult.Success => fail("Test unexpectedly passed")
    case TestResult.Assertion(msg) if msg.contains("counter hit max") => ()
    case other: TestResult.Failure => fail(s"wrong type of failure: ${other}")
  }

  it should "simulate and pass if finish asserted with success=1" in {
    val results = simulateTests(
      new ModuleWithTests,
      tests = TestChoice.Name("signal_pass"),
      timeout = 100
    )
    assertPass(results("signal_pass"))
  }

  it should "simulate and fail if finish asserted with success=0" in {
    val results = simulateTests(
      new ModuleWithTests,
      tests = TestChoice.Name("signal_fail"),
      timeout = 100
    )
    assertFail(results("signal_fail"))
  }

  it should "simulate and timeout if finish not asserted" in {
    val results = simulateTests(
      new ModuleWithTests,
      tests = TestChoice.Name("timeout"),
      timeout = 100
    )
    assertTimeout(100)(results("timeout"))
  }

  it should "simulate and fail early if assertion raised" in {
    val results = simulateTests(
      new ModuleWithTests,
      tests = TestChoice.Name("assertion"),
      timeout = 100
    )
    assertAssertion("counter hit max")(results("assertion"))
  }

  it should "run multiple passing simulations" in {
    val results = simulateTests(
      new ModuleWithTests,
      tests = TestChoice.Names(Seq("signal_pass", "signal_pass_2")),
      timeout = 100
    )
    results.all.foreach { case (name, result) =>
      assertPass(result)
    }
  }

  it should "run one passing and one failing simulation" in {
    val results = simulateTests(
      new ModuleWithTests,
      tests = TestChoice.Names(Seq("signal_pass", "signal_fail")),
      timeout = 100
    )
    assertPass(results("signal_pass"))
    assertFail(results("signal_fail"))
  }

  it should "simulate all tests" in {
    val results = simulateTests(
      new ModuleWithTests,
      tests = TestChoice.All,
      timeout = 100
    )
    assert(results.all.size == 11)

    assertFail(results("signal_fail"))
    assertTimeout(100)(results("timeout"))
    assertAssertion("counter hit max")(results("assertion"))
    assertTimeout(100)(results("check1"))
    assertTimeout(100)(results("check2"))
    assertTimeout(100)(results("bar"))
    assertPass(results("signal_pass"))
    assertPass(results("signal_pass_2"))
    assertTimeout(100)(results("with_monitor"))
    assertTimeout(100)(results("with_result"))
    assertTimeout(100)(results("foo"))
  }
}
