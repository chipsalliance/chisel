package chiselTests

import chisel3._
import chisel3.experimental.hierarchy._
import chisel3.experimental.inlinetest._
import chisel3.testers._
import chisel3.properties.Property
import chisel3.testing.scalatest.FileCheck
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
    TestConfiguration.runForCycles(10)
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
    TestConfiguration.runForCycles(10)
  }

  test("failing") { instance =>
    instance.io.in := 5.U(ioWidth.W)
    TestConfiguration(
      done = RegNext(true.B),
      success = instance.io.out =/= 5.U,
      cf"unexpected output"
    )
  }

  test("assertion") { instance =>
    instance.io.in := 5.U(ioWidth.W)
    chisel3.assert(instance.io.out =/= 5.U, "assertion fired in ModuleWithTests")
    TestConfiguration.runForCycles(10)
  }

  {
    import TestHarnessWithMonitorSocket._
    test("with_monitor") { instance =>
      instance.io.in := 5.U(ioWidth.W)
      assert(instance.io.out =/= 0.U)
      TestConfiguration.runForCycles(10)
    }
  }

  test("check2") { dut =>
    ProtocolChecks.check(2)(dut)
    TestConfiguration.runForCycles(10)
  }
}

@instantiable
class RawModuleWithTests(ioWidth: Int = 32) extends RawModule with HasTests {
  @public val io = IO(new ProtocolBundle(ioWidth))
  io.out := io.in
  test("passing") { instance =>
    instance.io.in := 3.U(ioWidth.W)
    assert(instance.io.out === 3.U)
    TestConfiguration.runForCycles(10)
  }
}

class InlineTestSpec extends AnyFlatSpec with Matchers with FileCheck with ChiselSim {
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
      | CHECK-NEXT:   input init
      | CHECK:        inst dut of ModuleWithTests
      |
      | CHECK:      public module test_ModuleWithTests_passing
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input init
      | CHECK-NEXT:   output done : UInt<1>
      | CHECK-NEXT:   output success : UInt<1>
      | CHECK:        inst dut of ModuleWithTests
      |
      | CHECK:      public module test_ModuleWithTests_failing
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input init
      | CHECK-NEXT:   output done : UInt<1>
      | CHECK-NEXT:   output success : UInt<1>
      | CHECK:        inst dut of ModuleWithTests
      |
      | CHECK:      public module test_ModuleWithTests_with_monitor
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input init
      | CHECK-NEXT:   output done : UInt<1>
      | CHECK-NEXT:   output success : UInt<1>
      | CHECK:        inst dut of ModuleWithTests
      | CHECK:        inst monitor of ProtocolMonitor
      | CHECK-NEXT:   connect monitor.clock, clock
      | CHECK-NEXT:   connect monitor.reset, init
      | CHECK-NEXT:   connect monitor.io.out, read(dut.monProbe).out
      | CHECK-NEXT:   connect monitor.io.in, read(dut.monProbe).in
      |
      | CHECK:      public module test_ModuleWithTests_check2
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input init
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
      | CHECK-NEXT:   input init : UInt<1>
      |
      | CHECK:      public module test_ModuleWithTests_passing
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input init : UInt<1>
      | CHECK-NEXT:   output done : UInt<1>
      | CHECK-NEXT:   output success : UInt<1>
      |
      | CHECK:      public module test_ModuleWithTests_failing
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input init : UInt<1>
      | CHECK-NEXT:   output done : UInt<1>
      | CHECK-NEXT:   output success : UInt<1>
      |
      | CHECK:      public module test_ModuleWithTests_with_monitor
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input init : UInt<1>
      |
      | CHECK:      public module test_ModuleWithTests_check2
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input init : UInt<1>
      | CHECK-NEXT:   output done : UInt<1>
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
      | CHECK-NEXT:   input init : UInt<1>
      | CHECK-NEXT:   output done : UInt<1>
      | CHECK-NEXT:   output success : UInt<1>
      """
    )
  }

  it should "simulate and pass if done asserted with success=1" in {
    simulateTests(
      new ModuleWithTests,
      tests = TestChoice.Globs("passing"),
      timeout = 100
    )
  }

  it should "simulate and fail if done asserted with success=0" in {
    val exception = intercept[chisel3.simulator.Exceptions.TestsFailed] {
      simulateTests(
        new ModuleWithTests,
        tests = TestChoice.Globs("failing"),
        timeout = 100
      )
    }
    exception.getMessage should include("ModuleWithTests tests: 0 passed, 1 failed")
    exception.getMessage should include("failures:")
    exception.getMessage should include("- failing: test signaled failure")
  }

  it should "simulate and fail early if assertion raised" in {
    val exception = intercept[chisel3.simulator.Exceptions.TestsFailed] {
      simulateTests(
        new ModuleWithTests,
        tests = TestChoice.Globs("assertion"),
        timeout = 100
      )
    }
    exception.getMessage should include("ModuleWithTests tests: 0 passed, 1 failed")
    exception.getMessage should include("failures:")
    (exception.getMessage should include).regex("- assertion: .*assertion fired")
  }

  it should "run multiple passing simulations" in {
    simulateTests(
      new ModuleWithTests,
      tests = TestChoice.Globs("passing", "with_monitor"),
      timeout = 100
    )
  }

  it should "run one passing and one failing-with-signal simulation" in {
    val exception = intercept[chisel3.simulator.Exceptions.TestsFailed] {
      simulateTests(
        new ModuleWithTests,
        tests = TestChoice.Globs("passing", "failing"),
        timeout = 100
      )
    }
    exception.getMessage should include("ModuleWithTests tests: 1 passed, 1 failed")
    exception.getMessage should include("failures:")
    exception.getMessage should include("- failing: test signaled failure")

    // Verify that granular results are available in the exception
    exception.results should have size 2
    exception.results.foreach { result =>
      result.testName match {
        case "passing" => result.result should be(TestResult.Success)
        case "failing" => {
          val TestResult.Failure(message) = result.result
          message should include("test signaled failure")
        }
      }
    }
  }

  it should "run one failing-with-assertion and one passing simulation" in {
    val exception = intercept[chisel3.simulator.Exceptions.TestsFailed] {
      simulateTests(
        new ModuleWithTests,
        tests = TestChoice.Globs("assertion", "passing"),
        timeout = 100
      )
    }
    exception.getMessage should include("ModuleWithTests tests: 1 passed, 1 failed")
    exception.getMessage should include("failures:")
    (exception.getMessage should include).regex("- assertion: .*assertion fired")
  }

  it should "run one failing-with-assertion, one passing, and one failing-with-signal simulation in any order" in {
    Array("passing", "failing", "assertion").permutations.foreach { testNames =>
      val exception = intercept[chisel3.simulator.Exceptions.TestsFailed] {
        simulateTests(
          new ModuleWithTests,
          tests = TestChoice.Globs(testNames),
          timeout = 100
        )
      }
      exception.getMessage should include("ModuleWithTests tests: 1 passed, 2 failed")
      exception.getMessage should include("failures:")
      exception.getMessage should include("- failing: test signaled failure")
      (exception.getMessage should include).regex("- assertion: .*assertion fired")

      // Verify that granular results are available in the exception
      exception.results should have size 3
      exception.results.foreach { result =>
        result.testName match {
          case "passing" => result.result should be(TestResult.Success)
          case "failing" => {
            val TestResult.Failure(message) = result.result
            message should include("test signaled failure")
          }
          case "assertion" => {
            val TestResult.Failure(message) = result.result
            message should include("assertion fired")
          }
        }
      }
    }
  }
}
