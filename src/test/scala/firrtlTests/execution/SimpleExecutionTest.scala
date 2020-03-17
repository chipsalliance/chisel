package firrtlTests.execution

import java.io.File

import firrtl.ir._
import firrtl.testutils._

sealed trait SimpleTestCommand
case class Step(n: Int) extends SimpleTestCommand
case class Invalidate(expStr: String) extends SimpleTestCommand
case class Poke(expStr: String, value: Int) extends SimpleTestCommand
case class Expect(expStr: String, value: Int) extends SimpleTestCommand

/**
  * This trait defines an interface to run a self-contained test circuit.
  */
trait TestExecution {
  def runEmittedDUT(c: Circuit, testDir: File): Unit
}

/**
  * A class that makes it easier to write execution-driven tests.
  * 
  * By combining a DUT body (supplied as a string without an enclosing
  * module or circuit) with a sequence of test operations, an
  * executable, self-contained Verilog testbench may be automatically
  * created and checked.
  * 
  * @note It is necessary to mix in a trait extending TestExecution
  * @note The DUT has two implicit ports, "clock" and "reset"
  * @note Execution of the command sequences begins after reset is deasserted
  * 
  * @see [[firrtlTests.execution.TestExecution]]
  * @see [[firrtlTests.execution.VerilogExecution]]
  * 
  * @example {{{
  * class AndTester extends SimpleExecutionTest with VerilogExecution {
  *   val body = "reg r : UInt<32>, clock with: (reset => (reset, UInt<32>(0)))"
  *   val commands = Seq(
  *     Expect("r", 0),
  *     Poke("r", 3),
  *     Step(1),
  *     Expect("r", 3)
  *   )
  * }
  * }}}
  */
abstract class SimpleExecutionTest extends FirrtlPropSpec {
  this: TestExecution =>

  /**
    * Text representing the body of the DUT. This is useful for testing
    * statement-level language features, and cuts out the overhead of
    * writing a top-level DUT module and having peeks/pokes point at
    * IOs.
    */
  val body: String

  /**
    * A sequence of commands (peeks, pokes, invalidates, steps) that
    * represents how the testbench will progress. The semantics are
    * inspired by chisel-testers.
    */
  def commands: Seq[SimpleTestCommand]

  private def interpretCommand(eth: ExecutionTestHelper, cmd: SimpleTestCommand) = cmd match {
    case Step(n) => eth.step(n)
    case Invalidate(expStr) => eth.invalidate(expStr)
    case Poke(expStr, value) => eth.poke(expStr, UIntLiteral(value))
    case Expect(expStr, value) => eth.expect(expStr, UIntLiteral(value))
  }

  private def runTest(): Unit = {
    val initial = ExecutionTestHelper(body)
    val test = commands.foldLeft(initial)(interpretCommand(_, _))
    val testName = this.getClass.getSimpleName
    val testDir = createTestDirectory(s"${testName}-generated-src")
    runEmittedDUT(test.emit, testDir)
  }

  property("Execution of the compiled Verilog for ExecutionTestHelper should succeed") {
    runTest()
  }
}
