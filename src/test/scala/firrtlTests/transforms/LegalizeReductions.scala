// See LICENSE for license details.

package firrtlTests.transforms

import firrtl._
import firrtl.ir.StringLit
import firrtl.testutils._

import org.scalatest.flatspec.AnyFlatSpec

import java.io.File

object LegalizeAndReductionsTransformSpec extends FirrtlRunners {
  private case class Test(
    name: String,
    op: String,
    input: BigInt,
    expected: BigInt,
    forceWidth: Option[Int] = None
  ) {
    def toFirrtl: String = {
      val width = forceWidth.getOrElse(input.bitLength)
      val inputLit = s"""UInt("h${input.toString(16)}")"""
      val expectLit = s"""UInt("h${expected.toString(16)}")"""
      val assertMsg = StringLit(s"Assertion failed! $op($inputLit) != $expectLit!\n").verilogEscape
      // Reset the register to something different from the input
      val regReset = if (input == 0) 1 else 0
      s"""
circuit $name :
  module $name :
    input clock : Clock
    input reset : UInt<1>

    node input = $inputLit
    node expected = $expectLit

    reg r : UInt<$width>, clock with : (reset => (reset, UInt($regReset)))
    r <= input
    node expr = $op(r)
    node equal = eq(expr, expected)

    reg count : UInt<2>, clock with : (reset => (reset, UInt(0)))
    count <= tail(add(count, UInt(1)), 1)

    when not(reset) :
      when eq(count, UInt(1)) :
        printf(clock, UInt(1), "input = %x, expected = %x, expr = %x, equal = %x\\n", input, expected, expr, equal)
        when not(equal) :
          printf(clock, UInt(1), $assertMsg)
          stop(clock, UInt(1), 1)
        else :
          stop(clock, UInt(1), 0)
"""
    }
  }

  private def executeTest(test: Test): Unit = {
    import firrtl.stage._
    import firrtl.options.TargetDirAnnotation
    val prefix = test.name
    val testDir = createTestDirectory(s"LegalizeReductions.$prefix")
    // Run FIRRTL
    val annos =
      FirrtlSourceAnnotation(test.toFirrtl) ::
      TargetDirAnnotation(testDir.toString) ::
      CompilerAnnotation(new MinimumVerilogCompiler) ::
      Nil
    val resultAnnos = (new FirrtlStage).transform(annos)
    val outputFilename = resultAnnos.collectFirst { case OutputFileAnnotation(f) => f }
    outputFilename.toRight(s"Output file not found!")
    // Copy Verilator harness
    val harness = new File(testDir, "top.cpp")
    copyResourceToFile(cppHarnessResourceName, harness)
    // Run Verilator
    verilogToCpp(prefix, testDir, Nil, harness, suppressVcd = true) #&&
    cppToExe(prefix, testDir) !
    loggingProcessLogger
    // Run binary
    if (!executeExpectingSuccess(prefix, testDir)) {
      throw new Exception("Test failed!") with scala.util.control.NoStackTrace
    }
  }
}


class LegalizeAndReductionsTransformSpec extends AnyFlatSpec {

  import LegalizeAndReductionsTransformSpec._

  behavior of "LegalizeAndReductionsTransform"

  private val tests =
    //   name                      primop  input                          expected  width
    Test("andreduce_ones",         "andr", BigInt("1"*68, 2),             1) ::
    Test("andreduce_zero",         "andr", 0,                             0, Some(68)) ::
    Test("orreduce_ones",          "orr",  BigInt("1"*68, 2),             1) ::
    Test("orreduce_high_one",      "orr",  BigInt("1" + "0"*67, 2),       1) ::
    Test("orreduce_zero",          "orr",  0,                             0, Some(68)) ::
    Test("xorreduce_high_one",     "xorr", BigInt("1" + "0"*67, 2),       1) ::
    Test("xorreduce_high_low_one", "xorr", BigInt("1" + "0"*66 + "1", 2), 0) ::
    Test("xorreduce_zero",         "xorr", 0,                             0, Some(68)) ::
    Nil

  for (test <- tests) {
    it should s"support ${test.name}" in {
      executeTest(test)
    }
  }

}
