// See LICENSE for license details.
package chisel3.iotesters

import java.io.File

import chisel3.{Module, Bits}
import chisel3.internal.HasId

import scala.collection.mutable.HashMap

import firrtl_interpreter.InterpretiveTester

private[iotesters] class FirrtlTerpBackend(
                                           dut: Module,
                                           firrtlIR: String,
                                           verbose: Boolean = true,
                                           logger: java.io.PrintStream = System.out,
                                           _base: Int = 16,
                                           _seed: Long = System.currentTimeMillis) extends Backend(_seed) {
  val interpretiveTester = new InterpretiveTester(firrtlIR)
  reset(5) // reset firrtl interpreter on construction

  val portNames = getDataNames(dut).toMap

  def poke(signal: HasId, value: BigInt, off: Option[Int]): Unit = {
    signal match {
      case port: Bits =>
        val name = portNames(port)
        interpretiveTester.poke(name, value)
        if (verbose) logger println s"  POKE ${name} <- ${bigIntToStr(value, _base)}"
      case _ =>
    }
  }

  def peek(signal: HasId, off: Option[Int]): BigInt = {
    signal match {
      case port: Bits =>
        val name = portNames(port)
        val result = interpretiveTester.peek(name)
        if (verbose) logger println s"  PEEK ${name} -> ${bigIntToStr(result, _base)}"
        result
      case _ => BigInt(rnd.nextInt)
    }
  }

  def expect(signal: HasId, expected: BigInt, msg: => String) : Boolean = {
    signal match {
      case port: Bits =>
        val name = portNames(port)
        val got = interpretiveTester.peek(name)
        val good = got == expected
        if (verbose) logger println s"""${msg}  EXPECT ${name} -> ${bigIntToStr(got, _base)} == ${bigIntToStr(expected, _base)} ${if (good) "PASS" else "FAIL"}"""
        good
      case _ => false
    }
  }

  def poke(path: String, value: BigInt): Unit = {
    assert(false)
  }

  def peek(path: String): BigInt = {
    assert(false)
    BigInt(rnd.nextInt)
  }

  def expect(path: String, expected: BigInt, msg: => String) : Boolean = {
    assert(false)
    false
  }

  def step(n: Int): Unit = {
    interpretiveTester.step(n)
  }

  def reset(n: Int = 1): Unit = {
    interpretiveTester.poke("reset", 1)
    interpretiveTester.step(n)
    interpretiveTester.poke("reset", 0)
  }

  def finish: Unit = Unit
}

private[iotesters] object setupFirrtlTerpBackend {
  def apply[T <: chisel3.Module](dutGen: () => T): (T, Backend) = {
    val rootDirPath = new File(".").getCanonicalPath()
    val testDirPath = s"${rootDirPath}/test_run_dir"
    val dir = new File(testDirPath)
    dir.mkdirs()

    val graph = new CircuitGraph
    val circuit = chisel3.Driver.elaborate(dutGen)
    val dut = (graph construct circuit).asInstanceOf[T]

    // Dump FIRRTL for debugging
    val firrtlIRFilePath = s"${testDirPath}/${circuit.name}.ir"
    chisel3.Driver.dumpFirrtl(circuit, Some(new File(firrtlIRFilePath)))
    val firrtlIR = chisel3.Driver.emit(dutGen)

    (dut, new FirrtlTerpBackend(dut, firrtlIR))
  }
}
