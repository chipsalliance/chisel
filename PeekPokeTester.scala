// See LICENSE for license details.

package chisel.iotesters

import chisel._

import scala.util.Random

// Provides a template to define tester transactions
trait PeekPokeTests {
  def t: Long
  def rnd: Random
  implicit def int(x: Boolean): BigInt
  implicit def int(x: Int):     BigInt
  implicit def int(x: Long):    BigInt
  implicit def int(x: Bits):    BigInt
  def println(msg: String = ""): Unit
  def reset(n: Int): Unit
  def step(n: Int): Unit
  def poke(path: String, x: BigInt): Unit
  def peek(path: String): BigInt
  def poke(signal: Bits, x: BigInt): Unit
  def pokeAt[T <: Bits](signal: Mem[T], x: BigInt, off: Int): Unit
  def peek(signal: Bits): BigInt
  def peekAt[T <: Bits](signal: Mem[T], off: Int): BigInt
  def expect(good: Boolean, msg: => String): Boolean
  def expect(signal: Bits, expected: BigInt, msg: => String = ""): Boolean
  def finish: Boolean
}

abstract class PeekPokeTester[+T <: Module](
                                            val dut: T,
                                            verbose: Boolean = true,
                                            _base: Int = 16,
                                            logFile: Option[String] = chiselMain.context.logFile,
                                            waveform: Option[String] = chiselMain.context.waveform,
                                            _backend: Option[Backend] = None,
                                            _seed: Long = System.currentTimeMillis) {

  implicit def longToInt(x: Long) = x.toInt

  implicit val logger = logFile match {
    case None    => System.out
    case Some(f) => new java.io.PrintStream(f)
  }

  def println(msg: String = "") {
    logger println msg
  }

  /****************************/
  /*** Simulation Interface ***/
  /****************************/
  logger println s"SEED ${_seed}"
  val cmd = chiselMain.context.testCmd.toList ++ (waveform match {
    case None    => Nil
    case Some(f) => logger println s"Waveform: $f" ; List(s"+waveform=$f")
  })
  val backend = _backend getOrElse (
    if (chiselMain.context.isVCS)
      new VCSBackend(dut, cmd, verbose, logger, _base, _seed)
    else
      new VerilatorBackend(dut, cmd, verbose, logger, _base, _seed)
  )

  /********************************/
  /*** Classic Tester Interface ***/
  /********************************/
  /* Simulation Time */
  private var simTime = 0L
  protected[iotesters] def incTime(n: Int) { simTime += n }
  def t = simTime

  /** Indicate a failure has occurred.  */
  private var failureTime = -1L
  private var ok = true
  def fail = if (ok) {
    failureTime = simTime
    ok = false
  }

  val rnd = backend.rnd

  /** Convert a Boolean to BigInt */
  implicit def int(x: Boolean): BigInt = if (x) 1 else 0
  /** Convert an Int to BigInt */
  implicit def int(x: Int):     BigInt = (BigInt(x >>> 1) << 1) | BigInt(x & 1)
  /** Convert a Long to BigInt */
  implicit def int(x: Long):    BigInt = (BigInt(x >>> 1) << 1) | BigInt(x & 1)
  /** Convert Bits to BigInt */
  implicit def int(x: Bits):    BigInt = x.litValue()

  def reset(n: Int = 1) {
    backend.reset(n)
  }

  def step(n: Int) {
    if (verbose) logger println s"STEP ${simTime} -> ${simTime+n}"
    backend.step(n)
    incTime(n)
  }

  def poke(path: String, value: BigInt) = backend.poke(path, value)

  def peek(path: String) = backend.peek(path)

  def poke(signal: Bits, value: BigInt) {
    if (!signal.isLit) backend.poke(signal, value, None)
  }

  def pokeAt[T <: Bits](data: Mem[T], value: BigInt, off: Int): Unit = {
    backend.poke(data, value, Some(off))
  }

  def peek(signal: Bits) = {
    if (!signal.isLit) backend.peek(signal, None) else signal.litValue()
  }

  def peekAt[T <: Bits](data: Mem[T], off: Int): BigInt = {
    backend.peek(data, Some(off))
  }

  def expect (good: Boolean, msg: => String): Boolean = {
    if (verbose) logger println s"""EXPECT ${msg} ${if (good) "PASS" else "FAIL"}"""
    if (!good) fail
    good
  }

  def expect(signal: Bits, expected: BigInt, msg: => String = ""): Boolean = {
    if (!signal.isLit) {
      val good = backend.expect(signal, expected, msg)
      if (!good) fail
      good
    } else expect(signal.litValue() == expected, s"${signal.litValue()} == $expected")
  }

  def finish: Boolean = {
    try {
      backend.finish
    } catch {
      // Depending on load and timing, we may get a TestApplicationException
      //  when the test application exits.
      //  Check the exit value.
      //  Anything other than 0 is an error.
      case e: TestApplicationException => if (e.exitVal != 0) fail
    }
    logger println s"""RAN ${simTime} CYCLES ${if (ok) "PASSED" else s"FAILED FIRST AT CYCLE ${failureTime}"}"""
    ok
  }
}
