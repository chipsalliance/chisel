// See LICENSE for license details.
package chisel3.iotesters

import chisel3.internal.InstanceId
import java.io.PrintStream

/**
  * define interface for ClassicTester backend implementations such as verilator and firrtl interpreter
  */

private[iotesters] abstract class Backend(private[iotesters] val _seed: Long = System.currentTimeMillis) {
  val rnd = new scala.util.Random(_seed)

  def poke(signal: InstanceId, value: BigInt, off: Option[Int])
          (implicit logger: PrintStream, verbose: Boolean, base: Int): Unit

  def peek(signal: InstanceId, off: Option[Int])
          (implicit logger: PrintStream, verbose: Boolean, base: Int): BigInt

  def poke(path: String, value: BigInt)
          (implicit logger: PrintStream, verbose: Boolean, base: Int): Unit

  def peek(path: String)
          (implicit logger: PrintStream, verbose: Boolean, base: Int): BigInt

  def expect(signal: InstanceId, expected: BigInt)
            (implicit logger: PrintStream, verbose: Boolean, base: Int): Boolean =
    expect(signal, expected, "")

  def expect(signal: InstanceId, expected: BigInt, msg: => String)
            (implicit logger: PrintStream, verbose: Boolean, base: Int): Boolean

  def expect(path: String, expected: BigInt)
            (implicit logger: PrintStream, verbose: Boolean, base: Int): Boolean =
    expect(path, expected, "")

  def expect(path: String, expected: BigInt, msg: => String)
            (implicit logger: PrintStream, verbose: Boolean, base: Int): Boolean

  def step(n: Int)(implicit logger: PrintStream): Unit

  def reset(n: Int): Unit

  def finish(implicit logger: PrintStream): Unit
}


