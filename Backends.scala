// See LICENSE for license details.
package chisel.iotesters

import chisel.internal.HasId

/**
  * define interface for ClassicTester backend implementations such as verilator and firrtl interpreter
  */

abstract class Backend(_seed: Long = System.currentTimeMillis) {
  val rnd = new scala.util.Random(_seed)

  def poke(signal: HasId, value: BigInt, off: Option[Int]): Unit

  def peek(signal: HasId, off: Option[Int]): BigInt

  def poke(path: String, value: BigInt): Unit

  def peek(path: String): BigInt

  def expect(signal: HasId, expected: BigInt, msg: => String = "") : Boolean

  def step(n: Int): Unit

  def reset(n: Int = 1): Unit

  def finish: Unit
}


