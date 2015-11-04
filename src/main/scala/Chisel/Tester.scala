// See LICENSE for license details.

package Chisel

import scala.util.Random

class Tester[+T <: Module](c: T, isTrace: Boolean = true) {
  def t: Int = 0
  var ok: Boolean = true  // TODO: get rid of this

  def rnd: Random = new Random()

  def peek(data: Bits): BigInt = 0
  def poke(data: Bits, x: BigInt): Unit = {}
  def expect(data: Bits, expected: BigInt): Boolean = true
  def step(n: Int): Unit = {}

  // TODO: unify and disambiguate expect(...)
  def expect(ok: Boolean, failureMsg: String): Boolean = true
}

object chiselMainOld {
  val wrapped = true
  val unwrapped = false

  def apply[T <: Module](args: Array[String], gen: () => T): T = gen()

  def apply[T <: Module](args: Array[String], gen: () => T, ftester: T => Tester[T]): T = gen()

  // Assumes gen needs to be wrapped in Module()
  def run[T <: Module] (args: Array[String], gen: () => T): T = gen()

  def run[T <: Module] (args: Array[String], gen: () => T, ftester: T => Tester[T]): T = gen()
}

object chiselMainTest {
  def apply[T <: Module](args: Array[String], gen: () => T)(tester: T => Tester[T]): T =
    chiselMainOld(args, gen, tester)
}
