// See LICENSE for license details.

package chisel3.util

import chisel3._
//import chisel3.core.ExplicitCompileOptions.Strict

/** A counter module
 *
  * @param n number of counts before the counter resets (or one more than the
  * maximum output value of the counter), need not be a power of two
  */
class Counter(val n: Int, val init: Int = 0) {
  require(n >= 0)
  require(init < n || (n == 0 && init == 0))
  val value = if (n > 1) Reg(init=init.U(log2Up(n).W)) else 0.U

  /** Increment the counter, returning whether the counter currently is at the
    * maximum and will wrap. The incremented value is registered and will be
    * visible on the next cycle.
    */
  def inc(): Bool = {
    if (n > 1) {
      val wrap = value === (n-1).asUInt
      value := value + 1.U
      if (!isPow2(n)) {
        when (wrap) { value := 0.U }
      }
      wrap
    } else {
      true.B
    }
  }

  /** Reset the counter to the init value.
   */
  def restart(): Unit = {
    if (n > 1) value := init.U
  }
}

object Counter
{
  /** Instantiate a [[Counter! counter]] with the specified number of counts.
    */
  def apply(n: Int): Counter = new Counter(n)

  /** Instantiate a [[Counter! counter]] with the specified number of counts and an init value.
   */
  def apply(n: Int, init: Int) = new Counter(n, init)

  /** Instantiate a [[Counter! counter]] with the specified number of counts and a gate.
   *
    * @param cond condition that controls whether the counter increments this cycle
    * @param n number of counts before the counter resets
    * @return tuple of the counter value and whether the counter will wrap (the value is at
    * maximum and the condition is true).
    *
    * @example {{{
    * val countOn = true.B // increment counter every clock cycle
    * val (counterValue, counterWrap) = Counter(countOn, 4)
    * when (counterValue === 3.U) {
    *   ...
    * }
    * }}}
    */
  def apply(cond: Bool, n: Int): (UInt, Bool) = {
    val c = new Counter(n)
    var wrap: Bool = null
    when (cond) { wrap = c.inc() }
    (c.value, cond && wrap)
  }

  /** Instantiate a [[Counter! counter]] with the specified number of counts, a gate, and a restart.
   */
  def apply(cond: Bool, n: Int, restart: Bool): (UInt, Bool) = apply(cond, n, restart, 0)

  /** Instantiate a [[Counter! counter]] with the specified number of counts, a gate, a restart, and an init.
   */
  def apply(cond: Bool, n: Int, restart: Bool, init: Int): (UInt, Bool) = {
    val c = new Counter(n, init)
    var wrap: Bool = null
    when (cond) { wrap = c.inc() }
    when (restart && cond) { c.restart() }
    (c.value, cond && wrap)
  }
}
