// See LICENSE for license details.

package chisel3.util

import chisel3._
import chisel3.internal.naming.chiselName  // can't use chisel3_ version because of compile order

/** Used to generate an inline (logic directly in the containing Module, no internal Module is created)
  * hardware counter.
  *
  * Typically instantiated with apply methods in [[Counter$ object Counter]]
  *
  * Does not create a new Chisel Module
  *
  * @example {{{
  *   val countOn = true.B // increment counter every clock cycle
  *   val (counterValue, counterWrap) = Counter(countOn, 4)
  *   when (counterValue === 3.U) {
  *     ...
  *   }
  * }}}
  */
@chiselName
class Counter private (r: Range) {
  require(r.length > 0, s"Counter range cannot be empty, got: $r")
  require(r.start >= 0 && r.end >= 0, s"Counter range must be positive, got: $r")

  private lazy val delta = math.abs(r.step)
  private lazy val width = math.max(log2Up(r.last + 1), log2Up(r.head + 1))

  /** Creates a counter with the specified number of steps.
    *
    * @param n number of steps before the counter resets
    */
  def this(n: Int) { this(0 until math.max(1, n)) }

  /** The current value of the counter. */
  val value = if (r.length > 1) RegInit(r.head.U(width.W)) else r.head.U

  /** The range of the counter values. */
  def range: Range = r

  /** Increments the counter by a step.
    *
    * @note The incremented value is registered and will be visible on the next clock cycle
    * @return whether the counter will wrap on the next clock cycle
    */
  def inc(): Bool = {
    if (r.length > 1) {
      val wrap = value === r.last.U

      if (r.step > 0) {
        // Increasing range
        value := value + delta.U
      } else {
        // Decreasing range
        value := value - delta.U
      }

      // We only need to explicitly wrap counters that don't start at zero, or
      // end on a power of two. Otherwise we just let the counter overflow
      // naturally to avoid wasting an extra mux.
      if (!(r.head == 0 && isPow2(r.last + delta))) {
        when(wrap) { value := r.head.U }
      }

      wrap
    } else {
      true.B
    }
  }
}

object Counter
{
  /** Instantiate a [[Counter! counter]] with the specified number of counts.
    */
  def apply(n: Int): Counter = new Counter(n)

  /** Instantiate a [[Counter! counter]] with the specified number of counts and a gate.
   *
    * @param cond condition that controls whether the counter increments this cycle
    * @param n number of counts before the counter resets
    * @return tuple of the counter value and whether the counter will wrap (the value is at
    * maximum and the condition is true).
    */
  @chiselName
  def apply(cond: Bool, n: Int): (UInt, Bool) = {
    val c = new Counter(n)
    val wrap = WireInit(false.B)
    when (cond) { wrap := c.inc() }
    (c.value, wrap)
  }

  /** Creates a counter that steps through a specified range of values.
    *
    * @param r the range of counter values
    * @param enable controls whether the counter increments this cycle
    * @return tuple of the counter value and whether the counter will wrap (the value is at
    * maximum and the condition is true).
    */
  @chiselName
  def apply(r: Range, enable: Bool = true.B): (UInt, Bool) = {
    val c = new Counter(r)
    val wrap = WireInit(false.B)
    when (enable) { wrap := c.inc() }
    (c.value, wrap)
  }
}
