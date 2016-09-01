// See LICENSE for license details.

package chisel3.util

import chisel3._
import chisel3.Strict.CompileOptions

/** A counter module
  * @param n number of counts before the counter resets (or one more than the
  * maximum output value of the counter), need not be a power of two
  */
class Counter(val n: Int) {
  require(n >= 0)
  val value = if (n > 1) Reg(init=UInt(0, log2Up(n))) else UInt(0)
  /** Increment the counter, returning whether the counter currently is at the
    * maximum and will wrap. The incremented value is registered and will be
    * visible on the next cycle.
    */
  def inc(): Bool = {
    if (n > 1) {
      val wrap = value === UInt(n-1)
      value := value + UInt(1)
      if (!isPow2(n)) {
        when (wrap) { value := UInt(0) }
      }
      wrap
    } else {
      Bool(true)
    }
  }
}

/** Counter Object
  * Example Usage:
  * {{{ val countOn = Bool(true) // increment counter every clock cycle
  * val (myCounterValue, myCounterWrap) = Counter(countOn, n)
  * when ( myCounterValue === UInt(3) ) { ... } }}}*/
object Counter
{
  def apply(n: Int): Counter = new Counter(n)
  def apply(cond: Bool, n: Int): (UInt, Bool) = {
    val c = new Counter(n)
    var wrap: Bool = null
    when (cond) { wrap = c.inc() }
    (c.value, cond && wrap)
  }
}
