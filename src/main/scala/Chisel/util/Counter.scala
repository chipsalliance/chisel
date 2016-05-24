// See LICENSE for license details.

package Chisel

/** A counter module
  * @param n number of counts before the counter resets (or one more than the
  * maximum output value of the counter), need not be a power of two
  */
class Counter(val n: Int) {
  require(n >= 0)
  val value = if (n > 1) Reg(init=0.asUInt(log2Up(n))) else 0.asUInt
  /** Increment the counter, returning whether the counter currently is at the
    * maximum and will wrap. The incremented value is registered and will be
    * visible on the next cycle.
    */
  def inc(): Bool = {
    if (n > 1) {
      val wrap = value === (n-1).asUInt
      value := value + 1.asUInt
      if (!isPow2(n)) {
        when (wrap) { value := 0.asUInt }
      }
      wrap
    } else {
      true.asBool
    }
  }
}

/** Counter Object
  * Example Usage:
  * {{{ val countOn = Bool(true) // increment counter every clock cycle
  * val myCounter = Counter(countOn, n)
  * when ( myCounter.value === 3.asUInt ) { ... } }}}*/
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
