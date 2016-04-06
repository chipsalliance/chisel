// See LICENSE for license details.

package Chisel

/** A counter module
  * @param n number of counts before the counter resets (or one more than the
  * maximum output value of the counter), need not be a power of two
  */
class Counter(val n: Int) {
  val value = if (n == 1) UInt(0) else Reg(init=UInt(0, log2Up(n)))
  /** Increment the counter, returning whether the counter currently is at the
    * maximum and will wrap. The incremented value is registered and will be
    * visible on the next cycle.
    */
  def inc(): Bool = {
    if (n == 1) {
      Bool(true)
    } else {
      val wrap = value === UInt(n-1)
      value := value + UInt(1)
      if (!isPow2(n))
        when (wrap) { value := UInt(0) }
      wrap
    }
  }
}

/** Counter Object
  * Example Usage:
  * {{{ val countOn = Bool(true) // increment counter every clock cycle
  * val myCounter = Counter(countOn, n)
  * when ( myCounter.value === UInt(3) ) { ... } }}}*/
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
