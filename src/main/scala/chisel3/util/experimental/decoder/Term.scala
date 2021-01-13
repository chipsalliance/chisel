package chisel3.util.experimental.decoder

private class Term(val value: BigInt, val mask: BigInt = 0) {
  var prime = true

  def covers(x: Term): Boolean = ((value ^ x.value) &~ mask | x.mask &~ mask).signum == 0

  def intersects(x: Term): Boolean = ((value ^ x.value) &~ mask &~ x.mask).signum == 0

  override def equals(that: Any): Boolean = that match {
    case x: Term => x.value == value && x.mask == mask
    case _ => false
  }

  override def hashCode: Int = value.toInt

  def <(that: Term): Boolean = value < that.value || value == that.value && mask < that.mask

  def similar(x: Term): Boolean = {
    val diff = value - x.value
    mask == x.mask && value > x.value && (diff & diff - 1) == 0
  }

  def merge(x: Term): Term = {
    prime = false
    x.prime = false
    val bit = value - x.value
    new Term(value &~ bit, mask | bit)
  }

  override def toString: String = value.toString(16) + "-" + mask.toString(16) + (if (prime) "p" else "")
}
