package chisel3.util.experimental.decoder

/** Term: boolean expression that can express bit in a bit vector to be "assert", "not asserted" or "don't care".
  *
  * @todo add val width
  *       no more Term: add function to [[chisel3.util.BitPat]]
  *       remove prime
  * @param value Literal bits, 1 -> 1, 0 -> 0, ? -> 0
  * @param mask  Mask bits, care -> 0, don't care -> 1
  * @example Suppose `F(a, b, c, d)` is a boolean function with four inputs.
  *          Then `Term(01?1)` represents `F(0, 1, ?, 1)` which in turn represents `F(0, 1, 0, 1) and F(0, 1, 1, 1)`.
  */
private class Term(val value: BigInt, val mask: BigInt = 0) {
  // I think this requirement should be satisfied, which also is a good self-explanation to user(delete this before upstreaming)
  require((value & mask) == 0, "The literal value of don't care bits cannot be 1.")

  /** a `var` indicate whether this term is a [[https://en.wikipedia.org/wiki/Implicant#Prime_implicant]]. */
  var prime = true

  /** Function to check all bits in `this` cover the correspond position in `x`.
    *
    * Rule to define coverage relationship among `0`, `1` and `?`:
    *   1. '?' covers '0' and '1', '0' covers '0', '1' covers '1'
    *   1. '1' doesn't cover '?', '1' doesn't cover '0'
    *   1. '0' doesn't cover '?', '0' doesn't cover '1'
    *
    * For all bits that `this` don't care, `that` can be `0`, `1`, `?`
    * For all bits that `this` care, `that` must be the same value and not masked.
    * {{{
    *    (mask & -1) | ((~mask) & ((value xnor x.value) & ~x.mask)) = -1
    * -> mask | ((~mask) & ((value xnor x.value) & ~x.mask)) = -1
    * -> mask | ((value xnor x.value) & ~x.mask) = -1
    * -> (~mask) & ~((value xnor x.value) & ~x.mask) = 0
    * -> (~mask) & (~(value xnor x.value) | x.mask) = 0
    * -> (~mask) & ((value ^^ x.value) | x.mask) = 0
    * -> ((value ^ x.value) &~ mask | x.mask &~ mask) = 0
    * }}}
    *
    * @param x to check is covered by `this` or not.
    * @return Whether `this` covers `x`
    */
  def covers(x: Term): Boolean = ((value ^ x.value) &~ mask | x.mask &~ mask).signum == 0

  /** Check whether two terms have the same value on all of the cared bits (intersection).
    *
    * {{{
    * value ^^ x.value                                       // bits that are different
    * (bits that are different) &~ mask                      // bits that are different and `this` care
    * (bits that are different and `this` care) &~ x.mask    // bits that are different and `both` care
    * (bits that are different and both care) == 0           // no (bits that are different and we both care) exists
    * no (bits that are different and we both care) exists   // all cared bits are the same, two terms intersect
    * }}}
    *
    * @param x Term to be checked with
    * @return Whether two terms intersect
    */
  def intersects(x: Term): Boolean = ((value ^ x.value) &~ mask &~ x.mask).signum == 0

  /** Two terms equal only when both of their values and masks are the same.
    *
    * @param that Term to be checked with
    * @return Whether two terms are equal
    */
  override def equals(that: Any): Boolean = that match {
    case x: Term => x.value == value && x.mask == mask
    case _ => false
  }

  /** if value equal for cache match in Decoder.
    *
    * @todo benchmark this.
    * @note for faster cache lookup, query haskCode in Map firstly, then use [[equals]] to check.
    */
  override def hashCode: Int = value.toInt

  /** "Smaller" comparator
    * If two terms have different value, then their order is determined by the value, or by the mask.
    *
    * @todo see [[Ordered]]
    * @param that Term to be compared with
    * @return Whether this term is smaller than the other
    */
  def <(that: Term): Boolean = value < that.value || value == that.value && mask < that.mask

  /** Merge two similar terms (when simplifying)
    * Rule of merging: '0' and '1' merge to '?'
    *
    * @todo return Option[Term], use flatMap.
    * @param x Term to be merged with
    * @return A new term representing the merge result
    */
  def merge(x: Term): Term = {
    require(similar(x), s"merge is only reasonable when $this similar $x")

    // if two term can be merged, then they both are not prime implicants.
    prime = false
    x.prime = false
    val bit = value - x.value
    new Term(value &~ bit, mask | bit)
  }

  /** Check whether two terms are similar.
    * Two terms are "similar" when they satisfy all the following rules:
    *   1. have the same mask ('?'s are at the same positions)
    *   1. values only differ by one bit
    *   1. the bit at the differed position of this term is '1' (that of the other term is '0')
    *
    * @example this = 11?0, x = 10?0 -> similar
    * @example this = 11??, x = 10?0 -> not similar, violated rule 1
    * @example this = 11?1, x = 10?0 -> not similar, violated rule 2
    * @example this = 10?0, x = 11?0 -> not similar, violated rule 3
    * @param x Term to be checked with
    * @return Whether this term is similar to the other
    */
  def similar(x: Term): Boolean = {
    val diff = value - x.value
    mask == x.mask && value > x.value && (diff & diff - 1) == 0
  }

  override def toString: String = value.toString(16) + "-" + mask.toString(16) + (if (prime) "p" else "")
}
