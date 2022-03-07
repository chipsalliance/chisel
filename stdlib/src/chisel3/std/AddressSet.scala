package chisel3.std

import chisel3._
import chisel3.util.log2Floor

/** AddressSets specify the address space managed by the manager
  * Base is the base address, and mask are the bits consumed by the manager
  * e.g: base=0x200, mask=0xff describes a device managing 0x200-0x2ff
  * e.g: base=0x1000, mask=0xf0f describes a device managing 0x1000-0x100f, 0x1100-0x110f, ...
  */
case class AddressSet(base: BigInt, mask: BigInt) extends Ordered[AddressSet] {
  // Forbid misaligned base address (and empty sets)
  require((base & mask) == 0, s"Mis-aligned AddressSets are forbidden, got: ${this.toString}")
  require(
    base >= 0,
    s"AddressSet negative base is ambiguous: $base"
  ) // TL2 address widths are not fixed => negative is ambiguous
  // We do allow negative mask (=> ignore all high bits)

  def contains(x: BigInt): Boolean = ((x ^ base) & ~mask) == 0
  def contains(x: UInt):   Bool = ((x ^ base.U).zext & (~mask).S) === 0.S

  // turn x into an address contained in this set
  def legalize(x: UInt): UInt = base.U | (mask.U & x)

  // overlap iff bitwise: both care (~mask0 & ~mask1) => both equal (base0=base1)
  def overlaps(x: AddressSet): Boolean = (~(mask | x.mask) & (base ^ x.base)) == 0
  // contains iff bitwise: x.mask => mask && contains(x.base)
  def contains(x: AddressSet): Boolean = ((x.mask | (base ^ x.base)) & ~mask) == 0

  // The number of bytes to which the manager must be aligned
  def alignment: BigInt = (mask + 1) & ~mask
  // Is this a contiguous memory range
  def contiguous: Boolean = alignment == mask + 1

  def finite: Boolean = mask >= 0
  def max: BigInt = { require(finite, "Max cannot be calculated on infinite mask"); base | mask }

  // Widen the match function to ignore all bits in imask
  def widen(imask: BigInt): AddressSet = AddressSet(base & ~imask, mask | imask)

  // Return an AddressSet that only contains the addresses both sets contain
  def intersect(x: AddressSet): Option[AddressSet] = {
    if (!overlaps(x)) {
      None
    } else {
      val r_mask = mask & x.mask
      val r_base = base | x.base
      Some(AddressSet(r_base, r_mask))
    }
  }

  def subtract(x: AddressSet): Seq[AddressSet] = {
    intersect(x) match {
      case None => Seq(this)
      case Some(remove) =>
        AddressSet.enumerateBits(mask & ~remove.mask).map { bit =>
          val nmask = (mask & (bit - 1)) | remove.mask
          val nbase = (remove.base ^ bit) & ~nmask
          AddressSet(nbase, nmask)
        }
    }
  }

  // AddressSets have one natural Ordering (the containment order, if contiguous)
  def compare(x: AddressSet): Int = {
    val primary = (this.base - x.base).signum // smallest address first
    val secondary = (x.mask - this.mask).signum // largest mask first
    if (primary != 0) primary else secondary
  }

  // We always want to see things in hex
  override def toString: String = {
    if (mask >= 0) {
      "AddressSet(0x%x, 0x%x)".format(base, mask)
    } else {
      "AddressSet(0x%x, ~0x%x)".format(base, ~mask)
    }
  }
}

object AddressSet {
  val Everything = AddressSet(0, -1)
  def misaligned(base: BigInt, size: BigInt, tail: Seq[AddressSet] = Seq()): Seq[AddressSet] = {
    if (size == 0) tail.reverse
    else {
      val maxBaseAlignment = base & (-base) // 0 for infinite (LSB)
      val maxSizeAlignment = BigInt(1) << log2Floor(size) // MSB of size
      val step =
        if (maxBaseAlignment == 0 || maxBaseAlignment > maxSizeAlignment)
          maxSizeAlignment
        else maxBaseAlignment
      misaligned(base + step, size - step, AddressSet(base, step - 1) +: tail)
    }
  }

  def unify(seq: Seq[AddressSet], bit: BigInt): Seq[AddressSet] = {
    // Pair terms up by ignoring 'bit'
    seq.distinct
      .groupBy(x => x.copy(base = x.base & ~bit))
      .map {
        case (key, seq) =>
          if (seq.size == 1) {
            seq.head // singleton -> unaffected
          } else {
            key.copy(mask = key.mask | bit) // pair - widen mask by bit
          }
      }
      .toList
  }

  def unify(seq: Seq[AddressSet]): Seq[AddressSet] = {
    val bits = seq.map(_.base).foldLeft(BigInt(0))(_ | _)
    AddressSet.enumerateBits(bits).foldLeft(seq) { case (acc, bit) => unify(acc, bit) }.sorted
  }

  def enumerateMask(mask: BigInt): Seq[BigInt] = {
    def helper(id: BigInt, tail: Seq[BigInt]): Seq[BigInt] =
      if (id == mask) (id +: tail).reverse else helper(((~mask | id) + 1) & mask, id +: tail)
    helper(0, Nil)
  }

  def enumerateBits(mask: BigInt): Seq[BigInt] = {
    def helper(x: BigInt): Seq[BigInt] = {
      if (x == 0) {
        Nil
      } else {
        val bit = x & (-x)
        bit +: helper(x & ~bit)
      }
    }
    helper(mask)
  }
}
