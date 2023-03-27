// SPDX-License-Identifier: Apache-2.0

package chisel3.util.experimental.decode

import chisel3.util.BitPat
import chisel3.InternalErrorException

import scala.annotation.tailrec
import scala.math.Ordered.orderingToOrdered
import scala.language.implicitConversions

/** A [[Minimizer]] implementation to use Quine-Mccluskey algorithm to minimize the [[TruthTable]].
  *
  * This algorithm can always find the best solution, but is a NP-Complete algorithm,
  * which means, for large-scale [[TruthTable]] minimization task, it will be really slow,
  * and might run out of memory of JVM stack.
  *
  * In this situation, users should consider switch to [[EspressoMinimizer]],
  * which uses heuristic algorithm providing a sub-optimized result.
  */
object QMCMinimizer extends Minimizer {
  private implicit def toImplicant(x: BitPat): Implicant = new Implicant(x)

  private class Implicant(val bp: BitPat) {
    var isPrime: Boolean = true

    def width = bp.getWidth

    override def equals(that: Any): Boolean = that match {
      case x: Implicant => bp.value == x.bp.value && bp.mask == x.bp.mask
      case _ => false
    }

    override def hashCode = bp.value.toInt

    /** Check whether two implicants have the same value on all of the cared bits (intersection).
      *
      * {{{
      * value ^^ x.value                                       // bits that are different
      * (bits that are different) & x.mask                     // bits that are different and `this` care
      * (bits that are different and `this` care) & y.mask     // bits that are different and `both` care
      * (bits that are different and both care) == 0           // no (bits that are different and we both care) exists
      * no (bits that are different and we both care) exists   // all cared bits are the same, two terms intersect
      * }}}
      *
      * @param y Implicant to be checked with
      * @return Whether two implicants intersect
      */
    def intersects(y: Implicant): Boolean = ((bp.value ^ y.bp.value) & bp.mask & y.bp.mask) == 0

    /** Check whether two implicants are similar.
      * Two implicants are "similar" when they satisfy all the following rules:
      *   1. have the same mask ('?'s are at the same positions)
      *   1. values only differ by one bit
      *   1. the bit at the differed position of this term is '1' (that of the other term is '0')
      *
      * @example this = 11?0, x = 10?0 -> similar
      * @example this = 11??, x = 10?0 -> not similar, violated rule 1
      * @example this = 11?1, x = 10?0 -> not similar, violated rule 2
      * @example this = 10?0, x = 11?0 -> not similar, violated rule 3
      * @param y Implicant to be checked with
      * @return Whether this term is similar to the other
      */
    def similar(y: Implicant): Boolean = {
      val diff = bp.value - y.bp.value
      bp.mask == y.bp.mask && bp.value > y.bp.value && (diff & diff - 1) == 0
    }

    /** Merge two similar implicants
      * Rule of merging: '0' and '1' merge to '?'
      *
      * @param y Term to be merged with
      * @return A new term representing the merge result
      */
    def merge(y: Implicant): Implicant = {
      require(similar(y), s"merge is only reasonable when $this is similar to $y")

      // if two term can be merged, then they both are not prime implicants.
      isPrime = false
      y.isPrime = false
      val bit = bp.value - y.bp.value
      new BitPat(bp.value &~ bit, bp.mask &~ bit, width)
    }

    /** Check all bits in `x` cover the correspond position in `y`.
      *
      * Rule to define coverage relationship among `0`, `1` and `?`:
      *   1. '?' covers '0' and '1', '0' covers '0', '1' covers '1'
      *   1. '1' doesn't cover '?', '1' doesn't cover '0'
      *   1. '0' doesn't cover '?', '0' doesn't cover '1'
      *
      * For all bits that `x` don't care, `y` can be `0`, `1`, `?`
      * For all bits that `x` care, `y` must be the same value and not masked.
      * {{{
      *    (~x.mask & -1) | ((x.mask) & ((x.value xnor y.value) & y.mask)) = -1
      * -> ~x.mask | ((x.mask) & ((x.value xnor y.value) & y.mask)) = -1
      * -> ~x.mask | ((x.value xnor y.value) & y.mask) = -1
      * -> x.mask & ~((x.value xnor y.value) & y.mask) = 0
      * -> x.mask & (~(x.value xnor y.value) | ~y.mask) = 0
      * -> x.mask & ((x.value ^ y.value) | ~y.mask) = 0
      * -> ((x.value ^ y.value) & x.mask | ~y.mask & x.mask) = 0
      * }}}
      *
      * @param y to check is covered by `x` or not.
      * @return Whether `x` covers `y`
      */
    def covers(y: Implicant): Boolean = ((bp.value ^ y.bp.value) & bp.mask | ~y.bp.mask & bp.mask) == 0

    override def toString = (if (!isPrime) "Non" else "") + "Prime" + bp.toString.replace("BitPat", "Implicant")
  }

  /**
    * If two terms have different value, then their order is determined by the value, or by the mask.
    */
  private implicit def ordering: Ordering[Implicant] = new Ordering[Implicant] {
    override def compare(x: Implicant, y: Implicant): Int =
      if (x.bp.value < y.bp.value || x.bp.value == y.bp.value && x.bp.mask > y.bp.mask) -1 else 1
  }

  /** Calculate essential prime implicants based on previously calculated prime implicants and all implicants.
    *
    * @param primes    Prime implicants
    * @param minterms All implicants
    * @return (a, b, c)
    *         a: essential prime implicants
    *         b: nonessential prime implicants
    *         c: implicants that are not cover by any of the essential prime implicants
    */
  private def getEssentialPrimeImplicants(
    primes:   Seq[Implicant],
    minterms: Seq[Implicant]
  ): (Seq[Implicant], Seq[Implicant], Seq[Implicant]) = {
    // primeCovers(i): implicants that `prime(i)` covers
    val primeCovers = primes.map(p => minterms.filter(p.covers))
    // eliminate prime implicants that can be covered by other prime implicants
    for (((icover, pi), i) <- (primeCovers.zip(primes)).zipWithIndex) {
      for (((jcover, pj), _) <- (primeCovers.zip(primes)).zipWithIndex.drop(i + 1)) {
        // we prefer prime implicants with wider implicants coverage
        if (icover.size > jcover.size && jcover.forall(pi.covers)) {
          // calculate essential prime implicants with `pj` eliminated from prime implicants table
          return getEssentialPrimeImplicants(primes.filter(_ != pj), minterms)
        }
      }
    }

    // implicants that only one prime implicant covers
    val essentiallyCovered = minterms.filter(t => primes.count(_.covers(t)) == 1)
    // essential prime implicants, prime implicants that covers only one implicant
    val essential = primes.filter(p => essentiallyCovered.exists(p.covers))
    // {nonessential} = {prime implicants} - {essential prime implicants}
    val nonessential = primes.filterNot(essential contains _)
    // implicants that no essential prime implicants covers
    val uncovered = minterms.filterNot(t => essential.exists(_.covers(t)))
    if (essential.isEmpty || uncovered.isEmpty)
      (essential, nonessential, uncovered)
    else {
      // now there are implicants (`uncovered`) that are covered by multiple nonessential prime implicants (`nonessential`)
      // need to reduce prime implicants
      val (a, b, c) = getEssentialPrimeImplicants(nonessential, uncovered)
      (essential ++ a, b, c)
    }
  }

  /** Use [[https://en.wikipedia.org/wiki/Petrick%27s_method]] to select a [[Seq]] of nonessential prime implicants
    * that covers all implicants that are not covered by essential prime implicants.
    *
    * @param implicants Nonessential prime implicants
    * @param minterms   Implicants that are not covered by essential prime implicants
    * @return Selected nonessential prime implicants
    */
  private def getCover(implicants: Seq[Implicant], minterms: Seq[Implicant]): Seq[Implicant] = {

    /* Calculate the implementation cost (using comparators) of a list of implicants, more don't cares is cheaper
     *
     * @param cover Implicant list
     * @return How many comparators need to implement this list of implicants
     */
    def getCost(cover: Seq[Implicant]): Int = cover.map(_.bp.mask.bitCount).sum

    /* Determine if one combination of prime implicants is cheaper when implementing as comparators.
     * Shorter term list is cheaper, term list with more don't cares is cheaper (less comparators)
     *
     * @param a    Operand a
     * @param b    Operand b
     * @return `a` < `b`
     */
    def cheaper(a: Seq[Implicant], b: Seq[Implicant]): Boolean = {
      val ca = getCost(a)
      val cb = getCost(b)

      /* If `a` < `b`
       *
       * Like comparing the dictionary order of two strings.
       * Define `a` < `b` if both `a` and `b` are empty.
       *
       * @param a Operand a
       * @param b Operand b
       * @return `a` < `b`
       */
      @tailrec
      def listLess(a: Seq[Implicant], b: Seq[Implicant]): Boolean =
        b.nonEmpty && (a.isEmpty || a.head < b.head || a.head == b.head && listLess(a.tail, b.tail))

      ca < cb || ca == cb && listLess(a.sortWith(_ < _), b.sortWith(_ < _))
    }

    // if there are no implicant that is not covered by essential prime implicants, which means all implicants are
    // covered by essential prime implicants, there is no need to apply Petrick's method
    if (minterms.nonEmpty) {
      // cover(i): nonessential prime implicants that covers `minterms(i)`
      val cover = minterms.map(m => implicants.filter(_.covers(m)))
      // all subsets of `cover`, NP algorithm, O(2 ^ len(cover))
      val all = cover.tail.foldLeft(cover.head.map(Set(_)))((c0, c1) => c0.flatMap(a => c1.map(a + _)))
      all.map(_.toList).reduceLeft((a, b) => if (cheaper(a, b)) a else b)
    } else
      Seq[Implicant]()
  }

  def minimize(table: TruthTable): TruthTable = {
    require(table.table.nonEmpty, "Truth table must not be empty")

    // extract decode table to inputs and outputs
    val (inputs, outputs) = table.table.unzip

    require(
      outputs.map(_.getWidth == table.default.getWidth).reduce(_ && _),
      "All output BitPats and default BitPat must have the same length"
    )
    require(
      if (inputs.toSeq.length > 1) inputs.tail.map(_.width == inputs.head.width).reduce(_ && _) else true,
      "All input BitPats must have the same length"
    )

    // make sure no two inputs specified in the truth table intersect
    for (t <- inputs.tails; if t.nonEmpty)
      for (u <- t.tail)
        require(!t.head.intersects(u), "truth table entries " + t.head + " and " + u + " overlap")

    // number of inputs
    val n = inputs.head.width
    // number of outputs
    val m = outputs.head.getWidth

    // for all outputs
    val minimized = (0 until m).flatMap(i => {
      val outputBp = BitPat("b" + "?" * (m - i - 1) + "1" + "?" * i)

      // Minterms, implicants that makes the output to be 1
      val mint: Seq[Implicant] =
        table.table.filter { case (_, t) => t.mask.testBit(i) && t.value.testBit(i) }.map(_._1).map(toImplicant)
      // Maxterms, implicants that makes the output to be 0
      val maxt: Seq[Implicant] =
        table.table.filter { case (_, t) => t.mask.testBit(i) && !t.value.testBit(i) }.map(_._1).map(toImplicant)
      // Don't cares, implicants that can produce either 0 or 1 as output
      val dc: Seq[Implicant] = table.table.filter { case (_, t) => !t.mask.testBit(i) }.map(_._1).map(toImplicant)

      val (implicants, defaultToDc) = table.default match {
        case x if x.mask.testBit(i) && !x.value.testBit(i) => // default to 0
          (mint ++ dc, false)
        case x if x.mask.testBit(i) && x.value.testBit(i) => // default to 1
          (maxt ++ dc, false)
        case x if !x.mask.testBit(i) => // default to ?
          (mint, true)
        case _ => throw new InternalErrorException("Match error: table.default=${table.default}")
      }

      implicants.foreach(_.isPrime = true)
      val cols = (0 to n).reverse.map(b => implicants.filter(b == _.bp.mask.bitCount))
      val mergeTable = cols.map(c => (0 to n).map(b => collection.mutable.Set(c.filter(b == _.bp.value.bitCount): _*)))

      // O(n ^ 3)
      for (i <- 0 to n) {
        for (j <- 0 until n - i) {
          mergeTable(i)(j).foreach(a =>
            mergeTable(i + 1)(j) ++= mergeTable(i)(j + 1).filter(_.similar(a)).map(_.merge(a))
          )
        }
        if (defaultToDc) {
          for (j <- 0 until n - i) {
            for (a <- mergeTable(i)(j).filter(_.isPrime)) {
              if (a.bp.mask.testBit(i) && !a.bp.value.testBit(i)) {
                // this bit is `0`
                val t = new BitPat(a.bp.value.setBit(i), a.bp.mask, a.width)
                if (!maxt.exists(_.intersects(t))) mergeTable(i + 1)(j) += t.merge(a)
              }
            }
            for (a <- mergeTable(i)(j + 1).filter(_.isPrime)) {
              if (a.bp.mask.testBit(i) && a.bp.value.testBit(i)) {
                // this bit is `1`
                val t = new BitPat(a.bp.value.clearBit(i), a.bp.mask, a.width)
                if (!maxt.exists(_.intersects(t))) mergeTable(i + 1)(j) += a.merge(t)
              }
            }
          }
        }
      }

      val primeImplicants = mergeTable.flatten.flatten.filter(_.isPrime).sortWith(_ < _)

      // O(len(primeImplicants) ^ 4)
      val (essentialPrimeImplicants, nonessentialPrimeImplicants, uncoveredImplicants) =
        getEssentialPrimeImplicants(primeImplicants, implicants)

      (essentialPrimeImplicants ++ getCover(nonessentialPrimeImplicants, uncoveredImplicants)).map(a =>
        (a.bp, outputBp)
      )
    })

    // special case for 0 and DontCare, if output is not couple to input
    if (minimized.isEmpty)
      table.copy(
        Seq(
          (
            BitPat(s"b${"?" * table.inputWidth}"),
            BitPat(s"b${"0" * table.outputWidth}")
          )
        )
      )
    else
      minimized.tail.foldLeft(table.copy(table = Seq(minimized.head))) {
        case (tb, t) =>
          if (tb.table.exists(x => x._1 == t._1)) {
            tb.copy(table = tb.table.map {
              case (k, v) =>
                if (k == t._1) {
                  def ones(bitPat: BitPat) = bitPat.rawString.zipWithIndex.collect { case ('1', x) => x }
                  (
                    k,
                    BitPat(
                      "b" + (0 until v.getWidth)
                        .map(i => if ((ones(v) ++ ones(t._2)).contains(i)) "1" else "?")
                        .mkString
                    )
                  )
                } else (k, v)
            })
          } else {
            tb.copy(table = tb.table :+ t)
          }
      }
  }
}
