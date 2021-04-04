package chisel3.util.experimental.minimizer

import scala.collection.mutable

import chisel3._
import chisel3.util.{BitPat, Cat}

object Minimizer {
  private val caches: mutable.Map[UInt, mutable.Map[String, Bool]] = mutable.Map[UInt, mutable.Map[String, Bool]]()

  def tableToPLA(inputs: UInt, default: BitPat, table: Seq[Seq[BitPat]]): UInt = {
    val cache = caches.getOrElseUpdate(inputs, mutable.Map[String, Bool]())
    val invInputs = ~inputs
    Cat(table.zipWithIndex.map{ case (a, i) =>
      val b: Bool =
        if (a.isEmpty)
          false.B
        else {
          Cat(a.map { t =>
            // share AND plane decode result.
            cache
              .getOrElseUpdate(
                t.toString,
                Cat((0 until t.getWidth).flatMap{ j => // TODO
                  if (t.mask.testBit(j)) {
                    Some(
                      if (t.value.testBit(j)) inputs(j)
                      else invInputs(j)
                    )
                  } else {
                    None
                  }
                })
                  // PLA AND plane
                  .andR()
              )
          })
            // PLA OR plane
            .orR()
      }
      if (default.mask.testBit(i) && default.value.testBit(i)) !b else b
    }.reverse)
  }

  private[minimizer] implicit class Implicant(x: BitPat) {
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
    def intersects(y: BitPat): Boolean = ((x.value ^ y.value) & x.mask & y.mask) == 0

    /** Check if two implicants are the same
      * @param y Implicant to be checked with
      * @return Whether two implicants are the same
      */
    def sameAs(y: BitPat): Boolean = x.mask == y.mask && x.value == y.value

    /** Merge two similar implicants if they are similar
      * Rule of merging: '0' and '1' merge to '?'
      * Two implicants are "similar" when they satisfy all the following rules:
      *   1. have the same mask ('?'s are at the same positions)
      *   1. values only differ by one bit
      *
      * @example this = 11?0, x = 10?0 -> similar
      * @example this = 11??, x = 10?0 -> not similar, violated rule 1
      * @example this = 11?1, x = 10?0 -> not similar, violated rule 2
      * @param y Implicant to be merged with
      * @return Merge result wrapped in [[Some]] if `x` and `y` are similar,
      *         [[None]] otherwise
      */
    def mergeIfSimilar(y: BitPat): Option[BitPat] = {
      val diff = x.value ^ y.value
      if (x.mask == y.mask && diff.bitCount == 1) { // similar
        Some(new BitPat(x.value &~ diff, x.mask &~ diff, x.getWidth))
      } else {
        None
      }
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
    def covers(y: BitPat): Boolean = ((x.value ^ y.value) & x.mask | ~y.mask & x.mask) == 0

    /** Expand implicant `x` to a [[Seq]] of implicants without touching the forbidden implicants specified by `maxt`.
      * @param maxt The forbidden list
      * @return     Expanded implicants
      */
    def expand(maxt: Seq[BitPat]): Seq[BitPat] = (0 until x.getWidth).flatMap{ i =>
      val newImplicant = x match {
        case x if x.mask.testBit(i) && !x.value.testBit(i) => // this bit is 0
          Some(new BitPat(x.value.setBit(i), x.mask, x.getWidth))
        case x if x.mask.testBit(i) && x.value.testBit(i) => // this bit is 1
          Some(new BitPat(x.value.clearBit(i), x.mask, x.getWidth))
        case x if !x.mask.testBit(i) => // this bit is ?
          None
      }
      newImplicant.flatMap(a => if (maxt.exists(a.intersects(_))) None else Some(a))
    }
  }

  /**
    * If two terms have different value, then their order is determined by the value, or by the mask.
    */
  private[minimizer] implicit def ordering: Ordering[BitPat] = (x: BitPat, y: BitPat) => {
    if (x.value < y.value || x.value == y.value && x.mask > y.mask) -1 else 1
  }
}

abstract class Minimizer {
  /** Minimize a multi-input multi-output logic function given by the truth table `table`, with function output values
    * on unspecified inputs treated as `default`, and return a minimized PLA-like representation of the function.
    *
    * Each bit of `table[]._1` encodes one 1-bit input variable of the logic function, and each bit of `default` and
    * `table[]._2` represents one 1-bit output value of the function.
    *
    * @param default  Default output values, can have don't cares
    * @param table    Truth table, can have don't cares in both inputs and outputs, specified as [(inputs, outputs), ...]
    * @return         Minimized AND plane structure for each outputs TODO
    *
    * @example {{{
    *            minimize(BitPat("b01?"), Seq(
    *                   BitPat("b TODO
    * }}}
    */
  def minimize(default: BitPat, table: Seq[(BitPat, BitPat)]): Seq[Seq[BitPat]] // TODO
}