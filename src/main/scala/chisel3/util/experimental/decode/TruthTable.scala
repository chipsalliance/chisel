// SPDX-License-Identifier: Apache-2.0

package chisel3.util.experimental.decode

import chisel3.util.BitPat
import scala.util.hashing.MurmurHash3
import scala.collection.mutable

sealed class TruthTable private (val table: Seq[(BitPat, BitPat)], val default: BitPat) {
  def inputWidth = table.head._1.getWidth

  def outputWidth = table.head._2.getWidth

  override def toString: String = {
    def writeRow(map: (BitPat, BitPat)): String =
      s"${map._1.rawString}->${map._2.rawString}"

    (table.map(writeRow) ++ Seq(s"${" " * (inputWidth + 2)}${default.rawString}")).mkString("\n")
  }

  def copy(table: Seq[(BitPat, BitPat)] = this.table, default: BitPat = this.default) =
    TruthTable(table, default)

  override def equals(y: Any): Boolean = {
    y match {
      case that: TruthTable => this.table == that.table && this.default == that.default
      case _ => false
    }
  }

  override lazy val hashCode: Int = MurmurHash3.productHash((table, default))
}

object TruthTable {

  /** Pad the input signals to equalize all input widths. Pads input signals
    *  to the maximum width found in the table.
    *
    * @param table the truth table whose rows will be padded
    * @return the same truth table but with inputs padded
    */
  private def padInputs(table: Iterable[(BitPat, BitPat)]): Iterable[(BitPat, BitPat)] = {
    val inputWidth = table.map(_._1.getWidth).max
    table.map {
      case (in, out) if inputWidth > in.width =>
        (BitPat.N(inputWidth - in.width) ## in, out)
      case (in, out) => (in, out)
    }
  }

  /** For each duplicated input, collect the outputs into a single Seq.
    *
    * @param table the truth table
    * @return a Seq of tuple of length 2, where the first element is the
    *         input and the second element is a Seq of OR-ed outputs
    *         for the input
    */
  private def mergeTableOnInputs(table: Iterable[(BitPat, BitPat)]): Seq[(BitPat, Seq[BitPat])] = {
    groupByIntoSeq(table)(_._1).map {
      case (input, mappings) =>
        input -> mappings.map(_._2)
    }
  }

  /** Merge two BitPats by OR-ing the values and masks, and setting the
    *  width to the max width among the two
    */
  private def merge(a: BitPat, b: BitPat): BitPat = {
    new BitPat(a.value | b.value, a.mask | b.mask, a.width.max(b.width))
  }

  /** Public method for calling with the Espresso decoder format fd
    *
    * For Espresso, for each output, a 1 means this product term belongs to the ON-set,
    *  a 0 means this product term has no meaning for the value of this function.
    * This is the same as the fd (or f) type in espresso.
    *
    * @param table the truth table
    * @param default the default BitPat is made up of a single bit type, either "?", "0" or "1".
    *                A default of "?" sets Espresso to fr-format, while a "0" or "1" sets it to the
    *                fd-format.
    * @param sort whether to sort the final truth table using BitPat.bitPatOrder
    * @return a fully built TruthTable
    */
  def fromEspressoOutput(table: Iterable[(BitPat, BitPat)], default: BitPat, sort: Boolean = true): TruthTable = {
    apply_impl(table, default, sort, false)
  }

  /** Public apply method to TruthTable. Calls apply_impl with the default value true of checkCollisions */
  def apply(table: Iterable[(BitPat, BitPat)], default: BitPat, sort: Boolean = true): TruthTable = {
    apply_impl(table, default, sort, true)
  }

  /** Convert a table and default output into a [[TruthTable]]. */
  private def apply_impl(
    table:           Iterable[(BitPat, BitPat)],
    default:         BitPat,
    sort:            Boolean,
    checkCollisions: Boolean
  ): TruthTable = {
    val paddedTable = padInputs(table)

    require(table.map(_._2.getWidth).toSet.size == 1, "output width not equal.")

    val mergedTable = mergeTableOnInputs(paddedTable)

    val finalTable: Seq[(BitPat, BitPat)] = mergedTable.map {
      case (input, outputs) =>
        val (result, noCollisions) = outputs.tail.foldLeft((outputs.head, checkCollisions)) {
          case ((acc, ok), o) => (merge(acc, o), ok && acc.overlap(o))
        }
        // Throw an error if checkCollisions is true but there are bits with a non-zero overlap.
        require(!checkCollisions || noCollisions, s"TruthTable conflict on merged row:\n  $input -> $outputs")
        (input, result)
    }

    import BitPat.bitPatOrder
    new TruthTable(if (sort) finalTable.sorted else finalTable, default)
  }

  /** Parse TruthTable from its string representation. */
  def fromString(tableString: String): TruthTable = {
    TruthTable(
      tableString
        .split("\n")
        .filter(_.contains("->"))
        .map(_.split("->").map(str => BitPat(s"b$str")))
        .map(bps => bps(0) -> bps(1))
        .toSeq,
      BitPat(s"b${tableString.split("\n").filterNot(_.contains("->")).head.replace(" ", "")}")
    )
  }

  /** consume 1 table, split it into up to 3 tables with the same default bits.
    *
    * @return table and its indexes from original bits.
    * @note
    * Since most of minimizer(like espresso) cannot handle a multiple default table.
    * It is useful to split a table into 3 tables based on the default type.
    */
  private[decode] def split(
    table: TruthTable
  ): Seq[(TruthTable, Seq[Int])] = {
    def bpFilter(bitPat: BitPat, indexes: Seq[Int]): BitPat =
      BitPat(s"b${bitPat.rawString.zipWithIndex.filter(b => indexes.contains(b._2)).map(_._1).mkString}")

    def tableFilter(indexes: Seq[Int]): Option[(TruthTable, Seq[Int])] = {
      if (indexes.nonEmpty)
        Some(
          (
            TruthTable(
              table.table.map { case (in, out) => in -> bpFilter(out, indexes) },
              bpFilter(table.default, indexes)
            ),
            indexes
          )
        )
      else None
    }

    def index(bitPat: BitPat, bpType: Char): Seq[Int] =
      bitPat.rawString.zipWithIndex.filter(_._1 == bpType).map(_._2)

    Seq('1', '0', '?').flatMap(t => tableFilter(index(table.default, t)))
  }

  /** consume tables, merge it into single table with different default bits.
    *
    * @note
    * Since most of minimizer(like espresso) cannot handle a multiple default table.
    * It is useful to split a table into 3 tables based on the default type.
    */
  private[decode] def merge(
    tables: Seq[(TruthTable, Seq[Int])]
  ): TruthTable = {
    def reIndex(bitPat: BitPat, table: TruthTable, indexes: Seq[Int]): Seq[(Char, Int)] =
      table.table
        .map(a => a._1.toString -> a._2)
        .collectFirst { case (k, v) if k == bitPat.toString => v }
        .getOrElse(BitPat.dontCare(indexes.size))
        .rawString
        .zip(indexes)
    def bitPat(indexedChar: Seq[(Char, Int)]) = BitPat(s"b${indexedChar
      .sortBy(_._2)
      .map(_._1)
      .mkString}")
    TruthTable(
      tables
        .flatMap(_._1.table.map(_._1))
        .map { key =>
          key -> bitPat(tables.flatMap { case (table, indexes) => reIndex(key, table, indexes) })
        },
      bitPat(tables.flatMap { case (table, indexes) => table.default.rawString.zip(indexes) })
    )
  }

  /** Similar to Seq.groupBy except that it preserves ordering of elements within each group */
  private def groupByIntoSeq[A, K](xs: Iterable[A])(f: A => K): Seq[(K, Seq[A])] = {
    val map = mutable.LinkedHashMap.empty[K, mutable.ListBuffer[A]]
    for (x <- xs) {
      val key = f(x)
      val l = map.getOrElseUpdate(key, mutable.ListBuffer.empty[A])
      l += x
    }
    map.view.map({ case (k, vs) => k -> vs.toList }).toList
  }
}
