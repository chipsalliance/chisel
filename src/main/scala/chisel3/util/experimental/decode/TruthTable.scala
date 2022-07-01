// SPDX-License-Identifier: Apache-2.0

package chisel3.util.experimental.decode

import chisel3.util.BitPat
import firrtl.Utils.groupByIntoSeq

sealed class TruthTable private (val table: Seq[(BitPat, BitPat)], val default: BitPat, val sort: Boolean) {
  def inputWidth = table.head._1.getWidth

  def outputWidth = table.head._2.getWidth

  override def toString: String = {
    def writeRow(map: (BitPat, BitPat)): String =
      s"${map._1.rawString}->${map._2.rawString}"

    (table.map(writeRow) ++ Seq(s"${" " * (inputWidth + 2)}${default.rawString}")).mkString("\n")
  }

  def copy(table: Seq[(BitPat, BitPat)] = this.table, default: BitPat = this.default, sort: Boolean = this.sort) =
    TruthTable(table, default, sort)

  override def equals(y: Any): Boolean = {
    y match {
      case y: TruthTable => toString == y.toString
      case _ => false
    }
  }
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

  /** For each duplicated input, combine the outputs into a single Seq.
    *
    * @param table the truth table
    * @return a Seq of tuple of length 2, where the first element is the
    *         input and the second element is a Seq of combined outputs
    *         for the input
    */
  private def mergeTableOnInputs(table: Iterable[(BitPat, BitPat)]): Seq[(BitPat, Seq[BitPat])] = {
    groupByIntoSeq(table)(_._1).map {
      case (input, mappings) =>
        input -> mappings.map {
          case (_, output) => output
        }
    }
  }

  /** Check for BitPats that have non-zero overlap
    *
    * Non-zero overlap means that for two BitPats a and b, there is at least
    * one bitpos where the indices at the bitpos for the two bits are
    * present in [(0, 1), (1, 0)].
    *
    * @param x List of BitPats to check for overlap
    * @return true if the overlap is non-zero, false otherwise
    */
  private def checkDups(x: BitPat*): Boolean = {
    if (x.size > 1) {
      val f = (a: BitPat, b: BitPat) => a.overlap(b)
      val combs: Seq[Boolean] = x.combinations(2).map { a => f(a.head, a.last) }.toSeq
      combs.reduce(_ | _)
    } else {
      false
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
    * For Espresso, for each output, a 1 means this product term belongs to the * ON-set, a 0 means this product term has no meaning for the value of this * function". The is the same as the fd (or f) type in espresso.
    *
    * @param table the truth table
    * @param default the default BitPat. This also determines the format sent
    *                to Espresso
    * @param sort whether to sort the final truth table or not
    */
  def fromEspressoOutput(table: Iterable[(BitPat, BitPat)], default: BitPat, sort: Boolean = false): TruthTable = {
    apply_impl(table, default, sort, false)
  }

  def apply(table: Iterable[(BitPat, BitPat)], default: BitPat, sort: Boolean = true): TruthTable = {
    apply_impl(table, default, sort, true)
  }

  /** Convert a table and default output into a [[TruthTable]]. */
  private def apply_impl(
    table:            Iterable[(BitPat, BitPat)],
    default:          BitPat,
    sort:             Boolean = true,
    espressoFDFormat: Boolean = false
  ): TruthTable = {
    val paddedTable = padInputs(table)

    require(table.map(_._2.getWidth).toSet.size == 1, "output width not equal.")

    val mergedTable = mergeTableOnInputs(paddedTable)

    val finalTable: Seq[(BitPat, BitPat)] = mergedTable.map(
      x =>
        ({
          if (espressoFDFormat)
            (x._1, x._2.reduce(merge(_, _)))
          else {
            require(checkDups(x._2: _*) == false, "TruthTable conflict")
            (x._1, x._2.head)
          }
        })
    )

    new TruthTable(finalTable, default, sort)
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
}
