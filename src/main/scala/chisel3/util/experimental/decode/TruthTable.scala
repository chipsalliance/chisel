// SPDX-License-Identifier: Apache-2.0

package chisel3.util.experimental.decode

import chisel3.util.BitPat

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

  /** Convert a table and default output into a [[TruthTable]]. */
  def apply(table: Iterable[(BitPat, BitPat)], default: BitPat, sort: Boolean = true): TruthTable = {
    val inputWidth = table.map(_._1.getWidth).max
    require(table.map(_._2.getWidth).toSet.size == 1, "output width not equal.")
    val outputWidth = table.map(_._2.getWidth).head
    val mergedTable = table.map {
      case (in, out) =>
        // pad input signals.
        (BitPat.dontCare(inputWidth - in.getWidth) ## in, out)
    }
      .groupBy(_._1.toString)
      .map {
        case (key, values) =>
          // merge same input inputs.
          values.head._1 -> BitPat(s"b${Seq
            .tabulate(outputWidth) { i =>
              val outputSet = values
                .map(_._2)
                .map(_.rawString)
                .map(_(i))
                .toSet
                .filterNot(_ == '?')
              require(
                outputSet.size != 2,
                s"TruthTable conflict in :\n${values.map { case (i, o) => s"${i.rawString}->${o.rawString}" }.mkString("\n")}"
              )
              outputSet.headOption.getOrElse('?')
            }
            .mkString}")
      }
      .toSeq
    import BitPat.bitPatOrder
    new TruthTable(if (sort) mergedTable.sorted else mergedTable, default, sort)
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
