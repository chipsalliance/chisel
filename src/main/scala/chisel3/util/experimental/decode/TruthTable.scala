// SPDX-License-Identifier: Apache-2.0

package chisel3.util.experimental.decode

import chisel3.util.BitPat

sealed class TruthTable(val table: Map[BitPat, BitPat], val default: BitPat) {

  def inputWidth = table.head._1.getWidth

  def outputWidth = table.head._2.getWidth

  override def toString: String = {
    def writeRow(map: (BitPat, BitPat)): String =
      s"${TruthTable.bpStr(map._1)}->${TruthTable.bpStr(map._2)}"

    (table.map(writeRow) ++ Seq(s"${" "*(inputWidth + 2)}${TruthTable.bpStr(default)}")).toSeq.sorted.mkString("\n")
  }

  def copy(table: Map[BitPat, BitPat] = this.table, default: BitPat = this.default) = new TruthTable(table, default)

  override def equals(y: Any): Boolean = {
    y match {
      case y: TruthTable => toString == y.toString
      case _ => false
    }
  }
}

object TruthTable {
  /** Parse TruthTable from its string representation. */
  def apply(tableString: String): TruthTable = {
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

  /** Convert a table and default output into a [[TruthTable]]. */
  def apply(table: Iterable[(BitPat, BitPat)], default: BitPat) = {
    require(table.map(_._1.getWidth).toSet.size == 1, "input width not equal.")
    require(table.map(_._2.getWidth).toSet.size == 1, "output width not equal.")
    val outputWidth = table.map(_._2.getWidth).head
    new TruthTable(table.toSeq.groupBy(_._1).map { case (key, values) =>
      // merge same input inputs.
      key -> BitPat(s"b${
        Seq.tabulate(outputWidth) { i =>
          val outputSet = values.map(_._2)
            .map(bpStr)
            .map(_ (i))
            .toSet
            .filterNot(_ == '?')
          require(outputSet.size != 2, s"TruthTable conflict in :\n${values.map { case (i, o) => s"${bpStr(i)}->${bpStr(o)}" }.mkString("\n")}")
          outputSet.headOption.getOrElse('?')
        }.mkString
      }")
    }, default)
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
      BitPat(s"b${bpStr(bitPat).zipWithIndex.filter(b => indexes.contains(b._2)).map(_._1).mkString}")

    def tableFilter(indexes: Seq[Int]): Option[(TruthTable, Seq[Int])] = {
      if(indexes.nonEmpty) Some((TruthTable(
        table.table.map { case (in, out) => in -> bpFilter(out, indexes) },
        bpFilter(table.default, indexes)
      ), indexes)) else None
    }

    def index(bitPat: BitPat, bpType: Char): Seq[Int] =
      bpStr(bitPat).zipWithIndex.filter(_._1 == bpType).map(_._2)

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
      bpStr(table.table.getOrElse(bitPat, BitPat.dontCare(indexes.size))).zip(indexes)
    def bitPat(indexedChar: Seq[(Char, Int)]) = BitPat(s"b${indexedChar
      .sortBy(_._2)
      .map(_._1)
      .mkString}")
    TruthTable(
      tables
        .flatMap(_._1.table.keys)
        .map { key =>
          key -> bitPat(tables.flatMap { case (table, indexes) => reIndex(key, table, indexes) })
        }
        .toMap,
      bitPat(tables.flatMap { case (table, indexes) => bpStr(table.default).zip(indexes) })
    )
  }

  private def bpStr(bitPat: BitPat) = bitPat.toString.drop(7).dropRight(1)
}
