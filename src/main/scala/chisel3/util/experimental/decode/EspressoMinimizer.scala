// SPDX-License-Identifier: Apache-2.0

package chisel3.util.experimental.decode

import chisel3.util.BitPat
import logger.LazyLogging

class EspressoMinimizer extends Minimizer with LazyLogging {
  def minimize(table: TruthTable): TruthTable = ???

  def espresso(table: TruthTable): TruthTable = {
    import scala.sys.process._
    require(true, "todo, require table.default has same types.")
    // @todo match decodeType here.
    val decodeType: String = ???
    val espressoTempFile = java.io.File.createTempFile("pla", "txt")
    espressoTempFile.deleteOnExit()
    new java.io.PrintWriter(espressoTempFile) {
      try {
        write(table.pla)
      } finally {
        close()
      }
    }
    logger.trace(s"espresso input:\n${scala.io.Source.fromFile(espressoTempFile)}")
    Seq("espresso", espressoTempFile.getPath).!!.toTruthTable(decodeType)
  }

  private implicit class EspressoTruthTable(table: TruthTable) {
    def pla: String =
      s""".i ${table.table.head._1.getWidth}
         |.o ${table.table.head._2.getWidth}
         |.type ${// match table.default here
      ???}
         |""".stripMargin +
        table.table.map((row: (BitPat, BitPat)) => s"${???}").mkString("\n")
  }

  private implicit class PLAToTruthTable(string: String) {
    def toTruthTable(decodeType: String): TruthTable = {
      // @todo: @yhr fix me
      ???
    }
  }

  private def split(table: TruthTable): (TruthTable, TruthTable, TruthTable) = ???

  private def merge(
    onTable:       TruthTable,
    offTable:      TruthTable,
    dontCareTable: TruthTable
  ): TruthTable = ???
}
