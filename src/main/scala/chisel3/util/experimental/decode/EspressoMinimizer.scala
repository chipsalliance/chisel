// SPDX-License-Identifier: Apache-2.0

package chisel3.util.experimental.decode

import chisel3.util.BitPat
import logger.LazyLogging

case object EspressoNotFoundException extends Exception

/** A [[Minimizer]] implementation to use espresso to minimize the [[TruthTable]].
  *
  * espresso uses heuristic algorithm providing a sub-optimized) result.
  * For implementation details, please refer to:
  * [[https://www.springerprofessional.de/en/logic-minimization-algorithms-for-vlsi-synthesis/13780088]]
  *
  * a espresso executable should be downloaded from [[https://github.com/chipsalliance/espresso]]
  *
  * If user want to user the this [[Minimizer]], a espresso executable should be added to system PATH environment.
  */
object EspressoMinimizer extends Minimizer with LazyLogging {
  def minimize(table: TruthTable): TruthTable =
    TruthTable.merge(TruthTable.split(table).map { case (table, indexes) => (espresso(table), indexes) })

  private def espresso(table: TruthTable): TruthTable = {
    def writeTable(table: TruthTable): String = {
      def invert(string: String) = string
        .replace('0', 't')
        .replace('1', '0')
        .replace('t', '1')
      val defaultType: Char = {
        val t = table.default.rawString.toCharArray.distinct
        require(t.length == 1, "Internal Error: espresso only accept unified default type.")
        t.head
      }
      val tableType: String = defaultType match {
        case '?' => "fr"
        case _   => "fd"
      }
      val rawTable = table.toString
        .split("\n")
        .filter(_.contains("->"))
        .mkString("\n")
        .replace("->", " ")
        .replace('?', '-')
      // invert all output, since espresso cannot handle default is on.
      val invertRawTable = rawTable
        .split("\n")
        .map(_.split(" "))
        .map(row => s"${row(0)} ${invert(row(1))}")
        .mkString("\n")
      s""".i ${table.inputWidth}
         |.o ${table.outputWidth}
         |.type ${tableType}
         |""".stripMargin ++ (if (defaultType == '1') invertRawTable else rawTable)
    }

    def readTable(espressoTable: String) = {
      def bitPat(espresso: String): BitPat = BitPat("b" + espresso.replace('-', '?'))

      val out = espressoTable
        .split("\n")
        .filterNot(_.startsWith("."))
        .map(_.split(' '))
        .map(row => bitPat(row(0)) -> bitPat(row(1)))
      // special case for 0 and DontCare, if output is not couple to input
      if (out.isEmpty)
        Array(
          (
            BitPat(s"b${"?" * table.inputWidth}"),
            BitPat(s"b${"0" * table.outputWidth}")
          )
        )
      else out
    }

    val input = writeTable(table)
    logger.trace(s"""espresso input table:
                    |$input
                    |""".stripMargin)
    val output =
      try {
        os.proc("espresso").call(stdin = input).out.chunks.mkString
      } catch {
        case e: java.io.IOException if e.getMessage.contains("error=2, No such file or directory") =>
          throw EspressoNotFoundException
      }
    logger.trace(s"""espresso output table:
                    |$output
                    |""".stripMargin)
    TruthTable.fromEspressoOutput(readTable(output), table.default)
  }
}
