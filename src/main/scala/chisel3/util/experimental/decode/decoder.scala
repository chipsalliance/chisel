// SPDX-License-Identifier: Apache-2.0

package chisel3.util.experimental.decode

import chisel3._
import chisel3.experimental.{ChiselAnnotation, annotate}
import chisel3.util.{BitPat, pla}
import chisel3.util.experimental.getAnnotations
import firrtl.annotations.Annotation
import logger.LazyLogging

object decoder extends LazyLogging {
  /** Use a specific [[Minimizer]] to generated decoded signals.
    *
    * @param minimizer  specific [[Minimizer]], can be [[QMCMinimizer]] or [[EspressoMinimizer]].
    * @param input      input signal that contains decode table input
    * @param truthTable [[TruthTable]] to decode user input.
    * @return decode table output.
    */
  def apply(minimizer: Minimizer, input: UInt, truthTable: TruthTable): UInt = {
    val minimizedTable = getAnnotations().collect {
      case DecodeTableAnnotation(_, in, out) => TruthTable(in) -> TruthTable(out)
    }.toMap.getOrElse(truthTable, minimizer.minimize(truthTable))
    if (minimizedTable.table.isEmpty) {
      val outputs = Wire(UInt(minimizedTable.default.getWidth.W))
      outputs := minimizedTable.default.value.U(minimizedTable.default.getWidth.W)
      outputs
    } else {
      val (plaInput, plaOutput) =
        pla(minimizedTable.table.toSeq, BitPat(minimizedTable.default.value.U(minimizedTable.default.getWidth.W)))

      annotate(new ChiselAnnotation {
        override def toFirrtl: Annotation =
          DecodeTableAnnotation(plaOutput.toTarget, truthTable.toString, minimizedTable.toString)
      })

      plaInput := input
      plaOutput
    }
  }

  /** Use [[EspressoMinimizer]] to generated decoded signals.
    *
    * @param input      input signal that contains decode table input
    * @param truthTable [[TruthTable]] to decode user input.
    * @return decode table output.
    */
  def espresso(input: UInt, truthTable: TruthTable): UInt = apply(EspressoMinimizer, input, truthTable)

  /** Use [[QMCMinimizer]] to generated decoded signals.
    *
    * @param input      input signal that contains decode table input
    * @param truthTable [[TruthTable]] to decode user input.
    * @return decode table output.
    */
  def qmc(input: UInt, truthTable: TruthTable): UInt = apply(QMCMinimizer, input, truthTable)

  /** try to use [[EspressoMinimizer]] to decode `input` by `truthTable`
    * if `espresso` not exist in your PATH environment it will fall back to [[QMCMinimizer]], and print a warning.
    *
    * @param input      input signal that contains decode table input
    * @param truthTable [[TruthTable]] to decode user input.
    * @return decode table output.
    */
  def apply(input: UInt, truthTable: TruthTable): UInt = {
    def qmcFallBack(input: UInt, truthTable: TruthTable) = {
      """fall back to QMC.
        |Quine-McCluskey is a NP complete algorithm, may run forever or run out of memory in large decode tables.
        |To get rid of this warning, please use `decoder.qmc` directly, or add espresso to your PATH.
        |""".stripMargin
      qmc(input, truthTable)
    }

    try espresso(input, truthTable) catch {
      case EspressoNotFoundException =>
        logger.error(s"espresso is not found in your PATH:\n${sys.env("PATH").split(":").mkString("\n")}".stripMargin)
        qmcFallBack(input, truthTable)
      case e: java.io.IOException =>
        logger.error(s"espresso failed to run with ${e.getMessage}")
        qmcFallBack(input, truthTable)
    }
  }
}
