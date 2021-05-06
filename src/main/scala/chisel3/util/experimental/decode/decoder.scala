// SPDX-License-Identifier: Apache-2.0

package chisel3.util.experimental.decode

import chisel3._
import chisel3.experimental.{ChiselAnnotation, annotate}
import chisel3.util.{BitPat, pla}
import chisel3.util.experimental.getAnnotations
import firrtl.annotations.Annotation
import logger.LazyLogging

object decoder extends LazyLogging {
  def apply(minimizer: Minimizer, input: UInt, truthTable: TruthTable): UInt = {
    val minimizedTable = getAnnotations().collect {
      case DecodeTableAnnotation(_, in, out) => TruthTable(in) -> TruthTable(out)
    }.toMap.getOrElse(
      {
        logger.trace(s"""Decoder Cache Hit!
                        |${truthTable.table}
                        |""".stripMargin)
        truthTable
      }, {
        val startTime = System.nanoTime()
        val minimizedTable = minimizer.minimize(truthTable)
        val totalTime = System.nanoTime() - startTime
        val totalTimeInSeconds = totalTime / 1e9
        val info = f"Logic Minimize with $minimizer finished in ${totalTimeInSeconds} second"
        if (totalTimeInSeconds > 5)
          logger.error(
            s"$info spends too long, consider using chisel3.util.experimental.DecodeTableAnnotation to cache decode result or switch to EspressoMinimizer."
          )
        else logger.trace(info)
        minimizedTable
      }
    )
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
}
