// SPDX-License-Identifier: Apache-2.0

package chisel3.util.experimental.decode

import chisel3._
import chisel3.experimental.{ChiselAnnotation, annotate}
import chisel3.util.{BitPat, pla}
import chisel3.util.experimental.getAnnotations
import firrtl.annotations.Annotation
import logger.LazyLogging

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}

object decoder extends LazyLogging {
  def apply(minimizer: Minimizer, input: UInt, truthTable: TruthTable, timeout: Int = 5): UInt = {
    val minimizedTable = getAnnotations().collect {
      case DecodeTableAnnotation(_, in, out) => TruthTable(in) -> TruthTable(out)
    }.toMap.getOrElse(
      {
        logger.warn(s"""Decoder Cache Hit!
                        |${truthTable.table}
                        |""".stripMargin)
        truthTable
      }, {
        val startTime = System.nanoTime()
        val future = Future(minimizer.minimize(truthTable))
        var now = System.nanoTime()
        while (!future.isCompleted) {
          val elapsed = (System.nanoTime() - now) / 1e9
          now = System.nanoTime()
          if(elapsed > timeout) logger.error(s"Minimizer has been executed for ${(System.nanoTime() - startTime) / 1e9} seconds.")
        }
        val minimizedTable = Await.result(future, Duration.Inf)
        val totalTime = System.nanoTime() - startTime
        val totalTimeInSeconds = totalTime / 1e9
        val info = f"Logic Minimize with $minimizer finished in $totalTimeInSeconds second"
        logger.warn(info)
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
