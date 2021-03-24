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
          VecInit(a.map { t =>
            // share AND plane decode result.
            cache
              .getOrElseUpdate(
                t.toString,
                Cat((0 until t.getWidth).flatMap{ i =>
                  if (t.mask.testBit(i)) {
                    Some(
                      if (t.value.testBit(i)) inputs(i)
                      else invInputs(i)
                    )
                  } else {
                    None
                  }
                })
                  // PLA AND plane
                  .andR()
              )
          }).asUInt()
            // PLA OR plane
            .orR()
      }
      if (default.mask.testBit(i) && default.value.testBit(i)) !b else b
    }.reverse)
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
    * @return         Minimized AND plane structure for each outputs
    *
    * @example {{{
    *            minimize(BitPat("b01?"), Seq(
    *                   BitPat("b TODO
    * }}}
    */
  def minimize(default: BitPat, table: Seq[(BitPat, BitPat)]): Seq[Seq[BitPat]]
}