// SPDX-License-Identifier: Apache-2.0

package chisel3.util.experimental.decode

import chisel3._
import chisel3.experimental.{ChiselAnnotation, annotate}
import chisel3.util.{BitPat, pla}
import chisel3.util.experimental.{BitSet, getAnnotations}
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
      case DecodeTableAnnotation(_, in, out) => TruthTable.fromString(in) -> TruthTable.fromString(out)
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


  /** Generate a decoder circuit that matches the input to each bitSet.
    *
    * The resulting circuit functions like the following but is optimized with a logic minifier.
    * {{{
    *   when(input === bitSets(0)) { output := b000001 }
    *   .elsewhen (input === bitSets(1)) { output := b000010 }
    *   ....
    *   .otherwise { if (errorBit) output := b100000 else output := DontCare }
    * }}}
    *
    * @param input input to the decoder circuit, width should be equal to bitSets.width
    * @param bitSets set of ports to be matched, all width should be the equal
    * @param errorBit whether generate an additional decode error bit at MSB of output.
    * @return decoded wire
    */
  def bitset(input: chisel3.UInt, bitSets: Seq[BitSet], errorBit: Boolean = false): chisel3.UInt =
    chisel3.util.experimental.decode.decoder(
      input,
      chisel3.util.experimental.decode.TruthTable.fromString(
        {
          bitSets.zipWithIndex.flatMap {
            case (bs, i) =>
              bs.terms.map(bp =>
                s"${bp.rawString}->${if (errorBit) "0"}${"0" * (bitSets.size - i - 1)}1${"0" * i}"
              )
          } ++ Seq(s"${if (errorBit) "1"}${"?" * bitSets.size}")
        }.mkString("\n")
      )
    )
}

object decodeAs  {
    /** Decode signals using a [[Bundle]] as the output
      *
      * @param b          the output bundle to be used as the decoded signal
      * @param input      input signal that contains decode table input
      * @param truthTable [[TruthTable]] to decode user input.
      * @return bundle with the decode output.
      *
      * @note The bundle width must match the TruthTable width.
      * @example
      * {{{
      * class OutputBundle extends Bundle {
      *   val s1 = UInt(2.W)
      *   val s2 = UInt(2.W)
      *   val s3 = Bool()
      * }
      * val selector = "b001".U
      * val signals = decodeAs(
      *   (new OutputBundle),
      *   selector,
      *   TruthTable(
      *     Array(
      *     BitPat("b001")  -> BitPat(1.U(2.W)) ## BitPat(1.U(2.W)) ## BitPat.Y(),
      *     BitPat("b?11")  -> BitPat(2.U(2.W)) ## BitPat(2.U(2.W)) ## BitPat.N(),
      *     ),                 BitPat(0.U(2.W)) ## BitPat(0.U(2.W)) ## BitPat.dontCare(1) // Default values
      *   )
      * )
      * val a = signals.s1 // Should be 1.U(2.W)
      * val b = signals.s2 // Should be 1.U(2.W)
      * val c = signals.s3 // Should be true.B
      * }}}
      */
    def apply[T <: Bundle](b: T, input: UInt, truthTable: TruthTable): T =
    decoder(input, truthTable).asTypeOf(b)
    def apply[T <: Bundle](minimizer: Minimizer, b: T, input: UInt, truthTable: TruthTable): T =
    decoder(minimizer, input, truthTable).asTypeOf(b)
}
