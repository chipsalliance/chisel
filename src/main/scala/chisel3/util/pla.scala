// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import chisel3._

object pla {

  /** Construct a [[https://en.wikipedia.org/wiki/Programmable_logic_array]] from specified table.
    *
    * Each position in the input matrix corresponds to an input variable where
    * `0` implies the corresponding input literal appears complemented in the product term.
    * `1` implies the input literal appears uncomplemented in the product term
    * `?` implies the input literal does not appear in the product term.
    *
    * For each output
    * a `1` means this product term makes the function value to `1`
    * and a `0` or `?` means this product term make the function value to `0`
    *
    * @param table A [[Seq]] of inputs -> outputs mapping
    * @param invert A [[BitPat]] specify which bit of the output should be inverted. `1` means the correspond position
    *               of the output should be inverted in the PLA, a `0` or a `?` means direct output from the OR matrix.
    * @return the (input, output) [[Wire]] of [[UInt]] of the constructed pla.
    * {{{
    *   // A 1-of-8 decoder (like the 74xx138) can be constructed as follow
    *   val (inputs, outputs) = pla(Seq(
    *     (BitPat("b000"), BitPat("b00000001")),
    *     (BitPat("b001"), BitPat("b00000010")),
    *     (BitPat("b010"), BitPat("b00000100")),
    *     (BitPat("b011"), BitPat("b00001000")),
    *     (BitPat("b100"), BitPat("b00010000")),
    *     (BitPat("b101"), BitPat("b00100000")),
    *     (BitPat("b110"), BitPat("b01000000")),
    *     (BitPat("b111"), BitPat("b10000000")),
    *   ))
    * }}}
    */
  def apply(table: Seq[(BitPat, BitPat)], invert: BitPat = BitPat("b0")): (UInt, UInt) = {
    require(table.nonEmpty, "pla table must not be empty")

    val (inputTerms, outputTerms) = table.unzip
    require(
      inputTerms.map(_.getWidth).distinct.size == 1,
      "all `BitPat`s in the input part of specified PLA table must have the same width"
    )
    require(
      outputTerms.map(_.getWidth).distinct.size == 1,
      "all `BitPat`s in the output part of specified PLA table must have the same width"
    )

    // now all inputs / outputs have the same width
    val numberOfInputs = inputTerms.head.getWidth
    val numberOfOutputs = outputTerms.head.getWidth

    val inverterMask = invert.value & invert.mask
    if (inverterMask.bitCount != 0)
      require(invert.getWidth == numberOfOutputs,
        "non-zero inverter mask must have the same width as the output part of specified PLA table"
      )

    // input wires of the generated PLA
    val inputs = Wire(UInt(numberOfInputs.W))
    val invInputs = ~inputs

    // output wires of the generated PLA
    val outputs = Wire(UInt(numberOfOutputs.W))

    // the AND matrix
    // use `term -> AND line` map to reuse AND matrix output lines
    val andMatrixOutputs: Map[String, Bool] = inputTerms.map { t =>
      t.toString -> Cat(
        Seq
          .tabulate(numberOfInputs) { i =>
            if (t.mask.testBit(i)) {
              Some(
                if (t.value.testBit(i)) inputs(i)
                else invInputs(i)
              )
            } else {
              None
            }
          }
          .flatten
      ).andR()
    }.toMap

    // the OR matrix
    val orMatrixOutputs: UInt = Cat(
        Seq
          .tabulate(numberOfOutputs) { i =>
            val andMatrixLines = table
              // OR matrix composed by input terms which makes this output bit a `1`
              .filter {
                case (_, or) => or.mask.testBit(i) && or.value.testBit(i)
              }.map {
                case (inputTerm, _) =>
                  andMatrixOutputs(inputTerm.toString)
              }
            if (andMatrixLines.isEmpty) false.B
            else Cat(andMatrixLines).orR()
          }
          .reverse
      )

    // the INV matrix, useful for decoders
    val invMatrixOutputs: UInt = Cat(
      Seq
        .tabulate(numberOfOutputs) { i =>
          if (inverterMask.testBit(i)) ~orMatrixOutputs(i)
          else                          orMatrixOutputs(i)
        }
        .reverse
    )

    outputs := invMatrixOutputs

    (inputs, outputs)
  }
}
