package chisel3.util.experimental

import chisel3._
import chisel3.util.{BitPat, Cat}

object pla {
  /** Construct a [[https://en.wikipedia.org/wiki/Programmable_logic_array]] from specified table.
    * @param table A [[Seq]] of inputs -> outputs mapping
    *              Each position in the input plane corresponds to an input variable where a `0` implies the corresponding
    *              input literal appears complemented in the product term, a `1` implies the input literal appears un-
    *              complemented in the product term, and `?` implies the input literal does not appear in the product term.
    *              For each output, a `1` means this product term makes the function value a `1`, and a `0` or `?` means
    *              this product term has no meaning for the value of this function.
    * @return (inputs, outputs) the input and output [[Wire]] of [[UInt]] of the constructed pla.
    * @example A 1-of-8 decoder (like the 74xx138) can be constructed as follow
    *          {{{
    *            val (inputs, outputs) = pla(Seq(
    *              (BitPat("b000"), BitPat("b00000001")),
    *              (BitPat("b001"), BitPat("b00000010")),
    *              (BitPat("b010"), BitPat("b00000100")),
    *              (BitPat("b011"), BitPat("b00001000")),
    *              (BitPat("b100"), BitPat("b00010000")),
    *              (BitPat("b101"), BitPat("b00100000")),
    *              (BitPat("b110"), BitPat("b01000000")),
    *              (BitPat("b111"), BitPat("b10000000")),
    *            ))
    *          }}}
    */
  def apply(table: Seq[(BitPat, BitPat)]): (UInt, UInt) = {
    val (inputTerms, outputTerms) = table.unzip
    require(
      if (inputTerms.length > 1)
        inputTerms.tail.map(_.getWidth == inputTerms.head.getWidth).reduce(_ && _)
      else
        true,
      "all `BitPat`s in input part of specified PLA table must have the same width"
    )
    require(
      if (outputTerms.length > 1)
        outputTerms.tail.map(_.getWidth == outputTerms.head.getWidth).reduce(_ && _)
      else
        true,
      "all `BitPat`s in output part of specified PLA table must have the same width"
    )

    // now all inputs / outputs have the same width
    val numberOfInputs  = inputTerms.head.getWidth
    val numberOfOutputs = outputTerms.head.getWidth

    // input wires of the generated PLA
    val inputs = Wire(UInt(numberOfInputs.W))
    val invInputs = ~inputs

    // output wires of the generated PLA
    val outputs = Wire(UInt(numberOfOutputs.W))

    // construct the AND plane
    val andPlaneOutputs = inputTerms.map{ t =>
      val andLine = Cat((0 until numberOfInputs).flatMap { i =>
        if (t.mask.testBit(i)) {
          Some(
            if (t.value.testBit(i)) inputs(i)
            else invInputs(i)
          )
        } else {
          None
        }
      }).andR()

      // using `t.toString` instead of `t` as index saves logic (reuse AND plane output lines)
      (t.toString, andLine)
    }.toMap

    outputs := Cat(
      (0 until numberOfOutputs).map { i =>
        val andPlaneLines = table
          // OR plane composed by input terms which makes this output bit a `1`
          .filter(b => b._2.mask.testBit(i) && b._2.value.testBit(i))
          .map { case (inputTerm, _) =>
            andPlaneOutputs(inputTerm.toString)
          }
        if (andPlaneLines.isEmpty) false.B
        else Cat(andPlaneLines).orR()
      }.reverse
    )

    (inputs, outputs)
  }
}
