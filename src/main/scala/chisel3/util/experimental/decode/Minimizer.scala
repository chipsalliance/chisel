// SPDX-License-Identifier: Apache-2.0

package chisel3.util.experimental.decode

abstract class Minimizer {

  /** Minimize a multi-input multi-output logic function given by the truth table `table`, with function output values
    * on unspecified inputs treated as `default`, and return a minimized PLA-like representation of the function.
    *
    * Each bit of `table[]._1` encodes one 1-bit input variable of the logic function, and each bit of `default` and
    * `table[]._2` represents one 1-bit output value of the function.
    *
    * @param table    Truth table, can have don't cares in both inputs and outputs, specified as [(inputs, outputs), ...]
    * @return         Minimized truth table, [(inputs, outputs), ...]
    *
    * @example {{{
    *          minimize(BitPat("b?"), Seq(
    *              (BitPat("b000"), BitPat("b0")),
    *              // (BitPat("b001"), BitPat("b?")),  // same as default, can be omitted
    *              // (BitPat("b010"), BitPat("b?")),  // same as default, can be omitted
    *              (BitPat("b011"), BitPat("b0")),
    *              (BitPat("b100"), BitPat("b1")),
    *              (BitPat("b101"), BitPat("b1")),
    *              (BitPat("b110"), BitPat("b0")),
    *              (BitPat("b111"), BitPat("b1")),
    *          ))
    * }}}
    */
  def minimize(table: TruthTable): TruthTable
}
