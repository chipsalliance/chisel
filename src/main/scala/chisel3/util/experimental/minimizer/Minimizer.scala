package chisel3.util.experimental.minimizer

import chisel3.util.BitPat

abstract class Minimizer {
  /** Minimize a multi-input multi-output logic function given by the truth table `table`, with function output values
    * on unspecified inputs treated as `default`, and return a truth table representing the minimized function.
    *
    * Each bit of `table[]._1` encodes one 1-bit input variable of the logic function, and each bit of `default` and
    * `table[]._2` represents one 1-bit output value of the function.
    *
    * @param table    Truth table, can have don't cares in both inputs and outputs, specified as [(inputs, outputs), ...]
    * @param default  Default output values, can have don't cares
    * @return         Minimized truth table
    */
  def minimize(default: BitPat, table: Seq[(BitPat, BitPat)]): Seq[(BitPat, BitPat)]
}