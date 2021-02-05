package chisel3.util.experimental.decoder

import chisel3._
import chisel3.util.BitPat

abstract class Decoder {
  /** Simplify a multi-input multi-output logic function given by the truth table `mapping`, with function output values
    * on unspecified inputs treated as `default`, and return the output values of the function on input values `addr`.
    *
    * Each bit of `addr` and `mapping[]._1` means one 1-bit input variable of the logic function, and each bit of
    * `default` and `mapping[]._2` represents one 1-bit output value of the function.
    *
    * @param addr     Input values
    * @param default  Default output values, can have don't cares
    * @param mapping  Truth table, can have don't cares in both inputs and outputs, specified as [(inputs, outputs), ...]
    * @return         Output values of the logic function when inputs are `addr`
    */
  def decode(addr: UInt, default: BitPat, mapping: Iterable[(BitPat, BitPat)]): UInt


}