package chisel3.util.experimental.hierarchy

import chisel3._
import chisel3.experimental.hierarchy.{Definition, Instance}
import chisel3.util.{BitPat, PLAModule}

object pla {

  private val definitions: collection.mutable.HashMap[(Seq[(BitPat, BitPat)], BitPat), Definition[PLAModule]] =
    collection.mutable.HashMap()

  /** Instance a [[PLA]], and return its io.
    *
    * @param table A [[Seq]] of inputs -> outputs mapping
    * @param invert A [[BitPat]] specify which bit of the output should be inverted. `1` means the correspond position
    *               of the output should be inverted in the PLA, a `0` or a `?` means direct output from the OR matrix.
    *
    * @return the (input, output) [[chisel3.Wire]] of [[UInt]] of the constructed pla.
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
    val pla = Instance(definitions.getOrElseUpdate((table, invert), Definition(new PLAModule(table, invert))))
    (pla.inputs, pla.outputs)
  }
}
