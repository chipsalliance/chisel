// SPDX-License-Identifier: Apache-2.0

package chisel3.util.circt

import chisel3._
import chisel3.experimental.{FlatIO, IntrinsicModule}
import chisel3.internal.Builder

import circt.Intrinsic

/** Create a module which generates a verilog $plusargs$value.  This returns a
  * value as indicated by the format string and a flag for whether the value
  * was found.
  */
private class PlusArgsValueIntrinsic[T <: Data](gen: T, str: String)
    extends IntrinsicModule("circt.plusargs.value", Map("FORMAT" -> str)) {
  val io = FlatIO(new Bundle {
    val found = Output(UInt(1.W))
    val result = Output(gen)
  })
}

object PlusArgsValue {

  /** Creates an intrinsic which calls $test$plusargs.
    *
    * @example {{{
    * b := PlusArgsValue(UInt<32.W>, "FOO=%d")
    * b.found
    * b.value
    * }}}
    */
  def apply[T <: Data](gen: T, str: String) = {
    if (gen.isSynthesizable) {
      val inst = Module(new PlusArgsValueIntrinsic(chiselTypeOf(gen), str))
      inst.io
    } else {
      val inst = Module(new PlusArgsValueIntrinsic(gen, str))
      inst.io
    }
  }
}
