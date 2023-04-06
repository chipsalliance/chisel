// SPDX-License-Identifier: Apache-2.0

package chisel3.util.circt

import scala.language.reflectiveCalls

import chisel3._
import chisel3.experimental.{FlatIO, IntrinsicModule}
import chisel3.internal.Builder

import circt.Intrinsic

/** Create a module which generates a verilog \$value\$plusargs.  This returns a
  * value as indicated by the format string and a flag for whether the value
  * was found.
  */
private class PlusArgsValueIntrinsic[T <: Data](gen: T, str: String)
    extends IntrinsicModule("circt_plusargs_value", Map("FORMAT" -> str)) {
  val io = FlatIO(new Bundle {
    val found = Output(Bool())
    val result = Output(gen)
  })
}

object PlusArgsValue {

  /** Creates an intrinsic which calls \$value\$plusargs.
    *
    * @example {{{
    * b := PlusArgsValue(UInt(32.W), "FOO=%d")
    * b.found
    * b.value
    * }}}
    */
  def apply[T <: Data](gen: T, str: String) = {
    Module(if (gen.isSynthesizable) {
      new PlusArgsValueIntrinsic(chiselTypeOf(gen), str)
    } else {
      new PlusArgsValueIntrinsic(gen, str)
    }).io
  }

  /** Creates an intrinsic which calls \$value\$plusargs and returns a default
    * value if the specified pattern is not found.
    *
    * @example {{{
    * v := PlusArgsValue(UInt(32.W), "FOO=%d", 42.U)
    * }}}
    */
  def apply[T <: Data](gen: T, str: String, default: T): T = {
    val result = apply(gen, str)
    Mux(result.found, result.result, default)
  }
}
