// SPDX-License-Identifier: Apache-2.0

package chisel3.util.circt

import chisel3._
import chisel3.experimental.IntrinsicModule
import chisel3.internal.Builder

import circt.Intrinsic

/** Create a module with a parameterized type which calls the verilog function
  * $test$plusargs to test for the existance of the string str in the
  * simulator command line.
  */
private class PlusArgsTestIntrinsic[T <: Data](gen: T, str: String)
    extends IntrinsicModule("circt.plusargs.test", Map("FORMAT" -> str)) {
  val found = IO(Output(UInt(1.W)))
}

object PlusArgsTest {

  /** Creates an intrinsic which calls $test$plusargs.
    *
    * @example {{{
    * b := PlusArgsTest(UInt<32.W>, "FOO")
    * }}}
    */
  def apply[T <: Data](gen: T, str: String): Data = {
    if (gen.isSynthesizable) {
      val inst = Module(new PlusArgsTestIntrinsic(chiselTypeOf(gen), str))
      inst.found
    } else {
      val inst = Module(new PlusArgsTestIntrinsic(gen, str))
      inst.found
    }
  }
}
