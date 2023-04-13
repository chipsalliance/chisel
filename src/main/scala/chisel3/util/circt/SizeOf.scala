// SPDX-License-Identifier: Apache-2.0

package chisel3.util.circt

import chisel3._
import chisel3.experimental.IntrinsicModule
import chisel3.internal.{Builder, BuilderContextCache}

/** Create a module with a parameterized type which returns the size of the type
  * as a compile-time constant.  This lets you write code which depends on the
  * results of type inference.
  */
private class SizeOfIntrinsic[T <: Data](gen: T) extends IntrinsicModule("circt_sizeof") {
  val i = IO(Input(gen))
  val size = IO(Output(UInt(32.W)))
}

object SizeOf {

  /** Creates an intrinsic which returns the size of a type.  The returned size
    * is after width inference, so you can use this to compute expressions based
    * on the inferred size of types.
    *
    * @example {{{
    * val a = Wire(UInt())
    * a := 1 << (SizeOf(a) - 1)
    * }}}
    */
  def apply[T <: Data](gen: T): Data = {
    val inst = Module(new SizeOfIntrinsic(chiselTypeOf(gen)))
    inst.i := gen
    inst.size
  }
}
