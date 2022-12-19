// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import chisel3._
import chisel3.experimental.{annotate, ChiselAnnotation, ExtModule}

import circt.intrinsic

/** Create a module with a parameterized type which returns the size of the type
  * as a compile-time constant.  This lets you write code which depends on the
  * results of type inference.
  */
private class SizeOfIntrinsic [T <: Data](gen: T) extends ExtModule {
  val i    = IO(Input(gen));
  val size = IO(Output(UInt(32.W)));
  annotate(new ChiselAnnotation {
    def toFirrtl =
      intrinsic(toTarget, "circt.sizeof")
  })
  override val desiredName = "SizeOf" + _id.toString()
}

object SizeOf {
  def apply[T <: Data](gen: T): Data = {
    val sizeOfInst  = Module(new SizeOfIntrinsic(chiselTypeOf(gen)));
    sizeOfInst.i := gen
    sizeOfInst.size
  }
}
