// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import chisel3._
import chisel3.experimental.{annotate, ChiselAnnotation, ExtModule}

import circt.intrinsic

// We need a unique id for each external module. We also want this to be 
// deterministic, so we can't use UUID.  This wouldn't be needed if we could
// make ExtModules without a desiredName.
private object SizeOfUniqueID {
  var value = 0
}

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
  def uuid = java.util.UUID.randomUUID.toString
  override val desiredName = "SizeOf" + SizeOfUniqueID.value.toString()
  SizeOfUniqueID.value = SizeOfUniqueID.value + 1
}

object SizeOf {
  def apply[T <: Data](gen: T): Data = {
    val sizeOfInst  = Module(new SizeOfIntrinsic(chiselTypeOf(gen)));
    sizeOfInst.i := gen
    sizeOfInst.size
  }
}
