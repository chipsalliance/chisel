// SPDX-License-Identifier: Apache-2.0

package chisel3.util.circt

import scala.util.hashing.MurmurHash3

import chisel3._
import chisel3.experimental.{annotate, ChiselAnnotation, ExtModule}
import chisel3.internal.{Builder, BuilderContextCache}

import circt.Intrinsic

// We have to unique designedName per type, be we can't due to name conflicts
// on bundles.  Thus we use a globally unique id.
private object SizeOfGlobalIDGen {
  private case object CacheKey extends BuilderContextCache.Key[Int]
  def getID() = {
    val oldID = Builder.contextCache.getOrElse(CacheKey, 0)
    Builder.contextCache.put(CacheKey, oldID + 1)
    oldID
  }
}

/** Create a module with a parameterized type which returns the size of the type
  * as a compile-time constant.  This lets you write code which depends on the
  * results of type inference.
  */
private class SizeOfIntrinsic[T <: Data](gen: T) extends ExtModule {
  val i = IO(Input(gen));
  val size = IO(Output(UInt(32.W)))
  annotate(new ChiselAnnotation {
    override def toFirrtl =
      Intrinsic(toTarget, "circt.sizeof")
  })
  override val desiredName = "SizeOf_" + SizeOfGlobalIDGen.getID()
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
    val sizeOfInst = Module(new SizeOfIntrinsic(chiselTypeOf(gen)))
    sizeOfInst.i := gen
    sizeOfInst.size
  }
}
