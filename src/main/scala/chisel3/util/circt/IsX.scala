package chisel3.util.circt

// SPDX-License-Identifier: Apache-2.0

package chisel3.util.circt

import chisel3._
import chisel3.experimental.{annotate, ChiselAnnotation, ExtModule}

import circt.Intrinsic

// We have to unique designedName per type, be we can't due to name conflicts
// on bundles.  Thus we use a globally unique id.
private object IsXGlobalIDGen {
  private var id: Int = 0
  def getID() = {
    this.synchronized {
      val oldID = id
      id = id + 1
      oldID
    }
  }
}

/** Create a module with a parameterized type which returns whether the input
  * is a verilog 'x'.
  */
private class IsXIntrinsic[T <: Data](gen: T) extends ExtModule {
  val i = IO(Input(gen))
  val found = IO(Output(UInt(1.W)))
  annotate(new ChiselAnnotation {
    override def toFirrtl =
      Intrinsic(toTarget, "circt.isX")
  })
  override val desiredName = "IsX_" + SizeOfGlobalIDGen.getID()
}

object IsX {

  /** Creates an intrinsic which returns whether the input is a verilog 'x'.
    *
    * @example {{{
    * b := isX(a)
    * }}}
    */
  def apply[T <: Data](gen: T): Data = {
    val inst = Module(new IsXIntrinsic(chiselTypeOf(gen)))
    inst.i := gen
    inst.found
  }
}
