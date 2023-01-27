// SPDX-License-Identifier: Apache-2.0

package chisel3.util.circt

import chisel3._
import chisel3.experimental.{annotate, ChiselAnnotation, ExtModule, FlatIO}

import circt.Intrinsic

// We have to unique designedName per type, be we can't due to name conflicts
// on bundles.  Thus we use a globally unique id.
private object PlusArgsValueGlobalIDGen {
  private var id: Int = 0
  def getID() = {
    this.synchronized {
      val oldID = id
      id = id + 1
      oldID
    }
  }
}

/** Create a module which generates a verilog $plusargs$value.  This returns a
  * single value as indicated by the format string.
  */
private class PlusArgsValueIntrinsic[T <: Data](gen: T, str: String) extends ExtModule(Map("FORMAT" -> str)) {
  val io = FlatIO(new Bundle {
      val found= Output(UInt(1.W))
    val result = Output(gen)
  }
  )
  annotate(new ChiselAnnotation {
    override def toFirrtl =
      Intrinsic(toTarget, "circt.plusargs.value")
  })
  override val desiredName = "PlusArgsValue_" + PlusArgsValueGlobalIDGen.getID()
}

object PlusArgsValue {

  /** Creates an intrinsic which returns whether the input is a verilog 'x'.
    *
    * @example {{{
    * b := isX(a)
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