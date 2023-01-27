// SPDX-License-Identifier: Apache-2.0

package chisel3.util.circt

import chisel3._
import chisel3.experimental.{annotate, ChiselAnnotation, ExtModule}

import circt.Intrinsic

// We have to unique designedName per type, be we can't due to name conflicts
// on bundles.  Thus we use a globally unique id.
private object PlusArgsTestGlobalIDGen {
  private var id: Int = 0
  def getID() = {
    this.synchronized {
      val oldID = id
      id = id + 1
      oldID
    }
  }
}

/** Create a module with a parameterized type which calls the verilog function
  * $test$plusargs to test for the existance of the string str in the
  * simulator command line.
  */
private class PlusArgsTestIntrinsic[T <: Data](gen: T, str: String) extends ExtModule(Map("FORMAT" -> str)) {
  val found = IO(Output(UInt(1.W)))
  annotate(new ChiselAnnotation {
    override def toFirrtl =
      Intrinsic(toTarget, "circt.plusargs.test")
  })
  override val desiredName = "PlusArgsTest_" + PlusArgsValueGlobalIDGen.getID()
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
