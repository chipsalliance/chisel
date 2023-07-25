package chisel3

import chisel3.internal.plugin.{autoNameRecursively, autoNameRecursivelyProduct, withName}
import chisel3.util.Mux1H
import circt.stage.ChiselStage.emitCHIRRTL

import scala.reflect.runtime.universe.showRaw

object PluginNamingSpec {

  def MuxT1H[T <: Data, U <: Data](sel: UInt, in: Seq[(T, U)]): (T, U) =
    (Mux1H(sel, in.map(_._1)), Mux1H(sel, in.map(_._2)))

  abstract class BaseIntf extends Module {
    val sel = IO(Input(UInt(2.W)))
    val ins = IO(Input(Vec(2, UInt(8.W))))
    val div = IO(Input(UInt(8.W)))
    val out1, out2 = IO(Output(UInt(8.W)))
  }

  // It's not uncommon to just assign the returned tuple to a val
  // In this case there are no names AND no prefixes
  class TupleVal extends BaseIntf {
    val foobar = MuxT1H(sel, ins.map(x => (x / div, x % div)))
//    val foobarTuple = (7.U, 15.U, 23)
//    println("U" * 80 + "\n" + withName("newName", (7.U, 15.U, 23)))
    def wire1 = Wire(UInt(8.W))
    def wire2 = Wire(UInt(16.W))

    // METHOD 1
//    val foobarTuple = chisel3.internal.plugin.autoNameRecursivelyProduct(List(Some("FOO"), Some("BAR"), None))((wire1, wire2, 23))
//    val foobarTuple = withName("AVAVAVAVA", (wire1, wire2, 23))

     // METHOD 2
//        val vv = chisel3.internal.plugin.autoNameRecursivelyProduct(List(Some("AA"), Some("BB"), None))(foobarTuple)
    val foobarTuple = (wire1, wire2, 23)
//    val vv = chis
    //
    //
    //
    //
    //    80 + "\n" + showRaw(vv))

    out1 := foobar._1 & 0xf0.U + foobarTuple._1
    out2 := foobar._2 & 0x0f.U + foobarTuple._2

  }

  def main(args: Array[String]): Unit = {
    val chirrtl = emitCHIRRTL(new TupleVal)
    println(chirrtl)
  }
}

object HackFest {
  def main(args: Array[String]): Unit = {
    class M extends Module {
      val foobarTuple = (3.U, 7, 15.U)
      val foobarIn = IO(Input(UInt()))
      val foobarOut = foobarTuple._1 + foobarTuple._3

      foobarTuple.productIterator.foreach(println)
      foobarTuple.productIterator.foreach {
        case d: Data =>
          d.suggestName("puppy")
        case _ =>

      }

    }
    println(emitCHIRRTL(new M))
  }
}
