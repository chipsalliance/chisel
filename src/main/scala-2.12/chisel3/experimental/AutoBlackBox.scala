package chisel3.experimental

import chisel3.{BlackBox, Data, Record}
import scala.collection.immutable.{ListMap, SeqMap}

trait AutoBlackBox extends BlackBox {
  import chisel3.util.experimental.SlangUtils._

  def verilog: String

  def signalFilter: String => Boolean = _ => true

  override val params = verilogParameter(ast).toMap

  override def desiredName = verilogModuleName(ast)

  lazy val ast = getVerilogAst(verilog)

  final val io = IO(new AutoBundleFromVerilog(ListMap(verilogModuleIO(getVerilogAst(verilog)): _*))(signalFilter))
}

class AutoBundleFromVerilog(allElements: ListMap[String, Data])(signalFilter: String => Boolean)
    extends Record
    with AutoCloneType {
  override def elements: ListMap[String, Data] = allElements.filter(n => signalFilter(n._1))
}
