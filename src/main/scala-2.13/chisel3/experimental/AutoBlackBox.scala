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

  final val io = IO(new AutoBundleFromVerilog(SeqMap.from(verilogModuleIO(ast)))(signalFilter))
}

class AutoBundleFromVerilog(allElements: SeqMap[String, Data])(signalFilter: String => Boolean)
    extends Record
    with AutoCloneType {
  override def elements: SeqMap[String, Data] = allElements.filter(n => signalFilter(n._1))
}
