package chisel3.experimental

import chisel3.{BlackBox, ChiselException, Data, Record}

import scala.collection.immutable.SeqMap

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
  def apply(data: String) = elements.getOrElse(data, throw new ChiselException(s"$data not found in Verilog IO: \n ${allElements.keys.mkString("\n")}"))
}
