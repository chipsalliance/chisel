// SPDX-License-Identifier: Apache-2.0

package chisel3.util.experimental

import chisel3._

import scala.collection.immutable.SeqMap
import chisel3.experimental.SourceInfo

class AutoBlackBox(
  val verilog:      String,
  val signalFilter: String => Boolean = _ => true
)(
  implicit val __sourceInfo: SourceInfo)
    extends FixedIOExtModule(
      ioGenerator = new AutoBundleFromVerilog(
        SeqMap.from(SlangUtils.verilogModuleIO(SlangUtils.getVerilogAst(verilog)))
      )(signalFilter)
    ) {
  override def desiredName = SlangUtils.verilogModuleName(SlangUtils.getVerilogAst(verilog))
  override def _sourceInfo = __sourceInfo
}

class AutoBundleFromVerilog(
  allElements:  SeqMap[String, Data]
)(signalFilter: String => Boolean)
    extends Record {
  val elements: SeqMap[String, Data] = allElements.filter(n => signalFilter(n._1)).map {
    case (k, v) =>
      k -> chisel3.reflect.DataMirror.internal.chiselTypeClone(v)
  }
  def apply(data: String) = elements.getOrElse(
    data,
    throw new ChiselException(
      s"$data not found in Verilog IO: \n ${allElements.keys.mkString("\n")}"
    )
  )
}
