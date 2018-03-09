package chisel3.tester

import chisel3.{Data, Element, Module, Record, Vec}
import chisel3.core.ActualDirection
import chisel3.experimental.DataMirror

object TesterUtils {
  /** Returns a Seq of (data reference, fully qualified element names) for the input.
    * name is the name of data
    */
  def getDataNames(name: String, data: Data): Seq[(Data, String)] = Seq(data -> name) ++ (data match {
    case e: Element => Seq()
    case b: Record => b.elements.toSeq flatMap {case (n, e) => getDataNames(s"${name}_$n", e)}
    case v: Vec[_] => v.zipWithIndex flatMap {case (e, i) => getDataNames(s"${name}_$i", e)}
  })

  // TODO: the naming facility should be part of infrastructure not backend
  def getPortNames(dut: Module) = (getDataNames("io", dut.io) ++ getDataNames("reset", dut.reset) ++ getDataNames("clock", dut.clock)).toMap

  def getIOPorts(dut: Module): (Seq[(Data, String)], Seq[(Data, String)]) = {
    getDataNames("io", dut.io) partition { case (e, _) => DataMirror.directionOf(e) == ActualDirection.Input }
  }
}
