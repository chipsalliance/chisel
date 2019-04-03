// See LICENSE for license details.

package chisel3.experimental

import chisel3.{ActualDirection, Data, Element, Record, SpecifiedDirection, Vec}
import chisel3.internal.firrtl.Width


/** Experimental hardware construction reflection API
  */
object DataMirror {
  def widthOf(target: Data): Width = target.width
  def specifiedDirectionOf(target: Data): SpecifiedDirection = target.specifiedDirection
  def directionOf(target: Data): ActualDirection = {
    requireIsHardware(target, "node requested directionality on")
    target.direction
  }

  // Returns the top-level module ports
  // TODO: maybe move to something like Driver or DriverUtils, since this is mainly for interacting
  // with compiled artifacts (vs. elaboration-time reflection)?
  def modulePorts(target: BaseModule): Seq[(String, Data)] = target.getChiselPorts

  // Returns all module ports with underscore-qualified names
  def fullModulePorts(target: BaseModule): Seq[(String, Data)] = {
    def getPortNames(name: String, data: Data): Seq[(String, Data)] = Seq(name -> data) ++ (data match {
      case _: Element => Seq()
      case r: Record => r.elements.toSeq flatMap { case (eltName, elt) => getPortNames(s"${name}_${eltName}", elt) }
      case v: Vec[_] => v.zipWithIndex flatMap { case (elt, index) => getPortNames(s"${name}_${index}", elt) }
    })
    modulePorts(target).flatMap { case (name, data) =>
      getPortNames(name, data).toList
    }
  }

  // Internal reflection-style APIs, subject to change and removal whenever.
  object internal { // scalastyle:ignore object.name
    def isSynthesizable(target: Data): Boolean = target.topBindingOpt.isDefined
    // For those odd cases where you need to care about object reference and uniqueness
    def chiselTypeClone[T<:Data](target: Data): T = {
      target.cloneTypeFull.asInstanceOf[T]
    }
  }
}
