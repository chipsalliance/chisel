// See LICENSE for license details.

package chisel3

import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.internal._

/** Element is a leaf data type: it cannot contain other [[Data]] objects. Example uses are for representing primitive
  * data types, like integers and bits.
  *
  * @define coll element
  */
abstract class Element extends Data {
  private[chisel3] final def allElements: Seq[Element] = Seq(this)
  def widthKnown: Boolean = width.known
  def name: String = getRef.name

  private[chisel3] override def bind(target: Binding, parentDirection: SpecifiedDirection) {
    binding = target
    val resolvedDirection = SpecifiedDirection.fromParent(parentDirection, specifiedDirection)
    direction = ActualDirection.fromSpecified(resolvedDirection)
  }

  private[chisel3] override def topBindingOpt: Option[TopBinding] = super.topBindingOpt match {
    // Translate Bundle lit bindings to Element lit bindings
    case Some(BundleLitBinding(litMap)) => litMap.get(this) match {
      case Some(litArg) => Some(ElementLitBinding(litArg))
      case _ => Some(DontCareBinding())
    }
    case topBindingOpt => topBindingOpt
  }

  private[chisel3] def litArgOption: Option[LitArg] = topBindingOpt match {
    case Some(ElementLitBinding(litArg)) => Some(litArg)
    case _ => None
  }

  override def litOption: Option[BigInt] = litArgOption.map(_.num)
  private[chisel3] def litIsForcedWidth: Option[Boolean] = litArgOption.map(_.forcedWidth)

  // provide bits-specific literal handling functionality here
  override private[chisel3] def ref: Arg = topBindingOpt match {
    case Some(ElementLitBinding(litArg)) => litArg
    case Some(BundleLitBinding(litMap)) => litMap.get(this) match {
      case Some(litArg) => litArg
      case _ => throwException(s"internal error: DontCare should be caught before getting ref")
    }
    case _ => super.ref
  }

  private[chisel3] def legacyConnect(that: Data)(implicit sourceInfo: SourceInfo): Unit = {
    // If the source is a DontCare, generate a DefInvalid for the sink,
    //  otherwise, issue a Connect.
    if (that == DontCare) {
      pushCommand(DefInvalid(sourceInfo, Node(this)))
    } else {
      pushCommand(Connect(sourceInfo, Node(this), that.ref))
    }
  }
}
