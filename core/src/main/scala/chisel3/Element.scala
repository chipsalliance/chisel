// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.experimental.SourceInfo
import chisel3.internal._

/** Element is a leaf data type: it cannot contain other [[Data]] objects. Example uses are for representing primitive
  * data types, like integers and bits.
  *
  * @define coll element
  */
abstract class Element extends Data {
  private[chisel3] final def allElements: Seq[Element] = Seq(this)
  def widthKnown:                         Boolean = width.known
  def name:                               String = getRef.name

  private[chisel3] override def bind(target: Binding, parentDirection: SpecifiedDirection): Unit = {
    this.maybeAddToParentIds(target)
    binding = target
    val resolvedDirection = SpecifiedDirection.fromParent(parentDirection, specifiedDirection)
    direction = ActualDirection.fromSpecified(resolvedDirection)
  }

  private[chisel3] override def topBindingOpt: Option[TopBinding] = super.topBindingOpt match {
    // Translate Bundle lit bindings to Element lit bindings
    case Some(BundleLitBinding(litMap)) =>
      litMap.get(this) match {
        case Some(litArg) => Some(ElementLitBinding(litArg))
        case _            => Some(DontCareBinding())
      }
    case Some(VecLitBinding(litMap)) =>
      litMap.get(this) match {
        case Some(litArg) => Some(ElementLitBinding(litArg))
        case _            => Some(DontCareBinding())
      }
    // TODO Do we even need this? Looking up things in the AggregateViewBinding is fine
    case Some(b @ AggregateViewBinding(viewMap)) =>
      viewMap.get(this) match {
        case Some(elt: Element) => Some(ViewBinding(elt))
        // TODO We could generate a reduced AggregateViewBinding, but is there a point?
        // Generating the new object would be somewhat slow, it's not clear if we should do this
        //   matching anyway
        case Some(data: Aggregate) => Some(b)
        case _ => throwException(s"Internal Error! $this missing from topBinding $b")
      }
    case topBindingOpt => topBindingOpt
  }

  private[chisel3] def litArgOption: Option[LitArg] = topBindingOpt match {
    case Some(ElementLitBinding(litArg)) => Some(litArg)
    case _                               => None
  }

  override def litOption:                Option[BigInt] = litArgOption.map(_.num)
  private[chisel3] def litIsForcedWidth: Option[Boolean] = litArgOption.map(_.forcedWidth)

  private[chisel3] def firrtlConnect(that: Data)(implicit sourceInfo: SourceInfo): Unit = {
    // If the source is a DontCare, generate a DefInvalid for the sink,
    //  otherwise, issue a Connect.
    if (that == DontCare) {
      pushCommand(DefInvalid(sourceInfo, lref))
    } else {
      pushCommand(Connect(sourceInfo, lref, that.ref))
    }
  }

  override def containsAFlipped = specifiedDirection match {
    case SpecifiedDirection.Flip | SpecifiedDirection.Input => true
    case _                                                  => false
  }
}
