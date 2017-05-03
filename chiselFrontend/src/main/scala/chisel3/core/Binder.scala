package chisel3.core

/**
* A Binder is a function from UnboundBinding to some Binding.
*
* These are used exclusively by Binding.bind and sealed in order to keep
* all of them in one place. There are two flavors of Binders:
* Non-terminal (returns another UnboundBinding): These are used to reformat an
*   UnboundBinding (like setting direction) before it is terminally bound.
* Terminal (returns any other Binding): Due to the nature of Bindings, once a
*   Data is bound to anything but an UnboundBinding, it is forever locked to
*   being that type (as it now represents something in the hardware graph).
*
* Note that some Binders require extra arguments to be constructed, like the
* enclosing Module.
*/

sealed trait Binder[Out <: Binding] extends Function1[UnboundBinding, Out]{
  def apply(in: UnboundBinding): Out
}

// THE NON-TERMINAL BINDERS
// These 'rebind' to another unbound node of different direction!
case object InputBinder extends Binder[UnboundBinding] {
  def apply(in: UnboundBinding) = UnboundBinding(Some(Direction.Input))
}
case object OutputBinder extends Binder[UnboundBinding] {
  def apply(in: UnboundBinding) = UnboundBinding(Some(Direction.Output))
}
case object FlippedBinder extends Binder[UnboundBinding] {
  def apply(in: UnboundBinding) = UnboundBinding(in.direction.map(_.flip))
  // TODO(twigg): flipping a None should probably be a warning/error
}
// The need for this should be transient.
case object NoDirectionBinder extends Binder[UnboundBinding] {
  def apply(in: UnboundBinding) = UnboundBinding(None)
}

// THE TERMINAL BINDERS
case object LitBinder extends Binder[LitBinding] {
  def apply(in: UnboundBinding) = LitBinding()
}

case class MemoryPortBinder(enclosure: UserModule) extends Binder[MemoryPortBinding] {
  def apply(in: UnboundBinding) = MemoryPortBinding(enclosure)
}

case class OpBinder(enclosure: UserModule) extends Binder[OpBinding] {
  def apply(in: UnboundBinding) = OpBinding(enclosure)
}

// Notice how PortBinder uses the direction of the UnboundNode
case class PortBinder(enclosure: BaseModule) extends Binder[PortBinding] {
  def apply(in: UnboundBinding) = PortBinding(enclosure, in.direction)
}

case class RegBinder(enclosure: UserModule) extends Binder[RegBinding] {
  def apply(in: UnboundBinding) = RegBinding(enclosure)
}

case class WireBinder(enclosure: UserModule) extends Binder[WireBinding] {
  def apply(in: UnboundBinding) = WireBinding(enclosure)
}

