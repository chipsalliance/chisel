package chisel3.experimental.hierarchy.core


// Marker Trait
trait IsStandIn[+T] {

  def parent: Option[IsHierarchical]
  def proto: T

  def toInstance:   Instance[T]
  def toDefinition: Definition[T]
  /** Determines whether another object is a clone of the same proxy proto
    *
    * @param a
    */
  def hasSameProto(a: Any): Boolean = {
    val aProto = a match {
      case is: IsStandIn[_] => is.proto
      case other => other
    }
    this == aProto || proto == aProto
  }
}

case class StandInIsInstantiable[T <: IsInstantiable](proto: T, parent: Option[IsHierarchical]) extends IsStandIn[T] {
  def toInstance:   Instance[T] = new Instance(StandIn(this))
  def toDefinition: Definition[T] = new Definition(StandIn(this))
}

// Wrapper Class
sealed trait Proxy[+T] {
  def proto: T
  def toDefinition = Definition(this)
  def toInstance = Instance(this)
}
// Used for when proxy implementation is pure
final case class Proto[T](proto: T, parent: Option[Proxy[IsHierarchical]]) extends Proxy[T]
// Used for when proxy implementation is not pure, and thus requires a mock up
final case class StandIn[T](isStandIn: IsStandIn[T]) extends Proxy[T] {
  def proto = isStandIn.proto
}

// Typeclass Trait
trait Proxifier[B] extends IsTypeclass[B] {
  type U
  type C = Proxy[U]
  def apply[A](b: B, context: Hierarchy[A])(implicit h: Hierarchicalizer[A]): C
}

// Typeclass Default Implementations
object Proxifier {
  implicit def isLookupable[L <: IsLookupable] = new Proxifier[L] {
    type U = L
    def apply[A](b: L, context: Hierarchy[A])(implicit h: Hierarchicalizer[A]) = Proto(b, h.hierarchy(context))
  }
  implicit def isContextual[L <: IsContextual] = new Proxifier[L] {
    type U = L
    def apply[A](b: L, context: Hierarchy[A])(implicit h: Hierarchicalizer[A]) = Proto(b, h.hierarchy(context))
  }
  implicit def isIsInstantiable[L <: IsInstantiable] = new Proxifier[L] {
    type U = L
    def apply[A](b: L, context: Hierarchy[A])(implicit h: Hierarchicalizer[A]) = h.hierarchy(context) match {
      case None => StandIn(StandInIsInstantiable(b, None))
      case Some(p: Proto[_]) => StandIn(StandInIsInstantiable(b, Some(p.proto)))
      case Some(StandIn(i: IsStandIn[_] with IsHierarchical)) => StandIn(StandInIsInstantiable(b, Some(i)))
      case Some(StandIn(i: IsStandIn[_] with IsInstantiable)) => StandIn(StandInIsInstantiable(b, i.parent))
    }
  }
}