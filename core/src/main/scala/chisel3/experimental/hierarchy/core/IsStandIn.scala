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
  def myContext: Option[IsHierarchical]
  def lookupContext: Option[IsHierarchical]
}
// Used for when proxy implementation is pure
final case class Proto[T](proto: T, parent: Option[Proxy[IsHierarchical]]) extends Proxy[T] {
  def myContext: Option[IsHierarchical] = parent match {
    case Some(p: Proxy[_]) => p.lookupContext
    case None => None
  }
  def lookupContext: Option[IsHierarchical] = proto match {
    case p: IsHierarchical => Some(p)
    case _ => myContext
  }
}
// Used for when proxy implementation is not pure, and thus requires a mock up
final case class StandIn[T](isStandIn: IsStandIn[T]) extends Proxy[T] {
  def proto = isStandIn.proto
  def myContext: Option[IsHierarchical] = isStandIn.parent
  def lookupContext: Option[IsHierarchical] = isStandIn match {
    case p: IsHierarchical => Some(p)
    case o => myContext
  }
}

// Typeclass Trait
trait Proxifier[V] extends IsTypeclass[V] {
  type U
  type R = Proxy[U]
  def apply[H](value: V, hierarchy: Hierarchy[H]): R
}

// Typeclass Default Implementations
object Proxifier {
  implicit def isIsInstantiable[L <: IsInstantiable, C <: IsHierarchical](implicit contexter: Contexter[C]) =
    new Proxifier[L] {
      type U = L
      def apply[H](b: L, hierarchy: Hierarchy[H]) = StandIn(StandInIsInstantiable(b, contexter.lookupContext(hierarchy)))
    }
}