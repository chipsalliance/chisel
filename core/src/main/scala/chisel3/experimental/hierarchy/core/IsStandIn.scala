package chisel3.experimental.hierarchy.core


// Marker Trait
trait IsStandIn[+T] {

  def parent: Option[IsHierarchicable]
  def proto: T
  /** Determines whether another object is a clone of the same underlying proto
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

// Wrapper Class
sealed trait Underlying[+T] {
  def proto: T
  def hierarchy: Option[Underlying[IsHierarchicable]]
  def toDefinition = Definition(this)
  def toInstance = Instance(this)
}
// Used for when underlying implementation is pure
final case class Proto[T](proto: T, parent: Option[Underlying[IsHierarchicable]]) extends Underlying[T] {
  def hierarchy: Option[Underlying[IsHierarchicable]] = proto match {
    case i: IsHierarchicable => Some(i.toUnderlying)
    case other => parent
  }
}
// Used for when underlying implementation is not pure, and thus requires a mock up
final case class StandIn[T](isStandIn: IsStandIn[T]) extends Underlying[T] {
  def hierarchy: Option[Underlying[IsHierarchicable]] = isStandIn match {
    case i: IsHierarchicable => Some(i.toUnderlying)
    case o => isStandIn.parent.map(_.toUnderlying)
  }
}

// Typeclass Trait
trait Viewable[B] extends IsTypeclass[B] {
  type C
  def apply[A](b: B, context: Hierarchy[A]): C
}

// Typeclass Default Implementations
object Viewable {
  implicit def isLookupable[L <: IsLookupable] = new Viewable[L] {
    type C = Underlying[L]
    def apply[A](b: L, context: Hierarchy[A]) = Proto(b, context.underlying.hierarchy)
  }
  implicit def isContextual[L <: IsContextual] = new Viewable[L] {
    type C = Underlying[L]
    def apply[A](b: L, context: Hierarchy[A]) = Proto(b, context.underlying.hierarchy)
  }
}