package chisel3.experimental.hierarchy.core


case class StandInIsInstantiable[T <: IsInstantiable](proto: T, parent: Option[IsContext]) extends IsStandIn[T] {
  def toInstance:   Instance[T] = new Instance(StandIn(this))
  def toDefinition: Definition[T] = new Definition(StandIn(this))
}


// Wrapper Class
sealed trait Proxy[+T] {
  def proto: T
  def toDefinition = Definition(this)
  def toInstance = Instance(this)
  def myContext: Context[IsContext]
  def lookupContext: Context[IsContext]
}

// Used for when proxy implementation is pure
final case class Proto[T](proto: T, parent: Option[Proxy[IsContext]]) extends Proxy[T] {
  def myContext: Context[IsContext] = parent match {
    case Some(p: Proxy[_]) => p.lookupContext
    case None => Context(None)
  }
  def lookupContext: Context[IsContext] = proto match {
    case p: IsContext => Context(Some(p))
    case _ => myContext
  }
}
// Used for when proxy implementation is not pure, and thus requires a mock up
final case class StandIn[T](isStandIn: IsStandIn[T]) extends Proxy[T] {
  def proto = isStandIn.proto
  def myContext: Context[IsContext] = Context(isStandIn.parent)
  def lookupContext: Context[IsContext] = isStandIn match {
    case p: IsContext => Context(Some(p))
    case o => myContext
  }
}

// Typeclass Trait
trait Proxifier[V]  {
  type U
  type R = Proxy[U]
  def apply[P](value: V, hierarchy: Hierarchy[P]): R
}

// Typeclass Default Implementations
object Proxifier {
  implicit def isIsInstantiable[L <: IsInstantiable, C <: IsContext](implicit contexter: Contexter[L, C]) =
    new Proxifier[L] {
      type U = L
      def apply[P](value: L, hierarchy: Hierarchy[P]) = StandIn(StandInIsInstantiable(value, contexter(value, hierarchy).context))
    }
}

trait ProxyDefiner[T] {
  def apply(f: => T): Proxy[T]
}

trait ProxyInstancer[T] {
  def apply(definition: Definition[T]): Proxy[T]
}