// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core
import scala.collection.mutable.HashMap
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{DefinitionTransform, InstanceTransform}
import scala.language.higherKinds // Necessary for lookupableIterable

/** Typeclass describing the return result (and type) when looking up a value of type V from a Wrapper[P]
  *
  * Default implementations are contained in the companion object
  */
trait Lookupable[-V] {
  def shareProto[P](p: Any, h: Wrapper[P]): Boolean = p match {
    case x: Proxy[_] => x.proto == h.proto
    case o => o == h.proto
  }

  type H
  def apply[P](getter: Wrapper[P], value: V): H
}

trait HierarchicalLookupable[V] extends Lookupable[V] {
  type H = Hierarchy[V]
}

trait ContextualLookupable[V] extends Lookupable[V] {
  type H = Contextual[V]
}

/** Default implementations of Lookupable */
object Lookupable {
  implicit val mg = new chisel3.internal.MacroGenerated {}

  /** Simple Lookupable implementation where the value is always returned, without modification */
  trait SimpleLookupable[V] extends Lookupable[V] {
    override type H = V
    override def apply[P](getter: Wrapper[P], value: V): H = value
  }

  // Lookups for primitive and simple objects

  implicit val lookupableInt = new SimpleLookupable[Int] {}
  implicit val lookupableByte = new SimpleLookupable[Byte] {}
  implicit val lookupableShort = new SimpleLookupable[Short] {}
  implicit val lookupableLong = new SimpleLookupable[Long] {}
  implicit val lookupableFloat = new SimpleLookupable[Float] {}
  implicit val lookupableChar = new SimpleLookupable[Char] {}
  implicit val lookupableBoolean = new SimpleLookupable[Boolean] {}
  implicit val lookupableBigInt = new SimpleLookupable[BigInt] {}

  // Lookups for simple containers of other values
  implicit def lookupableOption[B](implicit lookupable: Lookupable[B]) = new Lookupable[Option[B]] {
    override type H = Option[lookupable.H]
    override def apply[P](getter: Wrapper[P], value: Option[B]): H = value.map { x: B => lookupable[P](getter, x) }
  }

  implicit def lookupableEither[X, Y](implicit lookupableX: Lookupable[X], lookupableY: Lookupable[Y]) =
    new Lookupable[Either[X, Y]] {
      override type H = Either[lookupableX.H, lookupableY.H]
      override def apply[P](getter: Wrapper[P], value: Either[X, Y]): H = value.map { y: Y =>
        lookupableY[P](getter, y)
      }.left.map { x: X => lookupableX[P](getter, x) }
    }

  implicit def lookupableTuple2[X, Y](implicit lookupableX: Lookupable[X], lookupableY: Lookupable[Y]) =
    new Lookupable[Tuple2[X, Y]] {
      override type H = Tuple2[lookupableX.H, lookupableY.H]
      override def apply[P](getter: Wrapper[P], value: Tuple2[X, Y]): H =
        (lookupableX[P](getter, value._1), lookupableY(getter, value._2))
    }

  implicit def lookupableIterable[B, F[X] <: Iterable[X]](
    implicit lookupable: Lookupable[B]
  ) = new Lookupable[F[B]] {
    override type H = F[lookupable.H]
    override def apply[P](getter: Wrapper[P], value: F[B]): H = {
      val ret = value.asInstanceOf[Iterable[B]]
      ret.map { x: B => lookupable[P](getter, x) }.asInstanceOf[H]
    }

    //result.bind(newBinding)
    //result.setAllParents(Some(ViewParent))
    //result.forceName("view", Builder.viewNamespace)
    //result
  }

  implicit def lookupIsWrappable[U <: IsWrappable] = new Lookupable[U] {
    override type H = Instance[U]
    override def apply[P](getter: Wrapper[P], value: U): Instance[U] = {
      val d = IsWrappableDefinition(Raw(value))
      val t = IsWrappableTransparent(d)
      val m = IsWrappableMock(t, getter.proxy)
      m.toInstance
    }
  }

  implicit def isLookupable[V <: IsLookupable] = new SimpleLookupable[V] {}

  type IAux[V] = Lookupable[Instance[V]] { type H = Instance[V] }

  implicit def lookupInstance[V, P](implicit extensions: HierarchicalExtensions[V, P]): IAux[V] =
    new Lookupable[Instance[V]] {
      type H = Instance[V]
      def apply[P](getter: Wrapper[P], value: Instance[V]) = {
        // Converting Wrapper to Instance shouldn't error, as you can't create an Instance(this)
        if (shareProto(value.proxy, getter)) getter.asInstanceOf[Instance[V]]
        else {
          // Value should always have a parent because it is an instance, so this is safe to get
          val newParentWrapper = getter._lookup { _ => extensions.getProxyParent(value.proxy).get }(
            lookupUncloneableValue(extensions.parentExtensions),
            mg
          )
          val suffix = newParentWrapper match {
            case Instance(proxy)           => proxy.suffixProxy.toHierarchy._lookup { _ => value }(lookupInstance(extensions), mg)
            case Definition(proxy)         => value
            case ResolvedDefinition(proxy) => value
          }
          extensions.mock(suffix.proxy, newParentWrapper)
        }
      }
    }

  implicit def lookupDefinitive[V, P] = new Lookupable[Definitive[V]] {
    type H = Definitive[V]
    def apply[P](getter: Wrapper[P], value: Definitive[V]) = {
      // it is a definitive value, so we can return it!
      value
    }
  }
  type Aux[I, O] = Lookupable[I] { type H = O }

  implicit def lookupContextual[V, P](
    implicit extensions: ParameterExtensions[V, P]
  ): Aux[Contextual[V], Contextual[V]] = new Lookupable[Contextual[V]] {
    type H = Contextual[V]
    def apply[P](getter: Wrapper[P], value: Contextual[V]) = {
      // Converting Wrapper to Contextual shouldn't error, as you can't create an Instance(this)
      extensions.getProxyParent(value.proxy) match {
        case None                         => value
        case Some(p) if p == getter.proxy => value
        case Some(p)                      =>
          // Value should always have a parent because it is an instance, so this is safe to get
          val newParentWrapper = getter._lookup { _ => p }(lookupUncloneableValue(extensions.parentExtensions), mg)

          val suffixProxy = newParentWrapper match {
            case Instance(proxy) =>
              proxy.suffixProxy.toHierarchy._lookup { _ => value }(lookupContextual(extensions), mg)
            case Definition(proxy)         => value
            case ResolvedDefinition(proxy) => value
          }
          val x = extensions.mockContextual(suffixProxy, newParentWrapper)
          //println(s"Mocking contextual $x with suffix $suffixProxy and parent $newParentWrapper")
          x
      }
    }
  }

  //Used for looking up modules
  implicit def lookupUncloneableValue[V, P](
    implicit extensions: HierarchicalExtensions[V, P]
  ): HierarchicalLookupable[V] = new HierarchicalLookupable[V] {
    def apply[P](getter: Wrapper[P], value: V) = {
      val relevantGetter = getter match {
        case x: Hierarchy[_] =>
          x.getClosestParentOf(extensions.parentSelection) match {
            case Some(h: Hierarchy[_]) => h
            case None => getter
          }
      }
      //TODO This may be unnecessary if IsWrappables record a parent
      value match {
        case v if shareProto(v, relevantGetter) => relevantGetter.asInstanceOf[Hierarchy[V]]
        case other =>
          extensions.getParent(value) match {
            // I think i'm missing a case here, where a parent is a constructor argument to a child,
            // looked up from the grandparent.
            // In this case, we need to lookup value from getter's parent
            case None =>
              relevantGetter match {
                case i: Instance[_] =>
                  i.proxy.parentOpt match {
                    case None => extensions.toDefinition(value)
                    case Some(p: Proxy[_]) =>
                      p.toWrapper._lookup(_ => value)(lookupUncloneableValue(extensions), mg)
                  }
                case other => extensions.toDefinition(value)
              }
            case Some(p) =>
              val newParentHierarchy =
                relevantGetter._lookup(_ => p)(lookupUncloneableValue(extensions.parentExtensions), mg)
              extensions.mockValue(value, newParentHierarchy)
          }
      }
    }
    //def apply[P](setter: Setter[P], value: V) = apply(setter.toWrapper, value).toSetter
  }

  // Used for looking up data/membase
  implicit def lookupCloneableValue[V, P](implicit ce: CloneableExtensions[V, P]) = new Lookupable[V] {
    type H = V
    def apply[P](getter: Wrapper[P], value: V): V = ce.getParent(value) match {
      case None => value
      case Some(p) =>
        val newParentHierarchy = getter._lookup(_ => p)(lookupUncloneableValue(ce.parentExtensions), mg)
        ce.clone(value, newParentHierarchy)
    }
  }
}
