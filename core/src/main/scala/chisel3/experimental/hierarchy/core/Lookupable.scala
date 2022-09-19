// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core
import scala.collection.mutable.HashMap
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{DefinitionTransform, InstanceTransform}
import scala.language.higherKinds // Necessary for lookupableIterable
import Contextual.log

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
        val id = getter.identity * value.identity
        log(s"LookupInstance(0)$id: getter=${getter.debug}, value=${value.debug}")
        if (shareProto(value.proxy, getter)) getter.asInstanceOf[Instance[V]]
        else {
          // Value should always have a parent because it is an instance, so this is safe to get
          val oldParent = extensions.getProxyParent(value.proxy).get
          val rt = if(oldParent.isInstanceOf[Proxy[_]]) oldParent.asInstanceOf[Proxy[_]].debug else oldParent.toString
          log(s"LookupInstance(1)$id: oldParent=${rt}")
          val newParentWrapper = getter._lookup { _ => extensions.getProxyParent(value.proxy).get }(
            lookupUncloneableValue(extensions.parentExtensions),
            mg
          )
          log(s"LookupInstance(1)$id: newParent=${newParentWrapper.proxy.debug}")
          (oldParent, newParentWrapper.proxy) match {
            case (oldParent, newParent) if oldParent == newParent => value
            case (oldParent, newParent: RootProxy[_]) => value
            case (oldParent, newParent: InstanceProxy[_]) =>
              val suffix = newParent.suffixProxy.toHierarchy._lookup { _ => value }(lookupInstance(extensions), mg)
              if(value.proxy.suffixProxy == suffix.proxy) value else extensions.mock(suffix.proxy, newParentWrapper)
          }
          //val suffix = newParentWrapper match {
          //  case Instance(proxy) if proxy.suffixProxy == getter.proxy => value // value is outside scope of getter
          //  case Instance(proxy)           => proxy.suffixProxy.toHierarchy._lookup { _ => value }(lookupInstance(extensions), mg)
          //  case Definition(proxy)         => value
          //  case ResolvedDefinition(proxy) => value
          //}
          //extensions.mock(suffix.proxy, newParentWrapper)
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
      val id = value.identity * getter.identity
      // Converting Wrapper to Contextual shouldn't error, as you can't create an Instance(this)
      log(s"LookupContextual(0)$id: getter=${getter.debug}, value=${value.debug}")
      val ret = extensions.getProxyParent(value.proxy) match {
        case None                         => value
        case Some(p) if p == getter.proxy => value
        case Some(p)                      =>
          // Value should always have a parent because it is an instance, so this is safe to get
          val rt = if(p.isInstanceOf[Proxy[_]]) p.asInstanceOf[Proxy[_]].debug else p.toString
          log(s"LookupContextual(1)$id: parent=$rt")
          val newParentWrapper = getter._lookup { _ => p }(lookupUncloneableValue(extensions.parentExtensions), mg)
          log(s"LookupContextual(2)$id: newParent=${newParentWrapper.debug}")

          //(newParentWrapper.proxy, getter.proxy) match {
          //  case (newParentProxy, getterProxy) if newParentProxy == getterProxy => 
          //}

          val x = (p, newParentWrapper.proxy) match {
            case (oldParent, newParent) if oldParent == newParent =>
              log(s"LookupContextual(3)$id: newParent == oldParent, returning value ${value.debug}")
              value
            //case (oldParent, newParent: InstanceProxy[_]) if newParent.suffixProxy == getter.proxy =>
            //  log(s"LookupContextual(4)$id: newParent != oldParent, newParent.suffixProxy == getter.proxy, returning ${value.debug}")
            //  value
            case (oldParent, newParent: RootProxy[_]) =>
              log(s"LookupContextual(5)$id: newParent != oldParent, newParent=RootProxy")
              val y = extensions.mockContextual(value, newParentWrapper)
              log(s"LookupContextual(6)$id: newParent != oldParent, returning ${y.debug}")
              y
            case (oldParent, newParent: InstanceProxy[_]) =>
              log(s"LookupContextual(7)$id: newParent != oldParent, newParent=InstanceProxy, newParentsuffix = ${newParent.suffixProxy.debug}")
              val newSuffixProxy = newParent.suffixProxy.toHierarchy._lookup { _ => value.protoContextual}(lookupContextual(extensions), mg)
              val oldSuffixProxy = value.proxy.suffixProxyOpt
              log(s"LookupContextual(8)$id: newSuffixProxy = ${newSuffixProxy.debug}, oldSuffixProxy = ${oldSuffixProxy.map(_.debug)}")
              val y = if(Some(newSuffixProxy) != value.proxy.suffixProxyOpt) {
                log(s"LookupContextual(9)$id: Mocking!}")
                extensions.mockContextual(newSuffixProxy, newParentWrapper)
              } else {
                log(s"LookupContextual(A)$id: Not Mocking!}")
                value
              }
              log(s"LookupContextual(B)$id: newParent != oldParent, returning ${y.debug}")
              y
            //case Instance(newParent) if newParent.suffixProxy == getter.proxy => Some(value)
            //case Instance(newParent) => // if proxy.suffixProxy != getter.proxy =>
            //  log(s"LookupContextual(3)$id: contextParentSuffix${proxy.suffixProxy.debug}")
            //  //require(proxy.suffixProxy != getter.proxy, s"suffixProxy cannot equal getterProxy")
            //  val ret = proxy.suffixProxy.toHierarchy._lookup { _ => value }(lookupContextual(extensions), mg)
            //  log(s"LookupContextual(4)$id: ${ret.debug}")
            //  Some(ret)
          }
          //(suffixProxyOpt, newParentWrapper.proxy, getter.proxy) match {
          //  case (None, newParentProxy, getterProxy) if newParentProxy == getterProxy => extensions.mockContextual()
          //}
          //val x = if(suffixProxy.nonEmpty && suffixProxy.get != value) extensions.mockContextual(suffixProxy.get, newParentWrapper) else value
          ////val x = if(suffixProxy != value) extensions.mockContextual(suffixProxy, newParentWrapper) else value
          //println(s"Mocking contextual $x with suffix $suffixProxy and parent $newParentWrapper")
          x
      }
      log(s"LookupContextual(C)$id: ${getter.debug}?${value.debug} => ${ret.debug}")
      ret
    }
  }

  //Used for looking up modules
  implicit def lookupUncloneableValue[V, P](
    implicit extensions: HierarchicalExtensions[V, P]
  ): HierarchicalLookupable[V] = new HierarchicalLookupable[V] {
    def apply[P](getter: Wrapper[P], value: V) = {
      val id = getter.identity * value.hashCode()
      log(s"LookupUncloneableValue(0)$id: getter=${getter.debug}, value=${value}")
      val relevantGetter = getter match {
        case x: Hierarchy[_] =>
          x.getClosestParentOf(extensions.parentSelection) match {
            case Some(h: Hierarchy[_]) => h
            case None => getter
          }
      }
      log(s"LookupUncloneableValue(1)$id: relevantGetter=${relevantGetter.debug}")
      //TODO This may be unnecessary if IsWrappables record a parent
      val ret = value match {
        case v if shareProto(v, relevantGetter) => relevantGetter.asInstanceOf[Hierarchy[V]]
        case other =>
          extensions.getParent(value) match {
            // I think i'm missing a case here, where a parent is a constructor argument to a child,
            // looked up from the grandparent.
            // In this case, we need to lookup value from getter's parent
            case None =>
              relevantGetter match {
                case i: Instance[_] =>
                  log(s"LookupUncloneableValue(2)$id: ${i.proxy.parentOpt}")
                  i.proxy.parentOpt match {
                    case None =>
                      log(s"LookupUncloneableValue(3)$id: ${i.proxy.parentOpt}")
                      extensions.toDefinition(value)
                    case Some(p: Proxy[_]) =>
                      log(s"LookupUncloneableValue(4)$id: ${i.proxy.parentOpt}")
                      p.toWrapper._lookup(_ => value)(lookupUncloneableValue(extensions), mg)
                  }
                case other =>
                  log(s"LookupUncloneableValue(5)$id: notInstance: $other")
                  extensions.toDefinition(value)
              }
            case Some(p) =>
              log(s"LookupUncloneableValue(6)$id: p= $p")
              val newParentHierarchy =
                relevantGetter._lookup(_ => p)(lookupUncloneableValue(extensions.parentExtensions), mg)
              log(s"LookupUncloneableValue(6)$id: newParentHierarchy: ${newParentHierarchy.debug}")
              extensions.mock(value, newParentHierarchy)
              //value match {
              //  case i: InstanceProxy[V] => extensions.mockInstance(i.toInstance, newParentHierarchy)
              //  case o => extensions.mockValue(value, newParentHierarchy)
              //}
          }
      }
      log(s"LookupUncloneableValue(X)$id: ret=${ret}")
      ret
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
