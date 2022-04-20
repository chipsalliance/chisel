// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core
import java.util.IdentityHashMap
import scala.collection.mutable

/** Representation of a hierarchical version of an object
  * Has two subclasses, a DefinitionProxy and InstanceProxy, which are wrapped with Definition(..) and Instance(..)
  */
trait Proxy[+P] {

  /** @return Original object that we are proxy'ing */
  def proto: P

  private[chisel3] val proxyCache = new IdentityHashMap[Any, Proxy[Any]]()
  private[chisel3] val cache = new IdentityHashMap[Any, Any]()

  def cacheMe[V, R](protoValue: V, retValue: R): Unit = retValue match {
    case h: Hierarchy[_] => proxyCache.put(protoValue, h.proxy)
    case other => protoValue match {
      case _ => cache.put(protoValue, other)
    }
  }
  def retrieveMeAsWrapper[V](protoValue: V): Option[Any] = {
    if(proxyCache.containsKey(protoValue)) Some(proxyCache.get(protoValue).toWrapper) else None
  }
  def retrieveMe[V](protoValue: V): Option[Any] = {
    if(cache.containsKey(protoValue)) Some(cache.get(protoValue)) else None
  }

  /** @return a Definition wrapping this Proxy */
  //def toSetter: Setter[P]
  def toWrapper: Wrapper[P]
}

trait HierarchicalProxy[+P] extends Proxy[P] {
  //override def toSetter: HierarchySetter[P]
  def toHierarchy: Hierarchy[P]
  def toRoot: Root[P]
  def parentOpt: Option[Proxy[Any]]
  def toWrapper: Hierarchy[P] = toHierarchy
}

/** Proxy representing an Instance version of an object */
sealed trait InstanceProxy[+P] extends HierarchicalProxy[P] {

  /** The proxy which refers to the same proto, but whose root is one hierarchical step closer to the proto
    *
    * Example 0: if this Proxy refers to ~Top|Foo/bar:Bar, then suffixProxy refers to ~Top|Bar
    * Example 1: if this Proxy refers to ~Top|Top/foo:Foo/bar:Bar, then suffixProxy refers to ~Top|Foo/bar:Bar
    *
    * @return the suffixProxy proxy of this proxy
    */
  def suffixProxy: HierarchicalProxy[P]

  /** @return the InstanceProxy closest to the proto in the chain of suffixProxy proxy's */
  def localProxy: InstanceProxy[P] = suffixProxy match {
    case d: DefinitionProxy[P] => this
    case i: InstanceProxy[P]   => i.localProxy
  }

  override def proto = suffixProxy.proto

  /** @return an Instance wrapping this Proxy */
  def toInstance = new Instance(this)

  override def toRoot: Root[P] = suffixProxy.toRoot

  override def toHierarchy: Hierarchy[P] = toInstance

  //override def toSetter: HierarchySetter[P] = HierarchySetter(this)

  /** If this proxy was created by being looked up in a parent proxy, parent refers to that parent proxy.
    * @return parent proxy from whom this proxy was looked up from
    */
  def parentOpt: Option[Proxy[Any]]
}

/** InstanceProxy representing a new Instance which was instantiated in a proto.
  *
  * Note: Clone's parentOpt is always empty
  * Note: Clone's suffixProxy is always a DefinitionProxy
  *
  * E.g. it is the underlying proxy of Instance when a user writes:
  *   val inst = Instance(..)
  */
trait Clone[+P] extends InstanceProxy[P] {
  override def parentOpt: Option[Proxy[Any]] = None

  override def suffixProxy: RootProxy[P]
}

/** InstanceProxy representing an Instance which was created from converting an existing object into an Instance.
  *
  * Note: Transparent's parentOpt is always empty
  * Note: Transparent's suffixProxy is always a DefinitionProxy
  *
  * E.g. it is the underlying proxy of Instance when a user writes:
  *   val inst = myObject.toInstance
  */
trait Transparent[+P] extends InstanceProxy[P] {
  override def parentOpt: Option[Proxy[Any]] = None

  override def suffixProxy: RootProxy[P]
}

/** InstanceProxy representing an Instance which was created when a proto is looked up from a Hierarchy or Context
  *
  * Note: Mock's always have a non-empty parentOpt.
  * Note: Mock's suffixProxy is always an InstanceProxy
  *
  * E.g. it is the underlying proxy of child when a user writes:
  *   val parent = Instance(..)
  *   val child = parent.child
  */
trait Mock[+P] extends InstanceProxy[P] {

  /** @return Lineage of this Mock, e.g. the parent proxy from whom this Mock was looked up from */
  def parent: Proxy[Any]

  override def parentOpt: Option[Proxy[Any]] = Some(parent)

  override def suffixProxy: InstanceProxy[P]
}

trait RootProxy[+P] extends HierarchicalProxy[P] {
  def toHierarchy: Root[P]
  def toRoot: Root[P] = toHierarchy
  def toDefinition: Definition[P] = toHierarchy.toDefinition
  def predecessorOption: Option[RootProxy[P]]
  def builder: Option[Implementation]
  def parentOpt: Option[Proxy[Any]] = None
}


trait DefinitionProxy[+P] extends RootProxy[P] {
  //def toInterfaceProxy: InterfaceProxy[P]
  def builder: Option[Implementation]

  private[chisel3] var isResolved = false
  def predecessorOption = None
  override def toHierarchy: Definition[P] = new Definition(this)
}

trait DefinitiveProxy[P] extends Proxy[P] {
  def parent: Any
  var valueOpt: Option[P] = None
  var predecessorOpt: Option[DefinitiveProxy[_]] = None

  private[chisel3] val namer: mutable.ArrayBuffer[Any => Unit] = mutable.ArrayBuffer[Any => Unit]()

  def nonEmpty: Boolean = {
    valueOpt.nonEmpty || (predecessorOpt.nonEmpty && predecessorOpt.get.nonEmpty)
  }
  def isEmpty = !nonEmpty
  def isSet = {
    valueOpt.nonEmpty || predecessorOpt.nonEmpty
  }
  def toWrapper = Definitive(this)
}

trait SerializableDefinitiveProxy[P] extends DefinitiveProxy[P] {
  var func: Option[DefinitiveFunction[Any, Any]] = None

  def proto = {
    require(nonEmpty, s"Not empty!")
    if(cache.containsKey("value")) cache.get("value").asInstanceOf[P] else {
      val ret = valueOpt.orElse(predecessorOpt.map { case (p) => func.get(p.proto).asInstanceOf[P] } ).get
      //println(s"All namers of $this, ${namer.size}: ${namer.toList}")
      namer.map(f => chisel3.experimental.noPrefix { f(ret) } )
      cache.put("value", ret)
      ret
    }
  }

}
trait NonSerializableDefinitiveProxy[P] extends DefinitiveProxy[P] {
  var func: Option[Any => Any] = None

  def proto = {
    require(nonEmpty, s"Not empty!")
    if(cache.containsKey("value")) cache.get("value").asInstanceOf[P] else {
      val ret = valueOpt.orElse(predecessorOpt.map { case (p) => func.get(p.proto).asInstanceOf[P] } ).get
      //println(s"All namers of $this, ${namer.size}: ${namer.toList}")
      namer.map(f => chisel3.experimental.noPrefix { f(ret) } )
      cache.put("value", ret)
      ret
    }
  }
}
//TODO: In order to have nested definitive's, e.g. a definitive case class, we need to avoid using the proto as the key in the lookup.
//trait InterfaceProxy[+P] extends RootProxy[P] {
//  def predecessor: DeclarationProxy[P]
//  def toImplementationProxy: ImplementationProxy[P]
//
//  def proto: P = predecessor.proto
//  def builder: Option[ImplementationBuilder[P]] = predecessor.builder
//  def predecessorOption = Some(predecessor)
//  def toHierarchy: Interface[P] = new Interface(this)
//  def canInstantiate: Boolean = true
//}
//
//trait ImplementationProxy[+P] extends RootProxy[P] {
//  def predecessor: InterfaceProxy[P]
//  def toDefinitionProxy: DefinitionProxy[P]
//
//  def proto: P = predecessor.proto
//  def builder: Option[ImplementationBuilder[P]] = None
//  def predecessorOption = Some(predecessor)
//  def toHierarchy: Implementation[P] = new Implementation(this)
//  def canInstantiate: Boolean = true
//}
//
//trait DefinitionProxy[+P] extends RootProxy[P] {
//  def predecessor: ImplementationProxy[P]
//
//  def proto: P = predecessor.proto
//  def builder: Option[ImplementationBuilder[P]] = None
//  def predecessorOption = Some(predecessor)
//  def toHierarchy: Definition[P] = new Definition(this)
//  //override def toSetter: HierarchySetter[P] = HierarchySetter(this)
//  def canInstantiate: Boolean = true
//}

/** DefinitionProxy implementation for all proto's which extend IsWrappable
  *
  * TODO Move to IsWrappable.scala
  * @param proto underlying object we are creating a proxy of
  */
//final case class InstantiableDefinition[P](proto: P) extends DefinitionProxy[P] {
//  def implementChildren: DefinitionProxy[P] = this
//  def buildImplementation: DefinitionProxy[P] = this
//}

/** Transparent implementation for all proto's which extend IsWrappable
  *
  * Note: Clone is not needed for IsWrappables, as they cannot be instantiated via Instance(..)
  *
  * TODO Move to IsWrappable.scala
  * @param proto underlying object we are creating a proxy of
  */
//final case class InstantiableTransparent[P](suffixProxy: InstantiableDefinition[P]) extends Transparent[P]

/** Mock implementation for all proto's which extend IsWrappable
  *
  * Note: Clone is not needed for IsWrappables, as they cannot be instantiated via Instance(..)
  *
  * TODO Move to IsWrappable.scala
  * @param proto underlying object we are creating a proxy of
  */
//final case class InstantiableMock[P](suffixProxy: InstanceProxy[P], parent: Proxy[Any]) extends Mock[P]