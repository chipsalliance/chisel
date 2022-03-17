// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core
import java.util.IdentityHashMap

/** Representation of a hierarchical version of an object
  * Has two subclasses, a DefinitionProxy and InstanceProxy, which are wrapped with Definition(..) and Instance(..)
  */
sealed trait Proxy[+P] {

  /** @return Original object that we are proxy'ing */
  def proto: P

  /** Computes the new contextual given the original, proto Contextual (key) and the current contextual (c)
    *
    * We need two values here because the edits are stored according to the identity of the original, proto contextual value.
    * However, contextual values could be set in multiple contexts which grow in context. For example, a default
    * contextual value can be given, but when we have nested instances, each proto in that hierarchy path can give an
    * edit function. Thus, the contextual passed to the edit function may not be the same object as the original contextual.
    *
    * @param key original proto's contextual
    * @param contextual current contextual containing a value to be edited
    * @return a new contextual with the edited value
    */
  private[chisel3] def compute[T](key: Contextual[T], contextual: Contextual[T]): Contextual[T]

  // All user-specified contexts containing Contextual values
  def contexts: Seq[Context[P]]

  /** If this proxy was created by being looked up in a parent proxy, lineage refers to that parent proxy.
    * @return parent proxy from whom this proxy was looked up from
    */
  def lineageOpt: Option[Proxy[Any]]

  /** @return a Definition wrapping this Proxy */
  def toDefinition: Definition[P]
}

/** Proxy representing an Instance version of an object */
sealed trait InstanceProxy[+P] extends Proxy[P] {

  /** The proxy which refers to the same proto, but from a less-specific hierarchical path.
    *
    * Example 0: if this Proxy refers to ~Top|Foo/bar:Bar, then narrowerProxy refers to ~Top|Bar
    * Example 1: if this Proxy refers to ~Top|Top/foo:Foo/bar:Bar, then narrowerProxy refers to ~Top|Foo/bar:Bar
    *
    * @return the narrowerProxy proxy of this proxy
    */
  def narrowerProxy: Proxy[P]

  /** @return an Instance wrapping this Proxy */
  def toInstance = new Instance(this)

  /** @return the InstanceProxy closest to the proto in the chain of narrowerProxy proxy's */
  def localProxy: InstanceProxy[P] = narrowerProxy match {
    case d: DefinitionProxy[P] => this
    case i: InstanceProxy[P]   => i.localProxy
  }

  override def compute[T](key: Contextual[T], contextual: Contextual[T]): Contextual[T] = {
    val narrowerProxyContextual = narrowerProxy.compute(key, contextual)
    contexts.foldLeft(narrowerProxyContextual) { case (c, context) => context.compute(key, c) }
  }

  override def proto = narrowerProxy.proto
  override def toDefinition: Definition[P] = narrowerProxy.toDefinition
}

/** InstanceProxy representing a new Instance which was instantiated in a proto.
  *
  * Note: Clone's lineageOpt is always empty
  * Note: Clone's narrowerProxy is always a DefinitionProxy
  *
  * E.g. it is the underlying proxy of Instance when a user writes:
  *   val inst = Instance(..)
  */
trait Clone[+P] extends InstanceProxy[P] {
  override def lineageOpt: Option[Proxy[Any]] = None

  override def narrowerProxy: DefinitionProxy[P]
}

/** InstanceProxy representing an Instance which was created from converting an existing object into an Instance.
  *
  * Note: Transparent's lineageOpt is always empty
  * Note: Transparent's narrowerProxy is always a DefinitionProxy
  *
  * E.g. it is the underlying proxy of Instance when a user writes:
  *   val inst = myObject.toInstance
  */
trait Transparent[+P] extends InstanceProxy[P] {
  override def lineageOpt: Option[Proxy[Any]] = None

  override def narrowerProxy: DefinitionProxy[P]
}

/** InstanceProxy representing an Instance which was created when a proto is looked up from a Hierarchy or Context
  *
  * Note: Mock's always have a non-empty lineageOpt.
  * Note: Mock's narrowerProxy is always an InstanceProxy
  *
  * E.g. it is the underlying proxy of child when a user writes:
  *   val parent = Instance(..)
  *   val child = parent.child
  */
trait Mock[+P] extends InstanceProxy[P] {

  /** @return Lineage of this Mock, e.g. the parent proxy from whom this Mock was looked up from */
  def lineage: Proxy[Any]

  override def lineageOpt: Option[Proxy[Any]] = Some(lineage)

  override def narrowerProxy: InstanceProxy[P]
}

/** DefinitionProxy underlying a Definition
  *
  * E.g. it is the underlying proxy of defn when a user writes:
  *   val defn = Definition(..)
  */
trait DefinitionProxy[+P] extends Proxy[P] {
  override def contexts: Seq[Context[P]] = Nil
  override def compute[T](key: Contextual[T], contextual: Contextual[T]): Contextual[T] = {
    contexts.foldLeft(contextual) { case (c, context) => context.compute(key, c) }
  }
  override def lineageOpt: Option[Proxy[Any]] = None
  override def toDefinition = new Definition(this)
}

/** DefinitionProxy implementation for all proto's which extend IsInstantiable
  *
  * TODO Move to IsInstantiable.scala
  * @param proto underlying object we are creating a proxy of
  */
final case class InstantiableDefinition[P](proto: P) extends DefinitionProxy[P]

/** Transparent implementation for all proto's which extend IsInstantiable
  *
  * Note: Clone is not needed for IsInstantiables, as they cannot be instantiated via Instance(..)
  *
  * TODO Move to IsInstantiable.scala
  * @param proto underlying object we are creating a proxy of
  */
final case class InstantiableTransparent[P](narrowerProxy: InstantiableDefinition[P], contexts: Seq[Context[P]])
    extends Transparent[P]

/** Mock implementation for all proto's which extend IsInstantiable
  *
  * Note: Clone is not needed for IsInstantiables, as they cannot be instantiated via Instance(..)
  *
  * TODO Move to IsInstantiable.scala
  * @param proto underlying object we are creating a proxy of
  */
final case class InstantiableMock[P](narrowerProxy: InstanceProxy[P], lineage: Proxy[Any], contexts: Seq[Context[P]])
    extends Mock[P]
