// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core
import scala.collection.mutable

// ==========================================
// User-facing Traits
// ==========================================

// Context independent value. Use for immutable case classes which are true for all Hierarchy[_]
trait IsLookupable

// Context dependent that lacks a clone method, and so wraps itself in Hierarchy[_] to represent the context-specific value
trait IsWrappable

// ==========================================
// Library-facing Traits, NOT FOR USERS
// Additional library-facing traits are not in this file
// ==========================================


trait Extensions[V, P] {
  def getParent(x: V): Option[P]
  def parentExtensions: HierarchicalExtensions[P, P]
  def parentSelection: PartialFunction[Any, Hierarchy[P]]
}
trait CloneableExtensions[V, P] extends Extensions[V, P] {
  def clone(value: V, hierarchy: Hierarchy[P]): V
}

trait HierarchicalExtensions[V, P] extends Extensions[V, P] {
  def getProxyParent(x: Proxy[V]): Option[P]
  def buildDefinitiveFrom[X, Y](d: Definitive[X], f: DefinitiveFunction[X,Y]): DefinitiveProxy[Y]
  def buildDefinitiveFrom[X, Y](d: Definitive[X], f: X => Y): DefinitiveProxy[Y]
  def buildDefinitive[X](x: Option[X]): DefinitiveProxy[X]
  def buildDefinition(f: => V): DefinitionProxy[V]
  def buildInstance(root: Root[V]): Clone[V]
  def mock[P](value: Any, parent: Hierarchy[P]): Instance[V] = value match {
    case p: InstanceProxy[V] => mockInstance(p.toInstance, parent)
    case o: V                => mockValue(o, parent)
  }
  def mockInstance[P](value: Instance[V], parent: Hierarchy[P]): Instance[V]
  def mockValue[P](value: V, parent: Hierarchy[P]): Instance[V]
  def toDefinition(value: V): Definition[V]
}

