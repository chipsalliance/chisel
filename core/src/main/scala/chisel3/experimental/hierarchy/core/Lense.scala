// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core
import scala.collection.mutable
import java.util.IdentityHashMap

trait Lense[+P] {
  def proxy: Proxy[P]
  def top: TopLense[_]
  def toHierarchy: Hierarchy[P] = proxy match {
    case d: DefinitionProxy[P] => Definition(d)
    case i: InstanceProxy[P] => Instance(i)
  }
  private[chisel3] val edits = new IdentityHashMap[Contextual[Any], Any => Any]()
  private[chisel3] val cache = new IdentityHashMap[Any, Any]()

  private[chisel3] def addEdit[T](contextual: Contextual[T], edit: T => T): Unit = {
    require(!edits.containsKey(contextual), s"Cannot set $contextual more than once!")
    edits.put(contextual, edit.asInstanceOf[Any => Any])
  }
  private[chisel3] def getEdit[T](contextual: Contextual[T]): Edit[T] = {
    if(edits.containsKey(contextual)) Edit(contextual, edits.get(contextual).asInstanceOf[T => T]) else {
      Edit(contextual, {t: T => t})
    }
  }
  private[chisel3] def compute[T](key: Contextual[T], c: Contextual[T]): Contextual[T] = new Contextual(getEdit(key).edit(c.value))

  def _lookup[B, C](that: P => B)(
    implicit lookupable: Lookupable[B],
    macroGenerated:  chisel3.internal.MacroGenerated
  ): lookupable.S = {
    val protoValue = that(this.proxy.proto)
    if(cache.containsKey(protoValue)) cache.get(protoValue).asInstanceOf[lookupable.S] else {
      val retValue = lookupable.setter(protoValue, this)
      cache.put(protoValue, retValue)
      retValue
    }
  }
  def getter[B, C](protoValue: B)(
    implicit lookupable: Lookupable[B]
  ): lookupable.G = {
    if(cache.containsKey(protoValue)) cache.get(protoValue).asInstanceOf[lookupable.G] else {
      val retValue = lookupable.getter(protoValue, this)
      cache.put(protoValue, retValue)
      retValue
    }
  }
}
final case class TopLense[+P](proxy: Proxy[P]) extends Lense[P] {
  def top = this
}
final case class NestedLense[+P](proxy: Proxy[P], top: TopLense[_]) extends Lense[P] {

}

final case class Socket[T](contextual: Contextual[T], lense: Lense[_]) {
  def value: T = contextual.value
  def value_=(newValue: T): TopLense[_] = edit({_: T => newValue})
  def edit(f: T => T): TopLense[_] = {
    lense.addEdit(contextual, f)
    lense.top
  }
}
final case class Edit[T](contextual: Contextual[T], edit: T => T)