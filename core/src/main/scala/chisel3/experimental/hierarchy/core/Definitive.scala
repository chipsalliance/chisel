// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

import scala.collection.mutable.HashMap
import scala.collection.mutable
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{DefinitionTransform, InstanceTransform}
import java.util.IdentityHashMap

object Identity extends DefinitiveFunction[Any, Any] {
  def applyIt(i: Any): Any = i
}
case class Definitive[P] private[chisel3] (proxy: DefinitiveProxy[P]) extends Wrapper[P] {
  def value_=(v: P) = {
    require(!proxy.isSet, s"Cannot set a definitive twice! $proto is $value, cannot be set to $v")
    proxy.valueOpt = Some(v)
  }
  def value: P = proto
  def setAs(d: Definitive[P]): Unit = {
    require(!proxy.isSet, s"Cannot set a definitive twice! $proto is $value, cannot be set to $d")
    (d.proxy, proxy) match {
      case (x: SerializableDefinitiveProxy[P], y: SerializableDefinitiveProxy[P]) =>
        y.predecessorOpt = Some(x)
        y.func = Some(Identity)
      case (x, y: NonSerializableDefinitiveProxy[P]) =>
        y.predecessorOpt = Some(x)
        y.func = Some({any: Any => any})
      case (x: NonSerializableDefinitiveProxy[P], y: SerializableDefinitiveProxy[P]) =>
          require(false, "THIS IS BAD")

    }
  }

  def isEmpty = proxy.isEmpty
  def nonEmpty = !isEmpty
  def isSet = proxy.isSet

  def whenKnown[X](f: P => X): Definitive[X] = Definitive.buildFrom(this, f)
  def modify[X](f: DefinitiveFunction[P, X]): Definitive[X] = Definitive.buildFrom(this, f)
}

sealed trait DefinitiveFunction[-I, +O] {
  def applyIt(i: I): O
}

trait CustomDefinitiveFunction[-I, +O] extends DefinitiveFunction[I, O]

case class DerivedDefinitiveFunction[-I, +O](f: I => O) extends DefinitiveFunction[I, O] {
  def applyIt(i: I): O = f(i)
  override def toString = f.getClass().getName()
}

object Definitive {
  def buildFrom[P, X](d: Definitive[P], f: DefinitiveFunction[P, X])(implicit extensions: HierarchicalExtensions[_,_]): Definitive[X] = extensions.buildDefinitiveFrom(d, f).toWrapper
  def buildFrom[P, X](d: Definitive[P], f: P => X)(implicit extensions: HierarchicalExtensions[_,_]): Definitive[X] = extensions.buildDefinitiveFrom(d, f).toWrapper
  def apply[P](p: P)(implicit extensions: HierarchicalExtensions[_,_]): Definitive[P] = extensions.buildDefinitive(Some(p)).toWrapper
  def empty[P](implicit extensions: HierarchicalExtensions[_,_]): Definitive[P] = extensions.buildDefinitive(None).toWrapper
} 
