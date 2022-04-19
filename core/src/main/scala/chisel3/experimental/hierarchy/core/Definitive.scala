// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

import scala.collection.mutable.HashMap
import scala.collection.mutable
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{DefinitionTransform, InstanceTransform}
import java.util.IdentityHashMap

case class Definitive[P] private[chisel3] (proxy: DefinitiveProxy[P]) extends Wrapper[P] {
  def value_=(v: P) = {
    require(proxy.isEmpty, s"Cannot set a definitive twice! $proto is $value, cannot b eset to $v")
    proxy.valueOpt = Some(v)
  }
  def value: P = proto
  def isEmpty = proxy.isEmpty
  def nonEmpty = !isEmpty

  def whenKnown[X](f: P => X): Definitive[X] = Definitive.buildFrom(this, f)
}

object Definitive {
  def buildFrom[P, X](d: Definitive[P], f: P => X)(implicit extensions: HierarchicalExtensions[_,_]): Definitive[X] = extensions.buildDefinitiveFrom(d, f).toWrapper
  def apply[P](p: P)(implicit extensions: HierarchicalExtensions[_,_]): Definitive[P] = extensions.buildDefinitive(Some(p)).toWrapper
  def empty[P](implicit extensions: HierarchicalExtensions[_,_]): Definitive[P] = extensions.buildDefinitive(None).toWrapper
} 
