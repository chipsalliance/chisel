// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

import scala.collection.mutable.HashMap
import scala.collection.mutable
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{DefinitionTransform, InstanceTransform}
import java.util.IdentityHashMap



case class Contextual[P] private[chisel3] (proxy: ContextualProxy[P]) extends Wrapper[P] {
  def value_=(v: P) = {
    require(!proxy.hasDerivation, s"Cannot set a definitive twice! $proto is $value, cannot be set to $v")
    proxy.derivation = Some(ContextualToContextualDerivation(ContextualValue(v), Identity()))
  }
  def value: P = proto
  def setAs(d: Contextual[P]): Unit = {
    require(!proxy.hasDerivation, s"Cannot set a definitive twice! $proto is $value, cannot be set to $d")
    proxy.derivation = Some(ContextualToContextualDerivation(d.proxy, Identity()))
  }

  def isResolved = proxy.isResolved
  def hasDerivation = proxy.hasDerivation

  def values = proxy.values

  def modify[X](f: ParameterFunction): Contextual[X] = Contextual.buildFromDF(this, f)
  def combine[X](f: CombinerFunction): Definitive[X] = Definitive.buildFromCF(this, f)

  def printSuffix: Unit = {
    println(">" + this)
    if(proxy.suffixProxyOpt.nonEmpty) proxy.suffixProxyOpt.get.toContextual.printSuffix
  }
  private[chisel3] def absolutize(h: Hierarchy[_]): Unit = {
    val value = proxy.compute(h)
    require(value.nonEmpty)
    proxy.setValue(value.get)
  }
}


object Contextual {
  def buildFromDF[P, X](d: Contextual[P], f: ParameterFunction)(implicit extensions: ParameterExtensions[_,_]): Contextual[X] = extensions.buildContextualFrom(d, f).toWrapper
  def apply[P](p: P)(implicit extensions: ParameterExtensions[_,_]): Contextual[P] = extensions.buildContextual(Some(p)).toWrapper
  def empty[P](implicit extensions: ParameterExtensions[_,_]): Contextual[P] = extensions.buildContextual(None).toWrapper
} 
