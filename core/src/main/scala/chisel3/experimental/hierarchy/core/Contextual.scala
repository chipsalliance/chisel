// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

import scala.collection.mutable.HashMap
import scala.collection.mutable
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import chisel3.internal.sourceinfo._
import java.util.IdentityHashMap

case class Contextual[P] private[chisel3] (proxy: ContextualProxy[P]) extends Wrapper[P] {
  def value_=(v: P)(implicit sourceInfo: SourceInfo) = {
    require(!proxy.hasDerivation, s"Cannot set a definitive twice! $proto is $value, cannot be set to $v")
    proxy.derivation = Some(ContextualToContextualDerivation(ContextualValue(v, sourceInfo), Identity()))
  }
  def value: P = proto
  def setAs(d: Contextual[P]): Unit = {
    require(!proxy.hasDerivation, s"Cannot set a definitive twice! $proto is $value, cannot be set to $d")
    proxy.derivation = Some(ContextualToContextualDerivation(d.proxy, Identity()))
  }

  def isResolved = proxy.isResolved
  def hasDerivation = proxy.hasDerivation

  def values = proxy.values

  def modify[X](f:  ParameterFunction)(implicit sourceInfo: SourceInfo): Contextual[X] = Contextual.buildFromDF(this, f)
  def combine[X](f: CombinerFunction)(implicit sourceInfo: SourceInfo):  Definitive[X] = Definitive.buildFromCF(this, f)

  def printSuffix: Unit = {
    println(">" + this)
    if (proxy.suffixProxyOpt.nonEmpty) proxy.suffixProxyOpt.get.toContextual.printSuffix
  }
  private[chisel3] def absolutize(h: Hierarchy[_]): Unit = {
    val value = proxy.compute(h)
    require(value.nonEmpty, proxy.sourceInfo.makeMessage(i => s"$i: $this, ${proxy.hasDerivation}"))
    proxy.setValue(value.get)
  }
}

object Contextual {
  def buildFromDF[P, X](
    d: Contextual[P],
    f: ParameterFunction
  )(
    implicit extensions: ParameterExtensions[_, _], sourceInfo: SourceInfo
  ): Contextual[X] = extensions.buildContextualFrom(d, f, sourceInfo).toWrapper
  def apply[P](p: P)(implicit extensions: ParameterExtensions[_, _], sourceInfo: SourceInfo): Contextual[P] =
    extensions.buildContextual(Some(p), sourceInfo).toWrapper
  def empty[P](implicit extensions: ParameterExtensions[_, _], sourceInfo: SourceInfo): Contextual[P] =
    extensions.buildContextual(None, sourceInfo).toWrapper
}
