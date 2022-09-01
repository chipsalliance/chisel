// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

import scala.collection.mutable.HashMap
import scala.collection.mutable
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{DefinitionTransform, InstanceTransform}
import java.util.IdentityHashMap
import chisel3.internal.sourceinfo.SourceInfo

case class Definitive[P] private[chisel3] (proxy: DefinitiveProxy[P], sourceInfo: SourceInfo) extends Wrapper[P] {
  def value_=(v: P)(implicit si: SourceInfo) = {
    require(!proxy.hasDerivation, s"Cannot set a definitive twice!")
    proxy.derivation = Some(DefinitiveToDefinitiveDerivation(DefinitiveValue(v, si), Identity()))
  }
  def value: P = proto
  def setAs(d: Definitive[P]): Unit = {
    require(!proxy.hasDerivation, s"Cannot set a definitive twice!")
    proxy.derivation = Some(DefinitiveToDefinitiveDerivation(d.proxy, Identity()))
  }

  def hasDerivation = proxy.hasDerivation
  def valueOpt: Option[P] = proxy.compute

  def modify[X](f: ParameterFunction)(implicit si: SourceInfo): Definitive[X] = Definitive.buildFromDF(this, f)
}

object Definitive {
  def buildFromDF[P, X](
    d: Definitive[P],
    f: ParameterFunction
  )(
    implicit extensions: ParameterExtensions[_, _],
    sourceInfo: SourceInfo
  ): Definitive[X] = extensions.buildDefinitiveFrom(d, f, sourceInfo).toWrapper
  def buildFromCF[P, X](
    c: Contextual[P],
    f: CombinerFunction
  )(
    implicit extensions: ParameterExtensions[_, _],
    sourceInfo: SourceInfo
  ): Definitive[X] =
    extensions.buildDefinitiveFrom(c, f, sourceInfo).toWrapper
  def apply[P](p: P)(implicit extensions: ParameterExtensions[_, _], sourceInfo: SourceInfo): Definitive[P] =
    extensions.buildDefinitive(Some(p), sourceInfo).toWrapper
  def empty[P](implicit extensions: ParameterExtensions[_, _], sourceInfo: SourceInfo): Definitive[P] =
    extensions.buildDefinitive(None, sourceInfo).toWrapper
}
