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
    require(!proxy.hasDerivation, s"Cannot set a definitive twice!")
    proxy.derivation = Some(DefinitiveToDefinitiveDerivation(DefinitiveValue(v), Identity()))
  }
  def value: P = proto
  def setAs(d: Definitive[P]): Unit = {
    require(!proxy.hasDerivation, s"Cannot set a definitive twice!")
    proxy.derivation = Some(DefinitiveToDefinitiveDerivation(d.proxy, Identity()))
  }

  def hasDerivation = proxy.hasDerivation
  def valueOpt: Option[P] = proxy.compute

  def modify[X](f: ParameterFunction): Definitive[X] = Definitive.buildFromDF(this, f)
}

object Definitive {
  def buildFromDF[P, X](
    d: Definitive[P],
    f: ParameterFunction
  )(
    implicit extensions: ParameterExtensions[_, _]
  ): Definitive[X] = extensions.buildDefinitiveFrom(d, f).toWrapper
  def buildFromCF[P, X](
    c: Contextual[P],
    f: CombinerFunction
  )(
    implicit extensions: ParameterExtensions[_, _]
  ): Definitive[X] =
    extensions.buildDefinitiveFrom(c, f).toWrapper
  def apply[P](p: P)(implicit extensions: ParameterExtensions[_, _]): Definitive[P] =
    extensions.buildDefinitive(Some(p)).toWrapper
  def empty[P](implicit extensions: ParameterExtensions[_, _]): Definitive[P] =
    extensions.buildDefinitive(None).toWrapper
}
