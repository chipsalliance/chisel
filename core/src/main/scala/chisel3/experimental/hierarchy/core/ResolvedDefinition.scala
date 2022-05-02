// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

import scala.collection.mutable.HashMap
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{DefinitionTransform, InstanceTransform}
import java.util.IdentityHashMap

/** Represents a ResolvedDefinition of a proto, at the root of a hierarchical path
  *
  * @param proxy underlying representation of proto with correct internal state
  */
final case class ResolvedDefinition[+P] private[chisel3] (private[chisel3] proxy: DefinitionProxy[P]) extends Root[P] {
  override def toDefinition = Definition(proxy)
}

object ResolvedDefinition {
  def apply[P](proto: => P): ResolvedDefinition[P] =
    macro DefinitionTransform.apply[P]
  def do_apply[P](proto: => P)(implicit extensions: HierarchicalExtensions[P, _]): ResolvedDefinition[P] = {
    (new ResolvedDefinition(extensions.buildDefinition(proto)))
  }
}
