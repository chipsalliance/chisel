// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

import scala.collection.mutable.HashMap
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{DefinitionTransform, InstanceTransform}
import java.util.IdentityHashMap

/** Represents a Definition of a proto, at the root of a hierarchical path
  *
  * @param proxy underlying representation of proto with correct internal state
  */
final case class Definition[+P] private[chisel3] (private[chisel3] proxy: DefinitionProxy[P]) extends Root[P] {
  override def toDefinition = this
  private[chisel3] def toResolvedDefinition = {
    proxy.isResolved = true
    new ResolvedDefinition(proxy)
  }
}

object Definition {
  def apply[P](proto: => P): Definition[P] =
    macro DefinitionTransform.apply[P]
  def do_apply[P](proto: => P)(implicit extensions: HierarchicalExtensions[P, _]): Definition[P] = {
    (new Definition(extensions.buildDefinition(proto)))
  }
  //OLD
  //def withContext[P](proto: => P)(fs: (Context[P] => Unit)*): Definition[P] =
  //  macro WithContextDefinitionTransform.withContext[P]
  //def do_withContext[P](
  //  proto: => P
  //)(fs:         (Context[P] => Unit)*
  //)(
  //  implicit buildable: ProxyDefiner[P]
  //): Definition[P] = {
  //  val d = new Definition(buildable(proto))
  //  val context = d.proxy.toContext
  //  fs.foreach { f =>
  //    f(context)
  //  }
  //  //println(i.proxy.edits.values())
  //  d
  //}
}
