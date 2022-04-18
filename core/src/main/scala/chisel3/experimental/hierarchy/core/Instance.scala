// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core
import scala.collection.mutable.HashMap
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{InstanceTransform}
import java.util.IdentityHashMap

/** Represents an Instance of a proto, from a specific hierarchical path
  *
  * @param proxy underlying representation of proto with correct internal state
  */
final case class Instance[+P] private[chisel3] (private[chisel3] proxy: InstanceProxy[P]) extends Hierarchy[P] {

  override def toRoot = proxy.toRoot

  override def proxyAs[T]: InstanceProxy[P] with T = proxy.asInstanceOf[InstanceProxy[P] with T]
}

object Instance {
  def apply[P](root: Root[P]): Instance[P] =
    macro InstanceTransform.apply[P]
  def do_apply[P](root: Root[P])(implicit extensions: HierarchicalExtensions[P, _]): Instance[P] = {
    new Instance(extensions.buildInstance(root))
  }
}