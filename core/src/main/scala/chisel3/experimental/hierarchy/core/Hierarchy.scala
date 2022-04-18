// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core
import scala.collection.mutable.HashMap
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import java.util.IdentityHashMap

/** Represents a view of a proto from a specific hierarchical path */
trait Hierarchy[+P] extends Wrapper[P] {

  /** Finds the closest parent Instance/Hierarchy in proxy's parent which matches a partial function
    *
    * @param pf selection partial function
    * @return closest matching parent in parent which matches pf, if one does
    */
  def getClosestParentOf[T](pf: PartialFunction[Any, Hierarchy[T]]): Option[Hierarchy[T]] = {
    pf.lift(this)
      .orElse(proxy.parentOpt.flatMap {
        case d: DefinitionProxy[_] => d.toDefinition.getClosestParentOf(pf)
        case i: InstanceProxy[_]   => i.toInstance.getClosestParentOf(pf)
        case other => println(s"NONE!! $other"); None
      })
  }

  /** @return Return the proxy Definition[P] of this Hierarchy[P] */
  def toRoot: Root[P]

  private[chisel3] def proxy: HierarchicalProxy[P]
}
