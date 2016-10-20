// See LICENSE for license details.

package firrtl.passes
package memlib
import firrtl.ir._
import AnalysisUtils.eqMems
import firrtl.Mappers._


/** Resolves annotation ref to memories that exactly match (except name) another memory
 */
object ResolveMemoryReference extends Pass {

  def name = "Resolve Memory Reference"

  type AnnotatedMemories = collection.mutable.ArrayBuffer[DefAnnotatedMemory]

  /** If a candidate memory is identical except for name to another, add an
    *   annotation that references the name of the other memory.
    */
  def updateMemStmts(uniqueMems: AnnotatedMemories)(s: Statement): Statement = s match {
    case m: DefAnnotatedMemory =>
      uniqueMems find (x => eqMems(x, m)) match {
        case None =>
          uniqueMems += m
          m
        case Some(proto) => m copy (memRef = Some(proto.name))
      }
    case s => s map updateMemStmts(uniqueMems)
  }

  def run(c: Circuit) = {
    val uniqueMems = new AnnotatedMemories
    c copy (modules = c.modules map (_ map updateMemStmts(uniqueMems)))
  }
}
