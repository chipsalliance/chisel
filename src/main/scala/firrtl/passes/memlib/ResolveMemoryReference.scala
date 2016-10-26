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

  type AnnotatedMemories = collection.mutable.ArrayBuffer[(String, DefAnnotatedMemory)]

  /** If a candidate memory is identical except for name to another, add an
    *   annotation that references the name of the other memory.
    */
  def updateMemStmts(mname: String, uniqueMems: AnnotatedMemories)(s: Statement): Statement = s match {
    case m: DefAnnotatedMemory =>
      uniqueMems find (x => eqMems(x._2, m)) match {
        case None =>
          uniqueMems += (mname -> m)
          m
        case Some((module, proto)) => m copy (memRef = Some(module -> proto.name))
      }
    case s => s map updateMemStmts(mname, uniqueMems)
  }

  def run(c: Circuit) = {
    val uniqueMems = new AnnotatedMemories
    c copy (modules = c.modules map (m => m map updateMemStmts(m.name, uniqueMems)))
  }
}
