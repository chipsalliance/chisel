// See LICENSE for license details.

package firrtl.passes
package memlib
import firrtl._
import firrtl.ir._
import AnalysisUtils.eqMems
import firrtl.Mappers._
import firrtl.annotations._

/** A component, e.g. register etc. Must be declared only once under the TopAnnotation
  */
object NoDedupMemAnnotation {
  def apply(target: ComponentName): Annotation = Annotation(target, classOf[ResolveMemoryReference], s"nodedupmem!")

  def unapply(a: Annotation): Option[ComponentName] = a match {
    case Annotation(ComponentName(n, mn), _, "nodedupmem!") => Some(ComponentName(n, mn))
    case _ => None
  }
}

/** Resolves annotation ref to memories that exactly match (except name) another memory
 */
class ResolveMemoryReference extends Transform {
  def inputForm = MidForm
  def outputForm = MidForm

  type AnnotatedMemories = collection.mutable.ArrayBuffer[(String, DefAnnotatedMemory)]

  /** If a candidate memory is identical except for name to another, add an
    *   annotation that references the name of the other memory.
    */
  def updateMemStmts(mname: String, uniqueMems: AnnotatedMemories, noDeDupeMems: Seq[String])(s: Statement): Statement = s match {
    case m: DefAnnotatedMemory =>
      uniqueMems find (x => eqMems(x._2, m, noDeDupeMems)) match {
        case None =>
          uniqueMems += (mname -> m)
          m
        case Some((module, proto)) => m copy (memRef = Some(module -> proto.name))
      }
    case s => s map updateMemStmts(mname, uniqueMems, noDeDupeMems)
  }

  def run(c: Circuit, noDeDupeMems: Seq[String]) = {
    val uniqueMems = new AnnotatedMemories
    c copy (modules = c.modules map (m => m map updateMemStmts(m.name, uniqueMems, noDeDupeMems)))
  }
  def execute(state: CircuitState): CircuitState = {
    val noDedups = getMyAnnotations(state) match {
      case Nil => Seq.empty
      case annos =>
        annos.collect { case NoDedupMemAnnotation(ComponentName(cn, _)) => cn }
    }
    state.copy(circuit=run(state.circuit, noDedups))
  }
}
