// See LICENSE for license details.

package firrtl.passes
package memlib
import firrtl._
import firrtl.ir._
import firrtl.Mappers._
import firrtl.annotations._

/** A component, e.g. register etc. Must be declared only once under the TopAnnotation */
case class NoDedupMemAnnotation(target: ComponentName) extends SingleTargetAnnotation[ComponentName] {
  def duplicate(n: ComponentName) = NoDedupMemAnnotation(n)
}

/** Resolves annotation ref to memories that exactly match (except name) another memory
 */
class ResolveMemoryReference extends Transform {
  def inputForm = MidForm
  def outputForm = MidForm

  /** Helper class for determining when two memories are equivalent while igoring
    * irrelevant details like name and info
    */
  private class WrappedDefAnnoMemory(val underlying: DefAnnotatedMemory) {
    // Remove irrelevant details for comparison
    private def generic = underlying.copy(info = NoInfo, name = "", memRef = None)
    override def hashCode: Int = generic.hashCode
    override def equals(that: Any): Boolean = that match {
      case mem: WrappedDefAnnoMemory => this.generic == mem.generic
      case _ => false
    }
  }
  private def wrap(mem: DefAnnotatedMemory) = new WrappedDefAnnoMemory(mem)

  // Values are Tuple of Module Name and Memory Instance Name
  private type AnnotatedMemories = collection.mutable.HashMap[WrappedDefAnnoMemory, (String, String)]

  private def dedupable(noDedups: Map[String, Set[String]], module: String, memory: String): Boolean =
    noDedups.get(module).map(!_.contains(memory)).getOrElse(true)

  /** If a candidate memory is identical except for name to another, add an
    *   annotation that references the name of the other memory.
    */
  def updateMemStmts(mname: String,
                     existingMems: AnnotatedMemories,
                     noDedupMap: Map[String, Set[String]])
                    (s: Statement): Statement = s match {
    // If not dedupable, no need to add to existing (since nothing can dedup with it)
    // We just return the DefAnnotatedMemory as is in the default case below
    case m: DefAnnotatedMemory if dedupable(noDedupMap, mname, m.name) =>
      val wrapped = wrap(m)
      existingMems.get(wrapped) match {
        case proto @ Some(_) =>
          m.copy(memRef = proto)
        case None =>
          existingMems(wrapped) = (mname, m.name)
          m
      }
    case s => s.map(updateMemStmts(mname, existingMems, noDedupMap))
  }

  def run(c: Circuit, noDedupMap: Map[String, Set[String]]) = {
    val existingMems = new AnnotatedMemories
    val modulesx = c.modules.map(m => m.map(updateMemStmts(m.name, existingMems, noDedupMap)))
    c.copy(modules = modulesx)
  }
  def execute(state: CircuitState): CircuitState = {
    val noDedups = state.annotations.collect {
      case NoDedupMemAnnotation(ComponentName(cn, ModuleName(mn, _))) => mn -> cn
    }
    val noDedupMap: Map[String, Set[String]] = noDedups.groupBy(_._1).mapValues(_.map(_._2).toSet)
    state.copy(circuit = run(state.circuit, noDedupMap))
  }
}
