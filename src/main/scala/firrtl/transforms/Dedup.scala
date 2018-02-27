// See LICENSE for license details.

package firrtl
package transforms

import firrtl.ir._
import firrtl.Mappers._
import firrtl.annotations._
import firrtl.passes.PassException

// Datastructures
import scala.collection.mutable


/** A component, e.g. register etc. Must be declared only once under the TopAnnotation
  */
object NoDedupAnnotation {
  def apply(target: ModuleName): Annotation = Annotation(target, classOf[DedupModules], s"nodedup!")

  def unapply(a: Annotation): Option[ModuleName] = a match {
    case Annotation(ModuleName(n, c), _, "nodedup!") => Some(ModuleName(n, c))
    case _ => None
  }
}


// Only use on legal Firrtl. Specifically, the restriction of
//  instance loops must have been checked, or else this pass can
//  infinitely recurse
class DedupModules extends Transform {
  def inputForm = HighForm
  def outputForm = HighForm
  // Orders the modules of a circuit from leaves to root
  // A module will appear *after* all modules it instantiates
  private def buildModuleOrder(c: Circuit): Seq[String] = {
    val moduleOrder = mutable.ArrayBuffer.empty[String]
    def hasInstance(b: Statement): Boolean = {
      var has = false
      def onStmt(s: Statement): Statement = s map onStmt match {
        case DefInstance(i, n, m) =>
          if(!(moduleOrder contains m)) has = true
          s
        case WDefInstance(i, n, m, t) =>
          if(!(moduleOrder contains m)) has = true
          s
        case _ => s
      }
      onStmt(b)
      has
    }
    def addModule(m: DefModule): DefModule = m match {
      case Module(info, n, ps, b) =>
        if (!hasInstance(b)) moduleOrder += m.name
        m
      case e: ExtModule =>
        moduleOrder += m.name
        m
      case _ => m
    }

    while ((moduleOrder.size < c.modules.size)) {
      c.modules.foreach(m => if (!moduleOrder.contains(m.name)) addModule(m))
    }
    moduleOrder
  }

  // Finds duplicate Modules
  // Also changes DefInstances to instantiate the deduplicated module
  // Returns (Deduped Module name -> Seq of identical modules,
  //          Deuplicate Module name -> deduped module name)
  private def findDups(
      moduleOrder: Seq[String],
      moduleMap: Map[String, DefModule],
      noDedups: Seq[String]): (Map[String, Seq[DefModule]], Map[String, String]) = {
    // Module body -> Module name
    val dedupModules = mutable.HashMap.empty[String, String]
    // Old module name -> dup module name
    val dedupMap = mutable.HashMap.empty[String, String]
    // Deduplicated module name -> all identical modules
    val oldModuleMap = mutable.HashMap.empty[String, Seq[DefModule]]

    def onModule(m: DefModule): Unit = {
      def fixInstance(s: Statement): Statement = s map fixInstance match {
        case DefInstance(i, n, m) => DefInstance(i, n, dedupMap.getOrElse(m, m))
        case WDefInstance(i, n, m, t) => WDefInstance(i, n, dedupMap.getOrElse(m, m), t)
        case x => x
      }
      def removeInfo(stmt: Statement): Statement = stmt map removeInfo match {
        case sx: HasInfo => sx match {
          case s: DefWire => s.copy(info = NoInfo)
          case s: DefNode => s.copy(info = NoInfo)
          case s: DefRegister => s.copy(info = NoInfo)
          case s: DefInstance => s.copy(info = NoInfo)
          case s: WDefInstance => s.copy(info = NoInfo)
          case s: DefMemory => s.copy(info = NoInfo)
          case s: Connect => s.copy(info = NoInfo)
          case s: PartialConnect => s.copy(info = NoInfo)
          case s: IsInvalid => s.copy(info = NoInfo)
          case s: Attach => s.copy(info = NoInfo)
          case s: Stop => s.copy(info = NoInfo)
          case s: Print => s.copy(info = NoInfo)
          case s: Conditionally => s.copy(info = NoInfo)
        }
        case sx => sx
      }
      def removePortInfo(p: Port): Port = p.copy(info = NoInfo)


      val mx = m map fixInstance
      val mxx = (mx map removeInfo) map removePortInfo

      // If shouldn't dedup, just make it fail to be the same to any other modules
      val unique = if (!noDedups.contains(mxx.name)) "" else mxx.name

      val string = mxx match {
        case Module(i, n, ps, b) =>
          ps.map(_.serialize).mkString + b.serialize + unique
        case ExtModule(i, n, ps, dn, p) =>
          ps.map(_.serialize).mkString + dn + p.map(_.serialize).mkString + unique
      }
      dedupModules.get(string) match {
        case Some(dupname) =>
          dedupMap(mx.name) = dupname
          oldModuleMap(dupname) = oldModuleMap(dupname) :+ mx
        case None =>
          dedupModules(string) = mx.name
          oldModuleMap(mx.name) = Seq(mx)
      }
    }
    moduleOrder.foreach(n => onModule(moduleMap(n)))
    (oldModuleMap.toMap, dedupMap.toMap)
  }

  def run(c: Circuit, noDedups: Seq[String]): (Circuit, RenameMap) = {
    val moduleOrder = buildModuleOrder(c)
    val moduleMap = c.modules.map(m => m.name -> m).toMap

    val (oldModuleMap, dedupMap) = findDups(moduleOrder, moduleMap, noDedups)

    // Use old module list to preserve ordering
    val dedupedModules = c.modules.flatMap(m => oldModuleMap.get(m.name).map(_.head))

    val cname = CircuitName(c.main)
    val renameMap = RenameMap(dedupMap.map { case (from, to) =>
      logger.debug(s"[Dedup] $from -> $to")
      ModuleName(from, cname) -> List(ModuleName(to, cname))
    })

    (c.copy(modules = dedupedModules), renameMap)
  }

  def execute(state: CircuitState): CircuitState = {
    val noDedups = getMyAnnotations(state).collect { case NoDedupAnnotation(ModuleName(m, c)) => m }
    val (newC, renameMap) = run(state.circuit, noDedups)
    state.copy(circuit = newC, renames = Some(renameMap))
  }
}
