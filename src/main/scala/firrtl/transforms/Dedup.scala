// See LICENSE for license details.

package firrtl
package transforms

import firrtl.ir._
import firrtl.Mappers._
import firrtl.annotations._
import firrtl.passes.PassException

// Datastructures
import scala.collection.mutable

// Only use on legal Firrtl. Specifically, the restriction of
//  instance loops must have been checked, or else this pass can
//  infinitely recurse
class DedupModules extends Transform {
  def inputForm = HighForm
  def outputForm = HighForm
  def execute(state: CircuitState): CircuitState = CircuitState(run(state.circuit), state.form)
  def run(c: Circuit): Circuit = {
    val moduleOrder = mutable.ArrayBuffer.empty[String]
    val moduleMap = c.modules.map(m => m.name -> m).toMap
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
        if(!hasInstance(b)) moduleOrder += m.name
        m
      case e: ExtModule =>
        moduleOrder += m.name
        m
      case _ => m
    }

    while((moduleOrder.size < c.modules.size)) {
      c.modules.foreach(m => if(!moduleOrder.contains(m.name)) addModule(m))
    }

    // Module body -> Module name
    val dedupModules = mutable.HashMap.empty[String, String]
    // Old module name -> dup module name
    val dedupMap = mutable.HashMap.empty[String, String]
    // Dup module name -> all old module names
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
      val string = mxx match {
        case Module(i, n, ps, b) =>
          ps.map(_.serialize).mkString + b.serialize
        case ExtModule(i, n, ps, dn, p) =>
          ps.map(_.serialize).mkString + dn + p.map(_.serialize).mkString
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
    def mergeModules(ms: Seq[DefModule]) = {
      def mergeStatements(ss: Seq[Statement]): Statement = ss.head match {
        case Block(stmts) =>
          val inverted = invertSeqs(ss.map { case Block(s) => s })
          val finalStmts = inverted.map { jStmts => mergeStatements(jStmts) }
          Block(finalStmts.toSeq)
        case Conditionally(info, pred, conseq, alt) =>
          val finalConseq = mergeStatements(ss.map { case Conditionally(_, _, c, _) => c })
          val finalAlt = mergeStatements(ss.map { case Conditionally(_, _, _, a) => a })
          val finalInfo = ss.map { case Conditionally(i, _, _, _) => i }.reduce (_ ++ _)
          Conditionally(finalInfo, pred, finalConseq, finalAlt)
        case sx: HasInfo => sx match {
          case s: DefWire => s.copy(info = ss.map(getInfo).reduce(_ ++ _))
          case s: DefNode => s.copy(info = ss.map(getInfo).reduce(_ ++ _))
          case s: DefRegister => s.copy(info = ss.map(getInfo).reduce(_ ++ _))
          case s: DefInstance => s.copy(info = ss.map(getInfo).reduce(_ ++ _))
          case s: WDefInstance => s.copy(info = ss.map(getInfo).reduce(_ ++ _))
          case s: DefMemory => s.copy(info = ss.map(getInfo).reduce(_ ++ _))
          case s: Connect => s.copy(info = ss.map(getInfo).reduce(_ ++ _))
          case s: PartialConnect => s.copy(info = ss.map(getInfo).reduce(_ ++ _))
          case s: IsInvalid => s.copy(info = ss.map(getInfo).reduce(_ ++ _))
          case s: Attach => s.copy(info = ss.map(getInfo).reduce(_ ++ _))
          case s: Stop => s.copy(info = ss.map(getInfo).reduce(_ ++ _))
          case s: Print => s.copy(info = ss.map(getInfo).reduce(_ ++ _))
        }
        case s => s
      }
      def getInfo(s: Any): Info = s match {
        case sx: HasInfo => sx.info
        case _ => NoInfo
      }
      def invertSeqs[A](seq: Seq[Seq[A]]): Seq[Seq[A]] = {
        val finalSeq = collection.mutable.ArrayBuffer[Seq[A]]()
        for(j <- 0 until seq.head.size) {
          finalSeq += seq.map(s => s(j))
        }
        finalSeq.toSeq
      }
      val finalPorts = invertSeqs(ms.map(_.ports)).map { jPorts => 
        jPorts.tail.foldLeft(jPorts.head) { (p1, p2) =>
          Port(p1.info ++ p2.info, p1.name, p1.direction, p1.tpe)
        }
      }
      val finalInfo = ms.map(getInfo).reduce(_ ++ _)
      ms.head match {
        case e: ExtModule => ExtModule(finalInfo, e.name, finalPorts, e.defname, e.params)
        case e: Module => Module(finalInfo, e.name, finalPorts, mergeStatements(ms.collect { case m: Module => m.body}))
      }
    }
    moduleOrder.foreach(n => onModule(moduleMap(n)))

    // Use old module list to preserve ordering
    val dedupedModules = c.modules.flatMap { m => 
      oldModuleMap.get(m.name) match {
        case Some(modules) => Some(mergeModules(modules))
        case None => None
      }
    }
    c.copy(modules = dedupedModules)
  }
}
