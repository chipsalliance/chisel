package firrtl
package passes
import firrtl.Mappers.{ExpMap, StmtMap}

// Removes ValidIf as an optimization
object RemoveValidIf extends Pass {
   def name = "Remove ValidIfs"
   // Recursive. Removes ValidIf's
   private def onExp(e: Expression): Expression = {
      e map onExp match {
         case ValidIf(cond, value, tpe) => value
         case x => x
      }
   }
   // Recursive.
   private def onStmt(s: Stmt): Stmt = s map onStmt map onExp

   private def onModule(m: Module): Module = {
      m match {
         case m:InModule => InModule(m.info, m.name, m.ports, onStmt(m.body))
         case m:ExModule => m
      }
   }

   def run(c: Circuit): Circuit = Circuit(c.info, c.modules.map(onModule _), c.main)
}
