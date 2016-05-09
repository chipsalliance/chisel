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

   private def onModule(m: DefModule): DefModule = {
      m match {
         case m: Module => Module(m.info, m.name, m.ports, onStmt(m.body))
         case m: ExtModule => m
      }
   }

   def run(c: Circuit): Circuit = Circuit(c.info, c.modules.map(onModule _), c.main)
}
