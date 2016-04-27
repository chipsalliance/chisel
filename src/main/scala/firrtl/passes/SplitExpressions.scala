package firrtl
package passes

import firrtl.Mappers.{ExpMap, StmtMap}
import firrtl.Utils.{tpe, kind, gender, info}
import scala.collection.mutable


// Splits compound expressions into simple expressions
//  and named intermediate nodes
object SplitExpressions extends Pass {
   def name = "Split Expressions"
   private def onModule(m: InModule): InModule = {
      val namespace = Namespace(m)
      def onStmt(s: Stmt): Stmt = {
         val v = mutable.ArrayBuffer[Stmt]()
         // Splits current expression if needed
         // Adds named temporaries to v
         def split(e: Expression): Expression = e match {
            case e: DoPrim => {
               val name = namespace.newTemp
               v += DefNode(info(s), name, e)
               WRef(name, tpe(e), kind(e), gender(e))
            }
            case e: Mux => {
               val name = namespace.newTemp
               v += DefNode(info(s), name, e)
               WRef(name, tpe(e), kind(e), gender(e))
            }
            case e: ValidIf => {
               val name = namespace.newTemp
               v += DefNode(info(s), name, e)
               WRef(name, tpe(e), kind(e), gender(e))
            }
            case e => e
         }
         // Recursive. Splits compound nodes
         def onExp(e: Expression): Expression = {
            val ex = e map onExp
            ex match {
               case (_: DoPrim) => ex map split
               case v => v
            }
         }
         val x = s map onExp
         x match {
            case x: Begin => x map onStmt
            case x: Empty => x
            case x => {
               v += x
               if (v.size > 1) Begin(v.toVector)
               else v(0)
            }
         }
      }
      InModule(m.info, m.name, m.ports, onStmt(m.body))
   }
   def run(c: Circuit): Circuit = {
      val modulesx = c.modules.map( _ match {
         case (m:InModule) => onModule(m)
         case (m:ExModule) => m
      })
      Circuit(c.info, modulesx, c.main)
   }
}
