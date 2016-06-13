package firrtl
package passes

import firrtl.Mappers.{ExpMap, StmtMap}
import firrtl.Utils.{tpe, kind, gender, info}
import firrtl.ir._
import scala.collection.mutable


// Splits compound expressions into simple expressions
//  and named intermediate nodes
object SplitExpressions extends Pass {
   def name = "Split Expressions"
   private def onModule(m: Module): Module = {
      val namespace = Namespace(m)
      def onStmt(s: Statement): Statement = {
         val v = mutable.ArrayBuffer[Statement]()
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
            case EmptyStmt => x
            case x => {
               v += x
               if (v.size > 1) Begin(v.toVector)
               else v(0)
            }
         }
      }
      Module(m.info, m.name, m.ports, onStmt(m.body))
   }
   def run(c: Circuit): Circuit = {
      val modulesx = c.modules.map( _ match {
         case m: Module => onModule(m)
         case m: ExtModule => m
      })
      Circuit(c.info, modulesx, c.main)
   }
}
