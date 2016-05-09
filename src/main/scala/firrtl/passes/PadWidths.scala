package firrtl
package passes

import firrtl.Mappers.{ExpMap, StmtMap}
import firrtl.Utils.{tpe, long_BANG}

// Makes all implicit width extensions and truncations explicit
object PadWidths extends Pass {
   def name = "Pad Widths"
   private def width(t: Type): Int = long_BANG(t).toInt
   private def width(e: Expression): Int = width(tpe(e))
   // Returns an expression with the correct integer width
   private def fixup(i: Int)(e: Expression) = {
      def tx = tpe(e) match {
         case t: UIntType => UIntType(IntWidth(i))
         case t: SIntType => SIntType(IntWidth(i))
         // default case should never be reached
      }
      if (i > width(e))
         DoPrim(PAD_OP, Seq(e), Seq(i), tx)
      else if (i < width(e))
         DoPrim(BITS_SELECT_OP, Seq(e), Seq(i - 1, 0), tx)
      else e
   }
   // Recursive, updates expression so children exp's have correct widths
   private def onExp(e: Expression): Expression = {
      val sensitiveOps = Seq(
         LESS_OP, LESS_EQ_OP, GREATER_OP, GREATER_EQ_OP, EQUAL_OP,
         NEQUAL_OP, NOT_OP, AND_OP, OR_OP, XOR_OP, ADD_OP, SUB_OP,
         MUL_OP, DIV_OP, REM_OP, SHIFT_RIGHT_OP)
      val x = e map onExp
      x match {
         case Mux(cond, tval, fval, tpe) => {
            val tvalx = fixup(width(tpe))(tval)
            val fvalx = fixup(width(tpe))(fval)
            Mux(cond, tvalx, fvalx, tpe)
         }
         case DoPrim(op, args, consts, tpe) => op match {
            case _ if sensitiveOps.contains(op) => {
               val i = args.map(a => width(a)).foldLeft(0) {(a, b) => math.max(a, b)}
               x map fixup(i)
            }
            case DYN_SHIFT_LEFT_OP => {
               // special case as args aren't all same width
               val ax = fixup(width(tpe))(args(0))
               DoPrim(DSHLW_OP, Seq(ax, args(1)), consts, tpe)
            }
            case SHIFT_LEFT_OP => {
               // special case as arg should be same width as result
               val ax = fixup(width(tpe))(args(0))
               DoPrim(SHLW_OP, Seq(ax), consts, tpe)
            }
            case _ => x
         }
         case ValidIf(cond, value, tpe) => ValidIf(cond, fixup(width(tpe))(value), tpe)
         case x => x
      }
   }
   // Recursive. Fixes assignments and register initialization widths
   private def onStmt(s: Stmt): Stmt = {
      s map onExp match {
         case s: Connect => {
            val ex = fixup(width(s.loc))(s.exp)
            Connect(s.info, s.loc, ex)
         }
         case s: DefRegister => {
            val ex = fixup(width(s.tpe))(s.init)
            DefRegister(s.info, s.name, s.tpe, s.clock, s.reset, ex)
         }
         case s => s map onStmt
      }
   }
   private def onModule(m: DefModule): DefModule = {
      m match {
         case m: Module => Module(m.info, m.name, m.ports, onStmt(m.body))
         case m: ExtModule => m
      }
   }
   def run(c: Circuit): Circuit = Circuit(c.info, c.modules.map(onModule _), c.main)
}
