package firrtl
package passes

import firrtl.ir._
import firrtl.PrimOps._
import firrtl.Mappers._

// Makes all implicit width extensions and truncations explicit
object PadWidths extends Pass {
   def name = "Pad Widths"
   private def width(t: Type): Int = bitWidth(t).toInt
   private def width(e: Expression): Int = width(e.tpe)
   // Returns an expression with the correct integer width
   private def fixup(i: Int)(e: Expression) = {
      def tx = e.tpe match {
         case t: UIntType => UIntType(IntWidth(i))
         case t: SIntType => SIntType(IntWidth(i))
         // default case should never be reached
      }
      if (i > width(e)) {
         DoPrim(Pad, Seq(e), Seq(i), tx)
      } else if (i < width(e)) {
         val e2 = DoPrim(Bits, Seq(e), Seq(i - 1, 0), UIntType(IntWidth(i)))
         // Bit Select always returns UInt, cast if selecting from SInt
         e.tpe match {
            case UIntType(_) => e2
            case SIntType(_) => DoPrim(AsSInt, Seq(e2), Seq.empty, SIntType(IntWidth(i)))
         }
      } else {
        e
      }
   }
   // Recursive, updates expression so children exp's have correct widths
   private def onExp(e: Expression): Expression = {
      val sensitiveOps = Seq( Lt, Leq, Gt, Geq, Eq, Neq, Not, And, Or, Xor,
        Add, Sub, Mul, Div, Rem, Shr)
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
            case Dshl => {
               // special case as args aren't all same width
               val ax = fixup(width(tpe))(args(0))
               DoPrim(Dshlw, Seq(ax, args(1)), consts, tpe)
            }
            case Shl => {
               // special case as arg should be same width as result
               val ax = fixup(width(tpe))(args(0))
               DoPrim(Shlw, Seq(ax), consts, tpe)
            }
            case _ => x
         }
         case ValidIf(cond, value, tpe) => ValidIf(cond, fixup(width(tpe))(value), tpe)
         case x => x
      }
   }
   // Recursive. Fixes assignments and register initialization widths
   private def onStmt(s: Statement): Statement = {
      s map onExp match {
         case s: Connect => {
            val ex = fixup(width(s.loc))(s.expr)
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
