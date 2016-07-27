/*
Copyright (c) 2014 - 2016 The Regents of the University of
California (Regents). All Rights Reserved.  Redistribution and use in
source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
   * Redistributions of source code must retain the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer.
   * Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer in the documentation and/or other materials
     provided with the distribution.
   * Neither the name of the Regents nor the names of its contributors
     may be used to endorse or promote products derived from this
     software without specific prior written permission.
IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF
ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION
TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
MODIFICATIONS.
*/

package firrtl.passes

import scala.collection.mutable.{ArrayBuffer, HashSet, HashMap}

import firrtl._
import firrtl.ir._
import firrtl.Mappers._
import firrtl.PrimOps._

object InferReadWrite extends Pass {
  def name = "Infer ReadWrite Ports"

  def inferReadWrite(m: Module) = {
    import WrappedExpression.we
    val connects = HashMap[String, Expression]()
    val repl = HashMap[String, Expression]()
    val stmts = ArrayBuffer[Statement]()
    val zero = we(UIntLiteral(0, IntWidth(1)))
    val one = we(UIntLiteral(1, IntWidth(1)))

    def analyze(s: Statement): Unit = s match {
      case s: Connect  =>
        connects(s.loc.serialize) = s.expr
      case s: PartialConnect =>
        connects(s.loc.serialize) = s.expr
      case s: DefNode =>
        connects(s.name) = s.value
      case s: Block =>
        s.stmts foreach analyze
      case _ =>
    }

    def getProductTermsFromExp(e: Expression): Seq[Expression] =
      e match {
        // No ConstProp yet...
        case Mux(cond, tval, fval, _) if we(tval) == one && we(fval) == zero =>
          cond +: getProductTerms(cond.serialize)
        case DoPrim(op, args, consts, tpe) if op == And =>
          e +: (args flatMap getProductTermsFromExp)
        case _: WRef | _: SubField | _: SubIndex | _: SubAccess =>
          e +: getProductTerms(e.serialize)
        case _ =>
          List(e)
      }

    def getProductTerms(node: String): Seq[Expression] =
      if (connects contains node) getProductTermsFromExp(connects(node)) else Nil

    def checkComplement(a: Expression, b: Expression) = (a, b) match {
      case (_, DoPrim(op, args, _, _)) if op == Not =>
        args.head.serialize == a.serialize
      case (DoPrim(op, args, _, _), _) if op == Not =>
        args.head.serialize == b.serialize
      case (_, DoPrim(op, args, _, _)) if op == Eq =>
        args(0).serialize == a.serialize && we(args(1)) == zero ||
        args(1).serialize == a.serialize && we(args(0)) == zero
      case (DoPrim(op, args, _, _), _) if op == Eq =>
        args(0).serialize == b.serialize && we(args(1)) == zero ||
        args(1).serialize == b.serialize && we(args(0)) == zero
      case _ => false
    }

    def inferReadWrite(s: Statement): Statement = s map inferReadWrite match {
      case mem: DefMemory if mem.readLatency > 0 =>
        var idx = 0
        val bt = UIntType(IntWidth(1))
        val ut = UnknownType
        val ug = UNKNOWNGENDER
        val readers = HashSet[String]()
        val writers = HashSet[String]()
        val readwriters = ArrayBuffer[String]()
        for (w <- mem.writers ; r <- mem.readers) {
          val wp = getProductTerms(s"${mem.name}.$w.en")
          val rp = getProductTerms(s"${mem.name}.$r.en")
          if (wp exists (a => rp exists (b => checkComplement(a, b)))) {
            val rw = s"rw_$idx"
            val rw_exp = WSubField(WRef(mem.name, ut, NodeKind(), ug), rw, ut, ug)
            readwriters += rw
            readers += r
            writers += w
            repl(s"${mem.name}.$r.en")   = EmptyExpression
            repl(s"${mem.name}.$r.clk")  = EmptyExpression
            repl(s"${mem.name}.$r.addr") = EmptyExpression
            repl(s"${mem.name}.$r.data") = WSubField(rw_exp, "rdata", mem.dataType, MALE)
            repl(s"${mem.name}.$w.en")   = WSubField(rw_exp, "wmode", bt, FEMALE)
            repl(s"${mem.name}.$w.clk")  = EmptyExpression
            repl(s"${mem.name}.$w.addr") = EmptyExpression
            repl(s"${mem.name}.$w.data") = WSubField(rw_exp, "data", mem.dataType, FEMALE)
            repl(s"${mem.name}.$w.mask") = WSubField(rw_exp, "mask", ut, FEMALE)
            stmts += Connect(NoInfo, WSubField(rw_exp, "clk", ClockType, FEMALE),
              WRef("clk", ClockType, NodeKind(), MALE))
            stmts += Connect(NoInfo, WSubField(rw_exp, "en", bt, FEMALE),
              DoPrim(Or, List(connects(s"${mem.name}.$r.en"), connects(s"${mem.name}.$w.en")), Nil, bt))
            stmts += Connect(NoInfo, WSubField(rw_exp, "addr", ut, FEMALE),
              Mux(connects(s"${mem.name}.$w.en"), connects(s"${mem.name}.$w.addr"), 
                  connects(s"${mem.name}.$r.addr"), ut))
            idx += 1
          }
        }
        if (readwriters.isEmpty) mem else DefMemory(mem.info,
          mem.name, mem.dataType, mem.depth, mem.writeLatency, mem.readLatency,
          mem.readers filterNot readers, mem.writers filterNot writers,
          mem.readwriters ++ readwriters)
      case s => s
    }

    def replaceExp(e: Expression): Expression =
      e map replaceExp match {
        case e: WSubField => repl getOrElse (e.serialize, e)
        case e => e
      }

    def replaceStmt(s: Statement): Statement =
      s map replaceStmt map replaceExp match {
        case Connect(info, loc, exp) if loc == EmptyExpression => EmptyStmt 
        case s => s
      }
    
    analyze(m.body)
    Module(m.info, m.name, m.ports, Block((m.body map inferReadWrite map replaceStmt) +: stmts.toSeq))
  }

  def run (c:Circuit) = Circuit(c.info, c.modules map {
    case m: Module => inferReadWrite(m)
    case m: ExtModule => m
  }, c.main)
}
