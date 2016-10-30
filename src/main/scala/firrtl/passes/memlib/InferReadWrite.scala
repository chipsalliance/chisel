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
package memlib

import firrtl._
import firrtl.ir._
import firrtl.Mappers._
import firrtl.PrimOps._
import firrtl.Utils.{one, zero, BoolType}
import MemPortUtils.memPortField
import firrtl.passes.memlib.AnalysisUtils.{Connects, getConnects, getOrigin}
import WrappedExpression.weq
import Annotations._

case class InferReadWriteAnnotation(t: String, tID: TransID)
    extends Annotation with Loose with Unstable {
  val target = CircuitName(t)
  def duplicate(n: Named) = this.copy(t=n.name)
}

// This pass examine the enable signals of the read & write ports of memories
// whose readLatency is greater than 1 (usually SeqMem in Chisel).
// If any product term of the enable signal of the read port is the complement
// of any product term of the enable signal of the write port, then the readwrite
// port is inferred.
object InferReadWritePass extends Pass {
  def name = "Infer ReadWrite Ports"

  type Netlist = collection.mutable.HashMap[String, Expression]
  type Statements = collection.mutable.ArrayBuffer[Statement]
  type PortSet = collection.mutable.HashSet[String]

  private implicit def toString(e: Expression): String = e.serialize

  def getProductTerms(connects: Connects)(e: Expression): Seq[Expression] = e match {
    // No ConstProp yet...
    case Mux(cond, tval, fval, _) if weq(tval, one) && weq(fval, zero) =>
      getProductTerms(connects)(cond)
    // Visit each term of AND operation
    case DoPrim(op, args, consts, tpe) if op == And =>
      e +: (args flatMap getProductTerms(connects))
    // Visit connected nodes to references
    case _: WRef | _: WSubField | _: WSubIndex => connects get e match {
      case None => Seq(e)
      case Some(ex) => e +: getProductTerms(connects)(ex)
    }
    // Otherwise just return itself
    case _ => Seq(e)
  }

  def checkComplement(a: Expression, b: Expression) = (a, b) match {
    // b ?= Not(a)
    case (_, DoPrim(Not, args, _, _)) => weq(args.head, a)
    // a ?= Not(b)
    case (DoPrim(Not, args, _, _), _) => weq(args.head, b)
    // b ?= Eq(a, 0) or b ?= Eq(0, a)
    case (_, DoPrim(Eq, args, _, _)) =>
      weq(args.head, a) && weq(args(1), zero) ||
      weq(args(1), a) && weq(args.head, zero)
    // a ?= Eq(b, 0) or b ?= Eq(0, a)
    case (DoPrim(Eq, args, _, _), _) =>
      weq(args.head, b) && weq(args(1), zero) ||
      weq(args(1), b) && weq(args.head, zero)
    case _ => false
  }


  def replaceExp(repl: Netlist)(e: Expression): Expression =
    e map replaceExp(repl) match {
      case ex: WSubField => repl getOrElse (ex.serialize, ex)
      case ex => ex
    }

  def replaceStmt(repl: Netlist)(s: Statement): Statement =
    s map replaceStmt(repl) map replaceExp(repl) match {
      case Connect(_, EmptyExpression, _) => EmptyStmt 
      case sx => sx
    }
    
  def inferReadWriteStmt(connects: Connects,
                         repl: Netlist,
                         stmts: Statements)
                         (s: Statement): Statement = s match {
    // infer readwrite ports only for non combinational memories
    case mem: DefMemory if mem.readLatency > 0 =>
      val ut = UnknownType
      val ug = UNKNOWNGENDER
      val readers = new PortSet
      val writers = new PortSet
      val readwriters = collection.mutable.ArrayBuffer[String]()
      val namespace = Namespace(mem.readers ++ mem.writers ++ mem.readwriters)
      for (w <- mem.writers ; r <- mem.readers) {
        val wp = getProductTerms(connects)(memPortField(mem, w, "en"))
        val rp = getProductTerms(connects)(memPortField(mem, r, "en"))
        val wclk = getOrigin(connects)(memPortField(mem, w, "clk"))
        val rclk = getOrigin(connects)(memPortField(mem, r, "clk"))
        if (weq(wclk, rclk) && (wp exists (a => rp exists (b => checkComplement(a, b))))) {
          val rw = namespace newName "rw"
          val rwExp = createSubField(createRef(mem.name), rw)
          readwriters += rw
          readers += r
          writers += w
          repl(memPortField(mem, r, "clk"))  = EmptyExpression
          repl(memPortField(mem, r, "en"))   = EmptyExpression
          repl(memPortField(mem, r, "addr")) = EmptyExpression
          repl(memPortField(mem, r, "data")) = createSubField(rwExp, "rdata")
          repl(memPortField(mem, w, "clk"))  = EmptyExpression
          repl(memPortField(mem, w, "en"))   = createSubField(rwExp, "wmode")
          repl(memPortField(mem, w, "addr")) = EmptyExpression
          repl(memPortField(mem, w, "data")) = createSubField(rwExp, "wdata")
          repl(memPortField(mem, w, "mask")) = createSubField(rwExp, "wmask")
          stmts += Connect(NoInfo, createSubField(rwExp, "clk"), wclk)
          stmts += Connect(NoInfo, createSubField(rwExp, "en"),
             DoPrim(Or, Seq(connects(memPortField(mem, r, "en")),
                            connects(memPortField(mem, w, "en"))), Nil, BoolType))
          stmts += Connect(NoInfo, createSubField(rwExp, "addr"),
                        Mux(connects(memPortField(mem, w, "en")),
                            connects(memPortField(mem, w, "addr")),
                            connects(memPortField(mem, r, "addr")), UnknownType))
        }
      }
      if (readwriters.isEmpty) mem else mem copy (
        readers = mem.readers filterNot readers,
        writers = mem.writers filterNot writers,
        readwriters = mem.readwriters ++ readwriters)
    case sx => sx map inferReadWriteStmt(connects, repl, stmts)
  }

  def inferReadWrite(m: DefModule) = {
    val connects = getConnects(m)
    val repl = new Netlist
    val stmts = new Statements
    (m map inferReadWriteStmt(connects, repl, stmts)
       map replaceStmt(repl)) match {
      case m: ExtModule => m
      case m: Module => m copy (body = Block(m.body +: stmts))
    }
  }

  def run(c: Circuit) = c copy (modules = c.modules map inferReadWrite)
}

// Transform input: Middle Firrtl. Called after "HighFirrtlToMidleFirrtl"
// To use this transform, circuit name should be annotated with its TransId.
class InferReadWrite(transID: TransID) extends Transform with SimpleRun {
  def passSeq = Seq(
    InferReadWritePass,
    CheckInitialization,
    InferTypes,
    ResolveKinds,
    ResolveGenders
  )
  def execute(c: Circuit, map: AnnotationMap) = map get transID match {
    case Some(p) => p get CircuitName(c.main) match {
      case Some(InferReadWriteAnnotation(_, _)) => run(c, passSeq)
      case _ => sys.error("Unexpected annotation for InferReadWrite")
    }
    case _ => TransformResult(c)
  }
}
