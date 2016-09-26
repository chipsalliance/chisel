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

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import firrtl.PrimOps._
import MemPortUtils._

/** This pass generates delay reigsters for memories for verilog */
object VerilogMemDelays extends Pass {
  def name = "Verilog Memory Delays"
  val ug = UNKNOWNGENDER
  type Netlist = collection.mutable.HashMap[String, Expression]
  implicit def expToString(e: Expression): String = e.serialize
  private def NOT(e: Expression) = DoPrim(Not, Seq(e), Nil, BoolType)
  private def AND(e1: Expression, e2: Expression) = DoPrim(And, Seq(e1, e2), Nil, BoolType)

  def buildNetlist(netlist: Netlist)(s: Statement): Statement = {
    s match {
      case Connect(_, loc, expr) => kind(loc) match {
        case MemKind => netlist(loc) = expr
        case _ =>
      }
      case _ =>
    }
    s map buildNetlist(netlist)
  }

  def memDelayStmt(
      netlist: Netlist,
      namespace: Namespace,
      repl: Netlist)
      (s: Statement): Statement = s map memDelayStmt(netlist, namespace, repl) match {
    case s: DefMemory =>
      val ports = (s.readers ++ s.writers).toSet
      def newPortName(rw: String, p: String) = (for {
        idx <- Stream from 0
        newName = s"${rw}_${p}_$idx"
        if !ports(newName)
      } yield newName).head
      val rwMap = (s.readwriters map (rw =>
        rw -> (newPortName(rw, "r"), newPortName(rw, "w")))).toMap
      // 1. readwrite ports are split into read & write ports
      // 2. memories are transformed into combinational
      //    because latency pipes are added for longer latencies
      val mem = s copy (
        readers = (s.readers ++ (s.readwriters map (rw => rwMap(rw)._1))),
        writers = (s.writers ++ (s.readwriters map (rw => rwMap(rw)._2))),
        readwriters = Nil, readLatency = 0, writeLatency = 1)
      def pipe(e: Expression, // Expression to be piped
               n: Int, // pipe depth
               clk: Expression, // clock expression
               cond: Expression // condition for pipes
              ): (Expression, Seq[Statement]) = {
        // returns
        // 1) reference to the last pipe register
        // 2) pipe registers and connects
        val node = DefNode(NoInfo, namespace.newTemp, netlist(e))
        val wref = WRef(node.name, e.tpe, NodeKind, MALE)
        ((0 until n) foldLeft (wref, Seq[Statement](node))){case ((ex, stmts), i) =>
          val name = namespace newName s"${LowerTypes.loweredName(e)}_pipe_$i"
          val exx = WRef(name, e.tpe, RegKind, ug)
          (exx, stmts ++ Seq(DefRegister(NoInfo, name, e.tpe, clk, zero, exx)) ++
            (if (i < n - 1 && WrappedExpression.weq(cond, one)) Seq(Connect(NoInfo, exx, ex)) else {
              val condn = namespace newName s"${LowerTypes.loweredName(e)}_en"
              val condx = WRef(condn, BoolType, NodeKind, FEMALE)
              Seq(DefNode(NoInfo, condn, cond),
                  Connect(NoInfo, condx, cond),
                  Connect(NoInfo, exx, Mux(condx, ex, exx, e.tpe)))
            })
          )
        }
      }
      def readPortConnects(reader: String,
                           clk: Expression,
                           en: Expression,
                           addr: Expression) = Seq(
        Connect(NoInfo, memPortField(mem, reader, "clk"), clk),
        // connect latency pipes to read ports
        Connect(NoInfo, memPortField(mem, reader, "en"), en),
        Connect(NoInfo, memPortField(mem, reader, "addr"), addr)
      )
      def writePortConnects(writer: String,
                            clk: Expression,
                            en: Expression,
                            mask: Expression,
                            addr: Expression,
                            data: Expression) = Seq(
        Connect(NoInfo, memPortField(mem, writer, "clk"), clk),
        // connect latency pipes to write ports
        Connect(NoInfo, memPortField(mem, writer, "en"), en),
        Connect(NoInfo, memPortField(mem, writer, "mask"), mask),
        Connect(NoInfo, memPortField(mem, writer, "addr"), addr),
        Connect(NoInfo, memPortField(mem, writer, "data"), data)
      )
 

      Block(mem +: ((s.readers flatMap {reader =>
        // generate latency pipes for read ports (enable & addr)
        val clk = netlist(memPortField(s, reader, "clk"))
        val (en, ss1) = pipe(memPortField(s, reader, "en"), s.readLatency - 1, clk, one)
        val (addr, ss2) = pipe(memPortField(s, reader, "addr"), s.readLatency, clk, en)
        ss1 ++ ss2 ++ readPortConnects(reader, clk, en, addr)
      }) ++ (s.writers flatMap {writer =>
        // generate latency pipes for write ports (enable, mask, addr, data)
        val clk = netlist(memPortField(s, writer, "clk"))
        val (en, ss1) = pipe(memPortField(s, writer, "en"), s.writeLatency - 1, clk, one)
        val (mask, ss2) = pipe(memPortField(s, writer, "mask"), s.writeLatency - 1, clk, one)
        val (addr, ss3) = pipe(memPortField(s, writer, "addr"), s.writeLatency - 1, clk, one)
        val (data, ss4) = pipe(memPortField(s, writer, "data"), s.writeLatency - 1, clk, one)
        ss1 ++ ss2 ++ ss3 ++ ss4 ++ writePortConnects(writer, clk, en, mask, addr, data)
      }) ++ (s.readwriters flatMap {readwriter =>
        val (reader, writer) = rwMap(readwriter)
        val clk = netlist(memPortField(s, readwriter, "clk"))
        // generate latency pipes for readwrite ports (enable, addr, wmode, wmask, wdata)
        val (en, ss1) = pipe(memPortField(s, readwriter, "en"), s.readLatency - 1, clk, one)
        val (wmode, ss2) = pipe(memPortField(s, readwriter, "wmode"), s.writeLatency - 1, clk, one)
        val (wmask, ss3) = pipe(memPortField(s, readwriter, "wmask"), s.writeLatency - 1, clk, one)
        val (wdata, ss4) = pipe(memPortField(s, readwriter, "wdata"), s.writeLatency - 1, clk, one)
        val (raddr, ss5) = pipe(memPortField(s, readwriter, "addr"), s.readLatency, clk, AND(en, NOT(wmode)))
        val (waddr, ss6) = pipe(memPortField(s, readwriter, "addr"), s.writeLatency - 1, clk, one)
        repl(memPortField(s, readwriter, "rdata")) = memPortField(mem, reader, "data")
        ss1 ++ ss2 ++ ss3 ++ ss4 ++ ss5 ++ ss6 ++
        readPortConnects(reader, clk, en, raddr) ++
        writePortConnects(writer, clk, AND(en, wmode), wmask, waddr, wdata)
      })))
    case s: Connect => kind(s.loc) match {
      case MemKind => EmptyStmt
      case _ => s
    }
    case s => s
  }

  def replaceExp(repl: Netlist)(e: Expression): Expression = e match {
    case e: WSubField => repl get e match {
      case Some(ex) => ex
      case None => e
    }
    case e => e map replaceExp(repl)
  }

  def replaceStmt(repl: Netlist)(s: Statement): Statement =
    s map replaceStmt(repl) map replaceExp(repl)

  def memDelayMod(m: DefModule): DefModule = {
    val netlist = new Netlist
    val namespace = Namespace(m)
    val repl = new Netlist
    (m map buildNetlist(netlist)
       map memDelayStmt(netlist, namespace, repl)
       map replaceStmt(repl))
  }

  def run(c: Circuit): Circuit =
    c copy (modules = (c.modules map memDelayMod))
}
