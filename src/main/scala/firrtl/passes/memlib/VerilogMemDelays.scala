// See LICENSE for license details.

package firrtl.passes
package memlib

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import firrtl.PrimOps._
import MemPortUtils._

import collection.mutable

/** This pass generates delay reigsters for memories for verilog */
object VerilogMemDelays extends Pass {
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
      repl: Netlist,
      stmts: mutable.ArrayBuffer[Statement])
      (s: Statement): Statement = s.map(memDelayStmt(netlist, namespace, repl, stmts)) match {
    case sx: DefMemory =>
      val ports = (sx.readers ++ sx.writers).toSet
      def newPortName(rw: String, p: String) = (for {
        idx <- Stream from 0
        newName = s"${rw}_${p}_$idx"
        if !ports(newName)
      } yield newName).head
      val rwMap = (sx.readwriters map (rw =>
        rw -> (newPortName(rw, "r"), newPortName(rw, "w")))).toMap
      // 1. readwrite ports are split into read & write ports
      // 2. memories are transformed into combinational
      //    because latency pipes are added for longer latencies
      val mem = sx copy (
        readers = sx.readers ++ (sx.readwriters map (rw => rwMap(rw)._1)),
        writers = sx.writers ++ (sx.readwriters map (rw => rwMap(rw)._2)),
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
 
      stmts ++= ((sx.readers flatMap {reader =>
        // generate latency pipes for read ports (enable & addr)
        val clk = netlist(memPortField(sx, reader, "clk"))
        val (en, ss1) = pipe(memPortField(sx, reader, "en"), sx.readLatency - 1, clk, one)
        val (addr, ss2) = pipe(memPortField(sx, reader, "addr"), sx.readLatency, clk, en)
        ss1 ++ ss2 ++ readPortConnects(reader, clk, en, addr)
      }) ++ (sx.writers flatMap {writer =>
        // generate latency pipes for write ports (enable, mask, addr, data)
        val clk = netlist(memPortField(sx, writer, "clk"))
        val (en, ss1) = pipe(memPortField(sx, writer, "en"), sx.writeLatency - 1, clk, one)
        val (mask, ss2) = pipe(memPortField(sx, writer, "mask"), sx.writeLatency - 1, clk, one)
        val (addr, ss3) = pipe(memPortField(sx, writer, "addr"), sx.writeLatency - 1, clk, one)
        val (data, ss4) = pipe(memPortField(sx, writer, "data"), sx.writeLatency - 1, clk, one)
        ss1 ++ ss2 ++ ss3 ++ ss4 ++ writePortConnects(writer, clk, en, mask, addr, data)
      }) ++ (sx.readwriters flatMap {readwriter =>
        val (reader, writer) = rwMap(readwriter)
        val clk = netlist(memPortField(sx, readwriter, "clk"))
        // generate latency pipes for readwrite ports (enable, addr, wmode, wmask, wdata)
        val (en, ss1) = pipe(memPortField(sx, readwriter, "en"), sx.readLatency - 1, clk, one)
        val (wmode, ss2) = pipe(memPortField(sx, readwriter, "wmode"), sx.writeLatency - 1, clk, one)
        val (wmask, ss3) = pipe(memPortField(sx, readwriter, "wmask"), sx.writeLatency - 1, clk, one)
        val (wdata, ss4) = pipe(memPortField(sx, readwriter, "wdata"), sx.writeLatency - 1, clk, one)
        val (raddr, ss5) = pipe(memPortField(sx, readwriter, "addr"), sx.readLatency, clk, AND(en, NOT(wmode)))
        val (waddr, ss6) = pipe(memPortField(sx, readwriter, "addr"), sx.writeLatency - 1, clk, one)
        repl(memPortField(sx, readwriter, "rdata")) = memPortField(mem, reader, "data")
        ss1 ++ ss2 ++ ss3 ++ ss4 ++ ss5 ++ ss6 ++
        readPortConnects(reader, clk, en, raddr) ++
        writePortConnects(writer, clk, AND(en, wmode), wmask, waddr, wdata)
      }))
      mem // The mem stays put
    case sx: Connect => kind(sx.loc) match {
      case MemKind => EmptyStmt
      case _ => sx
    }
    case sx => sx
  }

  def replaceExp(repl: Netlist)(e: Expression): Expression = e match {
    case ex: WSubField => repl get ex match {
      case Some(exx) => exx
      case None => ex
    }
    case ex => ex map replaceExp(repl)
  }

  def replaceStmt(repl: Netlist)(s: Statement): Statement =
    s map replaceStmt(repl) map replaceExp(repl)

  def appendStmts(sx: Seq[Statement])(s: Statement): Statement = Block(s +: sx)

  def memDelayMod(m: DefModule): DefModule = {
    val netlist = new Netlist
    val namespace = Namespace(m)
    val repl = new Netlist
    val extraStmts = mutable.ArrayBuffer.empty[Statement]
    m.map(buildNetlist(netlist))
     .map(memDelayStmt(netlist, namespace, repl, extraStmts))
     .map(replaceStmt(repl))
     .map(appendStmts(extraStmts))
  }

  def run(c: Circuit): Circuit =
    c copy (modules = c.modules map memDelayMod)
}
