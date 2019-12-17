// See LICENSE for license details.

package firrtl.passes
package memlib

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import firrtl.traversals.Foreachers._
import firrtl.transforms
import firrtl.options.Dependency

import MemPortUtils._
import WrappedExpression._

import collection.mutable

object MemDelayAndReadwriteTransformer {
  // Representation of a group of signals and associated valid signals
  case class WithValid(valid: Expression, payload: Seq[Expression])

  // Grouped statements that are split into declarations and connects to ease ordering
  case class SplitStatements(decls: Seq[Statement], conns: Seq[Connect])

  // Utilities for generating hardware
  def NOT(e: Expression) = DoPrim(PrimOps.Not, Seq(e), Nil, BoolType)
  def AND(e1: Expression, e2: Expression) = DoPrim(PrimOps.And, Seq(e1, e2), Nil, BoolType)
  def connect(l: Expression, r: Expression): Connect = Connect(NoInfo, l, r)
  def condConnect(c: Expression)(l: Expression, r: Expression): Connect = connect(l, Mux(c, r, l, l.tpe))

  // Utilities for working with WithValid groups
  def connect(l: WithValid, r: WithValid): Seq[Connect] = {
    val paired = (l.valid +: l.payload) zip (r.valid +: r.payload)
    paired.map { case (le, re) => connect(le, re) }
  }

  def condConnect(l: WithValid, r: WithValid): Seq[Connect] = {
    connect(l.valid, r.valid) +: (l.payload zip r.payload).map { case (le, re) => condConnect(r.valid)(le, re) }
  }

  // Internal representation of a pipeline stage with an associated valid signal
  private case class PipeStageWithValid(idx: Int, ref: WithValid, stmts: SplitStatements = SplitStatements(Nil, Nil))

  // Utilities for creating legal names for registers
  private val metaChars = raw"[\[\]\.]".r
  private def flatName(e: Expression) = metaChars.replaceAllIn(e.serialize, "_")

  // Pipeline a group of signals with an associated valid signal. Gate registers when possible.
  def pipelineWithValid(ns: Namespace)(
    clock: Expression,
    depth: Int,
    src: WithValid,
    nameTemplate: Option[WithValid] = None): (WithValid, Seq[Statement], Seq[Connect]) = {

    def asReg(e: Expression) = DefRegister(NoInfo, e.serialize, e.tpe, clock, zero, e)
    val template = nameTemplate.getOrElse(src)

    val stages = Seq.iterate(PipeStageWithValid(0, src), depth + 1) { case prev =>
      def pipeRegRef(e: Expression) = WRef(ns.newName(s"${flatName(e)}_pipe_${prev.idx}"), e.tpe, RegKind)
      val ref = WithValid(pipeRegRef(template.valid), template.payload.map(pipeRegRef))
      val regs = (ref.valid +: ref.payload).map(asReg)
      PipeStageWithValid(prev.idx + 1, ref, SplitStatements(regs, condConnect(ref, prev.ref)))
    }
    (stages.last.ref, stages.flatMap(_.stmts.decls), stages.flatMap(_.stmts.conns))
  }
}

/**
  * This class performs the primary work of the transform: splitting readwrite ports into separate
  * read and write ports while simultaneously compiling memory latencies to combinational-read
  * memories with delay pipelines. It is represented as a class that takes a module as a constructor
  * argument, as it encapsulates the mutable state required to analyze and transform one module.
  *
  * @note The final transformed module is found in the (sole public) field [[transformed]]
  */
class MemDelayAndReadwriteTransformer(m: DefModule) {
  import MemDelayAndReadwriteTransformer._

  private val ns = Namespace(m)
  private val netlist = new collection.mutable.HashMap[WrappedExpression, Expression]
  private val exprReplacements = new collection.mutable.HashMap[WrappedExpression, Expression]
  private val newConns = new mutable.ArrayBuffer[Connect]

  private def findMemConns(s: Statement): Unit = s match {
    case Connect(_, loc, expr) if (kind(loc) == MemKind) => netlist(we(loc)) = expr
    case _ => s.foreach(findMemConns)
  }

  private def swapMemRefs(e: Expression): Expression = e map swapMemRefs match {
    case sf: WSubField => exprReplacements.getOrElse(we(sf), sf)
    case ex => ex
  }

  private def transform(s: Statement): Statement = s.map(transform) match {
    case mem: DefMemory =>
      // Per-memory bookkeeping
      val portNS = Namespace(mem.readers ++ mem.writers)
      val rMap = mem.readwriters.map(rw => (rw -> portNS.newName(s"${rw}_r"))).toMap
      val wMap = mem.readwriters.map(rw => (rw -> portNS.newName(s"${rw}_w"))).toMap
      val newReaders = mem.readers ++ mem.readwriters.map(rMap(_))
      val newWriters = mem.writers ++ mem.readwriters.map(wMap(_))
      val newMem = DefMemory(mem.info, mem.name, mem.dataType, mem.depth, 1, 0, newReaders, newWriters, Nil)
      val rCmdDelay = if (mem.readUnderWrite == ReadUnderWrite.Old) 0 else mem.readLatency
      val rRespDelay = if (mem.readUnderWrite == ReadUnderWrite.Old) mem.readLatency else 0
      val wCmdDelay = mem.writeLatency - 1

      val readStmts = (mem.readers ++ mem.readwriters).map { case r =>
        def oldDriver(f: String) = netlist(we(memPortField(mem, r, f)))
        def newField(f: String) = memPortField(newMem, rMap.getOrElse(r, r), f)
        val clk = oldDriver("clk")

        // Pack sources of read command inputs into WithValid object -> different for readwriter
        val enSrc = if (rMap.contains(r)) AND(oldDriver("en"), NOT(oldDriver("wmode"))) else oldDriver("en")
        val cmdSrc = WithValid(enSrc, Seq(oldDriver("addr")))
        val cmdSink = WithValid(newField("en"), Seq(newField("addr")))
        val (cmdPiped, cmdDecls, cmdConns) = pipelineWithValid(ns)(clk, rCmdDelay, cmdSrc, nameTemplate = Some(cmdSink))
        val cmdPortConns = connect(cmdSink, cmdPiped) :+ connect(newField("clk"), clk)

        // Pipeline read response using *last* command pipe stage enable as the valid signal
        val resp = WithValid(cmdPiped.valid, Seq(newField("data")))
        val respPipeNameTemplate = Some(resp.copy(valid = cmdSink.valid)) // base pipeline register names off field names
        val (respPiped, respDecls, respConns) = pipelineWithValid(ns)(clk, rRespDelay, resp, nameTemplate = respPipeNameTemplate)

        // Make sure references to the read data get appropriately substituted
        val oldRDataName = if (rMap.contains(r)) "rdata" else "data"
        exprReplacements(we(memPortField(mem, r, oldRDataName))) = respPiped.payload.head

        // Return all statements; they're separated so connects can go after all declarations
        SplitStatements(cmdDecls ++ respDecls, cmdConns ++ cmdPortConns ++ respConns)
      }

      val writeStmts = (mem.writers ++ mem.readwriters).map { case w =>
        def oldDriver(f: String) = netlist(we(memPortField(mem, w, f)))
        def newField(f: String) = memPortField(newMem, wMap.getOrElse(w, w), f)
        val clk = oldDriver("clk")

        // Pack sources of write command inputs into WithValid object -> different for readwriter
        val cmdSrc = if (wMap.contains(w)) {
          val en = AND(oldDriver("en"), oldDriver("wmode"))
          WithValid(en, Seq(oldDriver("addr"), oldDriver("wmask"), oldDriver("wdata")))
        } else {
          WithValid(oldDriver("en"), Seq(oldDriver("addr"), oldDriver("mask"), oldDriver("data")))
        }

        // Pipeline write command, connect to memory
        val cmdSink = WithValid(newField("en"), Seq(newField("addr"), newField("mask"), newField("data")))
        val (cmdPiped, cmdDecls, cmdConns) = pipelineWithValid(ns)(clk, wCmdDelay, cmdSrc, nameTemplate = Some(cmdSink))
        val cmdPortConns = connect(cmdSink, cmdPiped) :+ connect(newField("clk"), clk)

        // Return all statements; they're separated so connects can go after all declarations
        SplitStatements(cmdDecls, cmdConns ++ cmdPortConns)
      }

      newConns ++= (readStmts ++ writeStmts).flatMap(_.conns)
      Block(newMem +: (readStmts ++ writeStmts).flatMap(_.decls))
    case sx: Connect if kind(sx.loc) == MemKind => EmptyStmt // Filter old mem connections
    case sx => sx.map(swapMemRefs)
  }

  val transformed = m match {
    case mod: Module =>
      findMemConns(mod.body)
      mod.copy(body = Block(transform(mod.body) +: newConns.toSeq))
    case mod => mod
  }
}

object VerilogMemDelays extends Pass {

  override val prerequisites = firrtl.stage.Forms.LowForm :+ Dependency(firrtl.passes.RemoveValidIf)

  override val dependents =
    Seq( Dependency[VerilogEmitter],
         Dependency[SystemVerilogEmitter] )

  override def invalidates(a: Transform): Boolean = a match {
    case _: transforms.ConstantPropagation => true
    case _ => false
  }

  def transform(m: DefModule): DefModule = (new MemDelayAndReadwriteTransformer(m)).transformed
  def run(c: Circuit): Circuit = c.copy(modules = c.modules.map(transform))
}
