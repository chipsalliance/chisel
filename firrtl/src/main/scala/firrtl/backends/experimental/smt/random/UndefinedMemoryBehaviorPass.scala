// SPDX-License-Identifier: Apache-2.0

package firrtl.backends.experimental.smt.random

import firrtl.Utils.{isLiteral, BoolType}
import firrtl._
import firrtl.annotations.NoTargetAnnotation
import firrtl.backends.experimental.smt._
import firrtl.ir._
import firrtl.options.Dependency
import firrtl.passes.MemPortUtils.memPortField
import firrtl.passes.memlib.AnalysisUtils.Connects
import firrtl.passes.memlib.InferReadWritePass.checkComplement
import firrtl.passes.memlib.{AnalysisUtils, InferReadWritePass, VerilogMemDelays}
import firrtl.stage.Forms
import firrtl.transforms.RemoveWires

import scala.collection.mutable

/** Chooses which undefined memory behaviors should be instrumented. */
case class UndefinedMemoryBehaviorOptions(
  randomizeWriteWriteConflicts: Boolean = true,
  assertNoOutOfBoundsWrites:    Boolean = false,
  randomizeOutOfBoundsRead:     Boolean = true,
  randomizeDisabledReads:       Boolean = true,
  randomizeReadWriteConflicts:  Boolean = true)
    extends NoTargetAnnotation

/** Adds sources of randomness to model the various "undefined behaviors" of firrtl memory.
  * - Write/Write conflict: leads to arbitrary value written to write address
  * - Out-of-bounds write: assertion failure (disabled by default)
  * - Out-Of-bounds read: leads to arbitrary value being read
  * - Read w/ en=0: leads to arbitrary value being read
  * - Read/Write conflict: leads to arbitrary value being read
  */
object UndefinedMemoryBehaviorPass extends Transform with DependencyAPIMigration {
  override def prerequisites = Forms.LowForm
  override def optionalPrerequisiteOf = Seq(Dependency(VerilogMemDelays))
  override def invalidates(a: Transform) = a match {
    // this pass might destroy SSA form, as we add a wire for the data field of every read port
    case _: RemoveWires => true
    // TODO: should we add some optimization passes here? we could be generating some dead code.
    case _ => false
  }

  override protected def execute(state: CircuitState): CircuitState = {
    val opts = state.annotations.collect { case o: UndefinedMemoryBehaviorOptions => o }
    require(opts.size < 2, s"Multiple options: $opts")
    val opt = opts.headOption.getOrElse(UndefinedMemoryBehaviorOptions())

    val c = state.circuit.mapModule(onModule(_, opt))
    state.copy(circuit = c)
  }

  private def onModule(m: DefModule, opt: UndefinedMemoryBehaviorOptions): DefModule = m match {
    case mod: Module =>
      val mems = findMems(mod)
      if (mems.isEmpty) { mod }
      else {
        val namespace = Namespace(mod)
        val connects = AnalysisUtils.getConnects(mod)
        new InstrumentMems(opt, mems, connects, namespace).run(mod)
      }
    case other => other
  }

  /** finds all memory instantiations in a circuit */
  private def findMems(m: Module): List[DefMemory] = {
    val mems = mutable.ListBuffer[DefMemory]()
    m.foreachStmt(findMems(_, mems))
    mems.toList
  }
  private def findMems(s: Statement, mems: mutable.ListBuffer[DefMemory]): Unit = s match {
    case mem: DefMemory => mems.append(mem)
    case other => other.foreachStmt(findMems(_, mems))
  }
}

private class InstrumentMems(
  opt:       UndefinedMemoryBehaviorOptions,
  mems:      List[DefMemory],
  connects:  Connects,
  namespace: Namespace) {
  def run(m: Module): DefModule = {
    // ensure that all memories are the kind we can support
    mems.foreach(checkSupported(m.name, _))

    // transform circuit
    val body = m.body.mapStmt(transform)
    m.copy(body = Block(body +: newStmts.toList))
  }

  // used to replace memory signals like `m.r.data` in RHS expressions
  private val exprReplacements = mutable.HashMap[String, Expression]()
  // add new statements at the end of the circuit
  private val newStmts = mutable.ListBuffer[Statement]()
  // disconnect references so that they can be reassigned
  private val doDisconnect = mutable.HashSet[String]()

  // generates new expression replacements and immediately uses them
  private def transform(s: Statement): Statement = s.mapStmt(transform) match {
    case mem: DefMemory                                          => onMem(mem)
    case sx:  Connect if doDisconnect.contains(sx.loc.serialize) => EmptyStmt // Filter old mem connections
    case sx => sx.mapExpr(swapMemRefs)
  }
  private def swapMemRefs(e: Expression): Expression = e.mapExpr(swapMemRefs) match {
    case sf: RefLikeExpression => exprReplacements.getOrElse(sf.serialize, sf)
    case ex => ex
  }

  private def onMem(m: DefMemory): Statement = {
    // collect wire and random statement defines
    implicit val declarations: mutable.ListBuffer[Statement] = mutable.ListBuffer[Statement]()

    // cache for the expressions of memory inputs
    implicit val cache: mutable.HashMap[String, Expression] = mutable.HashMap[String, Expression]()

    // only for non power of 2 memories do we have to worry about reading or writing out of bounds
    val canBeOutOfBounds = !isPow2(m.depth)

    // only if we have at least two write ports, can there be conflicts
    val canHaveWriteWriteConflicts = m.writers.size > 1

    // only certain memory types exhibit undefined read/write conflicts
    val readWriteUndefined = (m.readLatency == m.writeLatency) && (m.readUnderWrite == ReadUnderWrite.Undefined)
    assert(
      m.readLatency == 0 || m.readLatency == m.writeLatency,
      "TODO: what happens if a sync read mem has asymmetrical latencies?"
    )

    // a write port is enabled iff mask & en
    val writeEn = m.writers.map { write =>
      val enRef = memPortField(m, write, "en")
      val maskRef = memPortField(m, write, "mask")

      val prods = getProductTerms(enRef) ++ getProductTerms(maskRef)
      val expr = Utils.and(readInput(m.info, enRef), readInput(m.info, maskRef))

      (expr, prods)
    }

    // implement the three undefined read behaviors
    m.readers.foreach { read =>
      // many memories have their read enable hard wired to true
      val canBeDisabled = !isTrue(readInput(m, read, "en"))
      val readEn = if (canBeDisabled) readInput(m, read, "en") else Utils.True()

      // collect signals that would lead to a randomization
      var doRand = List[Expression]()

      // randomize the read value when the address is out of bounds
      if (canBeOutOfBounds && opt.randomizeOutOfBoundsRead) {
        val addr = readInput(m, read, "addr")
        val cond = Utils.and(readEn, Utils.not(isInBounds(m.depth, addr)))
        val node = DefNode(m.info, namespace.newName(s"${m.name}_${read}_oob"), cond)
        declarations += node
        doRand = Reference(node) +: doRand
      }

      if (readWriteUndefined && opt.randomizeReadWriteConflicts) {
        val cond = readWriteConflict(m, read, writeEn)
        val node = DefNode(m.info, namespace.newName(s"${m.name}_${read}_rwc"), cond)
        declarations += node
        doRand = Reference(node) +: doRand
      }

      // randomize the read value when the read is disabled
      if (canBeDisabled && opt.randomizeDisabledReads) {
        val cond = Utils.not(readEn)
        val node = DefNode(m.info, namespace.newName(s"${m.name}_${read}_disabled"), cond)
        declarations += node
        doRand = Reference(node) +: doRand
      }

      // if there are no signals that would require a randomization, there is nothing to do
      if (doRand.isEmpty) {
        // nothing to do
      } else {
        val doRandName = s"${m.name}_${read}_do_rand"
        val doRandNode = if (doRand.size == 1) { doRand.head }
        else {
          val node = DefNode(m.info, namespace.newName(s"${m.name}_${read}_do_rand"), doRand.reduce(Utils.or))
          declarations += node
          Reference(node)
        }
        val doRandSignal = if (m.readLatency == 0) { doRandNode }
        else {
          val clock = readInput(m, read, "clk")
          val (signal, regDecls) = pipeline(m.info, clock, doRandName, doRandNode, m.readLatency)
          declarations ++= regDecls
          signal
        }

        // all old rhs references to m.r.data need to replace with m_r_data which might be random
        val dataRef = memPortField(m, read, "data")
        val dataWire = DefWire(m.info, namespace.newName(s"${m.name}_${read}_data"), m.dataType)
        declarations += dataWire
        exprReplacements(dataRef.serialize) = Reference(dataWire)

        // create a source of randomness and connect the new wire either to the actual data port or to the random value
        val randName = namespace.newName(s"${m.name}_${read}_rand_data")
        val random = DefRandom(m.info, randName, m.dataType, Some(readInput(m, read, "clk")), doRandSignal)
        declarations += random
        val data = Utils.mux(doRandSignal, Reference(random), dataRef)
        newStmts.append(Connect(m.info, Reference(dataWire), data))
      }
    }

    // write
    if (opt.randomizeWriteWriteConflicts) {
      writeWriteConflicts(m, writeEn)
    }

    // add an assertion that if the write is taking place, then the address must be in range
    if (canBeOutOfBounds && opt.assertNoOutOfBoundsWrites) {
      m.writers.zip(writeEn).foreach {
        case (write, (combinedEn, _)) =>
          val addr = readInput(m, write, "addr")
          val cond = Utils.implies(combinedEn, isInBounds(m.depth, addr))
          val clk = readInput(m, write, "clk")
          val a = Verification(Formal.Assert, m.info, clk, cond, Utils.True(), StringLit("out of bounds read"))
          newStmts.append(a)
      }
    }

    Block(m +: declarations.toList)
  }

  private def pipeline(
    info:    Info,
    clk:     Expression,
    prefix:  String,
    e:       Expression,
    latency: Int
  ): (Expression, Seq[Statement]) = {
    require(latency > 0)
    val regs = (1 to latency).map { i =>
      val name = namespace.newName(prefix + s"_r$i")
      DefRegister(info, name, e.tpe, clk, Utils.False(), Reference(name, e.tpe, RegKind, UnknownFlow))
    }
    val expr = regs.foldLeft(e) {
      case (prev, reg) =>
        newStmts.append(Connect(info, Reference(reg), prev))
        Reference(reg)
    }
    (expr, regs)
  }

  private def readWriteConflict(
    m:       DefMemory,
    read:    String,
    writeEn: Seq[(Expression, ProdTerms)]
  )(
    implicit cache: mutable.HashMap[String, Expression],
    declarations:   mutable.ListBuffer[Statement]
  ): Expression = {
    if (m.writers.isEmpty) return Utils.False()

    val readEn = readInput(m, read, "en")
    val readProd = getProductTerms(readEn)

    // create all conflict signals
    val conflicts = m.writers.zip(writeEn).map {
      case (write, (writeEn, writeProd)) =>
        if (isMutuallyExclusive(readProd, writeProd)) {
          Utils.False()
        } else {
          val name = namespace.newName(s"${m.name}_${read}_${write}_rwc")
          val bothEn = Utils.and(readEn, writeEn)
          val sameAddr = Utils.eq(readInput(m, read, "addr"), readInput(m, write, "addr"))
          // we need a wire because this condition might be used in a random statement
          val wire = DefWire(m.info, name, BoolType)
          declarations += wire
          newStmts.append(Connect(m.info, Reference(wire), Utils.and(bothEn, sameAddr)))
          Reference(wire)
        }
    }

    conflicts.reduce(Utils.or)
  }

  private type ProdTerms = Seq[Expression]
  private def writeWriteConflicts(
    m:       DefMemory,
    writeEn: Seq[(Expression, ProdTerms)]
  )(
    implicit cache: mutable.HashMap[String, Expression],
    declarations:   mutable.ListBuffer[Statement]
  ): Unit = {
    if (m.writers.size < 2) return

    // we first create all conflict signals:
    val conflict =
      m.writers
        .zip(writeEn)
        .zipWithIndex
        .flatMap {
          case ((w1, (en1, en1Prod)), i1) =>
            m.writers.zip(writeEn).drop(i1 + 1).map {
              case (w2, (en2, en2Prod)) =>
                if (isMutuallyExclusive(en1Prod, en2Prod)) {
                  (w1, w2) -> Utils.False()
                } else {
                  val name = namespace.newName(s"${m.name}_${w1}_${w2}_wwc")
                  val bothEn = Utils.and(en1, en2)
                  val sameAddr = Utils.eq(readInput(m, w1, "addr"), readInput(m, w2, "addr"))
                  // we need a wire because this condition might be used in a random statement
                  val wire = DefWire(m.info, name, BoolType)
                  declarations += wire
                  newStmts.append(Connect(m.info, Reference(wire), Utils.and(bothEn, sameAddr)))
                  (w1, w2) -> Reference(wire)
                }
            }
        }
        .toMap

    // now we calculate the new enable and data signals
    m.writers.zip(writeEn).zipWithIndex.foreach {
      case ((w1, (en1, _)), i1) =>
        val prev = m.writers.take(i1)
        val next = m.writers.drop(i1 + 1)

        // the write is enabled if the original enable is true and there are no prior conflicts
        val en = if (prev.isEmpty) {
          en1
        } else {
          val prevConflicts = prev.map(o => conflict(o, w1)).reduce(Utils.or)
          Utils.and(en1, Utils.not(prevConflicts))
        }

        // we write random data if there is a conflict with any of the next ports
        if (next.isEmpty) {
          // nothing to do, leave data as is
        } else {
          val nextConflicts = next.map(n => conflict(w1, n)).reduce(Utils.or)
          // if the conflict expression is more complex, create a node for the signal
          val hasConflict = nextConflicts match {
            case _: DoPrim | _: Mux =>
              val node = DefNode(m.info, namespace.newName(s"${m.name}_${w1}_wwc_active"), nextConflicts)
              declarations += node
              Reference(node)
            case _ => nextConflicts
          }

          // create the source of randomness
          val name = namespace.newName(s"${m.name}_${w1}_wwc_data")
          val random = DefRandom(m.info, name, m.dataType, Some(readInput(m, w1, "clk")), hasConflict)
          declarations.append(random)

          // generate new data input
          val data = Utils.mux(hasConflict, Reference(random), readInput(m, w1, "data"))
          newStmts.append(Connect(m.info, memPortField(m, w1, "data"), data))
          doDisconnect.add(memPortField(m, w1, "data").serialize)
        }

        // connect data enable signals
        val maskIsOne = isTrue(readInput(m, w1, "mask"))
        if (!maskIsOne) {
          newStmts.append(Connect(m.info, memPortField(m, w1, "mask"), Utils.True()))
          doDisconnect.add(memPortField(m, w1, "mask").serialize)
        }
        newStmts.append(Connect(m.info, memPortField(m, w1, "en"), en))
        doDisconnect.add(memPortField(m, w1, "en").serialize)
    }
  }

  /** check whether two signals can be proven to be mutually exclusive */
  private def isMutuallyExclusive(prodA: ProdTerms, prodB: ProdTerms): Boolean = {
    // this uses the same approach as the InferReadWrite pass
    val proofOfMutualExclusion = prodA.find(a => prodB.exists(b => checkComplement(a, b)))
    proofOfMutualExclusion.nonEmpty
  }

  /** memory inputs my not be read, only assigned to, thus we might need to add a wire to make them accessible */
  private def readInput(
    info:   Info,
    signal: RefLikeExpression
  )(
    implicit cache: mutable.HashMap[String, Expression],
    declarations:   mutable.ListBuffer[Statement]
  ): Expression =
    cache.getOrElseUpdate(
      signal.serialize, {
        // if it is a literal, we just return it
        val value = connects(signal.serialize)
        if (isLiteral(value)) {
          value
        } else {
          // otherwise we make a wire that refelect the value
          val wire = DefWire(info, copyName(signal), signal.tpe)
          declarations += wire

          // connect the old expression to the new wire
          val con = Connect(info, Reference(wire), value)
          newStmts.append(con)

          // use a reference to this new wire
          Reference(wire)
        }
      }
    )
  private def readInput(
    m:     DefMemory,
    port:  String,
    field: String
  )(
    implicit cache: mutable.HashMap[String, Expression],
    declarations:   mutable.ListBuffer[Statement]
  ): Expression =
    readInput(m.info, memPortField(m, port, field))

  private def copyName(ref: RefLikeExpression): String =
    namespace.newName(ref.serialize.replace('.', '_'))

  private def isInBounds(depth: BigInt, addr: Expression): Expression = {
    val width = getWidth(addr)
    // depth > addr (e.g. if the depth is 3, then the address must be in {0, 1, 2})
    DoPrim(PrimOps.Gt, List(UIntLiteral(depth, width), addr), List(), BoolType)
  }

  private def isPow2(v: BigInt): Boolean = ((v - 1) & v) == 0

  private def checkSupported(modName: String, m: DefMemory): Unit = {
    assert(m.readwriters.isEmpty, s"[$modName] Combined read/write ports are currently not supported!")
    if (m.writeLatency != 1) {
      throw new UnsupportedFeatureException(s"[$modName] memories with write latency > 1 (${m.name})")
    }
    if (m.readLatency > 1) {
      throw new UnsupportedFeatureException(s"[$modName] memories with read latency > 1 (${m.name})")
    }
  }

  private def getProductTerms(e: Expression): ProdTerms =
    InferReadWritePass.getProductTerms(connects)(e)

  /** tries to expand the expression based on the connects we collected */
  private def expandExpr(e: Expression, fuel: Int): Expression = {
    e match {
      case m @ Mux(cond, tval, fval, _) =>
        m.copy(cond = expandExpr(cond, fuel), tval = expandExpr(tval, fuel), fval = expandExpr(fval, fuel))
      case p @ DoPrim(_, args, _, _) =>
        p.copy(args = args.map(expandExpr(_, fuel)))
      case r: RefLikeExpression =>
        if (fuel > 0) {
          connects.get(r.serialize) match {
            case None       => r
            case Some(expr) => expandExpr(expr, fuel - 1)
          }
        } else {
          r
        }
      case other => other
    }
  }

  private def isTrue(e: Expression): Boolean = simplifyExpr(expandExpr(e, fuel = 2)) == Utils.True()

  private def simplifyExpr(e: Expression): Expression = {
    e // TODO: better simplification could improve the resulting circuit size
  }
}
