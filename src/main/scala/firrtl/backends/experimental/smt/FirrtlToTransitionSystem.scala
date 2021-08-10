// SPDX-License-Identifier: Apache-2.0
// Author: Kevin Laeufer <laeufer@cs.berkeley.edu>

package firrtl.backends.experimental.smt

import firrtl.annotations.{MemoryInitAnnotation, NoTargetAnnotation, PresetRegAnnotation}
import firrtl.bitWidth
import FirrtlExpressionSemantics.getWidth
import firrtl.backends.experimental.smt.random._
import firrtl.graph.MutableDiGraph
import firrtl.options.Dependency
import firrtl.passes.MemPortUtils.memPortField
import firrtl.passes.PassException
import firrtl.passes.memlib.VerilogMemDelays
import firrtl.stage.Forms
import firrtl.stage.TransformManager.TransformDependency
import firrtl.transforms.{DeadCodeElimination, EnsureNamedStatements, PropagatePresetAnnotations}
import firrtl.{
  ir,
  CircuitState,
  DependencyAPIMigration,
  MemoryArrayInit,
  MemoryInitValue,
  MemoryScalarInit,
  Namespace,
  Transform,
  Utils
}
import logger.LazyLogging

import scala.collection.mutable

// Contains code to convert a flat firrtl module into a functional transition system which
// can then be exported as SMTLib or Btor2 file.

private case class State(sym: SMTSymbol, init: Option[SMTExpr], next: Option[SMTExpr])
private case class Signal(name: String, e: BVExpr) { def toSymbol: BVSymbol = BVSymbol(name, e.width) }
private case class TransitionSystem(
  name:     String,
  inputs:   Array[BVSymbol],
  states:   Array[State],
  signals:  Array[Signal],
  outputs:  Set[String],
  assumes:  Set[String],
  asserts:  Set[String],
  fair:     Set[String],
  ufs:      List[BVFunctionSymbol] = List(),
  comments: Map[String, String] = Map(),
  header:   Array[String] = Array()) {
  def serialize: String = {
    (Iterator(name) ++
      ufs.map(u => u.toString) ++
      inputs.map(i => s"input ${i.name} : ${SMTExpr.serializeType(i)}") ++
      signals.map(s => s"${s.name} : ${SMTExpr.serializeType(s.e)} = ${s.e}") ++
      states.map(s => s"state ${s.sym} = [init] ${s.init} [next] ${s.next}")).mkString("\n")
  }
}

private case class TransitionSystemAnnotation(sys: TransitionSystem) extends NoTargetAnnotation

object FirrtlToTransitionSystem extends Transform with DependencyAPIMigration {
  override def prerequisites: Seq[Dependency[Transform]] = Forms.LowForm ++
    Seq(
      Dependency(VerilogMemDelays),
      Dependency(EnsureNamedStatements), // this is required to give assert/assume statements good names
      Dependency[PropagatePresetAnnotations]
    )
  override def invalidates(a: Transform): Boolean = false
  // since this pass only runs on the main module, inlining needs to happen before
  override def optionalPrerequisites: Seq[TransformDependency] = Seq(Dependency[firrtl.passes.InlineInstances])

  override protected def execute(state: CircuitState): CircuitState = {
    val circuit = state.circuit
    val presetRegs = state.annotations.collect {
      case PresetRegAnnotation(target) if target.module == circuit.main => target.ref
    }.toSet

    // collect all non-random memory initialization
    val memInit = state.annotations.collect { case a: MemoryInitAnnotation if !a.isRandomInit => a }
      .filter(_.target.module == circuit.main)
      .map(a => a.target.ref -> a.initValue)
      .toMap

    // module look up table
    val modules = circuit.modules.map(m => m.name -> m).toMap

    // collect uninterpreted module annotations
    val uninterpreted = state.annotations.collect {
      case a: UninterpretedModuleAnnotation =>
        UninterpretedModuleAnnotation.checkModule(modules(a.target.module), a)
        a.target.module -> a
    }.toMap

    // convert the main module
    val main = modules(circuit.main)
    val sys = main match {
      case x: ir.ExtModule =>
        throw new ExtModuleException(
          "External modules are not supported by the SMT backend. Use yosys if you need to convert Verilog."
        )
      case m: ir.Module =>
        new ModuleToTransitionSystem().run(m, presetRegs = presetRegs, memInit = memInit, uninterpreted = uninterpreted)
    }

    val sortedSys = TopologicalSort.run(sys)
    val anno = TransitionSystemAnnotation(sortedSys)
    state.copy(circuit = circuit, annotations = state.annotations :+ anno)
  }
}

private object UnsupportedException {
  val HowToRunStuttering: String =
    """
      |You can run the StutteringClockTransform which
      |replaces all clock inputs with a clock enable signal.
      |This is required not only for multi-clock designs, but also to
      |accurately model asynchronous reset which could happen even if there
      |isn't a clock edge.
      | If you are using the firrtl CLI, please add:
      |   -fct firrtl.backends.experimental.smt.StutteringClockTransform
      | If you are calling into firrtl programmatically you can use:
      |   RunFirrtlTransformAnnotation(Dependency[StutteringClockTransform])
      | To designate a clock to be the global_clock (i.e. the simulation tick), use:
      |   GlobalClockAnnotation(CircuitTarget(...).module(...).ref("your_clock")))
      |""".stripMargin
}

private class ExtModuleException(s: String) extends PassException(s)
private class AsyncResetException(s: String) extends PassException(s + UnsupportedException.HowToRunStuttering)
private class MultiClockException(s: String) extends PassException(s + UnsupportedException.HowToRunStuttering)
private class MissingFeatureException(s: String)
    extends PassException("Unfortunately the SMT backend does not yet support: " + s)

private class ModuleToTransitionSystem extends LazyLogging {

  def run(
    m:             ir.Module,
    presetRegs:    Set[String] = Set(),
    memInit:       Map[String, MemoryInitValue] = Map(),
    uninterpreted: Map[String, UninterpretedModuleAnnotation] = Map()
  ): TransitionSystem = {
    // first pass over the module to convert expressions; discover state and I/O
    val scan = new ModuleScanner(uninterpreted)
    m.foreachPort(scan.onPort)
    m.foreachStmt(scan.onStatement)

    // multi-clock support requires the StutteringClock transform to be run
    if (scan.clocks.size > 1) {
      throw new MultiClockException(s"The module ${m.name} has more than one clock: ${scan.clocks.mkString(", ")}")
    }

    // turn wires and nodes into signals
    val outputs = scan.outputs.toSet
    val constraints = scan.assumes.toSet
    val bad = scan.asserts.toSet
    val isSignal = (scan.wires ++ scan.nodes ++ scan.memSignals).toSet ++ outputs ++ constraints ++ bad
    val signals = scan.connects.filter { case (name, _) => isSignal.contains(name) }.map {
      case (name, expr) => Signal(name, expr)
    }

    // turn registers and memories into states
    val registers = scan.registers.map(r => r._1 -> r).toMap
    val regStates = scan.connects.filter(s => registers.contains(s._1)).map {
      case (name, nextExpr) =>
        val (_, width, resetExpr, initExpr) = registers(name)
        onRegister(name, width, resetExpr, initExpr, nextExpr, presetRegs)
    }
    // turn memories into state
    val memoryStatesAndOutputs = scan.memories.map(m => onMemory(m, scan.connects, memInit.get(m.name)))
    // replace pseudo assigns for memory outputs
    val memOutputs = memoryStatesAndOutputs.flatMap(_._2).toMap
    val signalsWithMem = signals.map { s =>
      if (memOutputs.contains(s.name)) {
        s.copy(e = memOutputs(s.name))
      } else { s }
    }
    // filter out any left-over self assignments (this happens when we have a registered read port)
      .filter(s =>
        s match {
          case Signal(n0, BVSymbol(n1, _)) if n0 == n1 => false
          case _                                       => true
        }
      )
    val states = regStates.toArray ++ memoryStatesAndOutputs.map(_._1)

    // generate comments from infos
    val comments = mutable.HashMap[String, String]()
    scan.infos.foreach {
      case (name, info) =>
        serializeInfo(info).foreach { infoString =>
          if (comments.contains(name)) { comments(name) += InfoSeparator + infoString }
          else { comments(name) = InfoPrefix + infoString }
        }
    }

    // inputs are original module inputs and any DefRandom signal
    val inputs = scan.inputs

    // module info to the comment header
    val header = serializeInfo(m.info).map(InfoPrefix + _).toArray

    val fair = Set[String]() // as of firrtl 1.4 we do not support fairness constraints

    // collect unique functions
    val ufs = scan.functionCalls.groupBy(_.name).map(_._2.head).toList

    TransitionSystem(
      m.name,
      inputs.toArray,
      states,
      signalsWithMem.toArray,
      outputs,
      constraints,
      bad,
      fair,
      ufs,
      comments.toMap,
      header
    )
  }

  private def onRegister(
    name:       String,
    width:      Int,
    resetExpr:  BVExpr,
    initExpr:   BVExpr,
    nextExpr:   BVExpr,
    presetRegs: Set[String]
  ): State = {
    assert(initExpr.width == width)
    assert(nextExpr.width == width)
    assert(resetExpr.width == 1)
    val sym = BVSymbol(name, width)
    val hasReset = initExpr != sym
    val isPreset = presetRegs.contains(name)
    assert(!isPreset || hasReset, s"Expected preset register $name to have a reset value, not just $initExpr!")
    if (hasReset) {
      val init = if (isPreset) Some(initExpr) else None
      val next = if (isPreset) nextExpr else BVIte(resetExpr, initExpr, nextExpr)
      State(sym, next = Some(next), init = init)
    } else {
      State(sym, next = Some(nextExpr), init = None)
    }
  }

  type Connects = Iterable[(String, BVExpr)]
  private def onMemory(m: ir.DefMemory, connects: Connects, initValue: Option[MemoryInitValue]): (State, Connects) = {
    checkMem(m)

    // map of inputs to the memory
    val inputs = connects.filter(_._1.startsWith(m.name)).toMap

    // derive the type of the memory from the dataType and depth
    val dataWidth = bitWidth(m.dataType).toInt
    val indexWidth = Utils.getUIntWidth(m.depth - 1).max(1)
    val memSymbol = ArraySymbol(m.name, indexWidth, dataWidth)

    // there could be a constant init
    val init = initValue.map(getInit(m, indexWidth, dataWidth, _))
    init.foreach(e => assert(e.dataWidth == memSymbol.dataWidth && e.indexWidth == memSymbol.indexWidth))

    // derive next state expression
    val next = if (m.writers.isEmpty) {
      memSymbol
    } else {
      m.writers.foldLeft[ArrayExpr](memSymbol) {
        case (prev, write) =>
          // update
          val addr = BVSymbol(memPortField(m, write, "addr").serialize, indexWidth)
          val data = BVSymbol(memPortField(m, write, "data").serialize, dataWidth)
          val update = ArrayStore(prev, index = addr, data = data)

          // update guard
          val en = BVSymbol(memPortField(m, write, "en").serialize, 1)
          val mask = BVSymbol(memPortField(m, write, "mask").serialize, 1)
          val alwaysEnabled = Seq(en, mask).forall(s => inputs(s.name) == True)
          if (alwaysEnabled) { update }
          else {
            ArrayIte(and(en, mask), update, prev)
          }
      }
    }

    val state = State(memSymbol, init, Some(next))

    // derive read expressions
    val readSignals = m.readers.map { read =>
      val addr = BVSymbol(memPortField(m, read, "addr").serialize, indexWidth)
      memPortField(m, read, "data").serialize -> ArrayRead(memSymbol, addr)
    }

    (state, readSignals)
  }

  private def getInit(m: ir.DefMemory, indexWidth: Int, dataWidth: Int, initValue: MemoryInitValue): ArrayExpr =
    initValue match {
      case MemoryScalarInit(value) => ArrayConstant(BVLiteral(value, dataWidth), indexWidth)
      case MemoryArrayInit(values) =>
        assert(
          values.length == m.depth,
          s"Memory ${m.name} of depth ${m.depth} cannot be initialized with an array of length ${values.length}!"
        )
        // in order to get a more compact encoding try to find the most common values
        val histogram = mutable.LinkedHashMap[BigInt, Int]()
        values.foreach(v => histogram(v) = 1 + histogram.getOrElse(v, 0))
        val baseValue = histogram.maxBy(_._2)._1
        val base = ArrayConstant(BVLiteral(baseValue, dataWidth), indexWidth)
        values.zipWithIndex
          .filterNot(_._1 == baseValue)
          .foldLeft[ArrayExpr](base) {
            case (array, (value, index)) =>
              ArrayStore(array, BVLiteral(index, indexWidth), BVLiteral(value, dataWidth))
          }
      case other => throw new RuntimeException(s"Unsupported memory init option: $other")
    }

  // TODO: add to BV expression library
  private def and(a: BVExpr, b: BVExpr): BVExpr = (a, b) match {
    case (True, True) => True
    case (True, x)    => x
    case (x, True)    => x
    case _            => BVOp(Op.And, a, b)
  }

  private val True = BVLiteral(1, 1)
  private def checkMem(m: ir.DefMemory): Unit = {
    assert(m.readLatency == 0, "Expected read latency to be 0. Did you run VerilogMemDelays?")
    assert(m.writeLatency == 1, "Expected read latency to be 1. Did you run VerilogMemDelays?")
    assert(
      m.dataType.isInstanceOf[ir.GroundType],
      s"Memory $m is of type ${m.dataType} which is not a ground type!"
    )
    assert(m.readwriters.isEmpty, "Combined read/write ports are not supported! Please split them up.")
  }

  private val InfoSeparator = ", "
  private val InfoPrefix = "@ "
  private def serializeInfo(info: ir.Info): Option[String] = info match {
    case ir.NoInfo => None
    case f: ir.FileInfo => Some(f.escaped)
    case m: ir.MultiInfo =>
      val infos = m.flatten
      if (infos.isEmpty) { None }
      else { Some(infos.map(_.escaped).mkString(InfoSeparator)) }
  }
}

// performas a first pass over the module collecting all connections, wires, registers, input and outputs
private class ModuleScanner(
  uninterpreted: Map[String, UninterpretedModuleAnnotation])
    extends LazyLogging {
  import FirrtlExpressionSemantics.getWidth

  private[firrtl] val inputs = mutable.ArrayBuffer[BVSymbol]()
  private[firrtl] val outputs = mutable.ArrayBuffer[String]()
  private[firrtl] val clocks = mutable.LinkedHashSet[String]()
  private[firrtl] val wires = mutable.ArrayBuffer[String]()
  private[firrtl] val nodes = mutable.ArrayBuffer[String]()
  private[firrtl] val memSignals = mutable.ArrayBuffer[String]()
  private[firrtl] val registers = mutable.ArrayBuffer[(String, Int, BVExpr, BVExpr)]()
  private[firrtl] val memories = mutable.ArrayBuffer[ir.DefMemory]()
  // DefNode, Connect, IsInvalid and VerificationStatement connections
  private[firrtl] val connects = mutable.ArrayBuffer[(String, BVExpr)]()
  private[firrtl] val asserts = mutable.ArrayBuffer[String]()
  private[firrtl] val assumes = mutable.ArrayBuffer[String]()
  // maps identifiers to their info
  private[firrtl] val infos = mutable.ArrayBuffer[(String, ir.Info)]()
  // Keeps track of (so far) unused memory (data) and uninterpreted module outputs.
  // This is used in order to delay declaring them for as long as possible.
  private val unusedOutputs = mutable.LinkedHashMap[String, BVExpr]()
  // ensure unique names for assert/assume signals
  private[firrtl] val namespace = Namespace()
  // keep track of all uninterpreted functions called
  private[firrtl] val functionCalls = mutable.ArrayBuffer[BVFunctionSymbol]()

  private[firrtl] def onPort(p: ir.Port): Unit = {
    if (isAsyncReset(p.tpe)) {
      throw new AsyncResetException(s"Found AsyncReset ${p.name}.")
    }
    namespace.newName(p.name)
    infos.append(p.name -> p.info)
    p.direction match {
      case ir.Input =>
        if (isClock(p.tpe)) {
          clocks.add(p.name)
        } else {
          inputs.append(BVSymbol(p.name, bitWidth(p.tpe).toInt))
        }
      case ir.Output =>
        if (!isClock(p.tpe)) { // we ignore clock outputs
          outputs.append(p.name)
        }
    }
  }

  private[firrtl] def onStatement(s: ir.Statement): Unit = s match {
    case DefRandom(info, name, tpe, _, _) =>
      namespace.newName(name)
      assert(!isClock(tpe), "rand should never be a clock!")
      // we model random sources as inputs and ignore the enable signal
      infos.append(name -> info)
      inputs.append(BVSymbol(name, bitWidth(tpe).toInt))
    case ir.DefWire(info, name, tpe) =>
      namespace.newName(name)
      if (!isClock(tpe) && !isAsyncReset(tpe)) {
        infos.append(name -> info)
        wires.append(name)
      }
    case ir.DefNode(info, name, expr) =>
      namespace.newName(name)
      if (!isClock(expr.tpe) && !isAsyncReset(expr.tpe)) {
        insertDummyAssignsForUnusedOutputs(expr)
        infos.append(name -> info)
        val e = onExpression(expr)
        nodes.append(name)
        connects.append((name, e))
      }
    case ir.DefRegister(info, name, tpe, _, reset, init) =>
      namespace.newName(name)
      insertDummyAssignsForUnusedOutputs(reset)
      insertDummyAssignsForUnusedOutputs(init)
      infos.append(name -> info)
      val width = bitWidth(tpe).toInt
      val resetExpr = onExpression(reset, 1)
      val initExpr = onExpression(init, width)
      registers.append((name, width, resetExpr, initExpr))
    case m: ir.DefMemory =>
      namespace.newName(m.name)
      infos.append(m.name -> m.info)
      val outputs = getMemOutputs(m)
      (getMemInputs(m) ++ outputs).foreach(memSignals.append(_))
      val dataWidth = bitWidth(m.dataType).toInt
      outputs.foreach(name => unusedOutputs(name) = BVSymbol(name, dataWidth))
      memories.append(m)
    case ir.Connect(info, loc, expr) =>
      if (!isGroundType(loc.tpe)) error("All connects should have been lowered to ground type!")
      if (!isClock(loc.tpe)) { // we ignore clock connections
        val name = loc.serialize
        insertDummyAssignsForUnusedOutputs(expr)
        infos.append(name -> info)
        connects.append((name, onExpression(expr, bitWidth(loc.tpe).toInt, allowNarrow = true)))
      }
    case i @ ir.IsInvalid(info, loc) =>
      if (!isGroundType(loc.tpe)) error("All connects should have been lowered to ground type!")
      throw new UnsupportedFeatureException(s"IsInvalid statements are not supported: ${i.serialize}")
    case ir.DefInstance(info, name, module, tpe) => onInstance(info, name, module, tpe)
    case s @ ir.Verification(op, info, _, pred, en, msg) =>
      if (op == ir.Formal.Cover) {
        logger.warn(s"WARN: Cover statement was ignored: ${s.serialize}")
      } else {
        insertDummyAssignsForUnusedOutputs(pred)
        insertDummyAssignsForUnusedOutputs(en)
        val name = s.name
        val predicate = onExpression(pred)
        val enabled = onExpression(en)
        val e = BVImplies(enabled, predicate)
        infos.append(name -> info)
        connects.append(name -> e)
        if (op == ir.Formal.Assert) {
          asserts.append(name)
        } else {
          assumes.append(name)
        }
      }
    case s: ir.Conditionally =>
      error(s"When conditions are not supported. Please run ExpandWhens: ${s.serialize}")
    case s: ir.PartialConnect =>
      error(s"PartialConnects are not supported. Please run ExpandConnects: ${s.serialize}")
    case s: ir.Attach =>
      error(s"Analog wires are not supported in the SMT backend: ${s.serialize}")
    case s: ir.Stop =>
      // we could wire up the stop condition as output for debug reasons
      logger.warn(s"WARN: Stop statements are currently not supported. Ignoring: ${s.serialize}")
    case s: ir.Print =>
      logger.warn(s"WARN: Print statements are not supported. Ignoring: ${s.serialize}")
    case other => other.foreachStmt(onStatement)
  }

  private def onInstance(info: ir.Info, name: String, module: String, tpe: ir.Type): Unit = {
    namespace.newName(name)
    if (!tpe.isInstanceOf[ir.BundleType]) error(s"Instance $name of $module has an invalid type: ${tpe.serialize}")
    if (uninterpreted.contains(module)) {
      onUninterpretedInstance(info: ir.Info, name: String, module: String, tpe: ir.Type)
    } else {
      // We treat all instances that aren't annotated as uninterpreted as blackboxes
      // this means that their outputs could be any value, no matter what their inputs are.
      logger.warn(
        s"WARN: treating instance $name of $module as blackbox. " +
          "Please flatten your hierarchy if you want to include submodules in the formal model."
      )
      val ports = tpe.asInstanceOf[ir.BundleType].fields
      // skip async reset ports
      ports.filterNot(p => isAsyncReset(p.tpe)).foreach { p =>
        if (!p.tpe.isInstanceOf[ir.GroundType]) error(s"Instance $name of $module has an invalid port type: $p")
        val isOutput = p.flip == ir.Default
        val pName = name + "." + p.name
        infos.append(pName -> info)
        // outputs of the submodule become inputs to our module
        if (isOutput) {
          if (isClock(p.tpe)) {
            clocks.add(pName)
          } else {
            inputs.append(BVSymbol(pName, bitWidth(p.tpe).toInt))
          }
        } else {
          if (!isClock(p.tpe)) { // we ignore clock outputs
            outputs.append(pName)
          }
        }
      }
    }
  }

  private def onUninterpretedInstance(info: ir.Info, instanceName: String, module: String, tpe: ir.Type): Unit = {
    val anno = uninterpreted(module)

    // sanity checks for ports were done already using the UninterpretedModule.checkModule function
    val ports = tpe.asInstanceOf[ir.BundleType].fields

    val outputs = ports.filter(_.flip == ir.Default).map(p => BVSymbol(p.name, bitWidth(p.tpe).toInt))
    val inputs = ports.filterNot(_.flip == ir.Default).map(p => BVSymbol(p.name, bitWidth(p.tpe).toInt))

    assert(anno.stateBits == 0, "TODO: implement support for uninterpreted stateful modules!")

    // for state-less (i.e. combinatorial) circuits, the outputs only depend on the inputs
    val args = inputs.map(i => BVSymbol(instanceName + "." + i.name, i.width)).toList
    outputs.foreach { out =>
      val functionName = anno.prefix + "." + out.name
      val call = BVFunctionCall(functionName, args, out.width)
      val wireName = instanceName + "." + out.name
      // remember which functions were called
      functionCalls.append(call.toSymbol)
      // insert the output definition right before its first use in an attempt to get SSA
      unusedOutputs(wireName) = call
      // treat these outputs as wires
      wires.append(wireName)
    }

    // we also treat the arguments as wires
    wires ++= args.map(_.name)
  }

  private val readInputFields = List("en", "addr")
  private val writeInputFields = List("en", "mask", "addr", "data")
  private def getMemInputs(m: ir.DefMemory): Iterable[String] = {
    assert(m.readwriters.isEmpty, "Combined read/write ports are not supported!")
    val p = m.name + "."
    m.writers.flatMap(w => writeInputFields.map(p + w + "." + _)) ++
      m.readers.flatMap(r => readInputFields.map(p + r + "." + _))
  }
  private def getMemOutputs(m: ir.DefMemory): Iterable[String] = {
    assert(m.readwriters.isEmpty, "Combined read/write ports are not supported!")
    val p = m.name + "."
    m.readers.map(r => p + r + ".data")
  }
  // inserts a dummy assign right before a memory/uninterpreted module output is used for the first time
  // example:
  // m.r.data <= m.r.data ; this is the dummy assign
  // test <= m.r.data     ; this is the first use of m.r.data
  private def insertDummyAssignsForUnusedOutputs(next: ir.Expression): Unit = if (unusedOutputs.nonEmpty) {
    val uses = mutable.ArrayBuffer[String]()
    findUnusedOutputUse(next)(uses)
    if (uses.nonEmpty) {
      val useSet = uses.toSet
      unusedOutputs.foreach {
        case (name, value) =>
          if (useSet.contains(name)) connects.append(name -> value)
      }
      useSet.foreach(name => unusedOutputs.remove(name))
    }
  }
  private def findUnusedOutputUse(e: ir.Expression)(implicit uses: mutable.ArrayBuffer[String]): Unit = e match {
    case s: ir.SubField =>
      val name = s.serialize
      if (unusedOutputs.contains(name)) uses.append(name)
    case other => other.foreachExpr(findUnusedOutputUse)
  }

  private case class Context() extends TranslationContext {}

  private def onExpression(e: ir.Expression, width: Int, allowNarrow: Boolean = false): BVExpr = {
    implicit val ctx: TranslationContext = Context()
    FirrtlExpressionSemantics.toSMT(e, width, allowNarrow)
  }
  private def onExpression(e: ir.Expression): BVExpr = {
    implicit val ctx: TranslationContext = Context()
    FirrtlExpressionSemantics.toSMT(e)
  }

  private def error(msg:        String):  Unit = throw new RuntimeException(msg)
  private def isGroundType(tpe: ir.Type): Boolean = tpe.isInstanceOf[ir.GroundType]
  private def isClock(tpe:      ir.Type): Boolean = tpe == ir.ClockType
  private def isAsyncReset(tpe: ir.Type): Boolean = tpe == ir.AsyncResetType
}

private object TopologicalSort {

  /** Ensures that all signals in the resulting system are topologically sorted.
    * This is necessary because [[firrtl.transforms.RemoveWires]] does
    * not sort assignments to outputs, submodule inputs nor memory ports.
    */
  def run(sys: TransitionSystem): TransitionSystem = {
    val inputsAndStates = sys.inputs.map(_.name) ++ sys.states.map(_.sym.name)
    val signalOrder = sort(sys.signals.map(s => s.name -> s.e), inputsAndStates)
    // TODO: maybe sort init expressions of states (this should not be needed most of the time)
    signalOrder match {
      case None => sys
      case Some(order) =>
        val signalMap = sys.signals.map(s => s.name -> s).toMap
        // we flatMap over `get` in order to ignore inputs/states in the order
        sys.copy(signals = order.flatMap(signalMap.get).toArray)
    }
  }

  private def sort(signals: Iterable[(String, SMTExpr)], globalSignals: Iterable[String]): Option[Iterable[String]] = {
    val known = new mutable.HashSet[String]() ++ globalSignals
    var needsReordering = false
    val digraph = new MutableDiGraph[String]
    signals.foreach {
      case (name, expr) =>
        digraph.addVertex(name)
        val uniqueDependencies = mutable.LinkedHashSet[String]() ++ findDependencies(expr)
        uniqueDependencies.foreach { d =>
          if (!known.contains(d)) { needsReordering = true }
          digraph.addPairWithEdge(name, d)
        }
        known.add(name)
    }
    if (needsReordering) {
      Some(digraph.linearize.reverse)
    } else { None }
  }

  private def findDependencies(expr: SMTExpr): List[String] = expr match {
    case BVSymbol(name, _)       => List(name)
    case ArraySymbol(name, _, _) => List(name)
    case other                   => other.children.flatMap(findDependencies)
  }
}
