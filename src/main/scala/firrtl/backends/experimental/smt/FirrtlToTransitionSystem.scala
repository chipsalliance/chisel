// SPDX-License-Identifier: Apache-2.0
// Author: Kevin Laeufer <laeufer@cs.berkeley.edu>

package firrtl.backends.experimental.smt

import firrtl.annotations.{MemoryInitAnnotation, NoTargetAnnotation, PresetRegAnnotation}
import FirrtlExpressionSemantics.getWidth
import firrtl.graph.MutableDiGraph
import firrtl.options.Dependency
import firrtl.passes.PassException
import firrtl.stage.Forms
import firrtl.stage.TransformManager.TransformDependency
import firrtl.transforms.PropagatePresetAnnotations
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
  comments: Map[String, String] = Map(),
  header:   Array[String] = Array()) {
  def serialize: String = {
    (Iterator(name) ++
      inputs.map(i => s"input ${i.name} : ${SMTExpr.serializeType(i)}") ++
      signals.map(s => s"${s.name} : ${SMTExpr.serializeType(s.e)} = ${s.e}") ++
      states.map(s => s"state ${s.sym} = [init] ${s.init} [next] ${s.next}")).mkString("\n")
  }
}

private case class TransitionSystemAnnotation(sys: TransitionSystem) extends NoTargetAnnotation

object FirrtlToTransitionSystem extends Transform with DependencyAPIMigration {
  // TODO: We only really need [[Forms.MidForm]] + LowerTypes, but we also want to fail if there are CombLoops
  // TODO: We also would like to run some optimization passes, but RemoveValidIf won't allow us to model DontCare
  //       precisely and PadWidths emits ill-typed firrtl.
  override def prerequisites: Seq[Dependency[Transform]] = Forms.LowForm
  override def invalidates(a: Transform): Boolean = false
  // since this pass only runs on the main module, inlining needs to happen before
  override def optionalPrerequisites: Seq[TransformDependency] = Seq(Dependency[firrtl.passes.InlineInstances])

  // We run the propagate preset annotations pass manually since we do not want to remove ValidIfs and other
  // Verilog emission passes.
  // Ideally we would go in and enable the [[PropagatePresetAnnotations]] to only depend on LowForm.
  private val presetPass = new PropagatePresetAnnotations
  override protected def execute(state: CircuitState): CircuitState = {
    // run the preset pass to extract all preset registers and remove preset reset signals
    val afterPreset = presetPass.execute(state)
    val circuit = afterPreset.circuit
    val presetRegs = afterPreset.annotations.collect {
      case PresetRegAnnotation(target) if target.module == circuit.main => target.ref
    }.toSet

    // collect all non-random memory initialization
    val memInit = afterPreset.annotations.collect { case a: MemoryInitAnnotation if !a.isRandomInit => a }
      .filter(_.target.module == circuit.main)
      .map(a => a.target.ref -> a.initValue)
      .toMap

    // convert the main module
    val main = circuit.modules.find(_.name == circuit.main).get
    val sys = main match {
      case x: ir.ExtModule =>
        throw new ExtModuleException(
          "External modules are not supported by the SMT backend. Use yosys if you need to convert Verilog."
        )
      case m: ir.Module =>
        new ModuleToTransitionSystem().run(m, presetRegs = presetRegs, memInit = memInit)
    }

    val sortedSys = TopologicalSort.run(sys)
    val anno = TransitionSystemAnnotation(sortedSys)
    state.copy(circuit = circuit, annotations = afterPreset.annotations :+ anno)
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
    m:          ir.Module,
    presetRegs: Set[String] = Set(),
    memInit:    Map[String, MemoryInitValue] = Map()
  ): TransitionSystem = {
    // first pass over the module to convert expressions; discover state and I/O
    val scan = new ModuleScanner(makeRandom)
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
    val memoryEncoding = new MemoryEncoding(makeRandom)
    val memoryStatesAndOutputs = scan.memories.map(m => memoryEncoding.onMemory(m, scan.connects, memInit.get(m.name)))
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
    val states = regStates.toArray ++ memoryStatesAndOutputs.flatMap(_._1)

    // generate comments from infos
    val comments = mutable.HashMap[String, String]()
    scan.infos.foreach {
      case (name, info) =>
        serializeInfo(info).foreach { infoString =>
          if (comments.contains(name)) { comments(name) += InfoSeparator + infoString }
          else { comments(name) = InfoPrefix + infoString }
        }
    }

    // inputs are original module inputs and any "random" signal we need for modelling
    val inputs = scan.inputs ++ randoms.values

    // module info to the comment header
    val header = serializeInfo(m.info).map(InfoPrefix + _).toArray

    val fair = Set[String]() // as of firrtl 1.4 we do not support fairness constraints
    TransitionSystem(
      m.name,
      inputs.toArray,
      states,
      signalsWithMem.toArray,
      outputs,
      constraints,
      bad,
      fair,
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

  private[firrtl] val randoms = mutable.LinkedHashMap[String, BVSymbol]()
  private def makeRandom(baseName: String, width: Int): BVExpr = {
    // TODO: actually ensure that there cannot be any name clashes with other identifiers
    val suffixes = Iterator(baseName) ++ (0 until 200).map(ii => baseName + "_" + ii)
    val name = suffixes.map(s => "RANDOM." + s).find(!randoms.contains(_)).get
    val sym = BVSymbol(name, width)
    randoms(name) = sym
    sym
  }
}

private class MemoryEncoding(makeRandom: (String, Int) => BVExpr) extends LazyLogging {
  type Connects = Iterable[(String, BVExpr)]
  def onMemory(
    defMem:    ir.DefMemory,
    connects:  Connects,
    initValue: Option[MemoryInitValue]
  ): (Iterable[State], Connects) = {
    // we can only work on appropriately lowered memories
    assert(
      defMem.dataType.isInstanceOf[ir.GroundType],
      s"Memory $defMem is of type ${defMem.dataType} which is not a ground type!"
    )
    assert(defMem.readwriters.isEmpty, "Combined read/write ports are not supported! Please split them up.")

    // collect all memory meta-data in a custom class
    val m = new MemInfo(defMem)

    // find all connections related to this memory
    val inputs = connects.filter(_._1.startsWith(m.prefix)).toMap

    // there could be a constant init
    val init = initValue.map(getInit(m, _))

    // parse and check read and write ports
    val writers = defMem.writers.map(w => new WritePort(m, w, inputs))
    val readers = defMem.readers.map(r => new ReadPort(m, r, inputs))

    // derive next state from all write ports
    assert(defMem.writeLatency == 1, "Only memories with write-latency of one are supported.")
    val next: ArrayExpr = if (writers.isEmpty) { m.sym }
    else {
      if (writers.length > 2) {
        throw new UnsupportedFeatureException(s"memories with 3+ write ports (${m.name})")
      }
      val validData = writers.foldLeft[ArrayExpr](m.sym) { case (sym, w) => w.writeTo(sym) }
      if (writers.length == 1) { validData }
      else {
        assert(writers.length == 2)
        val conflict = writers.head.doesConflict(writers.last)
        val conflictData = writers.head.makeRandomData("_write_write_collision")
        val conflictStore = ArrayStore(m.sym, writers.head.addr, conflictData)
        ArrayIte(conflict, conflictStore, validData)
      }
    }
    val state = State(m.sym, init, Some(next))

    // derive data signals from all read ports
    assert(defMem.readLatency >= 0)
    if (defMem.readLatency > 1) {
      throw new UnsupportedFeatureException(s"memories with read latency 2+ (${m.name})")
    }
    val readPortSignals = if (defMem.readLatency == 0) {
      readers.map { r =>
        // combinatorial read
        if (defMem.readUnderWrite != ir.ReadUnderWrite.New) {
          //logger.warn(s"WARN: Memory ${m.name} with combinatorial read port will always return the most recently written entry." +
          //  s" The read-under-write => ${defMem.readUnderWrite} setting will be ignored.")
        }
        // since we do a combinatorial read, the "old" data is the current data
        val data = r.readOld()
        r.data.name -> data
      }
    } else { Seq() }
    val readPortStates = if (defMem.readLatency == 1) {
      readers.map { r =>
        // we create a register for the read port data
        val next = defMem.readUnderWrite match {
          case ir.ReadUnderWrite.New =>
            throw new UnsupportedFeatureException(
              s"registered read ports that return the new value (${m.name}.${r.name})"
            )
          // the thing that makes this hard is to properly handle write conflicts
          case ir.ReadUnderWrite.Undefined =>
            val anyWriteToTheSameAddress = any(writers.map(_.doesConflict(r)))
            if (anyWriteToTheSameAddress == False) { r.readOld() }
            else {
              val readUnderWriteData = r.makeRandomData("_read_under_write_undefined")
              BVIte(anyWriteToTheSameAddress, readUnderWriteData, r.readOld())
            }
          case ir.ReadUnderWrite.Old => r.readOld()
        }
        State(r.data, init = None, next = Some(next))
      }
    } else { Seq() }

    (state +: readPortStates, readPortSignals)
  }

  private def getInit(m: MemInfo, initValue: MemoryInitValue): ArrayExpr = initValue match {
    case MemoryScalarInit(value) => ArrayConstant(BVLiteral(value, m.dataWidth), m.indexWidth)
    case MemoryArrayInit(values) =>
      assert(
        values.length == m.depth,
        s"Memory ${m.name} of depth ${m.depth} cannot be initialized with an array of length ${values.length}!"
      )
      // in order to get a more compact encoding try to find the most common values
      val histogram = mutable.LinkedHashMap[BigInt, Int]()
      values.foreach(v => histogram(v) = 1 + histogram.getOrElse(v, 0))
      val baseValue = histogram.maxBy(_._2)._1
      val base = ArrayConstant(BVLiteral(baseValue, m.dataWidth), m.indexWidth)
      values.zipWithIndex
        .filterNot(_._1 == baseValue)
        .foldLeft[ArrayExpr](base) {
          case (array, (value, index)) =>
            ArrayStore(array, BVLiteral(index, m.indexWidth), BVLiteral(value, m.dataWidth))
        }
    case other => throw new RuntimeException(s"Unsupported memory init option: $other")
  }

  private class MemInfo(m: ir.DefMemory) {
    val name = m.name
    val depth = m.depth
    // derrive the type of the memory from the dataType and depth
    val dataWidth = getWidth(m.dataType)
    val indexWidth = Utils.getUIntWidth(m.depth - 1).max(1)
    val sym = ArraySymbol(m.name, indexWidth, dataWidth)
    val prefix = m.name + "."
    val fullAddressRange = (BigInt(1) << indexWidth) == m.depth
    lazy val depthBV = BVLiteral(m.depth, indexWidth)
    def isValidAddress(addr: BVExpr): BVExpr = {
      if (fullAddressRange) { True }
      else {
        BVComparison(Compare.Greater, depthBV, addr, signed = false)
      }
    }
  }
  private abstract class MemPort(memory: MemInfo, val name: String, inputs: String => BVExpr) {
    val en:   BVSymbol = makeField("en", 1)
    val data: BVSymbol = makeField("data", memory.dataWidth)
    val addr: BVSymbol = makeField("addr", memory.indexWidth)
    protected def makeField(field: String, width: Int): BVSymbol = BVSymbol(memory.prefix + name + "." + field, width)
    // make sure that all widths are correct
    assert(inputs(en.name).width == en.width)
    assert(inputs(addr.name).width == addr.width)
    val enIsTrue: Boolean = inputs(en.name) == True
    def makeRandomData(suffix: String): BVExpr =
      makeRandom(memory.name + "_" + name + suffix, memory.dataWidth)
    def readOld(): BVExpr = {
      val canBeOutOfRange = !memory.fullAddressRange
      val canBeDisabled = !enIsTrue
      val data = ArrayRead(memory.sym, addr)
      val dataWithRangeCheck = if (canBeOutOfRange) {
        val outOfRangeData = makeRandomData("_addr_out_of_range")
        BVIte(memory.isValidAddress(addr), data, outOfRangeData)
      } else { data }
      val dataWithEnabledCheck = if (canBeDisabled) {
        val disabledData = makeRandomData("_not_enabled")
        BVIte(en, dataWithRangeCheck, disabledData)
      } else { dataWithRangeCheck }
      dataWithEnabledCheck
    }
  }
  private class WritePort(memory: MemInfo, name: String, inputs: String => BVExpr)
      extends MemPort(memory, name, inputs) {
    assert(inputs(data.name).width == data.width)
    val mask: BVSymbol = makeField("mask", 1)
    assert(inputs(mask.name).width == mask.width)
    val maskIsTrue: Boolean = inputs(mask.name) == True
    val doWrite: BVExpr = (enIsTrue, maskIsTrue) match {
      case (true, true)   => True
      case (true, false)  => mask
      case (false, true)  => en
      case (false, false) => and(en, mask)
    }
    def doesConflict(r: ReadPort): BVExpr = {
      val sameAddress = BVEqual(r.addr, addr)
      if (doWrite == True) { sameAddress }
      else { and(doWrite, sameAddress) }
    }
    def doesConflict(w: WritePort): BVExpr = {
      val bothWrite = and(doWrite, w.doWrite)
      val sameAddress = BVEqual(addr, w.addr)
      if (bothWrite == True) { sameAddress }
      else { and(doWrite, sameAddress) }
    }
    def writeTo(array: ArrayExpr): ArrayExpr = {
      val doUpdate = if (memory.fullAddressRange) doWrite else and(doWrite, memory.isValidAddress(addr))
      val update = ArrayStore(array, index = addr, data = data)
      if (doUpdate == True) update else ArrayIte(doUpdate, update, array)
    }

  }
  private class ReadPort(memory: MemInfo, name: String, inputs: String => BVExpr)
      extends MemPort(memory, name, inputs) {}

  private def and(a: BVExpr, b: BVExpr): BVExpr = (a, b) match {
    case (True, True) => True
    case (True, x)    => x
    case (x, True)    => x
    case _            => BVOp(Op.And, a, b)
  }
  private def or(a: BVExpr, b: BVExpr): BVExpr = BVOp(Op.Or, a, b)
  private val True = BVLiteral(1, 1)
  private val False = BVLiteral(0, 1)
  private def all(b: Iterable[BVExpr]): BVExpr = if (b.isEmpty) False else b.reduce((a, b) => and(a, b))
  private def any(b: Iterable[BVExpr]): BVExpr = if (b.isEmpty) True else b.reduce((a, b) => or(a, b))
}

// performas a first pass over the module collecting all connections, wires, registers, input and outputs
private class ModuleScanner(makeRandom: (String, Int) => BVExpr) extends LazyLogging {
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
  // keeps track of unused memory (data) outputs so that we can see where they are first used
  private val unusedMemOutputs = mutable.LinkedHashMap[String, Int]()
  // ensure unique names for assert/assume signals
  private val namespace = Namespace()

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
          inputs.append(BVSymbol(p.name, getWidth(p.tpe)))
        }
      case ir.Output =>
        if (!isClock(p.tpe)) { // we ignore clock outputs
          outputs.append(p.name)
        }
    }
  }

  private[firrtl] def onStatement(s: ir.Statement): Unit = s match {
    case ir.DefWire(info, name, tpe) =>
      namespace.newName(name)
      if (!isClock(tpe)) {
        infos.append(name -> info)
        wires.append(name)
      }
    case ir.DefNode(info, name, expr) =>
      namespace.newName(name)
      if (!isClock(expr.tpe)) {
        insertDummyAssignsForMemoryOutputs(expr)
        infos.append(name -> info)
        val e = onExpression(expr, name)
        nodes.append(name)
        connects.append((name, e))
      }
    case ir.DefRegister(info, name, tpe, _, reset, init) =>
      namespace.newName(name)
      insertDummyAssignsForMemoryOutputs(reset)
      insertDummyAssignsForMemoryOutputs(init)
      infos.append(name -> info)
      val width = getWidth(tpe)
      val resetExpr = onExpression(reset, 1, name + "_reset")
      val initExpr = onExpression(init, width, name + "_init")
      registers.append((name, width, resetExpr, initExpr))
    case m: ir.DefMemory =>
      namespace.newName(m.name)
      infos.append(m.name -> m.info)
      val outputs = getMemOutputs(m)
      (getMemInputs(m) ++ outputs).foreach(memSignals.append(_))
      val dataWidth = getWidth(m.dataType)
      outputs.foreach(name => unusedMemOutputs(name) = dataWidth)
      memories.append(m)
    case ir.Connect(info, loc, expr) =>
      if (!isGroundType(loc.tpe)) error("All connects should have been lowered to ground type!")
      if (!isClock(loc.tpe)) { // we ignore clock connections
        val name = loc.serialize
        insertDummyAssignsForMemoryOutputs(expr)
        infos.append(name -> info)
        connects.append((name, onExpression(expr, getWidth(loc.tpe), name)))
      }
    case ir.IsInvalid(info, loc) =>
      if (!isGroundType(loc.tpe)) error("All connects should have been lowered to ground type!")
      val name = loc.serialize
      infos.append(name -> info)
      connects.append((name, makeRandom(name + "_INVALID", getWidth(loc.tpe))))
    case ir.DefInstance(info, name, module, tpe) =>
      namespace.newName(name)
      if (!tpe.isInstanceOf[ir.BundleType]) error(s"Instance $name of $module has an invalid type: ${tpe.serialize}")
      // we treat all instances as blackboxes
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
            inputs.append(BVSymbol(pName, getWidth(p.tpe)))
          }
        } else {
          if (!isClock(p.tpe)) { // we ignore clock outputs
            outputs.append(pName)
          }
        }
      }
    case s @ ir.Verification(op, info, _, pred, en, msg) =>
      if (op == ir.Formal.Cover) {
        logger.warn(s"WARN: Cover statement was ignored: ${s.serialize}")
      } else {
        val name = namespace.newName(msgToName(op.toString, msg.string))
        val predicate = onExpression(pred, name + "_predicate")
        val enabled = onExpression(en, name + "_enabled")
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
  // inserts a dummy assign right before a memory output is used for the first time
  // example:
  // m.r.data <= m.r.data ; this is the dummy assign
  // test <= m.r.data     ; this is the first use of m.r.data
  private def insertDummyAssignsForMemoryOutputs(next: ir.Expression): Unit = if (unusedMemOutputs.nonEmpty) {
    implicit val uses = mutable.ArrayBuffer[String]()
    findUnusedMemoryOutputUse(next)
    if (uses.nonEmpty) {
      val useSet = uses.toSet
      unusedMemOutputs.foreach {
        case (name, width) =>
          if (useSet.contains(name)) connects.append(name -> BVSymbol(name, width))
      }
      useSet.foreach(name => unusedMemOutputs.remove(name))
    }
  }
  private def findUnusedMemoryOutputUse(e: ir.Expression)(implicit uses: mutable.ArrayBuffer[String]): Unit = e match {
    case s: ir.SubField =>
      val name = s.serialize
      if (unusedMemOutputs.contains(name)) uses.append(name)
    case other => other.foreachExpr(findUnusedMemoryOutputUse)
  }

  private case class Context(baseName: String) extends TranslationContext {
    override def getRandom(width: Int): BVExpr = makeRandom(baseName, width)
  }

  private def onExpression(e: ir.Expression, width: Int, randomPrefix: String): BVExpr = {
    implicit val ctx: TranslationContext = Context(randomPrefix)
    FirrtlExpressionSemantics.toSMT(e, width, allowNarrow = false)
  }
  private def onExpression(e: ir.Expression, randomPrefix: String): BVExpr = {
    implicit val ctx: TranslationContext = Context(randomPrefix)
    FirrtlExpressionSemantics.toSMT(e)
  }

  private def msgToName(prefix: String, msg: String): String = {
    // TODO: ensure that we can generate unique names
    prefix + "_" + msg.replace(" ", "_").replace("|", "")
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
