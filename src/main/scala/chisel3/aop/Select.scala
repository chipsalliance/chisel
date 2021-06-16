// SPDX-License-Identifier: Apache-2.0

package chisel3.aop

import chisel3._
import chisel3.experimental.{BaseModule, FixedPoint}
import chisel3.internal.HasId
import chisel3.internal.firrtl._
import firrtl.annotations.ReferenceTarget

import scala.collection.mutable

/** Use to select Chisel components in a module, after that module has been constructed
  * Useful for adding additional Chisel annotations or for use within an [[Aspect]]
  */
object Select {

  /** Return just leaf components of expanded node
    *
    * @param d Component to find leafs if aggregate typed. Intermediate fields/indicies are not included
    * @return
    */
  def getLeafs(d: Data): Seq[Data] = d match {
    case r: Record => r.getElements.flatMap(getLeafs)
    case v: Vec[_] => v.getElements.flatMap(getLeafs)
    case other => Seq(other)
  }

  /** Return all expanded components, including intermediate aggregate nodes
    *
    * @param d Component to find leafs if aggregate typed. Intermediate fields/indicies ARE included
    * @return
    */
  def getIntermediateAndLeafs(d: Data): Seq[Data] = d match {
    case r: Record => r +: r.getElements.flatMap(getIntermediateAndLeafs)
    case v: Vec[_] => v +: v.getElements.flatMap(getIntermediateAndLeafs)
    case other => Seq(other)
  }


  /** Collects all components selected by collector within module and all children modules it instantiates
    *   directly or indirectly
    * Accepts a collector function, rather than a collector partial function (see [[collectDeep]])
    * @param module Module to collect components, as well as all children module it directly and indirectly instantiates
    * @param collector Collector function to pick, given a module, which components to collect
    * @param tag Required for generics to work, should ignore this
    * @tparam T Type of the component that will be collected
    * @return
    */
  def getDeep[T](module: BaseModule)(collector: BaseModule => Seq[T]): Seq[T] = {
    check(module)
    val myItems = collector(module)
    val deepChildrenItems = instances(module).flatMap {
      i => getDeep(i)(collector)
    }
    myItems ++ deepChildrenItems
  }

  /** Collects all components selected by collector within module and all children modules it instantiates
    *   directly or indirectly
    * Accepts a collector partial function, rather than a collector function (see [[getDeep]])
    * @param module Module to collect components, as well as all children module it directly and indirectly instantiates
    * @param collector Collector partial function to pick, given a module, which components to collect
    * @param tag Required for generics to work, should ignore this
    * @tparam T Type of the component that will be collected
    * @return
    */
  def collectDeep[T](module: BaseModule)(collector: PartialFunction[BaseModule, T]): Iterable[T] = {
    check(module)
    val myItems = collector.lift(module)
    val deepChildrenItems = instances(module).flatMap {
      i => collectDeep(i)(collector)
    }
    myItems ++ deepChildrenItems
  }

  /** Selects all instances directly instantiated within given module
    * @param module
    * @return
    */
  def instances(module: BaseModule): Seq[BaseModule] = {
    check(module)
    module._component.get match {
      case d: DefModule => d.commands.collect {
        case i: DefInstance => i.id
      }
      case other => Nil
    }
  }

  /** Selects all registers directly instantiated within given module
    * @param module
    * @return
    */
  def registers(module: BaseModule): Seq[Data] = {
    check(module)
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case r: DefReg => r.id
      case r: DefRegInit => r.id
    }
  }

  /** Selects all ios directly contained within given module
    * @param module
    * @return
    */
  def ios(module: BaseModule): Seq[Data] = {
    check(module)
    module._component.get.asInstanceOf[DefModule].ports.map(_.id)
  }

  /** Selects all SyncReadMems directly contained within given module
    * @param module
    * @return
    */
  def syncReadMems(module: BaseModule): Seq[SyncReadMem[_]] = {
    check(module)
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case r: DefSeqMemory => r.id.asInstanceOf[SyncReadMem[_]]
    }
  }

  /** Selects all Mems directly contained within given module
    * @param module
    * @return
    */
  def mems(module: BaseModule): Seq[Mem[_]] = {
    check(module)
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case r: DefMemory => r.id.asInstanceOf[Mem[_]]
    }
  }

  /** Selects all arithmetic or logical operators directly instantiated within given module
    * @param module
    * @return
    */
  def ops(module: BaseModule): Seq[(String, Data)] = {
    check(module)
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case d: DefPrim[_] => (d.op.name, d.id)
    }
  }

  /** Selects a kind of arithmetic or logical operator directly instantiated within given module
    * The kind of operators are contained in [[chisel3.internal.firrtl.PrimOp]]
    * @param opKind the kind of operator, e.g. "mux", "add", or "bits"
    * @param module
    * @return
    */
  def ops(opKind: String)(module: BaseModule): Seq[Data] = {
    check(module)
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case d: DefPrim[_] if d.op.name == opKind => d.id
    }
  }

  /** Selects all wires in a module
    * @param module
    * @return
    */
  def wires(module: BaseModule): Seq[Data] = {
    check(module)
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case r: DefWire => r.id
    }
  }

  /** Selects all memory ports, including their direction and memory
    * @param module
    * @return
    */
  def memPorts(module: BaseModule): Seq[(Data, MemPortDirection, MemBase[_])] = {
    check(module)
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case r: DefMemPort[_] => (r.id, r.dir, r.source.id.asInstanceOf[MemBase[_ <: Data]])
    }
  }

  /** Selects all memory ports of a given direction, including their memory
    * @param dir The direction of memory ports to select
    * @param module
    * @return
    */
  def memPorts(dir: MemPortDirection)(module: BaseModule): Seq[(Data, MemBase[_])] = {
    check(module)
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case r: DefMemPort[_] if r.dir == dir => (r.id, r.source.id.asInstanceOf[MemBase[_ <: Data]])
    }
  }

  /** Selects all components who have been set to be invalid, even if they are later connected to
    * @param module
    * @return
    */
  def invalids(module: BaseModule): Seq[Data] = {
    check(module)
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case DefInvalid(_, arg) => getData(arg)
    }
  }

  /** Selects all components who are attached to a given signal, within a module
    * @param module
    * @return
    */
  def attachedTo(module: BaseModule)(signal: Data): Set[Data] = {
    check(module)
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case Attach(_, seq) if seq.contains(signal) => seq
    }.flatMap { seq => seq.map(_.id.asInstanceOf[Data]) }.toSet
  }

  /** Selects all connections to a signal or its parent signal(s) (if the signal is an element of an aggregate signal)
    * The when predicates surrounding each connection are included in the returned values
    *
    * E.g. if signal = io.foo.bar, connectionsTo will return all connections to io, io.foo, and io.bar
    * @param module
    * @param signal
    * @return
    */
  def connectionsTo(module: BaseModule)(signal: Data): Seq[PredicatedConnect] = {
    check(module)
    val sensitivitySignals = getIntermediateAndLeafs(signal).toSet
    val predicatedConnects = mutable.ArrayBuffer[PredicatedConnect]()
    val isPort = module._component.get.asInstanceOf[DefModule].ports.flatMap{ p => getIntermediateAndLeafs(p.id) }.contains(signal)
    var prePredicates: Seq[Predicate] = Nil
    var seenDef = isPort
    searchWhens(module, (cmd: Command, preds) => {
      cmd match {
        case cmd: Definition if cmd.id.isInstanceOf[Data] =>
          val x = getIntermediateAndLeafs(cmd.id.asInstanceOf[Data])
          if(x.contains(signal)) prePredicates = preds
        case Connect(_, loc@Node(d: Data), exp) =>
          val effected = getEffected(loc).toSet
          if(sensitivitySignals.intersect(effected).nonEmpty) {
            val expData = getData(exp)
            prePredicates.reverse.zip(preds.reverse).foreach(x => assert(x._1 == x._2, s"Prepredicates $x must match for signal $signal"))
            predicatedConnects += PredicatedConnect(preds.dropRight(prePredicates.size), d, expData, isBulk = false)
          }
        case BulkConnect(_, loc@Node(d: Data), exp) =>
          val effected = getEffected(loc).toSet
          if(sensitivitySignals.intersect(effected).nonEmpty) {
            val expData = getData(exp)
            prePredicates.reverse.zip(preds.reverse).foreach(x => assert(x._1 == x._2, s"Prepredicates $x must match for signal $signal"))
            predicatedConnects += PredicatedConnect(preds.dropRight(prePredicates.size), d, expData, isBulk = true)
          }
        case other =>
      }
    })
    predicatedConnects.toSeq
  }

  /** Selects all stop statements, and includes the predicates surrounding the stop statement
    *
    * @param module
    * @return
    */
  def stops(module: BaseModule): Seq[Stop]  = {
    val stops = mutable.ArrayBuffer[Stop]()
    searchWhens(module, (cmd: Command, preds: Seq[Predicate]) => {
      cmd match {
        case chisel3.internal.firrtl.Stop(_, clock, ret) => stops += Stop(preds, ret, getId(clock).asInstanceOf[Clock])
        case other =>
      }
    })
    stops.toSeq
  }

  /** Selects all printf statements, and includes the predicates surrounding the printf statement
    *
    * @param module
    * @return
    */
  def printfs(module: BaseModule): Seq[Printf] = {
    val printfs = mutable.ArrayBuffer[Printf]()
    searchWhens(module, (cmd: Command, preds: Seq[Predicate]) => {
      cmd match {
        case chisel3.internal.firrtl.Printf(_, clock, pable) => printfs += Printf(preds, pable, getId(clock).asInstanceOf[Clock])
        case other =>
      }
    })
    printfs.toSeq
  }

  // Checks that a module has finished its construction
  private def check(module: BaseModule): Unit = {
    require(module.isClosed, "Can't use Selector on modules that have not finished construction!")
    require(module._component.isDefined, "Can't use Selector on modules that don't have components!")
  }

  // Given a loc, return all subcomponents of id that could be assigned to in connect
  private def getEffected(a: Arg): Seq[Data] = a match {
    case Node(id: Data) => getIntermediateAndLeafs(id)
    case Slot(imm, name) => Seq(imm.id.asInstanceOf[Record].elements(name))
    case Index(imm, value) => getEffected(imm)
  }

  // Given an arg, return the corresponding id. Don't use on a loc of a connect.
  private def getId(a: Arg): HasId = a match {
    case Node(id) => id
    case l: ULit => l.num.U(l.w)
    case l: SLit => l.num.S(l.w)
    case l: FPLit => FixedPoint(l.num, l.w, l.binaryPoint)
    case other =>
      sys.error(s"Something went horribly wrong! I was expecting ${other} to be a lit or a node!")
  }

  private def getData(a: Arg): Data = a match {
    case Node(data: Data) => data
    case other =>
      sys.error(s"Something went horribly wrong! I was expecting ${other} to be Data!")
  }

  // Given an id, either get its name or its value, if its a lit
  private def getName(i: HasId): String = try {
    i.toTarget match {
      case r: ReferenceTarget =>
        val str = r.serialize
        str.splitAt(str.indexOf('>'))._2.drop(1)
    }
  } catch {
    case e: ChiselException => i.getOptionRef.get match {
      case l: LitArg => l.num.intValue.toString
    }
  }

  // Collects when predicates as it searches through a module, then applying processCommand to non-when related commands
  private def searchWhens(module: BaseModule, processCommand: (Command, Seq[Predicate]) => Unit) = {
    check(module)
    module._component.get.asInstanceOf[DefModule].commands.foldLeft((Seq.empty[Predicate], Option.empty[Predicate])) {
      (blah, cmd) =>
        (blah, cmd) match {
          case ((preds, o), cmd) => cmd match {
            case WhenBegin(_, Node(pred: Bool)) => (When(pred) +: preds, None)
            case WhenBegin(_, l: LitArg) if l.num == BigInt(1) => (When(true.B) +: preds, None)
            case WhenBegin(_, l: LitArg) if l.num == BigInt(0) => (When(false.B) +: preds, None)
            case other: WhenBegin =>
              sys.error(s"Something went horribly wrong! I was expecting ${other.pred} to be a lit or a bool!")
            case _: WhenEnd => (preds.tail, Some(preds.head))
            case AltBegin(_) if o.isDefined => (o.get.not +: preds, o)
            case _: AltBegin =>
              sys.error(s"Something went horribly wrong! I was expecting ${o} to be nonEmpty!")
            case OtherwiseEnd(_, _) => (preds.tail, None)
            case other =>
              processCommand(cmd, preds)
              (preds, o)
          }
        }
    }
  }

  trait Serializeable {
    def serialize: String
  }

  /** Used to indicates a when's predicate (or its otherwise predicate)
    */
  trait Predicate extends Serializeable {
    val bool: Bool
    def not: Predicate
  }

  /** Used to represent [[chisel3.when]] predicate
    *
    * @param bool the when predicate
    */
  case class When(bool: Bool) extends Predicate {
    def not: WhenNot = WhenNot(bool)
    def serialize: String = s"${getName(bool)}"
  }

  /** Used to represent the `otherwise` predicate of a [[chisel3.when]]
    *
    * @param bool the when predicate corresponding to this otherwise predicate
    */
  case class WhenNot(bool: Bool) extends Predicate {
    def not: When = When(bool)
    def serialize: String = s"!${getName(bool)}"
  }

  /** Used to represent a connection or bulk connection
    *
    * Additionally contains the sequence of when predicates seen when the connection is declared
    *
    * @param preds
    * @param loc
    * @param exp
    * @param isBulk
    */
  case class PredicatedConnect(preds: Seq[Predicate], loc: Data, exp: Data, isBulk: Boolean) extends Serializeable {
    def serialize: String = {
      val moduleTarget = loc.toTarget.moduleTarget.serialize
      s"$moduleTarget: when(${preds.map(_.serialize).mkString(" & ")}): ${getName(loc)} ${if(isBulk) "<>" else ":="} ${getName(exp)}"
    }
  }

  /** Used to represent a [[chisel3.stop]]
    *
    * @param preds
    * @param ret
    * @param clock
    */
  case class Stop(preds: Seq[Predicate], ret: Int, clock: Clock) extends Serializeable {
    def serialize: String = {
      s"stop when(${preds.map(_.serialize).mkString(" & ")}) on ${getName(clock)}: $ret"
    }
  }

  /** Used to represent a [[chisel3.printf]]
    *
    * @param preds
    * @param pable
    * @param clock
    */
  case class Printf(preds: Seq[Predicate], pable: Printable, clock: Clock) extends Serializeable {
    def serialize: String = {
      s"printf when(${preds.map(_.serialize).mkString(" & ")}) on ${getName(clock)}: $pable"
    }
  }
}
