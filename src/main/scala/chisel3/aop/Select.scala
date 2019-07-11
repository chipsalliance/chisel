// See LICENSE for license details.

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

  import scala.reflect.runtime.universe.TypeTag

  private def check(module: BaseModule): Unit = {
    require(module.isClosed, "Can't use Selector on modules that have not finished construction!")
    require(module._component.isDefined, "Can't use Selector on modules that don't have components!")
  }

  // Return just leaf components of expanded node
  private def getLeafs(d: HasId): Seq[HasId] = d match {
    case b: Bundle => b.getElements.flatMap(getLeafs)
    case v: Vec[_] => v.getElements.flatMap(getLeafs)
    case other => Seq(other)
  }

  // Return all expanded components, including intermediate aggregate nodes
  private def getIntermediateAndLeafs(d: HasId): Seq[HasId] = d match {
    case b: Bundle => b +: b.getElements.flatMap(getIntermediateAndLeafs)
    case v: Vec[_] => v +: v.getElements.flatMap(getIntermediateAndLeafs)
    case other => Seq(other)
  }

  // Given a loc, return all subcomponents of id that could be assigned to in connect
  private def getEffected(a: Arg): Seq[HasId] = a match {
    case Node(id) => getIntermediateAndLeafs(id)
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

  private def getName(i: HasId): String = try {
    i.toTarget match {
      case r: ReferenceTarget =>
        val str = r.serialize
        str.splitAt(str.indexOf('>'))._2.drop(1)
    }
  } catch {
    case e: ChiselException => i.getOptionRef.get match {
      case l: LitArg => l.num.intValue().toString
    }
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
  def getDeep[T](module: BaseModule)(collector: BaseModule => Seq[T])(implicit tag: TypeTag[T]): Seq[T] = {
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
  def collectDeep[T](module: BaseModule)(collector: PartialFunction[BaseModule, T])(implicit tag: TypeTag[T]): Iterable[T] = {
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
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case i: DefInstance => i.id
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
      case d: DefPrim[_] if d.name == opKind => d.id
    }
  }

  def wires(module: BaseModule): Seq[Data] = {
    check(module)
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case r: DefWire => r.id
    }
  }

  def memPorts(module: BaseModule): Seq[(Data, MemPortDirection, MemBase[_])] = {
    check(module)
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case r: DefMemPort[_] => (r.id, r.dir, r.source.id.asInstanceOf[MemBase[_ <: Data]])
    }
  }

  def memPorts(dir: MemPortDirection)(module: BaseModule): Seq[(Data, MemBase[_])] = {
    check(module)
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case r: DefMemPort[_] if r.dir == dir => (r.id, r.source.id.asInstanceOf[MemBase[_ <: Data]])
    }
  }

  def invalids(module: BaseModule): Seq[HasId] = {
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case DefInvalid(_, arg) => getId(arg)
    }
  }

  def attachedTo(module: BaseModule)(signal: HasId): Set[HasId] = {
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case Attach(_, seq) if seq.contains(signal) => seq
    }.flatMap { seq => seq.map(_.id) }.toSet
  }

  def connectionsTo(module: BaseModule)(signal: HasId): Seq[PredicatedConnect] = {
    check(module)
    val sensitivitySignals = getIntermediateAndLeafs(signal).toSet
    val predicatedConnects = mutable.ArrayBuffer[PredicatedConnect]()

    val isPort = module._component.get.asInstanceOf[DefModule].ports.flatMap{ p => getIntermediateAndLeafs(p.id) }.contains(signal)
    module._component.get.asInstanceOf[DefModule].commands.foldLeft((Seq.empty[Predicate], Option.empty[Predicate], isPort)) {
      (blah, cmd) =>
        //println(s"On cmd: $cmd with $blah")
      (blah, cmd) match {
        case ((preds, o, false), cmd: Definition) =>
          val x = getIntermediateAndLeafs(cmd.id)
          //println(s"Does ${cmd.name} with $x contain $signal? ${x.contains(signal)}")
          if(x.contains(signal)) (preds, None, true)
          else (preds, None, false)
        case ((preds, o, false), _) =>
          (preds, None, false)
        case ((preds, o, seenDef@true), cmd) => cmd match {
          case WhenBegin(_, Node(pred: Bool)) => (When(pred) +: preds, None, seenDef)
          case WhenBegin(_, l: LitArg) if l.num == BigInt(1) => (When(true.B) +: preds, None, seenDef)
          case WhenBegin(_, l: LitArg) if l.num == BigInt(0) => (When(false.B) +: preds, None, seenDef)
          case other: WhenBegin =>
            sys.error(s"Something went horribly wrong! I was expecting ${other.pred} to be a lit or a bool!")
          case _: WhenEnd => (preds.tail, Some(preds.head), seenDef)
          case AltBegin(_) if o.isDefined => (o.get.not +: preds, o, seenDef)
          case _: AltBegin =>
            sys.error(s"Something went horribly wrong! I was expecting ${o} to be nonEmpty!")
          case OtherwiseEnd(_, _) => (preds.tail, None, seenDef)
          case Connect(_, loc, exp) =>
            val effected = getEffected(loc).toSet
            if(sensitivitySignals.intersect(effected).nonEmpty) {
              val expId = getId(exp)
              predicatedConnects += PredicatedConnect(preds, loc.id, expId, false)
            }
            (preds, o, seenDef)
          case BulkConnect(_, loc, exp) =>
            val effected = getEffected(loc).toSet
            if(sensitivitySignals.intersect(effected).nonEmpty) {
              val expId = getId(exp)
              predicatedConnects += PredicatedConnect(preds, loc.id, expId, true)
            }
            (preds, o, seenDef)
          case other => (preds, o, seenDef)
        }
      }
    }
    predicatedConnects
  }

  trait Predicate {
    val bool: Bool
    def not: Predicate
    def serialize: String
  }

  case class When(bool: Bool) extends Predicate {
    def not: WhenNot = WhenNot(bool)
    def serialize: String = s"${getName(bool)}"
  }

  case class WhenNot(bool: Bool) extends Predicate {
    def not: When = When(bool)
    def serialize: String = s"!${getName(bool)}"
  }

  case class PredicatedConnect(preds: Seq[Predicate], loc: HasId, exp: HasId, isBulk: Boolean) {
    def serialize: String = {
      val moduleTarget = loc.toTarget.moduleTarget.serialize
      s"$moduleTarget: when(${preds.map(_.serialize).mkString(" & ")}): ${getName(loc)} ${if(isBulk) "<>" else ":="} ${getName(exp)}"
    }
  }

  case class Stop(ret: Int, clock: Clock)
  def stops(module: BaseModule): Seq[Stop]  = {
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case chisel3.internal.firrtl.Stop(_, clock, ret) => Stop(ret, getId(clock).asInstanceOf[Clock])
    }
  }

  case class Printf(pable: Printable, clock: Clock)
  def printfs(module: BaseModule): Seq[Printf] = {
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case chisel3.internal.firrtl.Printf(_, clock, pable) => Printf(pable, getId(clock).asInstanceOf[Clock])
    }
  }

}
