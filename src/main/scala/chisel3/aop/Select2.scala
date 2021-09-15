// SPDX-License-Identifier: Apache-2.0

package chisel3.aop

import chisel3._
import chisel3.internal.{HasId}
import chisel3.experimental.hierarchy.{Definition, Instance, Hierarchy}
import chisel3.experimental.BaseModule
import chisel3.experimental.FixedPoint
import chisel3.internal.firrtl.{Definition => IRDefinition, _}
import chisel3.internal.PseudoModule
import chisel3.internal.BaseModule.ModuleClone
import firrtl.annotations.ReferenceTarget

import scala.collection.mutable
import scala.reflect.runtime.universe.TypeTag
import chisel3.internal.naming.chiselName

/** Use to select Chisel components in a module, after that module has been constructed
  * Useful for adding additional Chisel annotations or for use within an [[Aspect]]
  */
object Select2 {
  implicit val mg = new chisel3.internal.MacroGenerated {}
  // Checks that a module has finished its construction
  private def check(module: Hierarchy[BaseModule]): Unit = {
    require(module.getProto.isClosed, "Can't use Selector on modules that have not finished construction!")
    require(module.getProto._component.isDefined, "Can't use Selector on modules that don't have components!")
  }


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

  def instancesOf[T <: BaseModule : TypeTag](parent: Hierarchy[BaseModule]): Seq[Instance[T]] = {
    check(parent)
    val ttag = implicitly[TypeTag[T]]
    parent.getProto._component.get match {
      case d: DefModule => d.commands.collect {
        case d: DefTypedInstance[ModuleClone[BaseModule]] if d.tag.tpe <:< ttag.tpe =>
          val x = d.id match {
            case m: ModuleClone[T] => parent._lookup { x =>
              Instance(Right(m.asInstanceOf[ModuleClone[T]]), d.tag)
            }
            case other: T => parent._lookup(x => other)
          }
          x.asInstanceOf[Instance[T]]
      }
      case other => Nil
    }
  }

  def allInstancesOf[T <: BaseModule : TypeTag](root: Hierarchy[BaseModule]): Seq[Instance[T]] = {
    val locals = instancesOf[T](root)
    val allLocalInstances = instances(root)
    locals ++ (allLocalInstances.flatMap(allInstancesOf[T]))
  }


  /** Selects all instances directly instantiated within given definition
    * @param module
    * @return
    */
  def instances(instance: Hierarchy[BaseModule]): Seq[Instance[BaseModule]] = {
    check(instance)
    instance.getProto._component.get match {
      case d: DefModule => d.commands.flatMap {
        case d: DefTypedInstance[BaseModule] =>
          val x = d.id match {
            case m: ModuleClone[BaseModule] => instance._lookup{x => Instance(Right(m), d.tag) }
            case other => instance._lookup(x => other)
          }
          Some(x)
        case _ => None
      }
      case other => Nil
    }
  }

  /** Selects all instances directly and indirectly instantiated within given module
    * @param module
    * @return
    */
  def definitionsOf[T <: BaseModule : TypeTag](parent: Hierarchy[BaseModule]): Seq[Definition[T]] = {
    type DefType = Definition[T]
    check(parent)
    val ttag = implicitly[TypeTag[T]]
    val defs = parent.getProto._component.get match {
      case d: DefModule => d.commands.collect {
        case d: DefTypedInstance[ModuleClone[BaseModule]] if d.tag.tpe <:< ttag.tpe =>
          val x = d.id match {
            case m: ModuleClone[T] => parent._lookup { x =>
              Definition(Right(m.asInstanceOf[ModuleClone[T]]), d.tag)
            }
            case other: T => parent._lookup(x => other)
          }
          x.asInstanceOf[Definition[T]]
      }
    }
    val (_, defList) = defs.foldLeft((Set.empty[DefType], List.empty[DefType])) { case ((set, list), definition: Definition[T]) =>
      if(set.contains(definition)) (set, list) else (set + definition, definition +: list)
    }
    defList
  }

  /** Selects all instances directly and indirectly instantiated within given module
    * @param module
    * @return
    */
  def allDefinitionsOf[T <: BaseModule : TypeTag](root: Hierarchy[BaseModule]): Seq[Definition[T]] = {
    type DefType = Definition[T]
    val defSet = mutable.HashSet[DefType]()
    val defList = mutable.ArrayBuffer[DefType]()
    def rec(hier: Hierarchy[BaseModule]): Unit = {
      val returnedDefs = definitionsOf[T](hier)
      returnedDefs.foreach { case d if !defSet.contains(d) =>
        defSet += d
        defList += d
        rec(d)
      }
    }
    rec(root)
    defList.toList
  }

  /** Selects all instances directly instantiated within given module
    * @param module
    * @return
    */
  def definitions(module: Hierarchy[BaseModule]): Seq[Definition[BaseModule]] = {
    type DefType = Definition[BaseModule]
    check(module)
    val defs = module.getProto._component.get match {
      case d: DefModule => d.commands.flatMap {
        case i: DefTypedInstance[BaseModule] => Some(new Definition(Right(i.id), i.tag))
        case _ => None
      }
      case other => Nil
    }
    val (_, defList) = defs.foldLeft((Set.empty[DefType], List.empty[DefType])) { case ((set, list), definition: Definition[BaseModule]) =>
      if(set.contains(definition)) (set, list) else (set + definition, definition +: list)
    }
    defList
  }


  /** Selects all registers directly instantiated within given module
    * @param module
    * @return
    */
  def registers(module: Hierarchy[BaseModule]): Seq[Data] = module._lookup { proto => Select.registers(proto) }

  /** Selects all ios directly contained within given module
    * @param module
    * @return
    */
  def ios(module: Hierarchy[BaseModule]): Seq[Data] = module._lookup(p => Select.ios(p))

  /** Selects all SyncReadMems directly contained within given module
    * @param module
    * @return
    */
  // TODO: Need to add lookupable for SyncReadMem's
  //def syncReadMems(module: Hierarchy[BaseModule]): Seq[SyncReadMem[_]] = module._lookup(p => Select.syncReadMems(p))

  /** Selects all Mems directly contained within given module
    * @param module
    * @return
    */
  // TODO: Need to add lookupable for Mem[_]
  //def mems(module: Hierarchy[BaseModule]): Seq[Mem[_]] = module._lookup(p => Select.mems(p))

  /** Selects all arithmetic or logical operators directly instantiated within given module
    * @param module
    * @return
    */
  // TODO: Add support for Tuples?
  //def ops(module: Hierarchy[BaseModule]): Seq[(String, Data)] = module._lookup(p => Select.ops(p))

  /** Selects a kind of arithmetic or logical operator directly instantiated within given module
    * The kind of operators are contained in [[chisel3.internal.firrtl.PrimOp]]
    * @param opKind the kind of operator, e.g. "mux", "add", or "bits"
    * @param module
    * @return
    */
  def ops(opKind: String)(module: Hierarchy[BaseModule]): Seq[Data] = module._lookup(p => Select.ops(opKind)(p))

  /** Selects all wires in a module
    * @param module
    * @return
    */
  def wires(module: Hierarchy[BaseModule]): Seq[Data] = module._lookup{m => Select.wires(m)}

  /** Selects all memory ports, including their direction and memory
    * @param module
    * @return
    */
  // TODO: Need to add lookupable for MemPortDirection
  //def memPorts(module: Hierarchy[BaseModule]): Seq[(Data, MemPortDirection, MemBase[_])] = module._lookup{m => Select.memPorts(m)}

  /** Selects all memory ports of a given direction, including their memory
    * @param dir The direction of memory ports to select
    * @param module
    * @return
    */
  // TODO
  //def memPorts(dir: MemPortDirection)(module: Hierarchy[BaseModule]): Seq[(Data, MemBase[_])] = module._lookup{m => Select.memPorts(dir)(m)}

  /** Selects all components who have been set to be invalid, even if they are later connected to
    * @param module
    * @return
    */
  def invalids(module: Hierarchy[BaseModule]): Seq[Data] = module._lookup(p => Select.invalids(p))

  ///** Selects all components who are attached to a given signal, within a module
  //  * @param module
  //  * @return
  //  */
  //def attachedTo(module: Hierarchy[BaseModule])(signal: Data): Set[Data] = {
  //  check(module)
  //  module._component.get.asInstanceOf[DefModule].commands.collect {
  //    case Attach(_, seq) if seq.contains(signal) => seq
  //  }.flatMap { seq => seq.map(_.id.asInstanceOf[Data]) }.toSet
  //}

  ///** Selects all connections to a signal or its parent signal(s) (if the signal is an element of an aggregate signal)
  //  * The when predicates surrounding each connection are included in the returned values
  //  *
  //  * E.g. if signal = io.foo.bar, connectionsTo will return all connections to io, io.foo, and io.bar
  //  * @param module
  //  * @param signal
  //  * @return
  //  */
  //def connectionsTo(module: Hierarchy[BaseModule])(signal: Data): Seq[PredicatedConnect] = {
  //  check(module)
  //  val sensitivitySignals = getIntermediateAndLeafs(signal).toSet
  //  val predicatedConnects = mutable.ArrayBuffer[PredicatedConnect]()
  //  val isPort = module._component.get.asInstanceOf[DefModule].ports.flatMap{ p => getIntermediateAndLeafs(p.id) }.contains(signal)
  //  var prePredicates: Seq[Predicate] = Nil
  //  var seenDef = isPort
  //  searchWhens(module, (cmd: Command, preds) => {
  //    cmd match {
  //      case cmd: IRDefinition if cmd.id.isInstanceOf[Data] =>
  //        val x = getIntermediateAndLeafs(cmd.id.asInstanceOf[Data])
  //        if(x.contains(signal)) prePredicates = preds
  //      case Connect(_, loc@Node(d: Data), exp) =>
  //        val effected = getEffected(loc).toSet
  //        if(sensitivitySignals.intersect(effected).nonEmpty) {
  //          val expData = getData(exp)
  //          prePredicates.reverse.zip(preds.reverse).foreach(x => assert(x._1 == x._2, s"Prepredicates $x must match for signal $signal"))
  //          predicatedConnects += PredicatedConnect(preds.dropRight(prePredicates.size), d, expData, isBulk = false)
  //        }
  //      case BulkConnect(_, loc@Node(d: Data), exp) =>
  //        val effected = getEffected(loc).toSet
  //        if(sensitivitySignals.intersect(effected).nonEmpty) {
  //          val expData = getData(exp)
  //          prePredicates.reverse.zip(preds.reverse).foreach(x => assert(x._1 == x._2, s"Prepredicates $x must match for signal $signal"))
  //          predicatedConnects += PredicatedConnect(preds.dropRight(prePredicates.size), d, expData, isBulk = true)
  //        }
  //      case other =>
  //    }
  //  })
  //  predicatedConnects.toSeq
  //}

  ///** Selects all stop statements, and includes the predicates surrounding the stop statement
  //  *
  //  * @param module
  //  * @return
  //  */
  //def stops(module: Hierarchy[BaseModule]): Seq[Stop]  = {
  //  val stops = mutable.ArrayBuffer[Stop]()
  //  searchWhens(module, (cmd: Command, preds: Seq[Predicate]) => {
  //    cmd match {
  //      case chisel3.internal.firrtl.Stop(_, clock, ret) => stops += Stop(preds, ret, getId(clock).asInstanceOf[Clock])
  //      case other =>
  //    }
  //  })
  //  stops.toSeq
  //}

  ///** Selects all printf statements, and includes the predicates surrounding the printf statement
  //  *
  //  * @param module
  //  * @return
  //  */
  //def printfs(module: Hierarchy[BaseModule]): Seq[Printf] = {
  //  val printfs = mutable.ArrayBuffer[Printf]()
  //  searchWhens(module, (cmd: Command, preds: Seq[Predicate]) => {
  //    cmd match {
  //      case chisel3.internal.firrtl.Printf(id, _, clock, pable) => printfs += Printf(id, preds, pable, getId(clock).asInstanceOf[Clock])
  //      case other =>
  //    }
  //  })
  //  printfs.toSeq
  //}

  //// Given a loc, return all subcomponents of id that could be assigned to in connect
  //private def getEffected(a: Arg): Seq[Data] = a match {
  //  case Node(id: Data) => getIntermediateAndLeafs(id)
  //  case Slot(imm, name) => Seq(imm.id.asInstanceOf[Record].elements(name))
  //  case Index(imm, value) => getEffected(imm)
  //}

  //// Given an arg, return the corresponding id. Don't use on a loc of a connect.
  //private def getId(a: Arg): HasId = a match {
  //  case Node(id) => id
  //  case l: ULit => l.num.U(l.w)
  //  case l: SLit => l.num.S(l.w)
  //  case l: FPLit => FixedPoint(l.num, l.w, l.binaryPoint)
  //  case other =>
  //    sys.error(s"Something went horribly wrong! I was expecting ${other} to be a lit or a node!")
  //}

  //private def getData(a: Arg): Data = a match {
  //  case Node(data: Data) => data
  //  case other =>
  //    sys.error(s"Something went horribly wrong! I was expecting ${other} to be Data!")
  //}

  //// Given an id, either get its name or its value, if its a lit
  //private def getName(i: HasId): String = try {
  //  i.toTarget match {
  //    case r: ReferenceTarget =>
  //      val str = r.serialize
  //      str.splitAt(str.indexOf('>'))._2.drop(1)
  //  }
  //} catch {
  //  case e: ChiselException => i.getOptionRef.get match {
  //    case l: LitArg => l.num.intValue.toString
  //  }
  //}

  //// Collects when predicates as it searches through a module, then applying processCommand to non-when related commands
  //private def searchWhens(module: Hierarchy[BaseModule], processCommand: (Command, Seq[Predicate]) => Unit) = {
  //  check(module)
  //  module._component.get.asInstanceOf[DefModule].commands.foldLeft((Seq.empty[Predicate], Option.empty[Predicate])) {
  //    (blah, cmd) =>
  //      (blah, cmd) match {
  //        case ((preds, o), cmd) => cmd match {
  //          case WhenBegin(_, Node(pred: Bool)) => (When(pred) +: preds, None)
  //          case WhenBegin(_, l: LitArg) if l.num == BigInt(1) => (When(true.B) +: preds, None)
  //          case WhenBegin(_, l: LitArg) if l.num == BigInt(0) => (When(false.B) +: preds, None)
  //          case other: WhenBegin =>
  //            sys.error(s"Something went horribly wrong! I was expecting ${other.pred} to be a lit or a bool!")
  //          case _: WhenEnd => (preds.tail, Some(preds.head))
  //          case AltBegin(_) if o.isDefined => (o.get.not +: preds, o)
  //          case _: AltBegin =>
  //            sys.error(s"Something went horribly wrong! I was expecting ${o} to be nonEmpty!")
  //          case OtherwiseEnd(_, _) => (preds.tail, None)
  //          case other =>
  //            processCommand(cmd, preds)
  //            (preds, o)
  //        }
  //      }
  //  }
  //}

}
