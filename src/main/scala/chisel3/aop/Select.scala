// SPDX-License-Identifier: Apache-2.0

package chisel3.aop

import chisel3._
import chisel3.internal.{HasId, PseudoModule}
import chisel3.experimental.BaseModule
import chisel3.internal.firrtl.{Definition => DefinitionIR, _}
import chisel3.experimental.hierarchy.core._
import chisel3.experimental.hierarchy.ModuleClone
import chisel3.reflect.DataMirror
import firrtl.annotations.ReferenceTarget

import scala.reflect.runtime.universe.TypeTag
import scala.collection.mutable

/** Use to select Chisel components in a module, after that module has been constructed
  * Useful for adding additional Chisel annotations or for use within an [[Aspect]]
  */
object Select {

  /** Return just leaf components of expanded node
    *
    * @param d Component to find leafs if aggregate typed. Intermediate fields/indicies are not included
    */
  @deprecated("Use DataMirror.collectLeafMembers instead")
  def getLeafs(d: Data): Seq[Data] = DataMirror.collectLeafMembers(d)

  /** Return all expanded components, including intermediate aggregate nodes
    *
    * @param d Component to find leafs if aggregate typed. Intermediate fields/indicies ARE included
    */
  @deprecated("Use DataMirror.collectAllMembers instead")
  def getIntermediateAndLeafs(d: Data): Seq[Data] = DataMirror.collectAllMembers(d)

  /** Selects all instances/modules directly instantiated within given definition
    *
    * @param parent
    */
  def instancesIn(parent: Hierarchy[BaseModule]): Seq[Instance[BaseModule]] = {
    check(parent)
    implicit val mg = new chisel3.internal.MacroGenerated {}
    parent.proto._component.get match {
      case d: DefModule =>
        d.commands.collect {
          case d: DefInstance =>
            d.id match {
              case p: IsClone[_] =>
                parent._lookup { x => new Instance(Clone(p)).asInstanceOf[Instance[BaseModule]] }
              case other: BaseModule =>
                parent._lookup { x => other }
            }
        }
      case other => Nil
    }
  }

  /** Selects all Instances of instances/modules directly instantiated within given module, of provided type
    *
    * @note IMPORTANT: this function requires summoning a TypeTag[T], which will fail if T is an inner class.
    * @note IMPORTANT: this function ignores type parameters. E.g. instancesOf[List[Int]] would return List[String].
    *
    * @param parent hierarchy which instantiates the returned Definitions
    */
  def instancesOf[T <: BaseModule: TypeTag](parent: Hierarchy[BaseModule]): Seq[Instance[T]] = {
    check(parent)
    implicit val mg = new chisel3.internal.MacroGenerated {}
    parent.proto._component.get match {
      case d: DefModule =>
        d.commands.flatMap {
          case d: DefInstance =>
            d.id match {
              case p: IsClone[_] =>
                val i = parent._lookup { x => new Instance(Clone(p)).asInstanceOf[Instance[BaseModule]] }
                if (i.isA[T]) Some(i.asInstanceOf[Instance[T]]) else None
              case other: BaseModule =>
                val i = parent._lookup { x => other }
                if (i.isA[T]) Some(i.asInstanceOf[Instance[T]]) else None
            }
          case other => None
        }
      case other => Nil
    }
  }

  /** Selects all Instances directly and indirectly instantiated within given root hierarchy, of provided type
    *
    * @note IMPORTANT: this function requires summoning a TypeTag[T], which will fail if T is an inner class.
    * @note IMPORTANT: this function ignores type parameters. E.g. allInstancesOf[List[Int]] would return List[String].
    *
    * @param root top of the hierarchy to search for instances/modules of given type
    */
  def allInstancesOf[T <: BaseModule: TypeTag](root: Hierarchy[BaseModule]): Seq[Instance[T]] = {
    val soFar = if (root.isA[T]) Seq(root.toInstance.asInstanceOf[Instance[T]]) else Nil
    val allLocalInstances = instancesIn(root)
    soFar ++ (allLocalInstances.flatMap(allInstancesOf[T]))
  }

  /** Selects the Definitions of all instances/modules directly instantiated within given module
    *
    * @param parent
    */
  def definitionsIn(parent: Hierarchy[BaseModule]): Seq[Definition[BaseModule]] = {
    type DefType = Definition[BaseModule]
    implicit val mg = new chisel3.internal.MacroGenerated {}
    check(parent)
    val defs = parent.proto._component.get match {
      case d: DefModule =>
        d.commands.collect {
          case i: DefInstance =>
            i.id match {
              case p: IsClone[_] =>
                parent._lookup { x => new Definition(Proto(p.getProto)).asInstanceOf[Definition[BaseModule]] }
              case other: BaseModule =>
                parent._lookup { x => other.toDefinition }
            }
        }
      case other => Nil
    }
    val (_, defList) = defs.foldLeft((Set.empty[DefType], List.empty[DefType])) {
      case ((set, list), definition: Definition[BaseModule]) =>
        if (set.contains(definition)) (set, list) else (set + definition, definition +: list)
    }
    defList.reverse
  }

  /** Selects all Definitions of instances/modules directly instantiated within given module, of provided type
    *
    * @note IMPORTANT: this function requires summoning a TypeTag[T], which will fail if T is an inner class.
    * @note IMPORTANT: this function ignores type parameters. E.g. definitionsOf[List[Int]] would return List[String].
    *
    * @param parent hierarchy which instantiates the returned Definitions
    */
  def definitionsOf[T <: BaseModule: TypeTag](parent: Hierarchy[BaseModule]): Seq[Definition[T]] = {
    check(parent)
    implicit val mg = new chisel3.internal.MacroGenerated {}
    type DefType = Definition[T]
    val defs = parent.proto._component.get match {
      case d: DefModule =>
        d.commands.flatMap {
          case d: DefInstance =>
            d.id match {
              case p: IsClone[_] =>
                val d = parent._lookup { x => new Definition(Clone(p)).asInstanceOf[Definition[BaseModule]] }
                if (d.isA[T]) Some(d.asInstanceOf[Definition[T]]) else None
              case other: BaseModule =>
                val d = parent._lookup { x => other.toDefinition }
                if (d.isA[T]) Some(d.asInstanceOf[Definition[T]]) else None
            }
          case other => None
        }
    }
    val (_, defList) = defs.foldLeft((Set.empty[DefType], List.empty[DefType])) {
      case ((set, list), definition: Definition[T]) =>
        if (set.contains(definition)) (set, list) else (set + definition, definition +: list)
    }
    defList.reverse
  }

  /** Selects all Definition's directly and indirectly instantiated within given root hierarchy, of provided type
    *
    * @note IMPORTANT: this function requires summoning a TypeTag[T], which will fail if T is an inner class, i.e.
    *   a class defined within another class.
    * @note IMPORTANT: this function ignores type parameters. E.g. allDefinitionsOf[List[Int]] would return List[String].
    *
    * @param root top of the hierarchy to search for definitions of given type
    */
  def allDefinitionsOf[T <: BaseModule: TypeTag](root: Hierarchy[BaseModule]): Seq[Definition[T]] = {
    type DefType = Definition[T]
    val allDefSet = mutable.HashSet[Definition[BaseModule]]()
    val defSet = mutable.HashSet[DefType]()
    val defList = mutable.ArrayBuffer[DefType]()
    def rec(hier: Definition[BaseModule]): Unit = {
      if (hier.isA[T] && !defSet.contains(hier.asInstanceOf[DefType])) {
        defSet += hier.asInstanceOf[DefType]
        defList += hier.asInstanceOf[DefType]
      }
      allDefSet += hier
      val allDefs = definitionsIn(hier)
      allDefs.collect {
        case d if !allDefSet.contains(d) => rec(d)
      }
    }
    rec(root.toDefinition)
    defList.toList
  }

  /** Collects all components selected by collector within module and all children modules it instantiates
    *   directly or indirectly
    * Accepts a collector function, rather than a collector partial function (see [[collectDeep]])
    *
    * @note This API will not work with the new experimental hierarchy package. Instead, use allInstancesOf or allDefinitionsOf.
    *
    * @param module Module to collect components, as well as all children module it directly and indirectly instantiates
    * @param collector Collector function to pick, given a module, which components to collect
    * @param tag Required for generics to work, should ignore this
    * @tparam T Type of the component that will be collected
    */
  def getDeep[T](module: BaseModule)(collector: BaseModule => Seq[T]): Seq[T] = {
    check(module)
    val myItems = collector(module)
    val deepChildrenItems = instances(module).flatMap { i =>
      getDeep(i)(collector)
    }
    myItems ++ deepChildrenItems
  }

  /** Collects all components selected by collector within module and all children modules it instantiates
    *   directly or indirectly
    * Accepts a collector partial function, rather than a collector function (see [[getDeep]])
    *
    * @note This API will not work with the new experimental hierarchy package. Instead, use allInstancesOf or allDefinitionsOf.
    *
    * @param module Module to collect components, as well as all children module it directly and indirectly instantiates
    * @param collector Collector partial function to pick, given a module, which components to collect
    * @param tag Required for generics to work, should ignore this
    * @tparam T Type of the component that will be collected
    */
  def collectDeep[T](module: BaseModule)(collector: PartialFunction[BaseModule, T]): Iterable[T] = {
    check(module)
    val myItems = collector.lift(module)
    val deepChildrenItems = instances(module).flatMap { i =>
      collectDeep(i)(collector)
    }
    myItems ++ deepChildrenItems
  }

  /** Selects all modules directly instantiated within given module
    *
    * @note This API will not work with the new experimental hierarchy package. Instead, use instancesIn or definitionsIn.
    *
    * @param module
    */
  def instances(module: BaseModule): Seq[BaseModule] = {
    check(module)
    module._component.get match {
      case d: DefModule =>
        d.commands.flatMap {
          case i: DefInstance =>
            i.id match {
              case m: ModuleClone[_] if !m._madeFromDefinition => None
              case _: PseudoModule =>
                throw new Exception(
                  "instances, collectDeep, and getDeep are currently incompatible with Definition/Instance!"
                )
              case other => Some(other)
            }
          case _ => None
        }
      case other => Nil
    }
  }

  /** Selects all registers directly instantiated within given module
    * @param module
    */
  def registers(module: BaseModule): Seq[Data] = {
    check(module)
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case r: DefReg     => r.id
      case r: DefRegInit => r.id
    }
  }

  /** Selects all Data ios on a given module
    *
    * Note that Property ios are not returned.
    *
    * @param module
    */
  def ios(module: BaseModule): Seq[Data] = {
    check(module)
    module._component.get.asInstanceOf[DefModule].ports.map(_.id).collect { case (d: Data) => d }
  }

  /** Selects all ios directly on a given Instance or Definition of a module
    * @param parent the Definition or Instance to get the IOs of
    */
  def ios[T <: BaseModule](parent: Hierarchy[T]): Seq[Data] = {
    check(parent)
    implicit val mg = new chisel3.internal.MacroGenerated {}
    parent._lookup { x => ios(parent.proto) }
  }

  /** Selects all SyncReadMems directly contained within given module
    * @param module
    */
  def syncReadMems(module: BaseModule): Seq[SyncReadMem[_]] = {
    check(module)
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case r: DefSeqMemory => r.id.asInstanceOf[SyncReadMem[_]]
    }
  }

  /** Selects all Mems directly contained within given module
    * @param module
    */
  def mems(module: BaseModule): Seq[Mem[_]] = {
    check(module)
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case r: DefMemory => r.id.asInstanceOf[Mem[_]]
    }
  }

  /** Selects all arithmetic or logical operators directly instantiated within given module
    * @param module
    */
  def ops(module: BaseModule): Seq[(String, Data)] = {
    check(module)
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case d: DefPrim[_] => (d.op.name, d.id)
    }
  }

  /** Selects a kind of arithmetic or logical operator directly instantiated within given module
    * The kind of operators are contained in `chisel3.internal.firrtl.PrimOp`
    * @param opKind the kind of operator, e.g. "mux", "add", or "bits"
    * @param module
    */
  def ops(opKind: String)(module: BaseModule): Seq[Data] = {
    check(module)
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case d: DefPrim[_] if d.op.name == opKind => d.id
    }
  }

  /** Selects all wires in a module
    * @param module
    */
  def wires(module: BaseModule): Seq[Data] = {
    check(module)
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case r: DefWire => r.id
    }
  }

  /** Selects all memory ports, including their direction and memory
    * @param module
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
    */
  def memPorts(dir: MemPortDirection)(module: BaseModule): Seq[(Data, MemBase[_])] = {
    check(module)
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case r: DefMemPort[_] if r.dir == dir => (r.id, r.source.id.asInstanceOf[MemBase[_ <: Data]])
    }
  }

  /** Selects all components who have been set to be invalid, even if they are later connected to
    * @param module
    */
  def invalids(module: BaseModule): Seq[Data] = {
    check(module)
    module._component.get.asInstanceOf[DefModule].commands.collect {
      case DefInvalid(_, arg) => getData(arg)
    }
  }

  /** Selects all components who are attached to a given signal, within a module
    * @param module
    */
  def attachedTo(module: BaseModule)(signal: Data): Set[Data] = {
    check(module)
    module._component.get
      .asInstanceOf[DefModule]
      .commands
      .collect {
        case Attach(_, seq) if seq.contains(signal) => seq
      }
      .flatMap { seq => seq.map(_.id.asInstanceOf[Data]) }
      .toSet
  }

  /** Selects all connections to a signal or its parent signal(s) (if the signal is an element of an aggregate signal)
    * The when predicates surrounding each connection are included in the returned values
    *
    * E.g. if signal = io.foo.bar, connectionsTo will return all connections to io, io.foo, and io.bar
    * @param module
    * @param signal
    */
  def connectionsTo(module: BaseModule)(signal: Data): Seq[PredicatedConnect] = {
    check(module)
    val sensitivitySignals = DataMirror.collectAllMembers(signal).toSet
    val predicatedConnects = mutable.ArrayBuffer[PredicatedConnect]()
    val isPort = module._component.get
      .asInstanceOf[DefModule]
      .ports
      .flatMap { port =>
        port.id match {
          case d: Data => DataMirror.collectAllMembers(d)
          case _ => Nil
        }
      }
      .contains(signal)
    var prePredicates: Seq[Predicate] = Nil
    var seenDef = isPort
    searchWhens(
      module,
      (cmd: Command, preds) => {
        cmd match {
          case cmd: DefinitionIR if cmd.id.isInstanceOf[Data] =>
            val x = DataMirror.collectAllMembers(cmd.id.asInstanceOf[Data])
            if (x.contains(signal)) prePredicates = preds
          case Connect(_, loc @ Node(d: Data), exp) =>
            val effected = getEffected(loc).toSet
            if (sensitivitySignals.intersect(effected).nonEmpty) {
              val expData = getData(exp)
              prePredicates.reverse
                .zip(preds.reverse)
                .foreach(x => assert(x._1 == x._2, s"Prepredicates $x must match for signal $signal"))
              predicatedConnects += PredicatedConnect(preds.dropRight(prePredicates.size), d, expData, isBulk = false)
            }
          case other =>
        }
      }
    )
    predicatedConnects.toSeq
  }

  /** Selects all stop statements, and includes the predicates surrounding the stop statement
    *
    * @param module
    */
  def stops(module: BaseModule): Seq[Stop] = {
    val stops = mutable.ArrayBuffer[Stop]()
    searchWhens(
      module,
      (cmd: Command, preds: Seq[Predicate]) => {
        cmd match {
          case chisel3.internal.firrtl.Stop(_, _, clock, ret) =>
            stops += Stop(preds, ret, getId(clock).asInstanceOf[Clock])
          case other =>
        }
      }
    )
    stops.toSeq
  }

  /** Selects all printf statements, and includes the predicates surrounding the printf statement
    *
    * @param module
    */
  def printfs(module: BaseModule): Seq[Printf] = {
    val printfs = mutable.ArrayBuffer[Printf]()
    searchWhens(
      module,
      (cmd: Command, preds: Seq[Predicate]) => {
        cmd match {
          case chisel3.internal.firrtl.Printf(id, _, clock, pable) =>
            printfs += Printf(id, preds, pable, getId(clock).asInstanceOf[Clock])
          case other =>
        }
      }
    )
    printfs.toSeq
  }

  // Checks that a module has finished its construction
  private def check(module: BaseModule): Unit = {
    require(module.isClosed, "Can't use Selector on modules that have not finished construction!")
    require(module._component.isDefined, "Can't use Selector on modules that don't have components!")
  }
  private def check(hierarchy: Hierarchy[BaseModule]): Unit = check(hierarchy.proto)

  // Given a loc, return all subcomponents of id that could be assigned to in connect
  private def getEffected(a: Arg): Seq[Data] = a match {
    case Node(id: Data) => DataMirror.collectAllMembers(id)
    case Slot(imm, name) => Seq(imm.id.asInstanceOf[Record].elements(name))
    case Index(imm, _)   => getEffected(imm)
    case _               => throw new InternalErrorException("Match error: a=$a")
  }

  // Given an arg, return the corresponding id. Don't use on a loc of a connect.
  private def getId(a: Arg): HasId = a match {
    case Node(id) => id
    case l: ULit => l.num.U(l.w)
    case l: SLit => l.num.S(l.w)
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
    case e: ChiselException =>
      i.getOptionRef.get match {
        case l: LitArg => l.num.intValue.toString
        case _ => throw new InternalErrorException("Match error: i.getOptionRef.get=${i.getOptionRef.get}")
      }
  }

  // Collects when predicates as it searches through a module, then applying processCommand to non-when related commands
  private def searchWhens(module: BaseModule, processCommand: (Command, Seq[Predicate]) => Unit) = {
    check(module)
    module._component.get.asInstanceOf[DefModule].commands.foldLeft((Seq.empty[Predicate], Option.empty[Predicate])) {
      (blah, cmd) =>
        (blah, cmd) match {
          case ((preds, o), cmd) =>
            cmd match {
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
    def not:       WhenNot = WhenNot(bool)
    def serialize: String = s"${getName(bool)}"
  }

  /** Used to represent the `otherwise` predicate of a [[chisel3.when]]
    *
    * @param bool the when predicate corresponding to this otherwise predicate
    */
  case class WhenNot(bool: Bool) extends Predicate {
    def not:       When = When(bool)
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
      s"$moduleTarget: when(${preds.map(_.serialize).mkString(" & ")}): ${getName(loc)} ${if (isBulk) "<>" else ":="} ${getName(exp)}"
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
  case class Printf(id: printf.Printf, preds: Seq[Predicate], pable: Printable, clock: Clock) extends Serializeable {
    def serialize: String = {
      s"printf when(${preds.map(_.serialize).mkString(" & ")}) on ${getName(clock)}: $pable"
    }
  }
}
