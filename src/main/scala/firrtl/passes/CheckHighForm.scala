// SPDX-License-Identifier: Apache-2.0

package firrtl.passes

import firrtl._
import firrtl.ir._
import firrtl.PrimOps._
import firrtl.Utils._
import firrtl.traversals.Foreachers._
import firrtl.options.Dependency

trait CheckHighFormLike { this: Pass =>
  type NameSet = collection.mutable.HashSet[String]

  private object ScopeView {
    def apply(): ScopeView = new ScopeView(new NameSet, List(new NameSet))
  }

  private class ScopeView private (moduleNS: NameSet, scopes: List[NameSet]) {
    require(scopes.nonEmpty)
    def declare(name: String): Unit = {
      moduleNS += name
      scopes.head += name
    }
    // ensures that the name cannot be used again, but prevent references to this name
    def addToNamespace(name: String): Unit = {
      moduleNS += name
    }
    def expandMPortVisibility(port: CDefMPort): Unit = {
      // Legacy CHIRRTL ports are visible in any scope where their parent memory is visible
      scopes.find(_.contains(port.mem)).getOrElse(scopes.head) += port.name
    }
    def legalDecl(name: String): Boolean = !moduleNS.contains(name)
    def legalRef(name:  String): Boolean = scopes.exists(_.contains(name))
    def childScope(): ScopeView = new ScopeView(moduleNS, new NameSet +: scopes)
  }

  // Custom Exceptions
  class NotUniqueException(info: Info, mname: String, name: String)
      extends PassException(s"$info: [module $mname] Reference $name does not have a unique name.")
  class InvalidLOCException(info: Info, mname: String)
      extends PassException(
        s"$info: [module $mname] Invalid connect to an expression that is not a reference or a WritePort."
      )
  class NegUIntException(info: Info, mname: String)
      extends PassException(s"$info: [module $mname] UIntLiteral cannot be negative.")
  class UndeclaredReferenceException(info: Info, mname: String, name: String)
      extends PassException(s"$info: [module $mname] Reference $name is not declared.")
  class PoisonWithFlipException(info: Info, mname: String, name: String)
      extends PassException(s"$info: [module $mname] Poison $name cannot be a bundle type with flips.")
  class MemWithFlipException(info: Info, mname: String, name: String)
      extends PassException(s"$info: [module $mname] Memory $name cannot be a bundle type with flips.")
  class IllegalMemLatencyException(info: Info, mname: String, name: String)
      extends PassException(
        s"$info: [module $mname] Memory $name must have non-negative read latency and positive write latency."
      )
  class RegWithFlipException(info: Info, mname: String, name: String)
      extends PassException(s"$info: [module $mname] Register $name cannot be a bundle type with flips.")
  class InvalidAccessException(info: Info, mname: String)
      extends PassException(s"$info: [module $mname] Invalid access to non-reference.")
  class ModuleNameNotUniqueException(info: Info, mname: String)
      extends PassException(s"$info: Repeat definition of module $mname")
  class DefnameConflictException(info: Info, mname: String, defname: String)
      extends PassException(s"$info: defname $defname of extmodule $mname conflicts with an existing module")
  class DefnameDifferentPortsException(info: Info, mname: String, defname: String)
      extends PassException(
        s"""$info: ports of extmodule $mname with defname $defname are different for an extmodule with the same defname"""
      )
  class ModuleNotDefinedException(info: Info, mname: String, name: String)
      extends PassException(s"$info: Module $name is not defined.")
  class IncorrectNumArgsException(info: Info, mname: String, op: String, n: Int)
      extends PassException(s"$info: [module $mname] Primop $op requires $n expression arguments.")
  class IncorrectNumConstsException(info: Info, mname: String, op: String, n: Int)
      extends PassException(s"$info: [module $mname] Primop $op requires $n integer arguments.")
  class NegWidthException(info: Info, mname: String)
      extends PassException(s"$info: [module $mname] Width cannot be negative.")
  class NegVecSizeException(info: Info, mname: String)
      extends PassException(s"$info: [module $mname] Vector type size cannot be negative.")
  class NegMemSizeException(info: Info, mname: String)
      extends PassException(s"$info: [module $mname] Memory size cannot be negative or zero.")
  class BadPrintfException(info: Info, mname: String, x: Char)
      extends PassException(s"$info: [module $mname] Bad printf format: " + "\"%" + x + "\"")
  class BadPrintfTrailingException(info: Info, mname: String)
      extends PassException(s"$info: [module $mname] Bad printf format: trailing " + "\"%\"")
  class BadPrintfIncorrectNumException(info: Info, mname: String)
      extends PassException(s"$info: [module $mname] Bad printf format: incorrect number of arguments")
  class InstanceLoop(info: Info, mname: String, loop: String)
      extends PassException(s"$info: [module $mname] Has instance loop $loop")
  class NoTopModuleException(info: Info, name: String)
      extends PassException(s"$info: A single module must be named $name.")
  class NegArgException(info: Info, mname: String, op: String, value: BigInt)
      extends PassException(s"$info: [module $mname] Primop $op argument $value < 0.")
  class LsbLargerThanMsbException(info: Info, mname: String, op: String, lsb: BigInt, msb: BigInt)
      extends PassException(s"$info: [module $mname] Primop $op lsb $lsb > $msb.")
  class ResetInputException(info: Info, mname: String, expr: Expression)
      extends PassException(s"$info: [module $mname] Abstract Reset not allowed as top-level input: ${expr.serialize}")
  class ResetExtModuleOutputException(info: Info, mname: String, expr: Expression)
      extends PassException(s"$info: [module $mname] Abstract Reset not allowed as ExtModule output: ${expr.serialize}")

  // Is Chirrtl allowed for this check? If not, return an error
  def errorOnChirrtl(info: Info, mname: String, s: Statement): Option[PassException]

  def run(c: Circuit): Circuit = {
    val errors = new Errors()
    val moduleGraph = new ModuleGraph
    val moduleNames = (c.modules.map(_.name)).toSet

    val intModuleNames = c.modules.view.collect({ case m: Module => m.name }).toSet

    c.modules.groupBy(_.name).filter(_._2.length > 1).flatMap(_._2).foreach { m =>
      errors.append(new ModuleNameNotUniqueException(m.info, m.name))
    }

    /** Strip all widths from types */
    def stripWidth(tpe: Type): Type = tpe match {
      case a: GroundType    => a.mapWidth(_ => UnknownWidth)
      case a: AggregateType => a.mapType(stripWidth)
    }

    val extmoduleCollidingPorts = c.modules.collect {
      case a: ExtModule => a
    }.groupBy(a => (a.defname, a.params.nonEmpty))
      .map {
        /* There are no parameters, so all ports must match exactly. */
        case (k @ (_, false), a) =>
          k -> a.map(_.copy(info = NoInfo)).map(_.ports.map(_.copy(info = NoInfo))).toSet
        /* If there are parameters, then only port names must match because parameters could parameterize widths.
         * This means that this check cannot produce false positives, but can have false negatives.
         */
        case (k @ (_, true), a) =>
          k -> a.map(_.copy(info = NoInfo)).map(_.ports.map(_.copy(info = NoInfo).mapType(stripWidth))).toSet
      }
      .filter(_._2.size > 1)

    c.modules.collect {
      case a: ExtModule =>
        a match {
          case ExtModule(info, name, _, defname, _) if (intModuleNames.contains(defname)) =>
            errors.append(new DefnameConflictException(info, name, defname))
          case _ =>
        }
        a match {
          case ExtModule(info, name, _, defname, params)
              if extmoduleCollidingPorts.contains((defname, params.nonEmpty)) =>
            errors.append(new DefnameDifferentPortsException(info, name, defname))
          case _ =>
        }
    }

    def checkHighFormPrimop(info: Info, mname: String, e: DoPrim): Unit = {
      def correctNum(ne: Option[Int], nc: Int): Unit = {
        ne match {
          case Some(i) if e.args.length != i =>
            errors.append(new IncorrectNumArgsException(info, mname, e.op.toString, i))
          case _ => // Do Nothing
        }
        if (e.consts.length != nc)
          errors.append(new IncorrectNumConstsException(info, mname, e.op.toString, nc))
      }

      def nonNegativeConsts(): Unit = {
        e.consts.filter(_ < 0).foreach { negC =>
          errors.append(new NegArgException(info, mname, e.op.toString, negC))
        }
      }

      e.op match {
        case Add | Sub | Mul | Div | Rem | Lt | Leq | Gt | Geq | Eq | Neq | Dshl | Dshr | And | Or | Xor | Cat | Dshlw |
            Clip | Wrap | Squeeze =>
          correctNum(Option(2), 0)
        case AsUInt | AsSInt | AsClock | AsAsyncReset | Cvt | Neq | Not =>
          correctNum(Option(1), 0)
        case AsFixedPoint | SetP =>
          correctNum(Option(1), 1)
        case Shl | Shr | Pad | Head | Tail | IncP | DecP =>
          correctNum(Option(1), 1)
          nonNegativeConsts()
        case Bits =>
          correctNum(Option(1), 2)
          nonNegativeConsts()
          if (e.consts.length == 2) {
            val (msb, lsb) = (e.consts(0), e.consts(1))
            if (lsb > msb) {
              errors.append(new LsbLargerThanMsbException(info, mname, e.op.toString, lsb, msb))
            }
          }
        case AsInterval =>
          correctNum(Option(1), 3)
        case Andr | Orr | Xorr | Neg =>
          correctNum(None, 0)
      }
    }

    def checkFstring(info: Info, mname: String, s: StringLit, i: Int): Unit = {
      val validFormats = "bdxc"
      val (percent, npercents) = s.string.foldLeft((false, 0)) {
        case ((percentx, n), b) if percentx && (validFormats contains b) =>
          (false, n + 1)
        case ((percentx, n), b) if percentx && b != '%' =>
          errors.append(new BadPrintfException(info, mname, b.toChar))
          (false, n)
        case ((percentx, n), b) =>
          (if (b == '%') !percentx else false /* %% -> percentx = false */, n)
      }
      if (percent) errors.append(new BadPrintfTrailingException(info, mname))
      if (npercents != i) errors.append(new BadPrintfIncorrectNumException(info, mname))
    }

    def checkValidLoc(info: Info, mname: String, e: Expression): Unit = e match {
      case _: UIntLiteral | _: SIntLiteral | _: DoPrim =>
        errors.append(new InvalidLOCException(info, mname))
      case _ => // Do Nothing
    }

    def checkHighFormW(info: Info, mname: => String)(w: Width): Unit = {
      w match {
        case wx: IntWidth if wx.width < 0 => errors.append(new NegWidthException(info, mname))
        case wx => // Do nothing
      }
    }

    def checkHighFormT(info: Info, mname: => String)(t: Type): Unit = {
      t.foreach(checkHighFormT(info, mname))
      t match {
        case tx: VectorType if tx.size < 0 =>
          errors.append(new NegVecSizeException(info, mname))
        case _: IntervalType =>
        case _ => t.foreach(checkHighFormW(info, mname))
      }
    }

    def validSubexp(info: Info, mname: String)(e: Expression): Unit = {
      e match {
        case _: Reference | _: SubField | _: SubIndex | _: SubAccess => // No error
        case _: WRef | _: WSubField | _: WSubIndex | _: WSubAccess | _: Mux | _: ValidIf => // No error
        case _ => errors.append(new InvalidAccessException(info, mname))
      }
    }

    def checkHighFormE(info: Info, mname: String, names: ScopeView)(e: Expression): Unit = {
      e match {
        case ex: Reference if !names.legalRef(ex.name) =>
          errors.append(new UndeclaredReferenceException(info, mname, ex.name))
        case ex: WRef if !names.legalRef(ex.name) =>
          errors.append(new UndeclaredReferenceException(info, mname, ex.name))
        case ex: UIntLiteral if ex.value < 0 =>
          errors.append(new NegUIntException(info, mname))
        case ex: DoPrim => checkHighFormPrimop(info, mname, ex)
        case _: Reference | _: WRef | _: UIntLiteral | _: Mux | _: ValidIf =>
        case ex: SubAccess => validSubexp(info, mname)(ex.expr)
        case ex => ex.foreach(validSubexp(info, mname))
      }
      e.foreach(checkHighFormW(info, mname + "/" + e.serialize))
      e.foreach(checkHighFormE(info, mname, names))
    }

    def checkName(info: Info, mname: String, names: ScopeView, canBeReference: Boolean)(name: String): Unit = {
      // Empty names are allowed for backwards compatibility reasons and
      // indicate that the entity has essentially no name.
      if (name.isEmpty) { assert(!canBeReference, "A statement with an empty name cannot be used as a reference!") }
      else {
        if (!names.legalDecl(name))
          errors.append(new NotUniqueException(info, mname, name))
        if (canBeReference) {
          names.declare(name)
        } else {
          names.addToNamespace(name)
        }
      }
    }

    def checkInstance(info: Info, child: String, parent: String): Unit = {
      if (!moduleNames(child))
        errors.append(new ModuleNotDefinedException(info, parent, child))
      // Check to see if a recursive module instantiation has occured
      val childToParent = moduleGraph.add(parent, child)
      if (childToParent.nonEmpty)
        errors.append(new InstanceLoop(info, parent, childToParent.mkString("->")))
    }

    def checkHighFormS(minfo: Info, mname: String, names: ScopeView)(s: Statement): Unit = {
      val info = get_info(s) match {
        case NoInfo => minfo
        case x      => x
      }
      val canBeReference = s match {
        case _: CanBeReferenced => true
        case _ => false
      }
      s.foreach(checkName(info, mname, names, canBeReference))
      s match {
        case DefRegister(info, name, tpe, _, reset, init) =>
          if (hasFlip(tpe))
            errors.append(new RegWithFlipException(info, mname, name))
        case sx: DefMemory =>
          if (sx.readLatency < 0 || sx.writeLatency <= 0)
            errors.append(new IllegalMemLatencyException(info, mname, sx.name))
          if (hasFlip(sx.dataType))
            errors.append(new MemWithFlipException(info, mname, sx.name))
          if (sx.depth <= 0)
            errors.append(new NegMemSizeException(info, mname))
        case sx:    DefInstance    => checkInstance(info, mname, sx.module)
        case sx:    Connect        => checkValidLoc(info, mname, sx.loc)
        case sx:    PartialConnect => checkValidLoc(info, mname, sx.loc)
        case sx:    Print          => checkFstring(info, mname, sx.string, sx.args.length)
        case _:     CDefMemory => errorOnChirrtl(info, mname, s).foreach { e => errors.append(e) }
        case mport: CDefMPort =>
          errorOnChirrtl(info, mname, s).foreach { e => errors.append(e) }
          names.expandMPortVisibility(mport)
        case sx => // Do Nothing
      }
      s.foreach(checkHighFormT(info, mname))
      s.foreach(checkHighFormE(info, mname, names))
      s match {
        case Conditionally(_, _, conseq, alt) =>
          checkHighFormS(minfo, mname, names.childScope())(conseq)
          checkHighFormS(minfo, mname, names.childScope())(alt)
        case _ => s.foreach(checkHighFormS(minfo, mname, names))
      }
    }

    def checkHighFormP(mname: String, names: ScopeView)(p: Port): Unit = {
      if (!names.legalDecl(p.name))
        errors.append(new NotUniqueException(NoInfo, mname, p.name))
      names.declare(p.name)
      checkHighFormT(p.info, mname)(p.tpe)
    }

    // Search for ResetType Ports of direction
    def findBadResetTypePorts(m: DefModule, dir: Direction): Seq[(Port, Expression)] = {
      val bad = to_flow(dir)
      for {
        port <- m.ports
        ref = WRef(port).copy(flow = to_flow(port.direction))
        expr <- create_exps(ref)
        if ((expr.tpe == ResetType) && (flow(expr) == bad))
      } yield (port, expr)
    }

    def checkHighFormM(m: DefModule): Unit = {
      val names = ScopeView()
      m.foreach(checkHighFormP(m.name, names))
      m.foreach(checkHighFormS(m.info, m.name, names))
      m match {
        case _:   Module =>
        case ext: ExtModule =>
          for ((port, expr) <- findBadResetTypePorts(ext, Output)) {
            errors.append(new ResetExtModuleOutputException(port.info, ext.name, expr))
          }
      }
    }

    c.modules.foreach(checkHighFormM)
    c.modules.filter(_.name == c.main) match {
      case Seq(topMod) =>
        for ((port, expr) <- findBadResetTypePorts(topMod, Input)) {
          errors.append(new ResetInputException(port.info, topMod.name, expr))
        }
      case _ => errors.append(new NoTopModuleException(c.info, c.main))
    }
    errors.trigger()
    c
  }
}

object CheckHighForm extends Pass with CheckHighFormLike {

  override def prerequisites = firrtl.stage.Forms.MinimalHighForm

  override def optionalPrerequisiteOf =
    Seq(
      Dependency(passes.ResolveKinds),
      Dependency(passes.InferTypes),
      Dependency(passes.ResolveFlows),
      Dependency[passes.InferWidths],
      Dependency[transforms.InferResets]
    )

  override def invalidates(a: Transform) = false

  class IllegalChirrtlMemException(info: Info, mname: String, name: String)
      extends PassException(s"$info: [module $mname] Memory $name has not been properly lowered from Chirrtl IR.")

  def errorOnChirrtl(info: Info, mname: String, s: Statement): Option[PassException] = {
    val memName = s match {
      case cm: CDefMemory => cm.name
      case cp: CDefMPort  => cp.mem
    }
    Some(new IllegalChirrtlMemException(info, mname, memName))
  }
}
