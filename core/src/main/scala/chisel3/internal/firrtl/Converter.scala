// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.firrtl

import chisel3._
import chisel3.experimental._
import chisel3.experimental.{NoSourceInfo, SourceInfo, SourceLine, UnlocatableSourceInfo}
import chisel3.properties.Property
import firrtl.{ir => fir}
import chisel3.internal.{castToInt, throwException, HasId}
import chisel3.EnumType
import scala.annotation.{nowarn, tailrec}
import scala.collection.immutable.{Queue, VectorBuilder}

@nowarn("msg=class Port") // delete when Port becomes private
private[chisel3] object Converter {
  // TODO modeled on unpack method on Printable, refactor?
  def unpack(pable: Printable, ctx: Component): (String, Seq[Arg]) = pable match {
    case Printables(pables) =>
      val (fmts, args) = pables.map(p => unpack(p, ctx)).unzip
      (fmts.mkString, args.flatten.toSeq)
    case PString(str) => (str.replaceAll("%", "%%"), List.empty)
    case format: FirrtlFormat =>
      ("%" + format.specifier, List(format.bits.ref))
    case Name(data)     => (data.ref.name, List.empty)
    case FullName(data) => (data.ref.fullName(ctx), List.empty)
    case Percent        => ("%%", List.empty)
  }

  private def reportInternalError(msg: String): Nothing = {
    val link = "https://github.com/chipsalliance/chisel3/issues/new"
    val fullMsg = s"Internal Error! $msg This is a bug in Chisel, please file an issue at '$link'"
    throwException(fullMsg)
  }

  def getRef(id: HasId, sourceInfo: SourceInfo): Arg =
    id.getOptionRef.getOrElse {
      val module = id._parent.map(m => s" '$id' was defined in module '$m'.").getOrElse("")
      val loc = sourceInfo.makeMessage(" " + _)
      reportInternalError(s"Could not get ref for '$id'$loc!$module")
    }

  private def clonedModuleIOError(mod: BaseModule, name: String, sourceInfo: SourceInfo): Nothing = {
    val loc = sourceInfo.makeMessage(" " + _)
    reportInternalError(s"Trying to convert a cloned IO of $mod inside of $mod itself$loc!")
  }

  def convert(info: SourceInfo): fir.Info = info match {
    case _:  NoSourceInfo => fir.NoInfo
    case sl: SourceLine   => fir.FileInfo.fromUnescaped(sl.serialize)
  }

  def convert(op: PrimOp): fir.PrimOp = firrtl.PrimOps.fromString(op.name)

  def convert(dir: MemPortDirection): firrtl.MPortDir = dir match {
    case MemPortDirection.INFER => firrtl.MInfer
    case MemPortDirection.READ  => firrtl.MRead
    case MemPortDirection.WRITE => firrtl.MWrite
    case MemPortDirection.RDWR  => firrtl.MReadWrite
  }

  // TODO
  //   * Memoize?
  //   * Move into the Chisel IR?
  def convert(arg: Arg, ctx: Component, info: SourceInfo): fir.Expression = arg match {
    case Node(id) =>
      convert(getRef(id, info), ctx, info)
    case Ref(name) =>
      fir.Reference(name, fir.UnknownType)
    case Slot(imm, name) =>
      fir.SubField(convert(imm, ctx, info), name, fir.UnknownType)
    case OpaqueSlot(imm) =>
      convert(imm, ctx, info)
    case Index(imm, ILit(idx)) =>
      fir.SubIndex(convert(imm, ctx, info), castToInt(idx, "Index"), fir.UnknownType)
    case Index(imm, value) =>
      fir.SubAccess(convert(imm, ctx, info), convert(value, ctx, info), fir.UnknownType)
    case ModuleIO(mod, name) =>
      if (mod eq ctx.id) fir.Reference(name, fir.UnknownType)
      else fir.SubField(fir.Reference(getRef(mod, info).name, fir.UnknownType), name, fir.UnknownType)
    case ModuleCloneIO(mod, name) =>
      if (mod eq ctx.id) clonedModuleIOError(mod, name, info)
      else fir.Reference(name)
    case u @ ULit(n, UnknownWidth()) =>
      fir.UIntLiteral(n, fir.IntWidth(u.minWidth))
    case ULit(n, w) =>
      fir.UIntLiteral(n, convert(w))
    case slit @ SLit(n, w) =>
      fir.SIntLiteral(n, convert(w))
      val unsigned = if (n < 0) (BigInt(1) << slit.width.get) + n else n
      val uint = convert(ULit(unsigned, slit.width), ctx, info)
      fir.DoPrim(firrtl.PrimOps.AsSInt, Seq(uint), Seq.empty, fir.UnknownType)
    // TODO Simplify
    case lit: ILit =>
      throw new InternalErrorException(s"Unexpected ILit: $lit")
    case PropertyLit(tpe, lit) => tpe.convert(lit, ctx, info)
    case e @ ProbeExpr(probe) =>
      fir.ProbeExpr(convert(probe, ctx, info))
    case e @ RWProbeExpr(probe) =>
      fir.RWProbeExpr(convert(probe, ctx, info))
    case e @ ProbeRead(probe) =>
      fir.ProbeRead(convert(probe, ctx, info))
    case other =>
      throw new InternalErrorException(s"Unexpected type in convert $other")
  }

  /** Convert Commands that map 1:1 to Statements */
  def convertSimpleCommand(cmd: Command, ctx: Component, typeAliases: Seq[String]): Option[fir.Statement] = cmd match {
    case e: DefPrim[_] =>
      val consts = e.args.collect { case ILit(i) => i }
      val args = e.args.flatMap {
        case _: ILit => None
        case other => Some(convert(other, ctx, e.sourceInfo))
      }
      val expr = e.op.name match {
        case "mux" =>
          assert(args.size == 3, s"Mux with unexpected args: $args")
          fir.Mux(args(0), args(1), args(2), fir.UnknownType)
        case _ =>
          fir.DoPrim(convert(e.op), args, consts, fir.UnknownType)
      }
      Some(fir.DefNode(convert(e.sourceInfo), e.name, expr))
    case e @ DefWire(info, id) =>
      Some(fir.DefWire(convert(info), e.name, extractType(id, info, typeAliases)))
    case e @ DefReg(info, id, clock) =>
      Some(
        fir.DefRegister(
          convert(info),
          e.name,
          extractType(id, info, typeAliases),
          convert(clock, ctx, info)
        )
      )
    case e @ DefRegInit(info, id, clock, reset, init) =>
      Some(
        fir.DefRegisterWithReset(
          convert(info),
          e.name,
          extractType(id, info, typeAliases),
          convert(clock, ctx, info),
          convert(reset, ctx, info),
          convert(init, ctx, info)
        )
      )
    case e @ DefMemory(info, id, t, size) =>
      Some(firrtl.CDefMemory(convert(info), e.name, extractType(t, info, typeAliases), size, false))
    case e @ DefSeqMemory(info, id, t, size, ruw) =>
      Some(firrtl.CDefMemory(convert(info), e.name, extractType(t, info, typeAliases), size, true, ruw))
    case e: DefMemPort[_] =>
      val info = e.sourceInfo
      Some(
        firrtl.CDefMPort(
          convert(e.sourceInfo),
          e.name,
          fir.UnknownType,
          e.source.fullName(ctx),
          Seq(convert(e.index, ctx, info), convert(e.clock, ctx, info)),
          convert(e.dir)
        )
      )
    case Connect(info, loc, exp) =>
      Some(fir.Connect(convert(info), convert(loc, ctx, info), convert(exp, ctx, info)))
    case PropAssign(info, loc, exp) =>
      Some(fir.PropAssign(convert(info), convert(loc, ctx, info), convert(exp, ctx, info)))
    case Attach(info, locs) =>
      Some(fir.Attach(convert(info), locs.map(l => convert(l, ctx, info))))
    case DefInvalid(info, arg) =>
      Some(fir.IsInvalid(convert(info), convert(arg, ctx, info)))
    case e @ DefInstance(info, id, _) =>
      Some(fir.DefInstance(convert(info), e.name, id.name))
    case e @ DefObject(info, obj) =>
      Some(fir.DefObject(convert(info), e.name, obj.className.name))
    case e @ Stop(_, info, clock, ret) =>
      Some(fir.Stop(convert(info), ret, convert(clock, ctx, info), firrtl.Utils.one, e.name))
    case e @ Printf(_, info, clock, pable) =>
      val (fmt, args) = unpack(pable, ctx)
      Some(
        fir.Print(
          convert(info),
          fir.StringLit(fmt),
          args.map(a => convert(a, ctx, info)),
          convert(clock, ctx, info),
          firrtl.Utils.one,
          e.name
        )
      )
    case e @ ProbeDefine(sourceInfo, sink, probeExpr) =>
      Some(fir.ProbeDefine(convert(sourceInfo), convert(sink, ctx, sourceInfo), convert(probeExpr, ctx, sourceInfo)))
    case e @ ProbeForceInitial(sourceInfo, probe, value) =>
      Some(fir.ProbeForceInitial(convert(sourceInfo), convert(probe, ctx, sourceInfo), convert(value, ctx, sourceInfo)))
    case e @ ProbeReleaseInitial(sourceInfo, probe) =>
      Some(fir.ProbeReleaseInitial(convert(sourceInfo), convert(probe, ctx, sourceInfo)))
    case e @ ProbeForce(sourceInfo, clock, cond, probe, value) =>
      Some(
        fir.ProbeForce(
          convert(sourceInfo),
          convert(clock, ctx, sourceInfo),
          convert(cond, ctx, sourceInfo),
          convert(probe, ctx, sourceInfo),
          convert(value, ctx, sourceInfo)
        )
      )
    case e @ ProbeRelease(sourceInfo, clock, cond, probe) =>
      Some(
        fir.ProbeRelease(
          convert(sourceInfo),
          convert(clock, ctx, sourceInfo),
          convert(cond, ctx, sourceInfo),
          convert(probe, ctx, sourceInfo)
        )
      )
    case e @ Verification(_, op, info, clk, pred, msg) =>
      val firOp = op match {
        case Formal.Assert => fir.Formal.Assert
        case Formal.Assume => fir.Formal.Assume
        case Formal.Cover  => fir.Formal.Cover
      }
      Some(
        fir.Verification(
          firOp,
          convert(info),
          convert(clk, ctx, info),
          convert(pred, ctx, info),
          firrtl.Utils.one,
          fir.StringLit(msg),
          e.name
        )
      )
    case _ => None
  }

  /** Trait used for tracking when or group regions. */
  private sealed trait RegionFrame

  /** Internal datastructure to help translate Chisel's flat Command structure to FIRRTL's AST
    *
    * In particular, when scoping is translated from flat with begin end to a nested datastructure
    *
    * @param when Current when Statement, holds info, condition, and consequence as they are
    *        available
    * @param outer Already converted Statements that precede the current when block in the scope in
    *        which the when is defined (ie. 1 level up from the scope inside the when)
    * @param alt Indicates if currently processing commands in the alternate (else) of the when scope
    */
  // TODO we should probably have a different structure in the IR to close elses
  private case class WhenFrame(when: fir.Conditionally, outer: VectorBuilder[fir.Statement], alt: Boolean)
      extends RegionFrame

  /** Internal datastructure to help convert optional groups to FIRRTL. */
  private case class GroupFrame(group: fir.GroupDefine, outer: VectorBuilder[fir.Statement]) extends RegionFrame

  /** Convert Chisel IR Commands into FIRRTL Statements
    *
    * @note ctx is needed because references to ports translate differently when referenced within
    *   the module in which they are defined vs. parent modules
    * @param cmds Chisel IR Commands to convert
    * @param ctx Component (Module) context within which we are translating
    * @param typeAliases Set of aliased type names to emit FIRRTL alias types for
    * @return FIRRTL Statement that is equivalent to the input cmds
    */
  def convert(cmds: Seq[Command], ctx: Component, typeAliases: Seq[String]): fir.Statement = {
    var stmts = new VectorBuilder[fir.Statement]()
    var scope: List[RegionFrame] = Nil
    var cmdsIt = cmds.iterator.buffered
    // Extra var because sometimes we want to push a Command to the head of cmdsIt
    // This is more efficient than changing the iterator
    var nextCmd: Command = null
    while (nextCmd != null || cmdsIt.hasNext) {
      val cmd = if (nextCmd != null) {
        val _nextCmd = nextCmd
        nextCmd = null
        _nextCmd
      } else {
        cmdsIt.next()
      }
      convertSimpleCommand(cmd, ctx, typeAliases) match {
        // Most Commands map 1:1
        case Some(stmt) =>
          stmts += stmt
        // When scoping logic does not map 1:1 and requires pushing/popping WhenFrames
        // Please see WhenFrame for more details
        case None =>
          cmd match {
            case WhenBegin(info, pred) =>
              val when = fir.Conditionally(convert(info), convert(pred, ctx, info), fir.EmptyStmt, fir.EmptyStmt)
              val frame = WhenFrame(when, stmts, false)
              stmts = new VectorBuilder[fir.Statement]
              scope = frame :: scope
            case WhenEnd(info, depth, _) =>
              val frame = scope.head.asInstanceOf[WhenFrame]
              val when =
                if (frame.alt) frame.when.copy(alt = fir.Block(stmts.result()))
                else frame.when.copy(conseq = fir.Block(stmts.result()))
              // Check if this when has an else
              cmdsIt.headOption match {
                case Some(AltBegin(_)) =>
                  assert(!frame.alt, "Internal Error! Unexpected when structure!") // Only 1 else per when
                  scope = frame.copy(when = when, alt = true) :: scope.tail
                  cmdsIt.next() // Consume the AltBegin
                  stmts = new VectorBuilder[fir.Statement]
                case _ => // Not followed by otherwise
                  // If depth > 0 then we need to close multiple When scopes so we add a new WhenEnd
                  // If we're nested we need to add more WhenEnds to ensure each When scope gets
                  // properly closed
                  if (depth > 0) {
                    nextCmd = WhenEnd(info, depth - 1, false)
                  }
                  stmts = frame.outer
                  stmts += when
                  scope = scope.tail
              }
            case OtherwiseEnd(info, depth) =>
              val frame = scope.head.asInstanceOf[WhenFrame]
              val when = frame.when.copy(alt = fir.Block(stmts.result()))
              // TODO For some reason depth == 1 indicates the last closing otherwise whereas
              //  depth == 0 indicates last closing when
              if (depth > 1) {
                nextCmd = OtherwiseEnd(info, depth - 1)
              }
              stmts = frame.outer
              stmts += when
              scope = scope.tail
            case GroupDefBegin(info, declaration) =>
              val groupDefine = fir.GroupDefine(convert(info), declaration.name, fir.EmptyStmt)
              val frame = GroupFrame(groupDefine, stmts)
              stmts = new VectorBuilder[fir.Statement]
              scope = frame :: scope
            case GroupDefEnd(info) =>
              val frame = scope.head.asInstanceOf[GroupFrame]
              val groupDefine = frame.group.copy(body = fir.Block(stmts.result()))
              stmts = frame.outer
              stmts += groupDefine
              scope = scope.tail
          }
      }
    }
    assert(scope.isEmpty)
    fir.Block(stmts.result())
  }

  def convert(width: Width): fir.Width = width match {
    case UnknownWidth()    => fir.UnknownWidth
    case KnownWidth(value) => fir.IntWidth(value)
  }

  private def firrtlUserDirOf(t: Data): SpecifiedDirection = t match {
    case t: Vec[_] =>
      SpecifiedDirection.fromParent(t.specifiedDirection, firrtlUserDirOf(t.sample_element))
    case t: Record if t._isOpaqueType =>
      SpecifiedDirection.fromParent(t.specifiedDirection, firrtlUserDirOf(t.elementsIterator.next()))
    case t => t.specifiedDirection
  }

  def extractType(baseType: Data, info: SourceInfo, typeAliases: Seq[String] = Seq.empty): fir.Type =
    extractType(baseType, false, info, true, true, typeAliases)

  def extractType(
    baseType:    Data,
    clearDir:    Boolean,
    info:        SourceInfo,
    checkProbe:  Boolean,
    checkConst:  Boolean,
    typeAliases: Seq[String]
  ): fir.Type = baseType match {
    // extract underlying type for probe
    case t: Data if (checkProbe && t.probeInfo.nonEmpty) =>
      if (t.probeInfo.get.writable) {
        fir.RWProbeType(extractType(t, clearDir, info, false, checkConst, typeAliases))
      } else {
        fir.ProbeType(extractType(t, clearDir, info, false, checkConst, typeAliases))
      }
    // extract underlying type for const
    case t: Data if (checkConst && t.isConst) =>
      fir.ConstType(extractType(t, clearDir, info, checkProbe, false, typeAliases))
    case _: Clock      => fir.ClockType
    case _: AsyncReset => fir.AsyncResetType
    case _: ResetType  => fir.ResetType
    case t: EnumType   => fir.UIntType(convert(t.width))
    case t: UInt       => fir.UIntType(convert(t.width))
    case t: SInt       => fir.SIntType(convert(t.width))
    case t: Analog => fir.AnalogType(convert(t.width))
    case t: Vec[_] =>
      val childClearDir = clearDir ||
        t.specifiedDirection == SpecifiedDirection.Input || t.specifiedDirection == SpecifiedDirection.Output
      // if Vector is a probe, don't emit Probe<...> on its elements
      fir.VectorType(extractType(t.sample_element, childClearDir, info, checkProbe, true, typeAliases), t.length)
    // Handle aliased bundles: Emit an AliasType directly
    case t: HasTypeAlias if t.finalizedAlias.exists { typeAliases.contains(_) } =>
      fir.AliasType(t.finalizedAlias.get)
    case t: Record => {
      val childClearDir = clearDir ||
        t.specifiedDirection == SpecifiedDirection.Input || t.specifiedDirection == SpecifiedDirection.Output
      // if Record is a probe, don't emit Probe<...> on its elements
      def eltField(elt: Data): fir.Field = (childClearDir, firrtlUserDirOf(elt)) match {
        case (true, _) =>
          fir.Field(getRef(elt, info).name, fir.Default, extractType(elt, true, info, checkProbe, true, typeAliases))
        case (false, SpecifiedDirection.Unspecified | SpecifiedDirection.Output) =>
          fir.Field(getRef(elt, info).name, fir.Default, extractType(elt, false, info, checkProbe, true, typeAliases))
        case (false, SpecifiedDirection.Flip | SpecifiedDirection.Input) =>
          fir.Field(getRef(elt, info).name, fir.Flip, extractType(elt, false, info, checkProbe, true, typeAliases))
      }
      if (!t._isOpaqueType)
        fir.BundleType(t._elements.toIndexedSeq.reverse.map { case (_, e) => eltField(e) })
      else
        extractType(t._elements.head._2, childClearDir, info, checkProbe, true, typeAliases)
    }
    case t: Property[_] => t.getPropertyType
  }

  def convert(name: String, param: Param): fir.Param = param match {
    case IntParam(value)    => fir.IntParam(name, value)
    case DoubleParam(value) => fir.DoubleParam(name, value)
    case StringParam(value) => fir.StringParam(name, fir.StringLit(value))
    case RawParam(value)    => fir.RawStringParam(name, value)
  }

  // TODO: Modify Panama CIRCT to account for type aliasing information. This is a temporary hack to
  // allow Panama CIRCT to compile
  def convert(
    port:   Port,
    topDir: SpecifiedDirection
  ): fir.Port = convert(port, Seq.empty, topDir)

  def convert(
    port:        Port,
    typeAliases: Seq[String],
    topDir:      SpecifiedDirection = SpecifiedDirection.Unspecified
  ): fir.Port = {
    val resolvedDir = SpecifiedDirection.fromParent(topDir, firrtlUserDirOf(port.id))
    val dir = resolvedDir match {
      case SpecifiedDirection.Unspecified | SpecifiedDirection.Output => fir.Output
      case SpecifiedDirection.Flip | SpecifiedDirection.Input         => fir.Input
    }
    val clearDir = resolvedDir match {
      case SpecifiedDirection.Input | SpecifiedDirection.Output     => true
      case SpecifiedDirection.Unspecified | SpecifiedDirection.Flip => false
    }
    val tpe = extractType(port.id, clearDir, port.sourceInfo, true, true, typeAliases)
    fir.Port(convert(port.sourceInfo), getRef(port.id, port.sourceInfo).name, dir, tpe)
  }

  def convert(component: Component, typeAliases: Seq[String]): fir.DefModule = component match {
    case ctx @ DefModule(id, name, ports, cmds) =>
      fir.Module(
        convert(id._getSourceLocator),
        name,
        (ports ++ ctx.secretPorts).map(p => convert(p, typeAliases)),
        convert(cmds ++ ctx.secretCommands, ctx, typeAliases)
      )
    case ctx @ DefBlackBox(id, name, ports, topDir, params) =>
      fir.ExtModule(
        convert(id._getSourceLocator),
        name,
        (ports ++ ctx.secretPorts).map(p => convert(p, typeAliases, topDir)),
        id.desiredName,
        params.keys.toList.sorted.map { name => convert(name, params(name)) }
      )
    case ctx @ DefIntrinsicModule(id, name, ports, topDir, params) =>
      fir.IntModule(
        convert(id._getSourceLocator),
        name,
        (ports ++ ctx.secretPorts).map(p => convert(p, typeAliases, topDir)),
        id.intrinsic,
        params.keys.toList.sorted.map { name => convert(name, params(name)) }
      )
    case ctx @ DefClass(id, name, ports, cmds) =>
      fir.DefClass(
        convert(id._getSourceLocator),
        name,
        ports.map(p => convert(p, typeAliases)),
        convert(cmds, ctx, typeAliases)
      )
  }

  def rec(decl: GroupDecl): fir.GroupDeclare = {
    val convention = decl.convention match {
      case GroupConvention.Bind => fir.GroupConvention.Bind
    }
    fir.GroupDeclare(convert(decl.sourceInfo), decl.name, convention, decl.children.map(rec))
  }

  def convert(circuit: Circuit): fir.Circuit = {
    val typeAliases: Seq[String] = circuit.typeAliases.map(_.name)
    fir.Circuit(
      fir.NoInfo,
      circuit.components.map(c => convert(c, typeAliases)),
      circuit.name,
      circuit.typeAliases.map(ta => fir.DefTypeAlias(convert(ta.sourceInfo), ta.name, ta.underlying)),
      circuit.groups.map(rec)
    )
  }

  // TODO Unclear if this should just be the default
  def convertLazily(circuit: Circuit): fir.Circuit = {
    val lazyModules = LazyList() ++ circuit.components
    val typeAliases: Seq[String] = circuit.typeAliases.map(_.name)
    fir.Circuit(
      fir.NoInfo,
      lazyModules.map(lm => convert(lm, typeAliases)),
      circuit.name,
      circuit.typeAliases.map(ta => {
        // To generate the correct FIRRTL type alias we need to always emit a BundleType.
        // This is not guaranteed if the alias name set contains this type alias's name itself
        // as otherwise an AliasType will be generated, resulting in self-referential FIRRTL
        // statements like `type Foo = Foo`.
        fir.DefTypeAlias(
          convert(ta.sourceInfo),
          ta.name,
          ta.underlying
        )
      })
    )
  }
}
