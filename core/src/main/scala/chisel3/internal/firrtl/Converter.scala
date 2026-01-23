// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.firrtl

import chisel3.{Placeholder => _, _}
import chisel3.experimental.{
  Analog,
  BaseModule,
  HasTypeAlias,
  NoSourceInfo,
  SourceInfo,
  SourceLine,
  UnlocatableSourceInfo
}
import chisel3.properties.Property
import firrtl.{ir => fir}
import chisel3.internal.{castToInt, throwException, HasId}
import chisel3.internal.firrtl.ir._
import chisel3.EnumType
import scala.annotation.tailrec
import scala.collection.immutable.{Queue, VectorBuilder, VectorMap}

private[chisel3] object Converter {

  def unpack(pable: Printable, ctx: Component, sourceInfo: SourceInfo): (String, Seq[Arg]) = {
    implicit val info: SourceInfo = sourceInfo
    val resolved = Printable.resolve(pable, ctx)
    val (fmt, data) = resolved.unpack
    (fmt, data.map(_.ref))
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

  def convertPrim(op: PrimOp, args: Seq[Arg], sourceInfo: SourceInfo, ctx: Component): fir.Expression = {
    val consts = args.collect { case ILit(i) => i }
    val argsx = args.flatMap {
      case _: ILit => None
      case other => Some(convert(other, ctx, sourceInfo))
    }
    op.name match {
      case "mux" =>
        assert(argsx.size == 3, s"Mux with unexpected args: $argsx")
        fir.Mux(argsx(0), argsx(1), argsx(2), fir.UnknownType)
      case _ =>
        fir.DoPrim(convert(op), argsx, consts, fir.UnknownType)
    }
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
    case LitIndex(imm, idx) =>
      fir.SubIndex(convert(imm, ctx, info), idx, fir.UnknownType)
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
    case u @ ULit(n, UnknownWidth) =>
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
    case PropExpr(info, tpe, op, args) =>
      fir.PropExpr(convert(info), tpe, op, args.map(convert(_, ctx, info)))
    case e: PrimExpr[_] =>
      convertPrim(e.op, e.args, info, ctx)
    case other =>
      throw new InternalErrorException(s"Unexpected type in convert $other")
  }

  /** Convert Commands that map 1:1 to Statements */
  def convertCommand(cmd: Command, ctx: Component, typeAliases: Seq[String]): fir.Statement = cmd match {
    case e: DefPrim[_] =>
      val expr = convertPrim(e.op, e.args, e.sourceInfo, ctx)
      fir.DefNode(convert(e.sourceInfo), e.name, expr)
    case e @ DefWire(info, id) =>
      fir.DefWire(convert(info), e.name, extractType(id, info, typeAliases))
    case e @ DefReg(info, id, clock) =>
      fir.DefRegister(
        convert(info),
        e.name,
        extractType(id, info, typeAliases),
        convert(clock, ctx, info)
      )
    case e @ DefRegInit(info, id, clock, reset, init) =>
      fir.DefRegisterWithReset(
        convert(info),
        e.name,
        extractType(id, info, typeAliases),
        convert(clock, ctx, info),
        convert(reset, ctx, info),
        convert(init, ctx, info)
      )
    case e @ DefMemory(info, id, t, size) =>
      firrtl.CDefMemory(convert(info), e.name, extractType(t, info, typeAliases), size, false)
    case e @ DefSeqMemory(info, id, t, size, ruw) =>
      firrtl.CDefMemory(convert(info), e.name, extractType(t, info, typeAliases), size, true, ruw)
    case e @ FirrtlMemory(
          info,
          id,
          t,
          size,
          readPortNames,
          writePortNames,
          readwritePortNames,
          readLatency,
          writeLatency
        ) =>
      fir.DefMemory(
        convert(info),
        e.name,
        extractType(t, info, typeAliases),
        size,
        writeLatency,
        readLatency,
        readPortNames,
        writePortNames,
        readwritePortNames
      )
    case e: DefMemPort[_] =>
      val info = e.sourceInfo
      firrtl.CDefMPort(
        convert(e.sourceInfo),
        e.name,
        fir.UnknownType,
        e.source.fullName(ctx),
        Seq(convert(e.index, ctx, info), convert(e.clock, ctx, info)),
        convert(e.dir)
      )
    case Connect(info, loc, exp) =>
      fir.Connect(convert(info), convert(loc, ctx, info), convert(exp, ctx, info))
    case PropAssign(info, loc, exp) =>
      fir.PropAssign(convert(info), convert(loc, ctx, info), convert(exp, ctx, info))
    case Attach(info, locs) =>
      fir.Attach(convert(info), locs.map(l => convert(l, ctx, info)))
    case DefInvalid(info, arg) =>
      fir.IsInvalid(convert(info), convert(arg, ctx, info))
    case e @ DefInstance(info, id, _) =>
      fir.DefInstance(convert(info), e.name, id.name)
    case e @ DefInstanceChoice(info, _, default, option, choices) =>
      fir.DefInstanceChoice(
        convert(info),
        e.name,
        default.name,
        option,
        choices.map { case (opt, mod) => (opt, mod.name) }
      )
    case e @ DefObject(info, _, className) =>
      fir.DefObject(convert(info), e.name, className)
    case e @ Stop(_, info, clock, ret) =>
      fir.Stop(convert(info), ret, convert(clock, ctx, info), firrtl.Utils.one, e.name)
    case e @ Printf(_, info, filename, clock, pable) =>
      val mkPrintf = filename match {
        case None => fir.Print.apply _
        case Some(f) =>
          val (ffmt, fargs) = unpack(f, ctx, info)
          fir.Fprint.apply(_, fir.StringLit(ffmt), fargs.map(a => convert(a, ctx, info)), _, _, _, _, _)
      }
      val (fmt, args) = unpack(pable, ctx, info)
      mkPrintf(
        convert(info),
        fir.StringLit(fmt),
        args.map(a => convert(a, ctx, info)),
        convert(clock, ctx, info),
        firrtl.Utils.one,
        e.name
      )
    case e @ Flush(info, filename, clock) =>
      val (fmt, args) = filename.map(unpack(_, ctx, info)).getOrElse(("", Seq.empty))
      val fn = Option.when(fmt.nonEmpty)(fir.StringLit(fmt))
      fir.Flush(convert(info), fn, args.map(a => convert(a, ctx, info)), convert(clock, ctx, info))
    case e @ ProbeDefine(sourceInfo, sink, probeExpr) =>
      fir.ProbeDefine(convert(sourceInfo), convert(sink, ctx, sourceInfo), convert(probeExpr, ctx, sourceInfo))
    case e @ ProbeForceInitial(sourceInfo, probe, value) =>
      fir.ProbeForceInitial(convert(sourceInfo), convert(probe, ctx, sourceInfo), convert(value, ctx, sourceInfo))
    case e @ ProbeReleaseInitial(sourceInfo, probe) =>
      fir.ProbeReleaseInitial(convert(sourceInfo), convert(probe, ctx, sourceInfo))
    case e @ ProbeForce(sourceInfo, clock, cond, probe, value) =>
      fir.ProbeForce(
        convert(sourceInfo),
        convert(clock, ctx, sourceInfo),
        convert(cond, ctx, sourceInfo),
        convert(probe, ctx, sourceInfo),
        convert(value, ctx, sourceInfo)
      )
    case e @ ProbeRelease(sourceInfo, clock, cond, probe) =>
      fir.ProbeRelease(
        convert(sourceInfo),
        convert(clock, ctx, sourceInfo),
        convert(cond, ctx, sourceInfo),
        convert(probe, ctx, sourceInfo)
      )
    case e @ Verification(_, op, info, clk, pred, pable) =>
      val (fmt, args) = unpack(pable, ctx, info)
      val firOp = op match {
        case Formal.Assert => fir.Formal.Assert
        case Formal.Assume => fir.Formal.Assume
        case Formal.Cover  => fir.Formal.Cover
      }
      fir.Verification(
        firOp,
        convert(info),
        convert(clk, ctx, info),
        convert(pred, ctx, info),
        firrtl.Utils.one,
        fir.StringLit(fmt),
        args.map(a => convert(a, ctx, info)),
        e.name
      )
    case i @ DefIntrinsic(info, intrinsic, args, params) =>
      fir.IntrinsicStmt(
        convert(info),
        intrinsic,
        args.map(a => convert(a, ctx, info)),
        params.map { case (k, v) => convert(k, v) }
      )
    case i @ DefIntrinsicExpr(info, intrinsic, id, args, params) =>
      val tpe = extractType(id, info, typeAliases)
      val expr = fir.IntrinsicExpr(
        intrinsic,
        args.map(a => convert(a, ctx, info)),
        params.map { case (k, v) => convert(k, v) },
        tpe
      )
      fir.DefNode(convert(info), i.name, expr)
    case When(info, pred, ifRegion, elseRegion) =>
      fir.Conditionally(
        convert(info),
        convert(pred, ctx, info),
        convert(ifRegion, ctx, typeAliases),
        if (elseRegion.nonEmpty) convert(elseRegion, ctx, typeAliases) else fir.EmptyStmt
      )
    case LayerBlock(info, layer, region) =>
      fir.LayerBlock(convert(info), layer, convert(region, ctx, typeAliases))
    case DefContract(info, ids, exprs) =>
      fir.EmptyStmt // dummy until the converter removed
    case Placeholder(info, block) =>
      convert(block, ctx, typeAliases)
    case FirrtlComment(text) =>
      fir.Comment(text)
    case DomainDefine(info, sink, source) =>
      fir.DomainDefine(convert(info), convert(sink, ctx, info), convert(source, ctx, info))
  }

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
    stmts.sizeHint(cmds)
    for (cmd <- cmds)
      stmts += convertCommand(cmd, ctx, typeAliases)
    fir.Block(stmts.result())
  }

  /** Convert Chisel IR Block into FIRRTL Statements
    *
    * @note ctx is needed because references to ports translate differently when referenced within
    *   the module in which they are defined vs. parent modules
    * @param block Chisel IR Block to convert
    * @param ctx Component (Module) context within which we are translating
    * @param typeAliases Set of aliased type names to emit FIRRTL alias types for
    * @return FIRRTL Statement that is equivalent to the input block
    */
  def convert(block: Block, ctx: Component, typeAliases: Seq[String]): fir.Statement = {
    val stmts = new VectorBuilder[fir.Statement]()
    val commands = block.getCommands()
    val secretCommands = block.getSecretCommands()
    (commands.knownSize, secretCommands.knownSize) match {
      case (-1, _)  => ()
      case (s, -1)  => stmts.sizeHint(s)
      case (s1, s2) => stmts.sizeHint(s1 + s2)
    }
    for (cmd <- commands)
      stmts += convertCommand(cmd, ctx, typeAliases)
    for (cmd <- secretCommands)
      stmts += convertCommand(cmd, ctx, typeAliases)
    fir.Block(stmts.result())
  }

  def convert(width: Width): fir.Width = width match {
    case UnknownWidth      => fir.UnknownWidth
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
        fir.RWProbeType(
          extractType(t, clearDir, info, false, checkConst, typeAliases),
          t.probeInfo.get.color.map(_.fullName)
        )
      } else {
        fir.ProbeType(
          extractType(t, clearDir, info, false, checkConst, typeAliases),
          t.probeInfo.get.color.map(_.fullName)
        )
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
    case t: Analog     => fir.AnalogType(convert(t.width))
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
    case t: domain.Type => fir.DomainType(t.domain.name)
  }

  def convert(name: String, param: Param): fir.Param = param match {
    case p: IntParam    => fir.IntParam(name, p.value)
    case p: DoubleParam => fir.DoubleParam(name, p.value)
    case p: StringParam => fir.StringParam(name, fir.StringLit(p.value))
    case p: PrintableParam => {
      val ctx = p.context._component.get
      val (fmt, _) = unpack(p.value, ctx, UnlocatableSourceInfo)
      fir.StringParam(name, fir.StringLit(fmt))
    }
    case p: RawParam => fir.RawStringParam(name, p.value)
  }

  def convert(param: TestParam): fir.TestParam = param match {
    case IntTestParam(value)    => fir.IntTestParam(value)
    case DoubleTestParam(value) => fir.DoubleTestParam(value)
    case StringTestParam(value) => fir.StringTestParam(value)
    case ArrayTestParam(value)  => fir.ArrayTestParam(value.map(convert))
    case MapTestParam(value)    => fir.MapTestParam(value.map { case (name, value) => (name, convert(value)) })
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
    fir.Port(
      convert(port.sourceInfo),
      getRef(port.id, port.sourceInfo).name,
      dir,
      tpe,
      port.associations.map(getRef(_, UnlocatableSourceInfo).name)
    )
  }

  def convert(component: Component, typeAliases: Seq[String]): fir.DefModule = component match {
    case ctx @ DefModule(id, name, public, layers, ports, block) =>
      fir.Module(
        convert(id._getSourceLocator),
        name,
        public,
        layers.map(_.fullName),
        (ports ++ ctx.secretPorts).map(p => convert(p, typeAliases)),
        convert(block, ctx, typeAliases)
      )
    case ctx @ DefBlackBox(id, name, ports, topDir, params, knownLayers, requirements) =>
      fir.ExtModule(
        convert(id._getSourceLocator),
        name,
        (ports ++ ctx.secretPorts).map(p => convert(p, typeAliases, topDir)),
        id.desiredName,
        params.keys.toList.sorted.map { name => convert(name, params(name)) },
        knownLayers.map(_.fullName),
        requirements
      )
    case ctx @ DefIntrinsicModule(id, name, ports, topDir, params) =>
      fir.IntModule(
        convert(id._getSourceLocator),
        name,
        (ports ++ ctx.secretPorts).map(p => convert(p, typeAliases, topDir)),
        id.intrinsic,
        params.keys.toList.sorted.map { name => convert(name, params(name)) }
      )
    case ctx @ DefClass(id, name, ports, block) =>
      fir.DefClass(
        convert(id._getSourceLocator),
        name,
        (ports ++ ctx.secretPorts).map(p => convert(p, typeAliases)),
        convert(block, ctx, typeAliases)
      )
    case ctx @ DefTestMarker(kind, name, module, params, sourceInfo) =>
      fir.TestMarker(
        kind match {
          case DefTestMarker.Formal     => fir.TestMarker.Formal
          case DefTestMarker.Simulation => fir.TestMarker.Simulation
        },
        convert(sourceInfo),
        name,
        module.name,
        convert(params).asInstanceOf[fir.MapTestParam]
      )
  }

  def convertLayer(layer: Layer): fir.Layer = {
    val config = layer.config match {
      case LayerConfig.Extract(outputDir) => fir.LayerConfig.Extract(outputDir)
      case LayerConfig.Inline             => fir.LayerConfig.Inline
    }
    fir.Layer(convert(layer.sourceInfo), layer.name, config, layer.children.map(convertLayer))
  }

  def convertOption(option: DefOption): fir.DefOption = {
    fir.DefOption(
      convert(option.sourceInfo),
      option.name,
      option.cases.map(optCase => fir.DefOptionCase(convert(optCase.sourceInfo), optCase.name))
    )
  }

  def convert(circuit: Circuit): fir.Circuit = {
    val typeAliases: Seq[String] = circuit.typeAliases.map(_.name)
    fir.Circuit(
      fir.NoInfo,
      circuit.components.map(c => convert(c, typeAliases)),
      circuit.name,
      circuit.typeAliases.map(ta => fir.DefTypeAlias(convert(ta.sourceInfo), ta.name, ta.underlying)),
      circuit.layers.map(convertLayer),
      circuit.options.map(convertOption)
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
