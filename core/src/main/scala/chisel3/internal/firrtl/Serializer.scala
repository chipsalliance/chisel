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
import firrtl.annotations.{Annotation, JsonProtocol}
import chisel3.internal.{castToInt, throwException, HasId}
import chisel3.internal.firrtl.ir._
import chisel3.EnumType
import scala.annotation.tailrec
import scala.collection.immutable.{Queue, VectorBuilder, VectorMap}

private[chisel3] object Serializer {
  private val NewLine = '\n'
  private val Indent = "  "

  // The version supported by the serializer.
  val version = "6.0.0"

  def getRef(id: HasId, sourceInfo: SourceInfo): Arg =
    id.getOptionRef.getOrElse {
      val module = id._parent.map(m => s" '$id' was defined in module '$m'.").getOrElse("")
      val loc = sourceInfo.makeMessage(" " + _)
      reportInternalError(s"Could not get ref for '$id'$loc!$module")
    }

  /** Generate a legal FIRRTL name. */
  private def legalize(name: String): String = name match {
    // If the name starts with a digit, then escape it with backticks.
    case _ if name.head.isDigit => s"`$name`"
    case _                      => name
  }

  /** create a new line with the appropriate indent */
  private def newLineAndIndent(inc: Int = 0)(implicit b: StringBuilder, indent: Int): Unit = {
    b += NewLine; doIndent(inc)
  }

  private def newLineNoIndent()(implicit b: StringBuilder): Unit = b += NewLine

  /** create indent, inc allows for a temporary increment */
  private def doIndent(inc: Int = 0)(implicit b: StringBuilder, indent: Int): Unit = {
    (0 until (indent + inc)).foreach { _ => b ++= Indent }
  }

  /** Serialize the given text as a "quoted" string. */
  private def quote(text: String)(implicit b: StringBuilder): Unit = {
    b ++= "\""
    b ++= text
    b ++= "\""
  }

  // TODO use makeMessage to get ':' filename col separator instead of space
  // Can we optimize the escaping?
  private def serialize(info: SourceInfo)(implicit b: StringBuilder, suppressSourceInfo: Boolean): Unit =
    info match {
      case sl: SourceLine if !suppressSourceInfo =>
        b ++= " @["; b ++= fir.FileInfo.fromUnescaped(sl.serialize).escaped; b ++= "]"
      case _ => ()
    }

  private def reportInternalError(msg: String): Nothing = {
    val link = "https://github.com/chipsalliance/chisel/issues/new"
    val fullMsg = s"Internal Error! $msg This is a bug in Chisel, please file an issue at '$link'"
    throwException(fullMsg)
  }

  private def clonedModuleIOError(mod: BaseModule, name: String, sourceInfo: SourceInfo): Nothing = {
    val loc = sourceInfo.makeMessage(" " + _)
    reportInternalError(s"Trying to convert a cloned IO of $mod inside of $mod itself$loc!")
  }

  // TODO modeled on unpack method on Printable, refactor?
  // Cannot just use Printable.unpack because it doesn't work right with nested expressions
  def unpack(pable: Printable, ctx: Component, sourceInfo: SourceInfo): (String, Seq[Arg]) = {
    implicit val info: SourceInfo = sourceInfo
    val resolved = Printable.resolve(pable, ctx)
    val (fmt, data) = resolved.unpack
    (fmt, data.map(_.ref))
  }

  private def serializeArgs(args: Seq[Arg], ctx: Component, info: SourceInfo)(implicit b: StringBuilder): Unit = {
    var first = true
    args.foreach { a =>
      if (!first) {
        b ++= ", "
      }
      first = false
      serialize(a, ctx, info)
    }
  }

  private def serializePrim(op: PrimOp, args: Seq[Arg], sourceInfo: SourceInfo, ctx: Component)(
    implicit b: StringBuilder
  ): Unit = {
    b ++= op.name; b += '('
    val last = args.size - 1
    args.zipWithIndex.foreach { case (arg, idx) =>
      serialize(arg, ctx, sourceInfo)
      if (idx != last) { b ++= ", " }
    }
    b += ')'
  }

  private def serialize(arg: Arg, ctx: Component, info: SourceInfo)(implicit b: StringBuilder): Unit = arg match {
    case Node(id)  => serialize(getRef(id, info), ctx, info)
    case Ref(name) => b ++= name
    // We don't need to legalize Slot names, firtool can parse subfields starting with digits
    case Slot(imm, name) => serialize(imm, ctx, info); b += '.'; b ++= legalize(name)
    case OpaqueSlot(imm) => serialize(imm, ctx, info)
    case LitIndex(imm, idx) =>
      serialize(imm, ctx, info); b += '['; b ++= idx.toString; b += ']'
    case Index(imm, ILit(idx)) =>
      serialize(imm, ctx, info); b += '['; b ++= castToInt(idx, "Index").toString; b += ']'
    case Index(imm, value) =>
      serialize(imm, ctx, info); b += '['; serialize(value, ctx, info); b += ']'
    case ModuleIO(mod, name) =>
      if (mod eq ctx.id) { b ++= name }
      else { b ++= getRef(mod, info).name; b += '.'; b ++= name }
    case ModuleCloneIO(mod, name) =>
      if (mod eq ctx.id) clonedModuleIOError(mod, name, info)
      else { b ++= name }
    case u @ ULit(n, w) =>
      val width = w match {
        case UnknownWidth => u.minWidth
        case w: KnownWidth => w.value
      }
      b ++= "UInt<"; b ++= width.toString; b ++= ">(0h"; b ++= n.toString(16); b += ')'
    case s @ SLit(n, w) =>
      val width = w match {
        case UnknownWidth => s.minWidth
        case w: KnownWidth => w.value
      }
      b ++= "SInt<"; b ++= width.toString; b ++= ">("
      if (n < 0) { b += '-' }
      b ++= "0h"; b ++= n.abs.toString(16); b += ')'
    case lit: ILit => b ++= lit.n.toString
    case PropertyLit(tpe, lit) =>
      // TODO can we not use FIRRTL types here?
      b ++= fir.Serializer.serialize(tpe.convert(lit, ctx, info))
    case e @ ProbeExpr(probe) =>
      b ++= "probe("; serialize(probe, ctx, info); b += ')'
    case e @ RWProbeExpr(probe) =>
      b ++= "rwprobe("; serialize(probe, ctx, info); b += ')'
    case e @ ProbeRead(probe) =>
      b ++= "read("; serialize(probe, ctx, info); b += ')'
    case PropExpr(_, tpe, op, args) =>
      b ++= op.toString; b += '('; serializeArgs(args, ctx, info); b += ')'
    case e: PrimExpr[_] =>
      serializePrim(e.op, e.args, info, ctx)
    case other =>
      throw new InternalErrorException(s"Unexpected type in convert $other")
  }

  private def serializeIntrinsic(
    ctx:         Component,
    info:        SourceInfo,
    name:        Option[String],
    id:          Option[Data],
    intrinsic:   String,
    args:        Seq[Arg],
    params:      Seq[(String, Param)],
    typeAliases: Seq[String]
  )(implicit b: StringBuilder, suppressSourceInfo: Boolean): Unit = {
    if (name.nonEmpty) {
      b ++= "node "; b ++= legalize(name.get); b ++= " = "
    }
    b ++= "intrinsic("; b ++= intrinsic;
    if (params.nonEmpty) {
      b += '<';
      val lastIdx = params.size - 1
      params.zipWithIndex.foreach { case ((name, param), idx) =>
        serialize(name, param)
        if (idx != lastIdx) b ++= ", "
      }
      b += '>'
    }
    if (id.nonEmpty) {
      b ++= " : "
      serializeType(id.get, info, typeAliases)
    }
    if (args.nonEmpty) {
      b ++= ", "
      serializeArgs(args, ctx, info)
    }
    b += ')'
    serialize(info)
  }

  /** Serialize Commands */
  private def serializeSimpleCommand(cmd: Command, ctx: Component, typeAliases: Seq[String])(
    implicit b:         StringBuilder,
    indent:             Int,
    suppressSourceInfo: Boolean
  ): Unit = cmd match {
    case e: DefPrim[_] =>
      b ++= "node "; b ++= legalize(e.name); b ++= " = ";
      serializePrim(e.op, e.args, e.sourceInfo, ctx)
      serialize(e.sourceInfo)
    case e @ DefWire(info, id) =>
      b ++= "wire "; b ++= legalize(e.name); b ++= " : "; serializeType(id, info, typeAliases); serialize(e.sourceInfo)
    case e @ DefReg(info, id, clock) =>
      b ++= "reg "; b ++= legalize(e.name); b ++= " : "; serializeType(id, info, typeAliases);
      b ++= ", "; serialize(clock, ctx, info)
      serialize(e.sourceInfo)
    case e @ DefRegInit(info, id, clock, reset, init) =>
      b ++= "regreset "; b ++= legalize(e.name); b ++= " : "; serializeType(id, info, typeAliases);
      b ++= ", "; serialize(clock, ctx, info)
      b ++= ", "; serialize(reset, ctx, info)
      b ++= ", "; serialize(init, ctx, info);
      serialize(e.sourceInfo)
    case e @ DefMemory(info, id, t, size) =>
      b ++= "cmem "; b ++= legalize(e.name); b ++= " : "; serializeType(t, info, typeAliases);
      b += '['; b ++= size.toString; b += ']'; serialize(e.sourceInfo)
    case e @ DefSeqMemory(info, id, t, size, ruw) =>
      b ++= "smem "; b ++= legalize(e.name); b ++= " : "; serializeType(t, info, typeAliases);
      b += '['; b ++= size.toString; b += ']';
      if (ruw != fir.ReadUnderWrite.Undefined) { // undefined is the default
        b ++= ", "; b ++= ruw.toString
      }
      serialize(e.sourceInfo)
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
      b ++= "mem "; b ++= legalize(e.name); b ++= " :"; serialize(e.sourceInfo); newLineAndIndent(1)
      b ++= "data-type => "; serializeType(t, info, typeAliases); newLineAndIndent(1)
      b ++= "depth => "; b ++= size.toString; newLineAndIndent(1)
      b ++= "read-latency => "; b ++= readLatency.toString; newLineAndIndent(1)
      b ++= "write-latency => "; b ++= writeLatency.toString; newLineAndIndent(1)
      readPortNames.foreach { r => b ++= "reader => "; b ++= legalize(r); newLineAndIndent(1) }
      writePortNames.foreach { w => b ++= "writer => "; b ++= legalize(w); newLineAndIndent(1) }
      readwritePortNames.foreach { r => b ++= "readwriter => "; b ++= legalize(r); newLineAndIndent(1) }
      b ++= "read-under-write => undefined"
    case e: DefMemPort[_] =>
      b ++= e.dir.toString; b ++= " mport "; b ++= legalize(e.name); b ++= " = "; b ++= legalize(e.source.fullName(ctx))
      b += '['; serialize(e.index, ctx, e.sourceInfo); b += ']'; b ++= ", "; serialize(e.clock, ctx, e.sourceInfo);
      serialize(e.sourceInfo)
    case Connect(info, loc, exp) =>
      b ++= "connect "; serialize(loc, ctx, info); b ++= ", "; serialize(exp, ctx, info); serialize(info)
    case PropAssign(info, loc, exp) =>
      b ++= "propassign "; serialize(loc, ctx, info); b ++= ", "; serialize(exp, ctx, info); serialize(info)
    case Attach(info, locs) =>
      b ++= "attach ("
      serializeArgs(locs, ctx, info)
      b += ')'; serialize(info)
    case DefInvalid(info, arg) =>
      b ++= "invalidate "; serialize(arg, ctx, info); serialize(info)
    case e @ DefInstance(info, id, _) =>
      b ++= "inst "; b ++= legalize(e.name); b ++= " of "; b ++= legalize(id.name); serialize(e.sourceInfo)
    case e @ DefInstanceChoice(info, _, default, option, choices) =>
      b ++= "instchoice "; b ++= legalize(e.name); b ++= " of "; b ++= legalize(default.name);
      b ++= ", "; b ++= legalize(option); b ++= " : "; serialize(e.sourceInfo)
      choices.foreach { case (choice, module) =>
        newLineAndIndent(1)
        b ++= legalize(choice); b ++= " => "; b ++= legalize(module.name)
      }
    case e @ DefObject(info, _, className) =>
      b ++= "object "; b ++= legalize(e.name); b ++= " of "; b ++= legalize(className); serialize(e.sourceInfo)
    case e @ Stop(_, info, clock, ret) =>
      b ++= "stop("; serialize(clock, ctx, info); b ++= ", UInt<1>(0h1), "; b ++= ret.toString; b += ')';
      val lbl = e.name
      if (lbl.nonEmpty) { b ++= " : "; b ++= legalize(lbl) }
      serialize(e.sourceInfo)
    case e @ Printf(_, info, filename, clock, pable) =>
      val (fmt, args) = unpack(pable, ctx, info)
      if (filename.isEmpty) b ++= "printf("; else b ++= "fprintf(";
      serialize(clock, ctx, info); b ++= ", UInt<1>(0h1), ";
      filename.foreach { fable =>
        val (ffmt, fargs) = unpack(fable, ctx, info)
        b ++= fir.StringLit(ffmt).escape; b ++= ", "
        fargs.foreach { a => serialize(a, ctx, info); b ++= ", " }
      }
      b ++= fir.StringLit(fmt).escape;
      args.foreach { a => b ++= ", "; serialize(a, ctx, info) }; b += ')'
      val lbl = e.name
      if (lbl.nonEmpty) { b ++= " : "; b ++= legalize(lbl) }
      serialize(e.sourceInfo)
    case e @ Flush(info, filename, clock) =>
      b ++= "fflush("; serialize(clock, ctx, info); b ++= ", UInt<1>(0h1)";
      filename.foreach { fable =>
        val (ffmt, fargs) = unpack(fable, ctx, info)
        b ++= ", "; b ++= fir.StringLit(ffmt).escape
        fargs.foreach { a => b ++= ", "; serialize(a, ctx, info) }
      }
      b += ')'; serialize(info)
    case e @ ProbeDefine(sourceInfo, sink, probeExpr) =>
      b ++= "define "; serialize(sink, ctx, sourceInfo); b ++= " = "; serialize(probeExpr, ctx, sourceInfo);
      serialize(sourceInfo)
    case e @ ProbeForceInitial(sourceInfo, probe, value) =>
      b ++= "force_initial("; serialize(probe, ctx, sourceInfo); b ++= ", "; serialize(value, ctx, sourceInfo);
      b += ')'; serialize(sourceInfo)
    case e @ ProbeReleaseInitial(sourceInfo, probe) =>
      b ++= "release_initial("; serialize(probe, ctx, sourceInfo); b += ')'; serialize(sourceInfo)
    case e @ ProbeForce(sourceInfo, clock, cond, probe, value) =>
      b ++= "force("; serializeArgs(Seq(clock, cond, probe, value), ctx, sourceInfo); b += ')'; serialize(sourceInfo)
    case e @ ProbeRelease(sourceInfo, clock, cond, probe) =>
      b ++= "release("; serializeArgs(Seq(clock, cond, probe), ctx, sourceInfo); b += ')'; serialize(sourceInfo)
    case e @ Verification(_, op, info, clk, pred, pable) =>
      val (fmt, args) = unpack(pable, ctx, info)
      b ++= op.toString; b += '('; serializeArgs(Seq(clk, pred), ctx, info); b ++= ", UInt<1>(0h1), ";
      b ++= fir.StringLit(fmt).escape;
      args.foreach { a => b ++= ", "; serialize(a, ctx, info) }; b += ')'
      val lbl = e.name
      if (lbl.nonEmpty) { b ++= " : "; b ++= legalize(lbl) }
      serialize(e.sourceInfo)
    case i @ DefIntrinsic(info, intrinsic, args, params) =>
      serializeIntrinsic(ctx, info, None, None, intrinsic, args, params, typeAliases)
    case i @ DefIntrinsicExpr(info, intrinsic, id, args, params) =>
      serializeIntrinsic(ctx, info, Some(i.name), Some(id), intrinsic, args, params, typeAliases)
    case FirrtlComment(text) =>
      // Because of split, iterator will always be non-empty even if text is empty
      val it = text.split("\n").iterator
      while ({
        val line = it.next()
        b ++= "; "; b ++= line
        if (it.hasNext) {
          newLineAndIndent()
        }
        it.hasNext
      }) ()
    case e @ DomainDefine(info, sink, source) =>
      b ++= "domain_define "
      serialize(sink, ctx, info)
      b ++= " = "
      serialize(source, ctx, info);
      serialize(info)
  }

  private def serializeCommand(cmd: Command, ctx: Component, typeAliases: Seq[String])(
    implicit indent:    Int,
    suppressSourceInfo: Boolean
  ): Iterator[String] = {
    cmd match {
      case When(info, pred, ifRegion, elseRegion) =>
        val start = {
          implicit val b = new StringBuilder
          doIndent(); b ++= "when "; serialize(pred, ctx, info); b ++= " :"; serialize(info)
          newLineNoIndent()
          Iterator(b.toString)
        }
        val middle =
          if (ifRegion.isEmpty) {
            implicit val b = new StringBuilder
            doIndent(1); b ++= "skip"
            newLineNoIndent()
            Iterator(b.toString)
          } else {
            ifRegion.flatMap(serializeCommand(_, ctx, typeAliases)(indent + 1, suppressSourceInfo))
          }
        val end = if (elseRegion.nonEmpty) {
          implicit val b = new StringBuilder
          doIndent(); b ++= "else :"
          newLineNoIndent()
          Iterator(b.toString) ++ elseRegion.flatMap(
            serializeCommand(_, ctx, typeAliases)(indent + 1, suppressSourceInfo)
          )
        } else Iterator.empty
        start ++ middle ++ end
      case LayerBlock(info, layer, region) =>
        val start = {
          implicit val b = new StringBuilder
          doIndent(); b ++= "layerblock "; b ++= layer; b ++= " :"; serialize(info)
          newLineNoIndent()
          Iterator(b.toString)
        }
        start ++ region.iterator.flatMap(serializeCommand(_, ctx, typeAliases)(indent + 1, suppressSourceInfo))
      case Placeholder(_, block) =>
        if (block.isEmpty) {
          implicit val b = new StringBuilder
          doIndent(); b ++= "skip"
          newLineNoIndent()
          Iterator(b.toString)
        } else {
          block.iterator.flatMap(serializeCommand(_, ctx, typeAliases)(indent, suppressSourceInfo))
        }
      case cmd @ DefContract(info, names, exprs) =>
        val start = {
          implicit val b = new StringBuilder
          doIndent()
          b ++= "contract"
          if (names.nonEmpty) {
            b ++= names.map(_.getRef.name).mkString(" ", ", ", "")
            b ++= " = "
            exprs.zipWithIndex.foreach { case (expr, idx) =>
              if (idx > 0) b ++= ", "
              serialize(expr, ctx, info)
            }
          }
          b ++= " :"
          serialize(info)
          newLineNoIndent()
          Iterator(b.toString)
        }
        start ++ cmd.region
          .getAllCommands()
          .flatMap(serializeCommand(_, ctx, typeAliases)(indent + 1, suppressSourceInfo))
      // TODO can we avoid checking 4 less common Commands every single time?
      case simple =>
        // TODO avoid Iterator boxing for every simple command
        implicit val b = new StringBuilder
        doIndent()
        serializeSimpleCommand(simple, ctx, typeAliases)
        newLineNoIndent()
        Iterator(b.toString)
    }
  }

  /** Serialize Chisel IR Block into FIRRTL Statements
    *
    * @note ctx is needed because references to ports translate differently when referenced within
    *   the module in which they are defined vs. parent modules
    * @param block Chisel IR Block to convert
    * @param ctx Component (Module) context within which we are translating
    * @param typeAliases Set of aliased type names to emit FIRRTL alias types for
    * @return Iterator[String] of the equivalent FIRRTL text
    */
  private def serialize(block: Block, ctx: Component, typeAliases: Seq[String])(
    implicit indent:    Int,
    suppressSourceInfo: Boolean
  ): Iterator[String] = {
    val commands = block.getCommands()
    val secretCommands = block.getSecretCommands()
    if (commands.isEmpty && secretCommands.isEmpty) {
      implicit val b = new StringBuilder
      doIndent(); b ++= "skip"
      newLineNoIndent()
      return Iterator(b.toString)
    } else {
      Iterator.empty[String] ++ (commands.iterator ++ secretCommands).flatMap(c =>
        serializeCommand(c, ctx, typeAliases)(indent, suppressSourceInfo)
      )
    }
  }

  private def serialize(width: Width)(implicit b: StringBuilder): Unit = width match {
    case KnownWidth(width) => b += '<'; b ++= width.toString; b += '>'
    case UnknownWidth      => // empty string
  }

  private def firrtlUserDirOf(t: Data): SpecifiedDirection = t match {
    case t: Vec[_] =>
      SpecifiedDirection.fromParent(t.specifiedDirection, firrtlUserDirOf(t.sample_element))
    case t: Record if t._isOpaqueType =>
      SpecifiedDirection.fromParent(t.specifiedDirection, firrtlUserDirOf(t.elementsIterator.next()))
    case t => t.specifiedDirection
  }

  def serializeType(baseType: Data, info: SourceInfo, typeAliases: Seq[String] = Seq.empty)(
    implicit b: StringBuilder
  ): Unit =
    serializeType(baseType, false, info, true, true, typeAliases)

  def serializeType(
    baseType:    Data,
    clearDir:    Boolean,
    info:        SourceInfo,
    checkProbe:  Boolean,
    checkConst:  Boolean,
    typeAliases: Seq[String]
  )(implicit b: StringBuilder): Unit = baseType match {
    // extract underlying type for probe
    case t: Data if (checkProbe && t.probeInfo.nonEmpty) =>
      if (t.probeInfo.get.writable) {
        b ++= "RWProbe<"
      } else {
        b ++= "Probe<"
      }
      serializeType(t, clearDir, info, false, checkConst, typeAliases)
      t.probeInfo.get.color.foreach { layer => b ++= s", ${layer.fullName}" }
      b += '>'
    // extract underlying type for const
    // TODO do we need !lastEmittedConst check?
    case t: Data if (checkConst && t.isConst) =>
      b ++= "const "
      serializeType(t, clearDir, info, checkProbe, false, typeAliases)
    case _: Clock      => b ++= "Clock"
    case _: AsyncReset => b ++= "AsyncReset"
    case _: ResetType  => b ++= "Reset"
    case t: EnumType   => b ++= "UInt"; serialize(t.width)
    case t: UInt       => b ++= "UInt"; serialize(t.width)
    case t: SInt       => b ++= "SInt"; serialize(t.width)
    case t: Analog     => b ++= "Analog"; serialize(t.width)
    case t: Vec[_] =>
      val childClearDir = clearDir ||
        t.specifiedDirection == SpecifiedDirection.Input || t.specifiedDirection == SpecifiedDirection.Output
      // if Vector is a probe, don't emit Probe<...> on its elements
      serializeType(t.sample_element, childClearDir, info, checkProbe, checkConst, typeAliases)
      b += '['; b ++= t.length.toString; b += ']'
    // Handle aliased bundles: Emit an AliasType directly
    case t: HasTypeAlias if t.finalizedAlias.exists { typeAliases.contains(_) } =>
      b ++= t.finalizedAlias.get
    case t: Record => {
      val childClearDir = clearDir ||
        t.specifiedDirection == SpecifiedDirection.Input || t.specifiedDirection == SpecifiedDirection.Output
      // if Record is a probe, don't emit Probe<...> on its elements
      def eltField(elt: Data): Unit = {
        (childClearDir, firrtlUserDirOf(elt)) match {
          case (false, SpecifiedDirection.Flip | SpecifiedDirection.Input) =>
            b ++= "flip "
          case _ => ()
        }
        b ++= legalize(getRef(elt, info).name); b ++= " : "
        serializeType(elt, childClearDir, info, checkProbe, true, typeAliases)
      }
      if (!t._isOpaqueType) {
        b ++= "{ "
        var first = true
        t._elements.toIndexedSeq.reverse.map { case (_, e) =>
          if (!first) {
            b ++= ", "
          }
          first = false
          eltField(e)
        }
        b += '}'
      } else {
        serializeType(t._elements.head._2, childClearDir, info, checkProbe, true, typeAliases)
      }
    }
    case t: Property[_] =>
      // TODO can we not use FIRRTL types here?
      b ++= fir.Serializer.serialize(t.getPropertyType)
    case t: domain.Type => b ++= "Domain of "; b ++= t.domain.name;
  }

  private def serialize(name: String, param: Param)(implicit b: StringBuilder): Unit = param match {
    case p: IntParam    => b ++= name; b ++= " = "; b ++= p.value.toString
    case p: DoubleParam => b ++= name; b ++= " = "; b ++= p.value.toString
    case p: StringParam => b ++= name; b ++= " = "; b ++= firrtl.ir.StringLit(p.value).escape
    case p: PrintableParam => {
      val ctx = p.context._component.get
      val (fmt, _) = unpack(p.value, ctx, UnlocatableSourceInfo)
      b ++= name; b ++= " = "; b ++= firrtl.ir.StringLit(fmt).escape
    }
    case p: RawParam =>
      b ++= name; b ++= " = "
      b += '\''; b ++= p.value.replace("'", "\\'"); b += '\''
  }

  private def serialize(param: TestParam)(implicit b: StringBuilder, indent: Int): Unit = param match {
    case IntTestParam(value)    => b ++= value.toString
    case DoubleTestParam(value) => b ++= value.toString
    case StringTestParam(value) => b ++= firrtl.ir.StringLit(value).escape
    case ArrayTestParam(value) =>
      b ++= "[";
      value.zipWithIndex.foreach { case (value, i) =>
        if (i > 0) b ++= ", "
        serialize(value)
      }
      b ++= "]"
    case MapTestParam(value) =>
      b ++= "{"
      value.keys.toSeq.sorted.zipWithIndex.foreach { case (name, i) =>
        if (i > 0) b ++= ", "
        b ++= name; b ++= " = "; serialize(value(name))
      }
      b ++= "}"
  }

  private def serialize(
    port:        Port,
    typeAliases: Seq[String],
    topDir:      SpecifiedDirection = SpecifiedDirection.Unspecified
  )(implicit b: StringBuilder, indent: Int, suppressSourceInfo: Boolean): Unit = {
    val resolvedDir = SpecifiedDirection.fromParent(topDir, firrtlUserDirOf(port.id))
    val dir = resolvedDir match {
      case SpecifiedDirection.Unspecified | SpecifiedDirection.Output => "output"
      case SpecifiedDirection.Flip | SpecifiedDirection.Input         => "input"
    }
    val clearDir = resolvedDir match {
      case SpecifiedDirection.Input | SpecifiedDirection.Output     => true
      case SpecifiedDirection.Unspecified | SpecifiedDirection.Flip => false
    }
    b ++= dir; b += ' '
    b ++= legalize(getRef(port.id, port.sourceInfo).name)
    b ++= " : "
    val tpe = serializeType(port.id, clearDir, port.sourceInfo, true, true, typeAliases)
    if (port.associations.nonEmpty) {
      b ++= " domains ["
      port.associations.zipWithIndex.foreach { case (assoc, i) =>
        if (i > 0) b ++= ", "
        b ++= legalize(getRef(assoc, UnlocatableSourceInfo).name)
      }
      b ++= "]"
    }
    serialize(port.sourceInfo)
  }

  // TODO what is typeAliases for? Should it be a Set?
  private def serialize(component: Component, typeAliases: Seq[String])(
    implicit indent:    Int,
    suppressSourceInfo: Boolean
  ): Iterator[String] = {
    component match {
      case ctx @ DefModule(id, name, public, layers, ports, block) =>
        val start = {
          implicit val b = new StringBuilder
          doIndent(0)
          if (public)
            b ++= "public "
          b ++= "module "; b ++= legalize(name);
          layers.foreach { l => b ++= " enablelayer "; b ++= l.fullName }
          b ++= " :"; serialize(id._getSourceLocator)
          (ports ++ ctx.secretPorts).foreach { p => newLineAndIndent(1); serialize(p, typeAliases) }
          newLineNoIndent() // add a blank line between port declaration and body
          newLineNoIndent() // newline for body, serialize(body) will indent
          b.toString
        }
        Iterator(start) ++ serialize(block, ctx, typeAliases)(indent + 1, suppressSourceInfo)

      case ctx @ DefBlackBox(id, name, ports, topDir, params, knownLayers, requirements) =>
        implicit val b = new StringBuilder
        doIndent(0); b ++= "extmodule "; b ++= legalize(name);
        if (knownLayers.nonEmpty) {
          b ++= knownLayers.map(_.fullName).mkString(" knownlayer ", ", ", "")
        }
        if (requirements.nonEmpty) {
          b ++= requirements.map(r => fir.StringLit(r).escape).mkString(" requires ", ", ", "")
        }
        b ++= " :"; serialize(id._getSourceLocator)
        (ports ++ ctx.secretPorts).foreach { p => newLineAndIndent(1); serialize(p, typeAliases, topDir) }
        newLineAndIndent(1); b ++= "defname = "; b ++= id.desiredName
        params.keys.toList.sorted.foreach { name =>
          newLineAndIndent(1); b ++= "parameter "; serialize(name, params(name))
        }
        Iterator(b.toString)

      case ctx @ DefIntrinsicModule(id, name, ports, topDir, params) =>
        implicit val b = new StringBuilder
        doIndent(0); b ++= "intmodule "; b ++= legalize(name); b ++= " :"; serialize(id._getSourceLocator)
        (ports ++ ctx.secretPorts).foreach { p => newLineAndIndent(1); serialize(p, typeAliases, topDir) }
        newLineAndIndent(1); b ++= "intrinsic = "; b ++= id.intrinsic
        params.keys.toList.sorted.foreach { name =>
          newLineAndIndent(1); b ++= "parameter "; serialize(name, params(name))
        }
        Iterator(b.toString)

      case ctx @ DefClass(id, name, ports, block) =>
        val start = {
          implicit val b = new StringBuilder
          doIndent(0); b ++= "class "; b ++= name; b ++= " :"; serialize(id._getSourceLocator)
          (ports ++ ctx.secretPorts).foreach { p => newLineAndIndent(1); serialize(p, typeAliases) }
          newLineNoIndent() // add a blank line between port declaration and body
          newLineNoIndent() // newline for body, serialize(body) will indent
          b.toString
        }
        Iterator(start) ++ serialize(block, ctx, typeAliases)(indent + 1, suppressSourceInfo)

      case ctx @ DefTestMarker(kind, name, module, params, sourceInfo) =>
        implicit val b = new StringBuilder
        doIndent(0); b ++= kind.toString; b ++= " "; b ++= legalize(name); b ++= " of "; b ++= legalize(module.name);
        b ++= " :";
        serialize(sourceInfo)
        params.value.keys.toSeq.sorted.foreach { case name =>
          newLineAndIndent(1); b ++= name; b ++= " = "; serialize(params.value(name))
        }
        Iterator(b.toString)
    }
  }

  private def serialize(layer: Layer)(implicit b: StringBuilder, indent: Int, suppressSourceInfo: Boolean): Unit = {
    newLineAndIndent()
    b ++= "layer "
    b ++= layer.name
    b ++= ", "
    layer.config match {
      case LayerConfig.Extract(outputDir) =>
        b ++= "bind"
        outputDir match {
          case Some(d) =>
            b ++= ", "
            quote(d)
          case None => ()
        }
      case LayerConfig.Inline =>
        b ++= "inline"
    }
    b ++= " :"
    serialize(layer.sourceInfo)
    layer.children.foreach(serialize(_)(b, indent + 1, suppressSourceInfo))
  }

  private def serialize(layers: Seq[Layer])(implicit indent: Int, suppressSourceInfo: Boolean): Iterator[String] = {
    if (layers.nonEmpty) {
      implicit val b = new StringBuilder
      layers.foreach(serialize(_)(b, indent, suppressSourceInfo))
      newLineNoIndent()
      Iterator(b.toString)
    } else Iterator.empty
  }

  private def serialize(ta: DefTypeAlias)(implicit b: StringBuilder, indent: Int): Unit = {
    b ++= "type "; b ++= ta.name; b ++= " = "
    b ++= fir.Serializer.serialize(ta.underlying) // TODO can we not use FIRRTL types here?
    // serialize(ta.sourceInfo) TODO: Uncomment once firtool accepts infos for type aliases
  }

  private def serialize(fieldType: domain.Field.Type)(implicit b: StringBuilder, indent: Int): Unit = {
    b ++= {
      fieldType match {
        case domain.Field.Boolean => "Bool"
        case domain.Field.Integer => "Integer"
        case domain.Field.String  => "String"
      }
    }
  }

  private def serialize(domain: Domain)(implicit b: StringBuilder, indent: Int): Unit = {
    newLineAndIndent()
    b ++= "domain "
    b ++= domain.name
    b ++= " :"
    domain.fields.map { case (name, tpe) =>
      newLineAndIndent(1)
      b ++= name
      b ++= " : "
      serialize(tpe)
    }
    newLineNoIndent()
  }

  private def serializeDomains(domains: Seq[Domain])(implicit indent: Int): Iterator[String] = {
    if (domains.isEmpty)
      return Iterator.empty

    implicit val b = new StringBuilder
    domains.foreach(serialize)
    newLineNoIndent()
    Iterator(b.toString)
  }

  // TODO make Annotation serialization lazy
  private def serialize(circuit: Circuit, annotations: Seq[Annotation]): Iterator[String] = {
    implicit val indent:             Int = 0
    implicit val suppressSourceInfo: Boolean = circuit.suppressSourceInfo
    val prelude = {
      implicit val b = new StringBuilder
      b ++= s"FIRRTL version $version\n"
      b ++= "circuit "; b ++= legalize(circuit.name); b ++= " :";
      if (annotations.nonEmpty) {
        b ++= "%["; b ++= JsonProtocol.serialize(annotations); b ++= "]";
      }
      Iterator(b.toString)
    }
    val options = if (circuit.options.nonEmpty) {
      implicit val b = new StringBuilder
      circuit.options.foreach { optGroup =>
        newLineAndIndent(1)
        b ++= s"option ${optGroup.name} :"
        serialize(optGroup.sourceInfo)
        optGroup.cases.foreach { optCase =>
          newLineAndIndent(2)
          b ++= optCase.name
          serialize(optCase.sourceInfo)
        }
        newLineNoIndent()
      }
      Iterator(b.toString)
    } else Iterator.empty
    val typeAliases = if (circuit.typeAliases.nonEmpty) {
      implicit val b = new StringBuilder
      circuit.typeAliases.foreach(ta => { b += NewLine; doIndent(1); serialize(ta) })
      b += NewLine
      Iterator(b.toString)
    } else Iterator.empty
    val layers = serialize(circuit.layers)(indent + 1, suppressSourceInfo)
    val domains = serializeDomains(circuit.domains)(indent + 1)
    // TODO what is typeAliases for? Should it be a Set?
    val typeAliasesSeq: Seq[String] = circuit.typeAliases.map(_.name)
    prelude ++
      options ++
      typeAliases ++
      layers ++
      domains ++
      circuit.components.iterator.zipWithIndex.flatMap { case (m, i) =>
        val newline = Iterator(if (i == 0) s"$NewLine" else s"${NewLine}${NewLine}")
        newline ++ serialize(m, typeAliasesSeq)(indent + 1, suppressSourceInfo)
      } ++
      Iterator(s"$NewLine")
  }

  def lazily(circuit: Circuit, annotations: Seq[Annotation]): Iterable[String] = new Iterable[String] {
    def iterator = serialize(circuit, annotations)
  }
}
