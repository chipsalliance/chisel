// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.firrtl
import chisel3._
import chisel3.experimental._
import chisel3.internal.sourceinfo.{NoSourceInfo, SourceInfo, SourceLine, UnlocatableSourceInfo}
import firrtl.{ir => fir}
import chisel3.internal.{HasId, castToInt, throwException}

import scala.annotation.{nowarn, tailrec}
import scala.collection.immutable.Queue
import scala.collection.immutable.LazyList // Needed for 2.12 alias

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
    case Name(data) => (data.ref.name, List.empty)
    case FullName(data) => (data.ref.fullName(ctx), List.empty)
    case Percent => ("%%", List.empty)
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
    case _: NoSourceInfo => fir.NoInfo
    case SourceLine(fn, line, col) => fir.FileInfo(fir.StringLit(s"$fn $line:$col"))
  }

  def convert(op: PrimOp): fir.PrimOp = firrtl.PrimOps.fromString(op.name)

  def convert(dir: MemPortDirection): firrtl.MPortDir = dir match {
    case MemPortDirection.INFER => firrtl.MInfer
    case MemPortDirection.READ => firrtl.MRead
    case MemPortDirection.WRITE => firrtl.MWrite
    case MemPortDirection.RDWR => firrtl.MReadWrite
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
    case slit @ SLit(n, w) => fir.SIntLiteral(n, convert(w))
      val unsigned = if (n < 0) (BigInt(1) << slit.width.get) + n else n
      val uint = convert(ULit(unsigned, slit.width), ctx, info)
      fir.DoPrim(firrtl.PrimOps.AsSInt, Seq(uint), Seq.empty, fir.UnknownType)
    // TODO Simplify
    case fplit @ FPLit(n, w, bp) =>
      val unsigned = if (n < 0) (BigInt(1) << fplit.width.get) + n else n
      val uint = convert(ULit(unsigned, fplit.width), ctx, info)
      val lit = bp.asInstanceOf[KnownBinaryPoint].value
      fir.DoPrim(firrtl.PrimOps.AsFixedPoint, Seq(uint), Seq(lit), fir.UnknownType)
    case intervalLit @ IntervalLit(n, w, bp) =>
      val unsigned = if (n < 0) (BigInt(1) << intervalLit.width.get) + n else n
      val uint = convert(ULit(unsigned, intervalLit.width), ctx, info)
      val lit = bp.asInstanceOf[KnownBinaryPoint].value
      fir.DoPrim(firrtl.PrimOps.AsInterval, Seq(uint), Seq(n, n, lit), fir.UnknownType)
    case lit: ILit =>
      throwException(s"Internal Error! Unexpected ILit: $lit")
  }

  /** Convert Commands that map 1:1 to Statements */
  def convertSimpleCommand(cmd: Command, ctx: Component): Option[fir.Statement] = cmd match {
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
      Some(fir.DefWire(convert(info), e.name, extractType(id, info)))
    case e @ DefReg(info, id, clock) =>
      Some(fir.DefRegister(convert(info), e.name, extractType(id, info), convert(clock, ctx, info),
                           firrtl.Utils.zero, convert(getRef(id, info), ctx, info)))
    case e @ DefRegInit(info, id, clock, reset, init) =>
      Some(fir.DefRegister(convert(info), e.name, extractType(id, info), convert(clock, ctx, info),
                           convert(reset, ctx, info), convert(init, ctx, info)))
    case e @ DefMemory(info, id, t, size) =>
      Some(firrtl.CDefMemory(convert(info), e.name, extractType(t, info), size, false))
    case e @ DefSeqMemory(info, id, t, size, ruw) =>
      Some(firrtl.CDefMemory(convert(info), e.name, extractType(t, info), size, true, ruw))
    case e: DefMemPort[_] =>
      val info = e.sourceInfo
      Some(firrtl.CDefMPort(convert(e.sourceInfo), e.name, fir.UnknownType,
             e.source.fullName(ctx), Seq(convert(e.index, ctx, info), convert(e.clock, ctx, info)), convert(e.dir)))
    case Connect(info, loc, exp) =>
      Some(fir.Connect(convert(info), convert(loc, ctx, info), convert(exp, ctx, info)))
    case BulkConnect(info, loc, exp) =>
      Some(fir.PartialConnect(convert(info), convert(loc, ctx, info), convert(exp, ctx, info)))
    case Attach(info, locs) =>
      Some(fir.Attach(convert(info), locs.map(l => convert(l, ctx, info))))
    case DefInvalid(info, arg) =>
      Some(fir.IsInvalid(convert(info), convert(arg, ctx, info)))
    case e @ DefInstance(info, id, _) =>
      Some(fir.DefInstance(convert(info), e.name, id.name))
    case e @ Stop(_, info, clock, ret) =>
      Some(fir.Stop(convert(info), ret, convert(clock, ctx, info), firrtl.Utils.one, e.name))
    case e @ Printf(_, info, clock, pable) =>
      val (fmt, args) = unpack(pable, ctx)
      Some(fir.Print(convert(info), fir.StringLit(fmt),
                     args.map(a => convert(a, ctx, info)), convert(clock, ctx, info), firrtl.Utils.one, e.name))
    case e @ Verification(_, op, info, clk, pred, msg) =>
      val firOp = op match {
        case Formal.Assert => fir.Formal.Assert
        case Formal.Assume => fir.Formal.Assume
        case Formal.Cover => fir.Formal.Cover
      }
      Some(fir.Verification(firOp, convert(info), convert(clk, ctx, info),
        convert(pred, ctx, info), firrtl.Utils.one, fir.StringLit(msg), e.name))
    case _ => None
  }

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
  private case class WhenFrame(when: fir.Conditionally, outer: Queue[fir.Statement], alt: Boolean)

  /** Convert Chisel IR Commands into FIRRTL Statements
    *
    * @note ctx is needed because references to ports translate differently when referenced within
    *   the module in which they are defined vs. parent modules
    * @param cmds Chisel IR Commands to convert
    * @param ctx Component (Module) context within which we are translating
    * @return FIRRTL Statement that is equivalent to the input cmds
    */
  def convert(cmds: Seq[Command], ctx: Component): fir.Statement = {
    @tailrec
    def rec(acc: Queue[fir.Statement],
            scope: List[WhenFrame])
           (cmds: Seq[Command]): Seq[fir.Statement] = {
      if (cmds.isEmpty) {
        assert(scope.isEmpty)
        acc
      } else convertSimpleCommand(cmds.head, ctx) match {
        // Most Commands map 1:1
        case Some(stmt) =>
          rec(acc :+ stmt, scope)(cmds.tail)
        // When scoping logic does not map 1:1 and requires pushing/popping WhenFrames
        // Please see WhenFrame for more details
        case None => cmds.head match {
          case WhenBegin(info, pred) =>
            val when = fir.Conditionally(convert(info), convert(pred, ctx, info), fir.EmptyStmt, fir.EmptyStmt)
            val frame = WhenFrame(when, acc, false)
            rec(Queue.empty, frame +: scope)(cmds.tail)
          case WhenEnd(info, depth, _) =>
            val frame = scope.head
            val when = if (frame.alt) frame.when.copy(alt = fir.Block(acc))
                       else frame.when.copy(conseq = fir.Block(acc))
            // Check if this when has an else
            cmds.tail.headOption match {
              case Some(AltBegin(_)) =>
                assert(!frame.alt, "Internal Error! Unexpected when structure!") // Only 1 else per when
                rec(Queue.empty, frame.copy(when = when, alt = true) +: scope.tail)(cmds.drop(2))
              case _ => // Not followed by otherwise
                // If depth > 0 then we need to close multiple When scopes so we add a new WhenEnd
                // If we're nested we need to add more WhenEnds to ensure each When scope gets
                // properly closed
                val cmdsx = if (depth > 0) WhenEnd(info, depth - 1, false) +: cmds.tail  else cmds.tail
                rec(frame.outer :+ when, scope.tail)(cmdsx)
            }
          case OtherwiseEnd(info, depth) =>
            val frame = scope.head
            val when = frame.when.copy(alt = fir.Block(acc))
            // TODO For some reason depth == 1 indicates the last closing otherwise whereas
            //  depth == 0 indicates last closing when
            val cmdsx = if (depth > 1) OtherwiseEnd(info, depth - 1) +: cmds.tail else cmds.tail
            rec(scope.head.outer :+ when, scope.tail)(cmdsx)
        }
      }
    }
    fir.Block(rec(Queue.empty, List.empty)(cmds))
  }

  def convert(width: Width): fir.Width = width match {
    case UnknownWidth() => fir.UnknownWidth
    case KnownWidth(value) => fir.IntWidth(value)
  }

  def convert(bp: BinaryPoint): fir.Width = bp match {
    case UnknownBinaryPoint => fir.UnknownWidth
    case KnownBinaryPoint(value) => fir.IntWidth(value)
  }

  private def firrtlUserDirOf(d: Data): SpecifiedDirection = d match {
    case d: Vec[_] =>
      SpecifiedDirection.fromParent(d.specifiedDirection, firrtlUserDirOf(d.sample_element))
    case d => d.specifiedDirection
  }

  def extractType(data: Data, info: SourceInfo): fir.Type = extractType(data, false, info)

  def extractType(data: Data, clearDir: Boolean, info: SourceInfo): fir.Type = data match {
    case _: Clock => fir.ClockType
    case _: AsyncReset => fir.AsyncResetType
    case _: ResetType => fir.ResetType
    case d: EnumType => fir.UIntType(convert(d.width))
    case d: UInt => fir.UIntType(convert(d.width))
    case d: SInt => fir.SIntType(convert(d.width))
    case d: FixedPoint => fir.FixedType(convert(d.width), convert(d.binaryPoint))
    case d: Interval => fir.IntervalType(d.range.lowerBound, d.range.upperBound, d.range.firrtlBinaryPoint)
    case d: Analog => fir.AnalogType(convert(d.width))
    case d: Vec[_] => fir.VectorType(extractType(d.sample_element, clearDir, info), d.length)
    case d: Record =>
      val childClearDir = clearDir ||
        d.specifiedDirection == SpecifiedDirection.Input || d.specifiedDirection == SpecifiedDirection.Output
      def eltField(elt: Data): fir.Field = (childClearDir, firrtlUserDirOf(elt)) match {
        case (true, _) => fir.Field(getRef(elt, info).name, fir.Default, extractType(elt, true, info))
        case (false, SpecifiedDirection.Unspecified | SpecifiedDirection.Output) =>
          fir.Field(getRef(elt, info).name, fir.Default, extractType(elt, false, info))
        case (false, SpecifiedDirection.Flip | SpecifiedDirection.Input) =>
          fir.Field(getRef(elt, info).name, fir.Flip, extractType(elt, false, info))
      }
      fir.BundleType(d.elements.toIndexedSeq.reverse.map { case (_, e) => eltField(e) })
    }

  def convert(name: String, param: Param): fir.Param = param match {
    case IntParam(value) => fir.IntParam(name, value)
    case DoubleParam(value) => fir.DoubleParam(name, value)
    case StringParam(value) => fir.StringParam(name, fir.StringLit(value))
    case RawParam(value) => fir.RawStringParam(name, value)
  }

  def convert(port: Port, topDir: SpecifiedDirection = SpecifiedDirection.Unspecified): fir.Port = {
    val resolvedDir = SpecifiedDirection.fromParent(topDir, port.dir)
    val dir = resolvedDir match {
      case SpecifiedDirection.Unspecified | SpecifiedDirection.Output => fir.Output
      case SpecifiedDirection.Flip | SpecifiedDirection.Input => fir.Input
    }
    val clearDir = resolvedDir match {
      case SpecifiedDirection.Input | SpecifiedDirection.Output => true
      case SpecifiedDirection.Unspecified | SpecifiedDirection.Flip => false
    }
    val info = UnlocatableSourceInfo // Unfortunately there is no source locator for ports ATM
    val tpe = extractType(port.id, clearDir, info)
    fir.Port(fir.NoInfo, getRef(port.id, info).name, dir, tpe)
  }

  def convert(component: Component): fir.DefModule = component match {
    case ctx @ DefModule(_, name, ports, cmds) =>
      fir.Module(fir.NoInfo, name, ports.map(p => convert(p)), convert(cmds.toList, ctx))
    case ctx @ DefBlackBox(id, name, ports, topDir, params) =>
      fir.ExtModule(fir.NoInfo, name, ports.map(p => convert(p, topDir)), id.desiredName,
                    params.map { case (name, p) => convert(name, p) }.toSeq)
  }

  def convert(circuit: Circuit): fir.Circuit =
    fir.Circuit(fir.NoInfo, circuit.components.map(convert), circuit.name)

  // TODO Unclear if this should just be the default
  def convertLazily(circuit: Circuit): fir.Circuit = {
    val lazyModules = LazyList() ++ circuit.components
    fir.Circuit(fir.NoInfo, lazyModules.map(convert), circuit.name)
  }
}

