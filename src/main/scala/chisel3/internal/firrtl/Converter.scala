// See LICENSE for license details.

package chisel3.internal.firrtl
import chisel3._
import chisel3.core.SpecifiedDirection
import chisel3.experimental._
import chisel3.internal.sourceinfo.{NoSourceInfo, SourceLine, SourceInfo}
import firrtl.{ir => fir}
import chisel3.internal.throwException

import scala.annotation.tailrec
import scala.collection.immutable.{Queue}

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

  // TODO memoize?
  // TODO move into the Chisel IR?
  def convert(arg: Arg, ctx: Component): fir.Expression = arg match {
    case Node(id) => convert(id.getRef, ctx)
    case Ref(name) => fir.Reference(name, fir.UnknownType)
    case Slot(imm, name) => fir.SubField(convert(imm, ctx), name, fir.UnknownType)
    case Index(imm, ILit(idx)) =>
      fir.SubIndex(convert(imm, ctx), idx.toInt, fir.UnknownType)
    case Index(imm, value) =>
      fir.SubAccess(convert(imm, ctx), convert(value, ctx), fir.UnknownType)
    case ModuleIO(mod, name) =>
      if (mod eq ctx.id) fir.Reference(name, fir.UnknownType)
      else fir.SubField(fir.Reference(mod.getRef.name, fir.UnknownType), name, fir.UnknownType)
    case ULit(n, w) => fir.UIntLiteral(n, convert(w))
    case slit @ SLit(n, w) => fir.SIntLiteral(n, convert(w))
      val unsigned = if (n < 0) (BigInt(1) << slit.width.get) + n else n
      val uint = convert(ULit(unsigned, slit.width), ctx)
      fir.DoPrim(firrtl.PrimOps.AsSInt, Seq(uint), Seq.empty, fir.UnknownType)
    // TODO Simplify
    case fplit @ FPLit(n, w, bp) =>
      val unsigned = if (n < 0) (BigInt(1) << fplit.width.get) + n else n
      val uint = convert(ULit(unsigned, fplit.width), ctx)
      val lit = bp.asInstanceOf[KnownBinaryPoint].value
      fir.DoPrim(firrtl.PrimOps.AsFixedPoint, Seq(uint), Seq(lit), fir.UnknownType)
    case lit: ILit => throwException(s"Internal Error! Unexpected ILit: $lit")
  }

  // alt indicates to a WhenEnd whether we're closing an alt or just a regular when
  // TODO we should probably have a different structure in the IR to close elses
  private case class WhenFrame(when: fir.Conditionally, outer: Queue[fir.Statement], alt: Boolean)

  // Whens markers are flat so scope must be inferred
  // TODO refactor
  def convert(cmds: Seq[Command], ctx: Component): fir.Statement = {
    @tailrec
    def rec(acc: Queue[fir.Statement],
            scope: List[WhenFrame])
           (cmds: Seq[Command]): Seq[fir.Statement] = {
      if (cmds.isEmpty) {
        assert(scope.isEmpty)
        acc
      } else cmds.head match {
        case e: DefPrim[_] =>
          val consts = e.args.collect { case ILit(i) => i }
          val args = e.args.flatMap {
            case _: ILit => None
            case other => Some(convert(other, ctx))
          }
          val expr = e.op.name match {
            case "mux" =>
              assert(args.size == 3, s"Mux with unexpected args: $args")
              fir.Mux(args(0), args(1), args(2), fir.UnknownType)
            case _ =>
              fir.DoPrim(convert(e.op), args, consts, fir.UnknownType)
          }
          val node = fir.DefNode(convert(e.sourceInfo), e.name, expr)
          rec(acc :+ node, scope)(cmds.tail)
        case e @ DefWire(info, id) =>
          val wire = fir.DefWire(convert(info), e.name, extractType(id))
          rec(acc :+ wire, scope)(cmds.tail)
        case e @ DefReg(info, id, clock) =>
          val reg = fir.DefRegister(convert(info), e.name, extractType(id), convert(clock, ctx),
                                    firrtl.Utils.zero, convert(id.getRef, ctx))
          rec(acc :+ reg, scope)(cmds.tail)
        case e @ DefRegInit(info, id, clock, reset, init) =>
          val reg = fir.DefRegister(convert(info), e.name, extractType(id), convert(clock, ctx),
                                    convert(reset, ctx), convert(init, ctx))
          rec(acc :+ reg, scope)(cmds.tail)
        case e @ DefMemory(info, id, t, size) =>
          val mem = firrtl.CDefMemory(convert(info), e.name, extractType(t), size, false)
          rec(acc :+ mem, scope)(cmds.tail)
        case e @ DefSeqMemory(info, id, t, size) =>
          val mem = firrtl.CDefMemory(convert(info), e.name, extractType(t), size, true)
          rec(acc :+ mem, scope)(cmds.tail)
        case e: DefMemPort[_] =>
          val port = firrtl.CDefMPort(convert(e.sourceInfo), e.name, fir.UnknownType,
            e.source.fullName(ctx), Seq(convert(e.index, ctx), convert(e.clock, ctx)), convert(e.dir))
          rec(acc :+ port, scope)(cmds.tail)
        case Connect(info, loc, exp) =>
          val con = fir.Connect(convert(info), convert(loc, ctx), convert(exp, ctx))
          rec(acc :+ con, scope)(cmds.tail)
        case BulkConnect(info, loc, exp) =>
          val con = fir.PartialConnect(convert(info), convert(loc, ctx), convert(exp, ctx))
          rec(acc :+ con, scope)(cmds.tail)
        case Attach(info, locs) =>
          val att = fir.Attach(convert(info), locs.map(l => convert(l, ctx)))
          rec(acc :+ att, scope)(cmds.tail)
        case DefInvalid(info, arg) =>
          val inv = fir.IsInvalid(convert(info), convert(arg, ctx))
          rec(acc :+ inv, scope)(cmds.tail)
        case e @ DefInstance(info, id, _) =>
          val inst = fir.DefInstance(convert(info), e.name, id.name)
          rec(acc :+ inst, scope)(cmds.tail)
        case WhenBegin(info, pred) =>
          val when = fir.Conditionally(convert(info), convert(pred, ctx),
                                       fir.EmptyStmt, fir.EmptyStmt)
          val frame = WhenFrame(when, acc, false)
          rec(Queue.empty, frame +: scope)(cmds.tail)
        case end @ WhenEnd(info, depth, _) =>
          val frame = scope.head
          val when = if (frame.alt) frame.when.copy(alt = fir.Block(acc))
                     else frame.when.copy(conseq = fir.Block(acc))
          cmds.tail.headOption match {
            case Some(AltBegin(_)) =>
              assert(!frame.alt, "Internal Error! Unexpected when structure!")
              rec(Queue.empty, frame.copy(when = when, alt = true) +: scope.tail)(cmds.drop(2))
            case _ => // Not followed by otherwise
              val cmdsx = if (depth > 0) WhenEnd(info, depth - 1, false) +: cmds.tail  else cmds.tail
              rec(frame.outer :+ when, scope.tail)(cmdsx)
          }
        case OtherwiseEnd(info, depth) =>
          val frame = scope.head
          val when = frame.when.copy(alt = fir.Block(acc))
          val cmdsx = if (depth > 1) OtherwiseEnd(info, depth - 1) +: cmds.tail else cmds.tail
          rec(scope.head.outer :+ when, scope.tail)(cmdsx)
        case Stop(info, clock, ret) =>
          val stop = fir.Stop(convert(info), ret, convert(clock, ctx), firrtl.Utils.one)
          rec(acc :+ stop, scope)(cmds.tail)
        case Printf(info, clock, pable) =>
          val (fmt, args) = unpack(pable, ctx)
          val p = fir.Print(convert(info), fir.StringLit(fmt),
                            args.map(a => convert(a, ctx)), convert(clock, ctx), firrtl.Utils.one)
          rec(acc :+ p, scope)(cmds.tail)
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

  def extractType(data: Data, clearDir: Boolean = false): fir.Type = data match {
    case _: Clock => fir.ClockType
    case d: UInt => fir.UIntType(convert(d.width))
    case d: SInt => fir.SIntType(convert(d.width))
    case d: FixedPoint => fir.FixedType(convert(d.width), convert(d.binaryPoint))
    case d: Analog => fir.AnalogType(convert(d.width))
    case d: Vec[_] => fir.VectorType(extractType(d.sample_element, clearDir), d.length)
    case d: Record =>
      val childClearDir = clearDir ||
        d.specifiedDirection == SpecifiedDirection.Input || d.specifiedDirection == SpecifiedDirection.Output
      def eltField(elt: Data): fir.Field = (childClearDir, firrtlUserDirOf(elt)) match {
        case (true, _) => fir.Field(elt.getRef.name, fir.Default, extractType(elt, true))
        case (false, SpecifiedDirection.Unspecified | SpecifiedDirection.Output) =>
          fir.Field(elt.getRef.name, fir.Default, extractType(elt, false))
        case (false, SpecifiedDirection.Flip | SpecifiedDirection.Input) =>
          fir.Field(elt.getRef.name, fir.Flip, extractType(elt, false))
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
    val tpe = extractType(port.id, clearDir)
    fir.Port(fir.NoInfo, port.id.getRef.name, dir, tpe)
  }

  def convert(component: Component): fir.DefModule = component match {
    case ctx @ DefModule(_, name, ports, cmds) =>
      fir.Module(fir.NoInfo, name, ports.map(p => convert(p)), convert(cmds, ctx))
    case ctx @ DefBlackBox(id, name, ports, topDir, params) =>
      fir.ExtModule(fir.NoInfo, name, ports.map(p => convert(p, topDir)), id.desiredName,
                    params.map { case (name, p) => convert(name, p) }.toSeq)
  }

  def convert(circuit: Circuit): fir.Circuit =
    fir.Circuit(fir.NoInfo, circuit.components.map(convert), circuit.name)
}

