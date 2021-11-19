// SPDX-License-Identifier: Apache-2.0

package firrtl.backends.experimental.rtlil

import java.io.Writer
import firrtl._
import firrtl.PrimOps._
import firrtl.ir._
import firrtl.Utils.{throwInternalError, _}
import firrtl.WrappedExpression._
import firrtl.traversals.Foreachers._
import firrtl.annotations._
import firrtl.options.Viewer.view
import firrtl.options.{CustomFileEmission, Dependency}
import firrtl.passes.LowerTypes
import firrtl.passes.MemPortUtils.memPortField
import firrtl.stage.{FirrtlOptions, TransformManager}

import scala.annotation.tailrec
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.language.postfixOps

case class EmittedRtlilCircuitAnnotation(name: String, value: String, outputSuffix: String)
    extends NoTargetAnnotation
    with CustomFileEmission {
  override protected def baseFileName(annotations: AnnotationSeq): String =
    view[FirrtlOptions](annotations).outputFileName.getOrElse(name)
  override protected def suffix: Option[String] = Some(outputSuffix)
  override def getBytes:         Iterable[Byte] = value.getBytes
}
case class EmittedRtlilModuleAnnotation(name: String, value: String, outputSuffix: String)
    extends NoTargetAnnotation
    with CustomFileEmission {
  override protected def baseFileName(annotations: AnnotationSeq): String =
    view[FirrtlOptions](annotations).outputFileName.getOrElse(name)
  override protected def suffix: Option[String] = Some(outputSuffix)
  override def getBytes:         Iterable[Byte] = value.getBytes
}

private[firrtl] class RtlilEmitter extends SeqTransform with Emitter with DependencyAPIMigration {

  override def prerequisites: Seq[TransformManager.TransformDependency] =
    Seq(
      Dependency[firrtl.transforms.CombineCats],
      Dependency(firrtl.passes.memlib.VerilogMemDelays)
    ) ++: firrtl.stage.Forms.LowFormOptimized

  override def outputSuffix: String = ".il"
  val tab = "  "

  override def transforms: Seq[Transform] = new TransformManager(prerequisites).flattenedTransformOrder

  def emit(state: CircuitState, writer: Writer): Unit = {
    val cs = runTransforms(state)
    val emissionOptions = new EmissionOptions(cs.annotations)
    val moduleMap = cs.circuit.modules.map(m => m.name -> m).toMap
    cs.circuit.modules.foreach {
      case DescribedMod(d, pds, m: Module) =>
        val renderer = new RtlilRender(d, pds, m, moduleMap, cs.circuit.main, emissionOptions)(writer)
        renderer.emit_rtlil()
      case m: Module =>
        val renderer = new RtlilRender(m, moduleMap, cs.circuit.main, emissionOptions)(writer)
        renderer.emit_rtlil()
      case _ => // do nothing
    }
  }

  override def execute(state: CircuitState): CircuitState = {
    val writerToString =
      (writer: java.io.StringWriter) => writer.toString.replaceAll("""(?m) +$""", "") // trim trailing whitespace

    val newAnnos = state.annotations.flatMap {
      case EmitCircuitAnnotation(a) if this.getClass == a =>
        val writer = new java.io.StringWriter
        emit(state, writer)
        Seq(
          EmittedRtlilModuleAnnotation(state.circuit.main, writerToString(writer), outputSuffix)
        )

      case EmitAllModulesAnnotation(a) if this.getClass == a =>
        val cs = runTransforms(state)
        val emissionOptions = new EmissionOptions(cs.annotations)
        val moduleMap = cs.circuit.modules.map(m => m.name -> m).toMap

        cs.circuit.modules.flatMap {
          case DescribedMod(d, pds, module: Module) =>
            val writer = new java.io.StringWriter
            val renderer = new RtlilRender(d, pds, module, moduleMap, cs.circuit.main, emissionOptions)(writer)
            renderer.emit_rtlil()
            Some(
              EmittedRtlilModuleAnnotation(module.name, writerToString(writer), outputSuffix)
            )
          case module: Module =>
            val writer = new java.io.StringWriter
            val renderer = new RtlilRender(module, moduleMap, cs.circuit.main, emissionOptions)(writer)
            renderer.emit_rtlil()
            Some(
              EmittedRtlilModuleAnnotation(module.name, writerToString(writer), outputSuffix)
            )
          case _ => None
        }
      case _ => Seq()
    }
    state.copy(annotations = newAnnos ++ state.annotations)
  }

  private class RtlilRender(
    description:      Seq[Description],
    portDescriptions: Map[String, Seq[Description]],
    m:                Module,
    moduleMap:        Map[String, DefModule],
    circuitName:      String,
    emissionOptions:  EmissionOptions
  )(
    implicit writer: Writer) {
    def this(
      m:               Module,
      moduleMap:       Map[String, DefModule],
      circuitName:     String,
      emissionOptions: EmissionOptions
    )(
      implicit writer: Writer
    ) = {
      this(Seq(), Map.empty, m, moduleMap, circuitName, emissionOptions)(writer)
    }

    private val netlist:   mutable.LinkedHashMap[WrappedExpression, InfoExpr] = mutable.LinkedHashMap()
    private val namespace: Namespace = Namespace(m)

    private val portdefs:         ArrayBuffer[Seq[Any]] = ArrayBuffer[Seq[Any]]()
    private val declares:         ArrayBuffer[Seq[Any]] = ArrayBuffer()
    private val instdeclares:     mutable.Map[String, InstInfo] = mutable.Map()
    private val assigns:          ArrayBuffer[Seq[Any]] = ArrayBuffer()
    private val attachSynAssigns: ArrayBuffer[Seq[Any]] = ArrayBuffer()
    private val processes:        ArrayBuffer[Seq[Any]] = ArrayBuffer()
    // Used to determine type of initvar for initializing memories
    private val initials:     ArrayBuffer[Seq[Any]] = ArrayBuffer()
    private val formals:      ArrayBuffer[Seq[Any]] = ArrayBuffer()
    private val moduleTarget: ModuleTarget = CircuitTarget(circuitName).module(m.name)

    private def getLeadingTabs(x: Any): String = {
      x match {
        case seq: Seq[_] =>
          val head = seq.takeWhile(_ == tab).mkString
          val tail = seq.dropWhile(_ == tab).headOption.map(getLeadingTabs).getOrElse(tab)
          head + tail
        case _ => tab
      }
    }

    private def emit(x: Any)(implicit w: Writer): Unit = {
      this.emitCol(x, 0, getLeadingTabs(x))(writer)
    }

    private def emit(x: Any, top: Int)(implicit w: Writer): Unit = {
      emitCol(x, top, "")(writer)
    }

    private def emitCol(x: Any, top: Int, tabs: String)(implicit w: Writer): Unit = {
      x match {
        case e: SrcInfo   => w.write(e.str_rep)
        case e: Reference => w.write(ref_to_name(e))
        case e: ValidIf   => emitCol(Seq(e.value), top + 1, tabs)(writer)
        case e: WSubField => w.write(SrcInfo(e).str_rep)
        case e: WSubAccess =>
          w.write("\\" + s"${LowerTypes.loweredName(e.expr)} [ ${LowerTypes.loweredName(e.index)} ]")
        case e: Literal => w.write(bigint_to_str_rep(e.value, get_type_width(e.tpe)))
        case t: GroundType => w.write(stringify(t))
        case t: VectorType =>
          emit(t.tpe, top + 1)(writer)
          w.write(s"[${t.size - 1}:0]")
        case s: String => w.write(s)
        case i: Int    => w.write(i.toString)
        case i: Long   => w.write(i.toString)
        case i: BigInt => w.write(bigint_to_str_rep(i, if (i > 0) i.bitLength else i.bitLength + 1))
        case i: Info =>
          infos_to_attr(i) match {
            case Some(attr) =>
              w.write(attr)
            case None =>
          }
        case s: Seq[Any] =>
          s.foreach { e => emitCol(e, top + 1, tabs)(writer) }
          if (top == 0)
            w.write("\n")
        case x => throwInternalError(s"trying to emit unsupported operation: $x")
      }
    }

    private def build_netlist(s: Statement): Unit = {
      s.foreach(build_netlist)
      s match {
        case sx: Connect   => netlist(sx.loc) = InfoExpr(sx.info, sx.expr)
        case _:  IsInvalid => error("Should have removed these!")
        // TODO Since only register update and memories use the netlist anymore, I think nodes are unnecessary
        case sx: DefNode =>
          val e = WRef(sx.name, sx.value.tpe, NodeKind, SourceFlow)
          netlist(e) = InfoExpr(sx.info, sx.value)
        case _ =>
      }
    }

    @tailrec
    private def remove_root(ex: Expression): Expression = ex match {
      case ex: WSubField =>
        ex.expr match {
          case e: WSubField => remove_root(e)
          case _: WRef      => WRef(ex.name, ex.tpe, InstanceKind, UnknownFlow)
        }
      case _ => throwInternalError(s"shouldn't be here: remove_root($ex)")
    }

    private def stringify(tpe: GroundType): String = tpe match {
      case _: UIntType | _: AnalogType =>
        val wx = bitWidth(tpe)
        if (wx > 1) s"width $wx" else ""
      case _: SIntType =>
        val wx = bitWidth(tpe)
        if (wx > 1) s"signed width $wx" else "signed"
      case ClockType | AsyncResetType => ""
      case _                          => throwInternalError(s"trying to write unsupported type in the Rtlil Emitter: $tpe")
    }

    private def stringify(param: Param): String = param match {
      case IntParam(name, value) =>
        val lit =
          if (value.isValidInt) {
            s"$value"
          } else {
            val blen = value.bitLength
            if (value > 0) s"$blen'd$value" else s"-${blen + 1}'sd${value.abs}"
          }
        s"parameter \\$name $lit"
      case DoubleParam(name, value)    => s"parameter \\$name $value"
      case StringParam(name, value)    => s"parameter \\$name ${value.verilogEscape}"
      case RawStringParam(name, value) => s"parameter \\$name $value"
    }

    // turn strings into Seq[String] verilog comments
    private def build_comment(desc: String): Seq[Seq[String]] = {
      val lines = desc.split("\n").toSeq
      lines.tail.map {
        case ""       => Seq("#")
        case nonEmpty => Seq("#", nonEmpty)
      }
    }
    private def build_attribute(attr: String): Seq[Seq[String]] = {
      Seq(Seq("attribute \\") ++ Seq(attr))
    }

    private def build_description(d: Seq[Description]): Seq[Seq[String]] = d.flatMap {
      case DocString(desc) => build_comment(desc.string)
      case Attribute(attr) => build_attribute(attr.string)
    }

    // Turn ports into Seq[String] and add to portdefs
    private def build_ports(): Unit = {
      def padToMax(strs: Seq[String]): Seq[String] = {
        val len = if (strs.nonEmpty) strs.map(_.length).max else 0
        strs.map(_.padTo(len, ' '))
      }

      // Turn directions into strings (and AnalogType into inout)
      val dirs = m.ports.map {
        case Port(_, _, dir, tpe) =>
          (dir, tpe) match {
            case (_, AnalogType(_)) => "inout " // padded to length of output
            case (Input, _)         => "input "
            case (Output, _)        => "output"
          }
      }
      // Turn types into strings, all ports must be GroundTypes
      val tpes = m.ports.map {
        case Port(_, _, _, tpe: GroundType) => stringify(tpe)
        case port: Port => error(s"Trying to emit non-GroundType Port $port")
      }

      // dirs are already padded
      (dirs, padToMax(tpes), m.ports).zipped.toSeq.zipWithIndex.foreach {
        case ((dir, tpe, Port(info, name, _, _)), i) =>
          portDescriptions.get(name).map { d =>
            portdefs += Seq("")
            portdefs ++= build_description(d)
          }
          portdefs += Seq("wire ", tpe, " ", dir, " ", i + 1, " \\", name, info)
      }
    }

    private def infos_to_attr(info: Info): Option[String] = {
      def info_extract(info: Info, prev: Seq[String] = Seq()): Seq[String] = info match {
        case FileInfo(str) =>
          val (file, line, col) = FileInfo(str).split
          prev :+ (file + ":" + line + "." + col)
        case MultiInfo(infos) =>
          infos.foldLeft(prev)((a, b) => {
            info_extract(b, a)
          })
        case NoInfo =>
          prev
      }
      val srcinfo = info_extract(info)
      if (srcinfo.isEmpty)
        Option.empty
      else
        Option("attribute \\src \"" + srcinfo.mkString("|") + "\"")
    }

    private def string_to_rtlil_name(name: String): String = {
      if (name.head == '_') {
        "$" + name
      } else {
        "\\" + name
      }
    }

    private def ref_to_name(ref: Reference): String = {
      string_to_rtlil_name(ref.name)
    }

    private def regUpdate(r: Expression, clk: Expression, reset: Expression, init: Expression) = {
      val procName = namespace.newName("$process$" + this.m.name)
      val regTempName = "\\" + r.serialize + procName
      val loweredReset = SrcInfo(reset)
      val loweredClk = SrcInfo(clk)
      val loweredInit = SrcInfo(init)
      val loweredReg = SrcInfo(r)
      def addUpdate(info: Info, expr: Expression, tabs: Seq[String]): Seq[Seq[Any]] = expr match {
        case m: Mux =>
          if (m.tpe == ClockType) throw EmitterException("Cannot emit clock muxes directly")
          if (m.tpe == AsyncResetType) throw EmitterException("Cannot emit async reset muxes directly")

          val (eninfo, tinfo, finfo) = MultiInfo.demux(info)
          lazy val _if: Seq[Seq[Any]] =
            Seq(Seq(tabs, eninfo), Seq(tabs, "switch ", SrcInfo(m.cond, eninfo).str_rep)) ++ (
              if (infos_to_attr(tinfo).nonEmpty)
                Seq(Seq(tabs, tab, tinfo), Seq(tabs, tab, "case 1'1"))
              else
                Seq(Seq(tabs, tab, "case 1'1"))
            )
          lazy val _else: Seq[Seq[Any]] = infos_to_attr(finfo) match {
            case Some(_) =>
              Seq(Seq(tabs, tab, finfo), Seq(tabs, tab, "case"))
            case None =>
              Seq(Seq(tabs, tab, "case"))
          }
          lazy val _ifNot: Seq[Seq[Any]] =
            Seq(Seq(tabs, eninfo), Seq(tabs, "switch ", SrcInfo(m.cond, eninfo).str_rep)) ++ (
              if (infos_to_attr(finfo).nonEmpty)
                Seq(Seq(tabs, tab, finfo), Seq(tabs, tab, "case 1'0"))
              else
                Seq(Seq(tabs, tab, "case 1'0"))
            )
          lazy val _end = Seq(Seq(tabs, "end"))
          lazy val _true = addUpdate(tinfo, m.tval, Seq(tab, tab) ++ tabs)
          lazy val _false = addUpdate(finfo, m.fval, Seq(tab, tab) ++ tabs)
          /* For a Mux assignment, there are five possibilities, with one subcase for asynchronous reset:
           *   1. Both the true and false condition are self-assignments; do nothing
           *   2. The true condition is a self-assignment; invert the false condition and use that only
           *   3. The false condition is a self-assignment
           *      a) The reset is asynchronous; emit both 'if' and a trivial 'else' to avoid latches
           *      b) The reset is synchronous; skip the false condition
           *   4. The false condition is a Mux; use the true condition and use 'else if' for the false condition
           *   5. Default; use both the true and false conditions
           */
          (m.tval, m.fval) match {
            case (t, f) if weq(t, r) && weq(f, r) => Nil
            case (t, _) if weq(t, r)              => _ifNot ++ _false ++ _end
            case (_, f) if weq(f, r) =>
              m.cond.tpe match {
                case AsyncResetType => (_if ++ _true ++ _else) ++ _true ++ _end
                case _              => _if ++ _true ++ _end
              }
            case _ => (_if ++ _true ++ _else) ++ _false ++ _end
          }
        case e =>
          Seq(Seq(tabs, "assign ", regTempName, " ", SrcInfo(e, info).str_rep))
      }
      if (weq(init, r)) { // Synchronous Reset
        val InfoExpr(info, e) = netlist(r)
        processes += Seq(info)
        processes += Seq("wire ", r.tpe, " ", regTempName)
        processes += Seq("process ", procName)
        processes += Seq("assign ", regTempName, " ", loweredInit.str_rep)
        processes ++= addUpdate(info, e, Seq(tab))
        processes += Seq(tab, "sync posedge ", clk)
        processes += Seq(tab, tab, "update ", SrcInfo(r).str_rep, " ", regTempName)
        processes += Seq("end")
      } else { // Asynchronous Reset
        assert(reset.tpe == AsyncResetType, "Error! Synchronous reset should have been removed!")
        val tv = init
        val InfoExpr(finfo, fv) = netlist(r)
        processes += Seq(finfo)
        processes += Seq("wire ", r.tpe, " ", regTempName)
        processes += Seq("process ", procName)
        processes += Seq("assign ", regTempName, " ", loweredInit.str_rep)
        processes ++= addUpdate(NoInfo, Mux(reset, tv, fv, mux_type_and_widths(tv, fv)), Seq.empty)
        processes += Seq("sync posedge ", loweredClk.str_rep)
        processes += Seq(tab, "update ", loweredReset.str_rep, " ", regTempName)
        processes += Seq("sync posedge ", reset)
        processes += Seq(tab, "update ", loweredReg.str_rep, " ", regTempName)
        processes += Seq("end")
      }
    }

    private def bigint_to_str_rep(bigInt: BigInt, width: BigInt): String = {
      if (width > 31) {
        var bigboi = bigInt
        var widthcnt = width
        var concatlist: Seq[String] = List()

        while (widthcnt > 32) {
          val lowbits = bigboi & 0xffffffff
          concatlist = concatlist :+ "%d'%s".format(32, lowbits.toString(2))
          bigboi >>= 32
          widthcnt -= 32
        }
        concatlist = concatlist :+ "%d'%s".format(widthcnt, bigboi.toString(2))
        "{ " + concatlist.reverse.mkString(" ") + " }"
      } else
        "%d'%s".format(width, bigInt.toString(2))
    }

    private case class InstInfo(inst_name: String, mod_name: String, info: Info) {
      val conns:  mutable.Map[String, String] = mutable.Map()
      var params: Seq[String] = Seq()
      def getConnection(port: String): Option[String] = {
        conns.get(port)
      }
      def addConnection(port: String, targetValue: String): Unit = {
        conns(port) = targetValue
      }
    }

    private case class SrcInfo(str_rep: String, signed: Boolean, width: BigInt)
    private object SrcInfo {
      def apply(e: Expression, i: Info = NoInfo): SrcInfo = e match {
        case InfoExpr(info, expr) =>
          SrcInfo(expr, MultiInfo(info, i))
        case x: Reference =>
          SrcInfo(ref_to_name(x), x.tpe.isInstanceOf[SIntType], get_type_width(x.tpe))
        case x: Literal =>
          val width = x.width.asInstanceOf[IntWidth].width
          SrcInfo(bigint_to_str_rep(x.value, width), x.isInstanceOf[SIntLiteral], width)
        case x @ DoPrim(op, args, consts, tpe) =>
          op match {
            case Cat =>
              SrcInfo(
                Seq(" { ", args.map(SrcInfo(_).str_rep).mkString(" "), " }").mkString,
                tpe.isInstanceOf[SIntType],
                get_type_width(tpe)
              )
            case Head =>
              val src0 = SrcInfo(args.head)
              SrcInfo(
                Seq(src0.str_rep, " [", (src0.width - 1).toInt, ":", consts.head.toInt, "]").mkString,
                tpe.isInstanceOf[SIntType],
                get_type_width(tpe)
              )
            case Tail =>
              val src0 = SrcInfo(args.head)
              SrcInfo(
                Seq(src0.str_rep, " [", (src0.width - 1 - consts.head).toInt, ":0]").mkString,
                tpe.isInstanceOf[SIntType],
                get_type_width(tpe)
              )
            case Pad =>
              val src0 = SrcInfo(args.head)
              if (src0.width >= consts.head)
                SrcInfo(
                  Seq(src0.str_rep, " [", (consts.head - 1).toInt, ":0]").mkString,
                  tpe.isInstanceOf[SIntType],
                  get_type_width(tpe)
                )
              else if (src0.signed)
                SrcInfo(
                  Seq(
                    " { ",
                    s"${src0.str_rep} [${src0.width - 1}] " * (consts.head - src0.width).toInt,
                    src0.str_rep,
                    " }"
                  ).mkString,
                  tpe.isInstanceOf[SIntType],
                  get_type_width(tpe)
                )
              else
                SrcInfo(
                  Seq(" { ", (consts.head - src0.width).toInt, "'0 ", src0.str_rep, " }").mkString,
                  tpe.isInstanceOf[SIntType],
                  get_type_width(tpe)
                )
            case _ =>
              val tempNetName = namespace.newName("$_PRIM_EX")
              if (infos_to_attr(i).nonEmpty) declares += Seq(i)
              declares += Seq("wire ", x.tpe, " ", tempNetName)
              assigns ++= output_expr(tempNetName, x, i)
              SrcInfo(tempNetName, x.tpe.isInstanceOf[SIntType], get_type_width(x.tpe))
          }
        case x @ SubField(Reference(modname, _, InstanceKind, _), portname, _, _) =>
          val currentPortConn = instdeclares(modname).getConnection(portname)
          if (currentPortConn.isEmpty) {
            val tempNetName = "\\" + LowerTypes.loweredName(x)
            if (infos_to_attr(i).nonEmpty) declares += Seq(i)
            declares += Seq("wire ", x.tpe, " ", tempNetName)
            instdeclares(modname).addConnection(portname, tempNetName)
            SrcInfo(tempNetName, x.tpe.isInstanceOf[SIntType], get_type_width(x.tpe))
          } else {
            SrcInfo(currentPortConn.get, x.tpe.isInstanceOf[SIntType], get_type_width(x.tpe))
          }
        case x: SubField =>
          SrcInfo("\\" + LowerTypes.loweredName(x), x.tpe.isInstanceOf[SIntType], get_type_width(x.tpe))
        case x: Mux =>
          val tempNetName = namespace.newName("$_MUX_EX")
          if (infos_to_attr(i).nonEmpty) declares += Seq(i)
          declares += Seq("wire ", x.tpe, " ", tempNetName)
          assigns ++= output_expr(tempNetName, e, i)
          SrcInfo(tempNetName, x.tpe.isInstanceOf[SIntType], get_type_width(x.tpe))
        case x =>
          throw EmitterException(s"Internal error! unhandled value $x passed to SrcInfo()")
      }
    }

    private def emit_streams(): Unit = {
      build_description(description).foreach(emit(_))
      emit(Seq("# Generated by firrtl.RtlilEmitter (FIRRTL Version ", BuildInfo.version + ")"))
      emit(Seq("autoidx 1"))
      emit(Seq("attribute \\cells_not_processed 1"))
      emit(Seq("module \\", m.name, m.info))
      for (x <- portdefs) emit(Seq(tab, x))
      for (x <- declares) emit(Seq(tab, x))
      for ((_, x) <- instdeclares) {
        emit(Seq(tab, "attribute \\module_not_derived 1"))
        emit(Seq(tab, x.info))
        emit(Seq(tab, "cell \\", x.mod_name, " \\", x.inst_name))
        for (p <- x.params) emit(Seq(tab, tab, p))
        for ((a, b) <- x.conns) emit(Seq(tab, tab, "connect \\", a, " ", b))
        emit(Seq(tab, "end"))
      }
      for (x <- assigns) emit(Seq(tab, x))
      for (x <- processes) emit(Seq(tab, x))
      for (x <- attachSynAssigns) emit(Seq(tab, x))
      for (x <- initials) emit(Seq(tab, x))
      emit(Seq("end"))
      emit(Seq())
    }

    private def primop_to_cell(p: PrimOp): String = p match {
      case Not  => "$not"
      case Neg  => "$neg"
      case Andr => "$reduce_and"
      case Orr  => "$reduce_or"
      case Xorr => "$reduce_xor"
      case And  => "$and"
      case Or   => "$or"
      case Xor  => "$xor"
      case Shl  => "$shl"
      case Dshl => "$shl"
      case Eq   => "$eq"
      case Lt   => "$lt"
      case Leq  => "$le"
      case Neq  => "$ne"
      case Geq  => "$ge"
      case Gt   => "$gt"
      case Add  => "$add"
      case Addw => "$add"
      case Sub  => "$sub"
      case Subw => "$sub"
      case Mul  => "$mul"
      case Div  => "$div"
      case Rem  => "$rem"
      case _ =>
        throwInternalError(
          "Internal Error! primop %s shouldn't have propagated this far!".format(p.serialize)
        );
    }

    private def unary_cells = List("$not", "$neg", "$reduce_and", "$reduce_or", "$reduce_xor")
    private def get_type_width(e: Type): BigInt = { // just trust me bro, its lofirrtl
      e.asInstanceOf[GroundType].width.asInstanceOf[IntWidth].width
    }

    private def emit_cell(
      i:           Info,
      name:        String,
      params:      Seq[(String, String)],
      connections: Seq[(String, String)]
    ): Seq[Seq[Any]] = {
      Seq(Seq(i), Seq("cell ", name, " ", namespace.newName(name + "$" + m.name))) ++
        params.map { p => Seq(tab, "parameter \\", p._1, " ", p._2) } ++
        connections.map { c => Seq(tab, "connect \\", c._1, " ", c._2) } ++
        Seq(Seq("end"))
    }

    private def emit_unary_cell(cell: String, src: SrcInfo, target: String, tgt_width: BigInt): Seq[Seq[Any]] = {
      emit_cell(
        NoInfo,
        cell,
        Seq(
          (
            "A_SIGNED",
            if (src.signed) { "1" }
            else { "0" }
          ),
          ("A_WIDTH", src.width.toString),
          ("Y_WIDTH", tgt_width.toString)
        ),
        Seq(("A", src.str_rep), ("Y", target))
      )
    }

    private def emit_binary_cell(
      cell:      String,
      src_a:     SrcInfo,
      src_b:     SrcInfo,
      target:    String,
      tgt_width: BigInt
    ): Seq[Seq[Any]] = {
      emit_cell(
        NoInfo,
        cell,
        Seq(
          (
            "A_SIGNED",
            if (src_a.signed) "1" else "0"
          ),
          ("A_WIDTH", src_a.width.toString),
          (
            "B_SIGNED",
            if (src_b.signed) "1" else "0"
          ),
          ("B_WIDTH", src_b.width.toString),
          ("Y_WIDTH", tgt_width.toString)
        ),
        Seq(("A", src_a.str_rep), ("B", src_b.str_rep), ("Y", target))
      )
    }

    @tailrec
    private def output_expr(n: String, d: Expression, i: Info): Seq[Seq[Any]] = d match {
      case UIntLiteral(_, _) | SIntLiteral(_, _) | Reference(_, _, _, _) | SubField(_, _, _, _) =>
        Seq(Seq("connect ", n, " ", SrcInfo(d, i).str_rep))
      case InfoExpr(info, expr) =>
        output_expr(n, expr, MultiInfo(Seq(i, info)))
      case Mux(cond, tval, fval, tpe) =>
        val (eninfo, tinfo, finfo) = MultiInfo.demux(i)
        val csrc = SrcInfo(cond, eninfo)
        val tsrc = SrcInfo(tval, tinfo)
        val fsrc = SrcInfo(fval, finfo)
        emit_cell(
          i,
          "$mux",
          Seq(("WIDTH", get_type_width(tpe).toString)),
          Seq(("A", fsrc.str_rep), ("B", tsrc.str_rep), ("S", csrc.str_rep), ("Y", n))
        )
      case DoPrim(op, args, consts, _) =>
        val sources = args.map(SrcInfo(_, i))
        val src0 = sources.head
        if (sources.map(_.width).contains(-1)) return Seq()
        op match {
          case AsSInt | AsUInt | AsClock | AsAsyncReset =>
            Seq(Seq("connect ", n, " ", src0))
          case Cvt =>
            if (src0.signed)
              Seq(Seq("connect ", n, " ", src0))
            else
              Seq(Seq("connect ", n, " { 1'0 ", src0, " }"))
          case Bits =>
            if (consts.head == consts.last)
              Seq(Seq("connect ", n, " ", src0, " [", consts.head.toInt, "]"))
            else
              Seq(Seq("connect ", n, " ", src0, " [", consts.head.toInt, ":", consts.last.toInt, "]"))
          case Shr | Shl =>
            val prim = if (op == Shr) (if (src0.signed) "$sshr" else "$shr") else "$shl"
            emit_binary_cell(
              prim,
              src0,
              SrcInfo(bigint_to_str_rep(consts.head, consts.head.bitLength), signed = false, consts.head.bitLength),
              n,
              get_type_width(d.tpe)
            )
          case Add =>
            if (src0.signed && sources(1).signed) {
              val src0_ext = SrcInfo(s"{ ${src0.str_rep} [${src0.width - 1}] ${src0.str_rep} }", true, src0.width + 1)
              val src1_ext = SrcInfo(
                s"{ ${sources(1).str_rep} [${sources(1).width - 1}] ${sources(1).str_rep} }",
                true,
                sources(1).width + 1
              )
              emit_binary_cell("$add", src0_ext, src1_ext, n, get_type_width(d.tpe))
            } else {
              emit_binary_cell("$add", src0, sources(1), n, get_type_width(d.tpe))
            }
          case Dshr | Dshl =>
            val prim = if (op == Dshr) (if (src0.signed) "$sshr" else "$shr") else "$shl"
            emit_binary_cell(prim, src0, sources(1), n, get_type_width(d.tpe))
          case Cat =>
            Seq(Seq("connect ", n, " { ", sources.map(_.str_rep).mkString(" "), " }"))
          case Head =>
            Seq(Seq("connect ", n, " ", src0, " [", (src0.width - 1).toInt, ":", consts.head.toInt, "]"))
          case Tail =>
            Seq(Seq("connect ", n, " ", src0, " [", (src0.width - 1 - consts.head).toInt, ":0]"))
          case Pad =>
            if (src0.width >= consts.head)
              Seq(Seq("connect ", n, " ", src0, " [", (consts.head - 1).toInt, ":0]"))
            else if (src0.signed)
              Seq(
                Seq("connect ", n) ++
                  Seq(
                    " { ",
                    s"${src0.str_rep} [${src0.width - 1}] " * (consts.head - src0.width).toInt,
                    src0.str_rep,
                    " }"
                  )
              )
            else
              Seq(Seq("connect ", n, " { ", (consts.head - src0.width).toInt, "'0 ", src0, " }"))
          case _ =>
            val cell = primop_to_cell(op)
            if (unary_cells.contains(cell))
              Seq(i) +: emit_unary_cell(cell, src0, n, get_type_width(d.tpe))
            else
              Seq(i) +: emit_binary_cell(cell, src0, sources(1), n, get_type_width(d.tpe))
        }
      case unk =>
        throw EmitterException(s"Internal error! unhandled output expression $unk passed to output_expr()")
    }

    private def build_streams(s: Statement): Unit = {
      val withoutDescription = s match {
        case DescribedStmt(d, stmt) =>
          stmt match {
            case _: IsDeclaration =>
              declares ++= build_description(d)
            case _ =>
          }
          stmt
        case stmt => stmt
      }
      withoutDescription.foreach(build_streams)
      withoutDescription match {
        case DefInstance(info, name, mdle, _) =>
          val (module, params) = moduleMap(mdle) match {
            case DescribedMod(_, _, ExtModule(_, _, _, extname, params)) => (extname, params)
            case DescribedMod(_, _, Module(_, name, _, _))               => (name, Seq.empty)
            case ExtModule(_, _, _, extname, params)                     => (extname, params)
            case Module(_, name, _, _)                                   => (name, Seq.empty)
          }
          instdeclares(name) = InstInfo(name, module, info)
          instdeclares(name).params = if (params.nonEmpty) params.map(stringify) else Seq()
        case WDefInstanceConnector(info, name, mdle, _, portCons) =>
          val (_, params) = moduleMap(mdle) match {
            case DescribedMod(_, _, ExtModule(_, _, _, extname, params)) => (extname, params)
            case DescribedMod(_, _, Module(_, name, _, _))               => (name, Seq.empty)
            case ExtModule(_, _, _, extname, params)                     => (extname, params)
            case Module(_, name, _, _)                                   => (name, Seq.empty)
          }
          instdeclares(name) = InstInfo(name, mdle, info)
          instdeclares(name).params = if (params.nonEmpty) params.map(stringify) else Seq()
          for ((port, ref) <- portCons) {
            val portName = SrcInfo(remove_root(port)).str_rep.tail
            if (instdeclares(name).getConnection(portName).nonEmpty) {
              assigns ++= output_expr(instdeclares(name).getConnection(portName).get, ref, NoInfo)
            } else {
              instdeclares(name).addConnection(SrcInfo(remove_root(port)).str_rep.tail, SrcInfo(ref).str_rep)
            }
          }
        case Connect(info, loc @ WRef(_, _, PortKind | WireKind | InstanceKind, _), expr) =>
          assigns ++= output_expr(ref_to_name(loc), expr, info)
        case Connect(info, SubField(Reference(modname, _, InstanceKind, _), portname, _, _), expr) =>
          if (instdeclares(modname).getConnection(portname).nonEmpty) {
            assigns ++= output_expr(instdeclares(modname).getConnection(portname).get, expr, NoInfo)
          } else {
            instdeclares(modname).addConnection(portname, SrcInfo(expr, info).str_rep)
          }
        case sx: DefWire =>
          declares += Seq(sx.info)
          declares += Seq("wire ", sx.tpe, " ", string_to_rtlil_name(sx.name))
        case sx: DefRegister =>
          val options = emissionOptions.getRegisterEmissionOption(moduleTarget.ref(sx.name))
          val e = WRef(sx.name, sx.tpe, ExpKind, UnknownFlow)
          declares += Seq(sx.info)
          declares += Seq("wire ", sx.tpe, " ", string_to_rtlil_name(sx.name))
          if (options.useInitAsPreset)
            regUpdate(e, sx.clock, sx.reset, e)
          else
            regUpdate(e, sx.clock, sx.reset, sx.init)
        case sx: DefNode =>
          declares += Seq(sx.info)
          declares += Seq("wire ", sx.value.tpe, " ", string_to_rtlil_name(sx.name))
          assigns ++= output_expr(string_to_rtlil_name(sx.name), sx.value, sx.info)
        case x @ Verification(value, info, _, pred, en, _) =>
          value match {
            case Formal.Assert =>
              formals += emit_cell(
                info,
                "$assert",
                Seq(),
                Seq(("A", SrcInfo(pred).str_rep), ("EN", SrcInfo(en).str_rep))
              )
            case Formal.Assume =>
              formals += emit_cell(
                info,
                "$assume",
                Seq(),
                Seq(("A", SrcInfo(pred).str_rep), ("EN", SrcInfo(en).str_rep))
              )
            case Formal.Cover =>
              formals += emit_cell(
                info,
                "$cover",
                Seq(),
                Seq(("A", SrcInfo(pred).str_rep), ("EN", SrcInfo(en).str_rep))
              )
          }
        case x @ DefMemory(i, name, tpe, depth, wlat, rlat, rd, wr, rdwr, runderw) =>
          val options = emissionOptions.getMemoryEmissionOption(moduleTarget.ref(name))
          val hasComplexRW = rdwr.nonEmpty && (rlat != 1)
          if (rlat > 1 || wlat != 1 || hasComplexRW)
            throw EmitterException(
              Seq(
                s"Memory $name is too complex to emit directly.",
                "Consider running VerilogMemDelays to simplify complex memories.",
                "Alternatively, add the --repl-seq-mem flag to replace memories with blackboxes."
              ).mkString(" ")
            )
          val dataWidth = bitWidth(tpe)
          val maxDataValue = (BigInt(1) << dataWidth.toInt) - 1

          def checkValueRange(value: BigInt, at: String): Unit = {
            if (value > maxDataValue)
              throw EmitterException(
                s"Memory $at cannot be initialized with value: $value. Too large (> $maxDataValue)!"
              )
          }
          declares += Seq("memory width ", dataWidth.toString, " size ", depth.toString, " \\", name)
          options.initValue match {
            case MemoryArrayInit(values) =>
              values.zipWithIndex.foreach {
                case (value, addr) =>
                  checkValueRange(value, s"$name[$addr]")
                  initials ++= emit_cell(
                    i,
                    "$meminit_v2",
                    Seq(
                      ("MEMID", "\"\\\\" + name + "\""),
                      ("ABITS", "32"),
                      ("WIDTH", dataWidth.toString),
                      ("WORDS", "1"),
                      ("PRIORITY", addr.toString)
                    ),
                    Seq(
                      ("ADDR", addr.toString),
                      ("DATA", bigint_to_str_rep(value, dataWidth)),
                      ("EN", bigint_to_str_rep(BigInt(2).pow(dataWidth.toInt) - BigInt(1), dataWidth))
                    )
                  )
              }

            case MemoryScalarInit(value) =>
              for (addr <- 0 until depth.intValue) {
                initials ++= emit_cell(
                  i,
                  "$meminit_v2",
                  Seq(
                    ("MEMID", "\"\\\\" + name + "\""),
                    ("ABITS", "32"),
                    ("WIDTH", dataWidth.toString),
                    ("WORDS", "1"),
                    ("PRIORITY", addr.toString)
                  ),
                  Seq(
                    ("ADDR", addr.toString),
                    ("DATA", bigint_to_str_rep(value, dataWidth)),
                    ("EN", bigint_to_str_rep(BigInt(2).pow(dataWidth.toInt) - BigInt(1), dataWidth))
                  )
                )
              }
            case MemoryRandomInit =>
              println(s"Memory $name cannot be initialized with random data, RTLIL cannot express this.")
              println("Leaving memory uninitialized.")
            case MemoryFileInlineInit(_, _) =>
              throw EmitterException(s"Memory $name cannot be initialized from a file, RTLIL cannot express this.")
          }
          for (r <- rd) {
            val data = memPortField(x, r, "data")
            val addr = memPortField(x, r, "addr")
            val en = memPortField(x, r, "en")
            val hasClk = if (rlat == 1) { "1'1" }
            else { "1'0" }
            val clkSrc = netlist(memPortField(x, r, "clk")).expr
            val transparent = runderw match {
              case ReadUnderWrite.New       => "1'1"
              case ReadUnderWrite.Old       => "1'0"
              case ReadUnderWrite.Undefined => "1'x"
            }
            declares += Seq("wire ", data.tpe, " ", SrcInfo(data).str_rep)
            assigns ++= emit_cell(
              i,
              "$memrd",
              Seq(
                ("ABITS", get_type_width(addr.tpe).toString),
                ("MEMID", "\"\\\\" + name + "\""),
                ("WIDTH", get_type_width(data.tpe).toString),
                ("CLK_ENABLE", hasClk),
                ("CLK_POLARITY", "1'1"),
                ("TRANSPARENT", transparent)
              ),
              Seq(
                ("CLK", SrcInfo(clkSrc, i).str_rep),
                ("EN", if (rlat == 1) SrcInfo(netlist(en), i).str_rep else "1'1"),
                ("ADDR", SrcInfo(netlist(addr), i).str_rep),
                ("DATA", SrcInfo(data, i).str_rep)
              )
            )
          }
          for (w <- wr) {
            val data = memPortField(x, w, "data")
            val addr = memPortField(x, w, "addr")
            val en = memPortField(x, w, "en")
            val mask = memPortField(x, w, "mask")
            val enSrc = SrcInfo(netlist(en))
            val maskSrc = SrcInfo(netlist(mask))
            if (maskSrc.width > 1) {
              throw EmitterException("Compound type memory write ports arent fully supported yet.")
            }
            var memwr_enmask = enSrc.str_rep
            if (bitWidth(data.tpe) != 1) {
              memwr_enmask = namespace.newName("$memwr_enmask$" + m.name)
              declares += Seq("wire signed width ", bitWidth(data.tpe).toInt, " ", memwr_enmask)
              assigns ++= emit_cell(
                i,
                "$and",
                Seq(
                  ("A_SIGNED", "1"),
                  ("B_SIGNED", "1"),
                  ("A_WIDTH", bitWidth(en.tpe).toString()),
                  ("B_WIDTH", maskSrc.width.toString()),
                  ("Y_WIDTH", bitWidth(data.tpe).toString())
                ),
                Seq(("A", enSrc.str_rep), ("B", maskSrc.str_rep), ("Y", memwr_enmask))
              )
            }
            val hasClk = if (wlat == 1) { "1'1" }
            else { "1'0" }
            val clkSrc = netlist(memPortField(x, w, "clk")).expr
            assigns ++= emit_cell(
              i,
              "$memwr",
              Seq(
                ("ABITS", get_type_width(addr.tpe).toString),
                ("MEMID", "\"\\\\" + name + "\""),
                ("WIDTH", get_type_width(data.tpe).toString),
                ("CLK_ENABLE", hasClk),
                ("CLK_POLARITY", "1'1"),
                ("PRIORITY", "32'1")
              ),
              Seq(
                ("CLK", SrcInfo(clkSrc).str_rep),
                ("EN", memwr_enmask),
                ("ADDR", SrcInfo(netlist(addr)).str_rep),
                ("DATA", SrcInfo(netlist(data)).str_rep)
              )
            )
          }
        case sx: Attach =>
          for (set <- sx.exprs.toSet.subsets(2)) {
            val (a, b) = set.toSeq match {
              case Seq(x, y) => (x, y)
            }
            attachSynAssigns += Seq("connect ", SrcInfo(a, sx.info).str_rep, " ", SrcInfo(b, sx.info).str_rep)
          }
        case _ =>
      }
    }

    def emit_rtlil(): DefModule = {
      build_netlist(m.body)
      build_ports()
      build_streams(m.body)
      emit_streams()
      m
    }
  }
}

private[firrtl] class EmissionOptionMap[V <: EmissionOption](val df: V) {
  private val m = collection.mutable.HashMap[ReferenceTarget, V]().withDefaultValue(df)
  def +=(elem: (ReferenceTarget, V)): EmissionOptionMap.this.type = {
    if (m.contains(elem._1))
      throw EmitterException(s"Multiple EmissionOption for the target ${elem._1} (${m(elem._1)} ; ${elem._2})")
    m += elem
    this
  }
  def apply(key: ReferenceTarget): V = m.apply(key)
}

private[firrtl] class EmissionOptions(annotations: AnnotationSeq) {
  // Private so that we can present an immutable API
  private val memoryEmissionOption = new EmissionOptionMap[MemoryEmissionOption](
    annotations.collectFirst { case a: CustomDefaultMemoryEmission => a }.getOrElse(MemoryEmissionOptionDefault)
  )
  private val registerEmissionOption = new EmissionOptionMap[RegisterEmissionOption](
    annotations.collectFirst { case a: CustomDefaultRegisterEmission => a }.getOrElse(RegisterEmissionOptionDefault)
  )
  private val wireEmissionOption = new EmissionOptionMap[WireEmissionOption](WireEmissionOptionDefault)
  private val portEmissionOption = new EmissionOptionMap[PortEmissionOption](PortEmissionOptionDefault)
  private val nodeEmissionOption = new EmissionOptionMap[NodeEmissionOption](NodeEmissionOptionDefault)
  private val connectEmissionOption = new EmissionOptionMap[ConnectEmissionOption](ConnectEmissionOptionDefault)

  def getMemoryEmissionOption(target: ReferenceTarget): MemoryEmissionOption =
    memoryEmissionOption(target)

  def getRegisterEmissionOption(target: ReferenceTarget): RegisterEmissionOption =
    registerEmissionOption(target)

  def getWireEmissionOption(target: ReferenceTarget): WireEmissionOption =
    wireEmissionOption(target)

  def getPortEmissionOption(target: ReferenceTarget): PortEmissionOption =
    portEmissionOption(target)

  def getNodeEmissionOption(target: ReferenceTarget): NodeEmissionOption =
    nodeEmissionOption(target)

  def getConnectEmissionOption(target: ReferenceTarget): ConnectEmissionOption =
    connectEmissionOption(target)

  def emitMemoryInitAsNoSynth: Boolean = {
    val annos = annotations.collect { case a @ (MemoryNoSynthInit | MemorySynthInit) => a }
    annos match {
      case Seq()                  => true
      case Seq(MemoryNoSynthInit) => true
      case Seq(MemorySynthInit)   => false
      case _ =>
        throw new FirrtlUserException(
          "There should only be at most one memory initialization option annotation, got $other"
        )
    }
  }

  private val emissionAnnos = annotations.collect {
    case m: SingleTargetAnnotation[ReferenceTarget] @unchecked with EmissionOption => m
  }

  annotations.foreach {
    case a: Annotation if a.dedup.nonEmpty =>
      val (_, _, target) = a.dedup.get
      if (!target.isLocal) {
        throw new FirrtlUserException(
          s"At least one dedupable annotation did not deduplicate: got non-local annotation $a from [[DedupAnnotationsTransform]]"
        )
      }
    case _ =>
  }

  // using multiple foreach instead of a single partial function as an Annotation can gather multiple EmissionOptions for simplicity
  emissionAnnos.foreach {
    case a: MemoryEmissionOption => memoryEmissionOption += ((a.target, a))
    case _ =>
  }
  emissionAnnos.foreach {
    case a: RegisterEmissionOption => registerEmissionOption += ((a.target, a))
    case _ =>
  }
  emissionAnnos.foreach {
    case a: WireEmissionOption => wireEmissionOption += ((a.target, a))
    case _ =>
  }
  emissionAnnos.foreach {
    case a: PortEmissionOption => portEmissionOption += ((a.target, a))
    case _ =>
  }
  emissionAnnos.foreach {
    case a: NodeEmissionOption => nodeEmissionOption += ((a.target, a))
    case _ =>
  }
  emissionAnnos.foreach {
    case a: ConnectEmissionOption => connectEmissionOption += ((a.target, a))
    case _ =>
  }
}
