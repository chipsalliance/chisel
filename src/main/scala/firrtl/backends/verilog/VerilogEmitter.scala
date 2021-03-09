package firrtl

import java.io.Writer

import firrtl.ir._
import firrtl.PrimOps._
import firrtl.Utils._
import firrtl.WrappedExpression._
import firrtl.traversals.Foreachers._
import firrtl.annotations.{CircuitTarget, MemoryLoadFileType, ReferenceTarget, SingleTargetAnnotation}
import firrtl.passes.LowerTypes
import firrtl.passes.MemPortUtils._
import firrtl.stage.TransformManager
import firrtl.transforms.FixAddingNegativeLiterals

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object VerilogEmitter {
  private val unaryOps: Set[PrimOp] = Set(Andr, Orr, Xorr, Neg, Not)

  // To make uses more self-documenting
  private val isUnaryOp: PrimOp => Boolean = unaryOps

  /** Maps a [[PrimOp]] to a precedence number, lower number means higher precedence
    *
    * Only the [[PrimOp]]s contained in this map will be inlined. [[PrimOp]]s
    * like [[Neg]] are not in this map because inlining them may result
    * in illegal verilog like '--2sh1'
    */
  private val precedenceMap: Map[PrimOp, Int] = {
    val precedenceSeq = Seq(
      Set(Head, Tail, Bits, Shr, Pad), // Shr and Pad emit as bit select
      unaryOps,
      Set(Mul, Div, Rem),
      Set(Add, Sub, Addw, Subw),
      Set(Dshl, Dshlw, Dshr),
      Set(Lt, Leq, Gt, Geq),
      Set(Eq, Neq),
      Set(And),
      Set(Xor),
      Set(Or)
    )
    precedenceSeq.zipWithIndex.foldLeft(Map.empty[PrimOp, Int]) {
      case (map, (ops, idx)) => map ++ ops.map(_ -> idx)
    }
  }

  /** true if op1 has equal precendence to op2
    */
  private def precedenceEq(op1: PrimOp, op2: PrimOp): Boolean = {
    precedenceMap(op1) == precedenceMap(op2)
  }

  /** true if op1 has greater precendence than op2
    */
  private def precedenceGt(op1: PrimOp, op2: PrimOp): Boolean = {
    precedenceMap(op1) < precedenceMap(op2)
  }
}

class VerilogEmitter extends SeqTransform with Emitter {
  import VerilogEmitter._

  def inputForm = LowForm
  def outputForm = LowForm

  override def prerequisites = firrtl.stage.Forms.AssertsRemoved ++
    firrtl.stage.Forms.LowFormOptimized

  override def optionalPrerequisiteOf = Seq.empty

  val outputSuffix = ".v"
  val tab = "  "
  def AND(e1: WrappedExpression, e2: WrappedExpression): Expression = {
    if (e1 == e2) e1.e1
    else if ((e1 == we(zero)) | (e2 == we(zero))) zero
    else if (e1 == we(one)) e2.e1
    else if (e2 == we(one)) e1.e1
    else DoPrim(And, Seq(e1.e1, e2.e1), Nil, UIntType(IntWidth(1)))
  }
  def wref(n:         String, t: Type) = WRef(n, t, ExpKind, UnknownFlow)
  def remove_root(ex: Expression): Expression = ex match {
    case ex: WSubField =>
      ex.expr match {
        case (e: WSubField) => remove_root(e)
        case (_: WRef)      => WRef(ex.name, ex.tpe, InstanceKind, UnknownFlow)
      }
    case _ => throwInternalError(s"shouldn't be here: remove_root($ex)")
  }

  /** Turn Params into Verilog Strings */
  def stringify(param: Param): String = param match {
    case IntParam(name, value) =>
      val lit =
        if (value.isValidInt) {
          s"$value"
        } else {
          val blen = value.bitLength
          if (value > 0) s"$blen'd$value" else s"-${blen + 1}'sd${value.abs}"
        }
      s".$name($lit)"
    case DoubleParam(name, value)    => s".$name($value)"
    case StringParam(name, value)    => s".${name}(${value.verilogEscape})"
    case RawStringParam(name, value) => s".$name($value)"
  }
  def stringify(tpe: GroundType): String = tpe match {
    case (_: UIntType | _: SIntType | _: AnalogType) =>
      val wx = bitWidth(tpe) - 1
      if (wx > 0) s"[$wx:0]" else ""
    case ClockType | AsyncResetType => ""
    case _                          => throwInternalError(s"trying to write unsupported type in the Verilog Emitter: $tpe")
  }
  private def getLeadingTabs(x: Any): String = {
    x match {
      case seq: Seq[_] =>
        val head = seq.takeWhile(_ == tab).mkString
        val tail = seq.dropWhile(_ == tab).lift(0).map(getLeadingTabs).getOrElse(tab)
        head + tail
      case _ => tab
    }
  }
  def emit(x: Any)(implicit w: Writer): Unit = {
    emitCol(x, 0, getLeadingTabs(x), 0)
  }
  private def emitCast(e: Expression): Any = e.tpe match {
    case (t: UIntType) => e
    case (t: SIntType) => Seq("$signed(", e, ")")
    case ClockType     => e
    case AnalogType(_) => e
    case _             => throwInternalError(s"unrecognized cast: $e")
  }
  def emit(x: Any, top: Int)(implicit w: Writer): Unit = {
    emitCol(x, top, "", 0)
  }
  private val maxCol = 120
  private def emitCol(x: Any, top: Int, tabs: String, colNum: Int)(implicit w: Writer): Int = {
    def writeCol(contents: String): Int = {
      if ((contents.size + colNum) > maxCol) {
        w.write("\n")
        w.write(tabs)
        w.write(contents)
        tabs.size + contents.size
      } else {
        w.write(contents)
        colNum + contents.size
      }
    }

    def cast(e: Expression): Any = e.tpe match {
      case (t: UIntType) => e
      case (t: SIntType) => Seq("$signed(", e, ")")
      case ClockType     => e
      case AnalogType(_) => e
      case _             => throwInternalError(s"unrecognized cast: $e")
    }
    x match {
      case (e: DoPrim) => emitCol(op_stream(e), top + 1, tabs, colNum)
      case (e: Mux) => {
        if (e.tpe == ClockType) {
          throw EmitterException("Cannot emit clock muxes directly")
        }
        if (e.tpe == AsyncResetType) {
          throw EmitterException("Cannot emit async reset muxes directly")
        }
        emitCol(Seq(e.cond, " ? ", cast(e.tval), " : ", cast(e.fval)), top + 1, tabs, colNum)
      }
      case (e: ValidIf)    => emitCol(Seq(cast(e.value)), top + 1, tabs, colNum)
      case (e: WRef)       => writeCol(e.serialize)
      case (e: WSubField)  => writeCol(LowerTypes.loweredName(e))
      case (e: WSubAccess) => writeCol(s"${LowerTypes.loweredName(e.expr)}[${LowerTypes.loweredName(e.index)}]")
      case (e: WSubIndex)  => writeCol(e.serialize)
      case (e: Literal)    => v_print(e, colNum)
      case (e: VRandom)    => writeCol(s"{${e.nWords}{`RANDOM}}")
      case (t: GroundType) => writeCol(stringify(t))
      case (t: VectorType) =>
        emit(t.tpe, top + 1)
        writeCol(s"[${t.size - 1}:0]")
      case (s: String) => writeCol(s)
      case (i: Int)    => writeCol(i.toString)
      case (i: Long)   => writeCol(i.toString)
      case (i: BigInt) => writeCol(i.toString)
      case (i: Info) =>
        i match {
          case NoInfo => colNum // Do nothing
          case f: FileInfo =>
            val escaped = FileInfo.escapedToVerilog(f.escaped)
            w.write(s" // @[$escaped]")
            colNum
          case m: MultiInfo =>
            val escaped = FileInfo.escapedToVerilog(m.flatten.map(_.escaped).mkString(" "))
            w.write(s" // @[$escaped]")
            colNum
        }
      case (s: Seq[Any]) =>
        val nextColNum = s.foldLeft(colNum) {
          case (colNum, e) => emitCol(e, top + 1, tabs, colNum)
        }
        if (top == 0) {
          w.write("\n")
          0
        } else {
          nextColNum
        }
      case x => throwInternalError(s"trying to emit unsupported operator: $x")
    }
  }

  //;------------- PASS -----------------
  def v_print(e: Expression, colNum: Int)(implicit w: Writer) = e match {
    case UIntLiteral(value, IntWidth(width)) =>
      val contents = s"$width'h${value.toString(16)}"
      w.write(contents)
      colNum + contents.size
    case SIntLiteral(value, IntWidth(width)) =>
      val stringLiteral = value.toString(16)
      val contents = stringLiteral.head match {
        case '-' if value == FixAddingNegativeLiterals.minNegValue(width) => s"$width'sh${stringLiteral.tail}"
        case '-'                                                          => s"-$width'sh${stringLiteral.tail}"
        case _                                                            => s"$width'sh${stringLiteral}"
      }
      w.write(contents)
      colNum + contents.size
    case _ => throwInternalError(s"attempt to print unrecognized expression: $e")
  }

  // NOTE: We emit SInts as regular Verilog unsigned wires/regs so the real type of any SInt
  // reference is actually unsigned in the emitted Verilog. Thus we must cast refs as necessary
  // to ensure Verilog operations are signed.
  def op_stream(doprim: DoPrim): Seq[Any] = {
    def parenthesize(e: Expression, isFirst: Boolean): Any = doprim.op match {
      // these PrimOps emit either {..., a0, ...} or a0 so they never need parentheses
      case Shl | Cat | Cvt | AsUInt | AsSInt | AsClock | AsAsyncReset => e
      case _ =>
        e match {
          case e: DoPrim =>
            op_stream(e) match {
              /** DoPrims like AsUInt simply emit Seq(a0), so we need to
                * recursively check whether a0 needs to be parenthesized
                */
              case Seq(passthrough: Expression) => parenthesize(passthrough, isFirst)

              /* Parentheses are never needed if precedence is greater
               * Otherwise, the first expression does not need parentheses if
               * - it's precedence is equal AND
               * - the ops are not unary operations (which all have equal precedence)
               */
              case other =>
                val noParens =
                  precedenceGt(e.op, doprim.op) ||
                    (isFirst && precedenceEq(e.op, doprim.op) && !isUnaryOp(e.op))
                if (noParens) other else Seq("(", other, ")")
            }

          /** Mux args should always have parens because Mux has the lowest precedence
            */
          case _: Mux => Seq("(", e, ")")
          case _ => e
        }
    }

    // Cast to SInt, don't cast multiple times
    def doCast(e: Expression): Any = e match {
      case DoPrim(AsSInt, Seq(arg), _, _) => doCast(arg)
      case slit: SIntLiteral => slit
      case other => Seq("$signed(", other, ")")
    }
    def castIf(e: Expression, isFirst: Boolean = false): Any = {
      if (doprim.args.exists(_.tpe.isInstanceOf[SIntType])) {
        e.tpe match {
          case _: SIntType => doCast(e)
          case _ => throwInternalError(s"Unexpected non-SInt type for $e in $doprim")
        }
      } else {
        parenthesize(e, isFirst)
      }
    }
    def cast(e: Expression, isFirst: Boolean = false): Any = doprim.tpe match {
      case _: UIntType => parenthesize(e, isFirst)
      case _: SIntType => doCast(e)
      case _ => throwInternalError(s"Unexpected type for $e in $doprim")
    }
    def castAs(e: Expression, isFirst: Boolean = false): Any = e.tpe match {
      case _: UIntType => parenthesize(e, isFirst)
      case _: SIntType => doCast(e)
      case _ => throwInternalError(s"Unexpected type for $e in $doprim")
    }
    def a0: Expression = doprim.args.head
    def a1: Expression = doprim.args(1)
    def c0: Int = doprim.consts.head.toInt
    def c1: Int = doprim.consts(1).toInt

    def castCatArgs(a0: Expression, a1: Expression): Seq[Any] = {
      val a0Seq = a0 match {
        case cat @ DoPrim(PrimOps.Cat, args, _, _) => castCatArgs(args.head, args(1))
        case _                                     => Seq(cast(a0))
      }
      val a1Seq = a1 match {
        case cat @ DoPrim(PrimOps.Cat, args, _, _) => castCatArgs(args.head, args(1))
        case _                                     => Seq(cast(a1))
      }
      a0Seq ++ Seq(",") ++ a1Seq
    }

    doprim.op match {
      case Add  => Seq(castIf(a0, true), " + ", castIf(a1))
      case Addw => Seq(castIf(a0, true), " + ", castIf(a1))
      case Sub  => Seq(castIf(a0, true), " - ", castIf(a1))
      case Subw => Seq(castIf(a0, true), " - ", castIf(a1))
      case Mul  => Seq(castIf(a0, true), " * ", castIf(a1))
      case Div  => Seq(castIf(a0, true), " / ", castIf(a1))
      case Rem  => Seq(castIf(a0, true), " % ", castIf(a1))
      case Lt   => Seq(castIf(a0, true), " < ", castIf(a1))
      case Leq  => Seq(castIf(a0, true), " <= ", castIf(a1))
      case Gt   => Seq(castIf(a0, true), " > ", castIf(a1))
      case Geq  => Seq(castIf(a0, true), " >= ", castIf(a1))
      case Eq   => Seq(castIf(a0, true), " == ", castIf(a1))
      case Neq  => Seq(castIf(a0, true), " != ", castIf(a1))
      case Pad =>
        val w = bitWidth(a0.tpe)
        val diff = c0 - w
        if (w == BigInt(0) || diff <= 0) Seq(a0)
        else
          doprim.tpe match {
            // Either sign extend or zero extend.
            // If width == BigInt(1), don't extract bit
            case (_: SIntType) if w == BigInt(1) => Seq("{", c0, "{", a0, "}}")
            case (_: SIntType)                   => Seq("{{", diff, "{", parenthesize(a0, true), "[", w - 1, "]}},", a0, "}")
            case (_) => Seq("{{", diff, "'d0}, ", a0, "}")
          }
      // Because we don't support complex Expressions, all casts are ignored
      // This simplifies handling of assignment of a signed expression to an unsigned LHS value
      //   which does not require a cast in Verilog
      case AsUInt | AsSInt | AsClock | AsAsyncReset => Seq(a0)
      case Dshlw                                    => Seq(cast(a0), " << ", parenthesize(a1, false))
      case Dshl                                     => Seq(cast(a0), " << ", parenthesize(a1, false))
      case Dshr =>
        doprim.tpe match {
          case (_: SIntType) => Seq(cast(a0), " >>> ", parenthesize(a1, false))
          case (_) => Seq(cast(a0), " >> ", parenthesize(a1, false))
        }
      case Shl => if (c0 > 0) Seq("{", cast(a0), s", $c0'h0}") else Seq(cast(a0))
      case Shr if c0 >= bitWidth(a0.tpe) =>
        error("Verilog emitter does not support SHIFT_RIGHT >= arg width")
      case Shr if c0 == (bitWidth(a0.tpe) - 1) => Seq(parenthesize(a0, true), "[", bitWidth(a0.tpe) - 1, "]")
      case Shr                                 => Seq(parenthesize(a0, true), "[", bitWidth(a0.tpe) - 1, ":", c0, "]")
      case Neg                                 => Seq("-", cast(a0, true))
      case Cvt =>
        a0.tpe match {
          case (_: UIntType) => Seq("{1'b0,", cast(a0), "}")
          case (_: SIntType) => Seq(cast(a0))
        }
      case Not  => Seq("~", parenthesize(a0, true))
      case And  => Seq(castAs(a0, true), " & ", castAs(a1))
      case Or   => Seq(castAs(a0, true), " | ", castAs(a1))
      case Xor  => Seq(castAs(a0, true), " ^ ", castAs(a1))
      case Andr => Seq("&", cast(a0, true))
      case Orr  => Seq("|", cast(a0, true))
      case Xorr => Seq("^", cast(a0, true))
      case Cat  => "{" +: (castCatArgs(a0, a1) :+ "}")
      // If selecting zeroth bit and single-bit wire, just emit the wire
      case Bits if c0 == 0 && c1 == 0 && bitWidth(a0.tpe) == BigInt(1) => Seq(a0)
      case Bits if c0 == c1                                            => Seq(parenthesize(a0, true), "[", c0, "]")
      case Bits                                                        => Seq(parenthesize(a0, true), "[", c0, ":", c1, "]")
      // If selecting zeroth bit and single-bit wire, just emit the wire
      case Head if c0 == 1 && bitWidth(a0.tpe) == BigInt(1) => Seq(a0)
      case Head if c0 == 1                                  => Seq(parenthesize(a0, true), "[", bitWidth(a0.tpe) - 1, "]")
      case Head =>
        val msb = bitWidth(a0.tpe) - 1
        val lsb = bitWidth(a0.tpe) - c0
        Seq(parenthesize(a0, true), "[", msb, ":", lsb, "]")
      case Tail if c0 == (bitWidth(a0.tpe) - 1) => Seq(parenthesize(a0, true), "[0]")
      case Tail                                 => Seq(parenthesize(a0, true), "[", bitWidth(a0.tpe) - c0 - 1, ":0]")
    }
  }

  /**
    * Gets a reference to a verilog renderer. This is used by the current standard verilog emission process
    * but allows access to individual portions, in particular, this function can be used to generate
    * the header for a verilog file without generating anything else.
    *
    * @param m         the start module
    * @param moduleMap a way of finding other modules
    * @param writer    where rendering will be placed
    * @return          the render reference
    */
  def getRenderer(m: Module, moduleMap: Map[String, DefModule])(implicit writer: Writer): VerilogRender = {
    new VerilogRender(m, moduleMap)(writer)
  }

  /**
    * Gets a reference to a verilog renderer. This is used by the current standard verilog emission process
    * but allows access to individual portions, in particular, this function can be used to generate
    * the header for a verilog file without generating anything else.
    *
    * @param descriptions comments to be emitted
    * @param m            the start module
    * @param moduleMap    a way of finding other modules
    * @param writer       where rendering will be placed
    * @return             the render reference
    */
  def getRenderer(
    descriptions: Seq[DescriptionAnnotation],
    m:            Module,
    moduleMap:    Map[String, DefModule]
  )(
    implicit writer: Writer
  ): VerilogRender = {
    val newMod = new AddDescriptionNodes().executeModule(m, descriptions)

    newMod match {
      case DescribedMod(d, pds, m: Module) =>
        new VerilogRender(d, pds, m, moduleMap, "", new EmissionOptions(Seq.empty))(writer)
      case m: Module => new VerilogRender(m, moduleMap)(writer)
    }
  }

  def addFormalStatement(
    formals: mutable.Map[Expression, ArrayBuffer[Seq[Any]]],
    clk:     Expression,
    en:      Expression,
    stmt:    Seq[Any],
    info:    Info,
    msg:     StringLit
  ): Unit = {
    throw EmitterException(
      "Cannot emit verification statements in Verilog" +
        "(2001). Use the SystemVerilog emitter instead."
    )
  }

  /**
    * Store Emission option per Target
    * Guarantee only one emission option per Target
    */
  private[firrtl] class EmissionOptionMap[V <: EmissionOption](val df: V) {
    private val m = collection.mutable.HashMap[ReferenceTarget, V]().withDefaultValue(df)
    def +=(elem: (ReferenceTarget, V)): EmissionOptionMap.this.type = {
      if (m.contains(elem._1))
        throw EmitterException(s"Multiple EmissionOption for the target ${elem._1} (${m(elem._1)} ; ${elem._2})")
      m += (elem)
      this
    }
    def apply(key: ReferenceTarget): V = m.apply(key)
  }

  /** Provide API to retrieve EmissionOptions based on the provided [[AnnotationSeq]]
    *
    * @param annotations : AnnotationSeq to be searched for EmissionOptions
    */
  private[firrtl] class EmissionOptions(annotations: AnnotationSeq) {
    // Private so that we can present an immutable API
    private val memoryEmissionOption = new EmissionOptionMap[MemoryEmissionOption](MemoryEmissionOptionDefault)
    private val registerEmissionOption = new EmissionOptionMap[RegisterEmissionOption](RegisterEmissionOptionDefault)
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

    private val emissionAnnos = annotations.collect {
      case m: SingleTargetAnnotation[ReferenceTarget] @unchecked with EmissionOption => m
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

  /**
    * Used by getRenderer, it has machinery to produce verilog from IR.
    * Making this a class allows access to particular parts of the verilog emission.
    *
    * @param description      a description of the start module
    * @param portDescriptions a map of port name to description
    * @param m                the start module
    * @param moduleMap        a map of modules so submodules can be discovered
    * @param writer           where rendered information is placed.
    */
  class VerilogRender(
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
    def this(m: Module, moduleMap: Map[String, DefModule])(implicit writer: Writer) = {
      this(Seq(), Map.empty, m, moduleMap, "", new EmissionOptions(Seq.empty))(writer)
    }

    val netlist = mutable.LinkedHashMap[WrappedExpression, InfoExpr]()
    val namespace = Namespace(m)
    namespace.newName("_RAND") // Start rand names at _RAND_0
    def build_netlist(s: Statement): Unit = {
      s.foreach(build_netlist)
      s match {
        case sx: Connect   => netlist(sx.loc) = InfoExpr(sx.info, sx.expr)
        case sx: IsInvalid => error("Should have removed these!")
        // TODO Since only register update and memories use the netlist anymore, I think nodes are
        // unnecessary
        case sx: DefNode =>
          val e = WRef(sx.name, sx.value.tpe, NodeKind, SourceFlow)
          netlist(e) = InfoExpr(sx.info, sx.value)
        case _ =>
      }
    }

    val portdefs = ArrayBuffer[Seq[Any]]()
    // maps ifdef guard to declaration blocks
    val ifdefDeclares: mutable.Map[String, ArrayBuffer[Seq[Any]]] = mutable.Map().withDefault { key =>
      val value = ArrayBuffer[Seq[Any]]()
      ifdefDeclares(key) = value
      value
    }
    val declares = ArrayBuffer[Seq[Any]]()
    val instdeclares = ArrayBuffer[Seq[Any]]()
    val assigns = ArrayBuffer[Seq[Any]]()
    val attachSynAssigns = ArrayBuffer.empty[Seq[Any]]
    val attachAliases = ArrayBuffer.empty[Seq[Any]]
    // No (aka synchronous) always blocks, keyed by clock
    val noResetAlwaysBlocks = mutable.LinkedHashMap[Expression, ArrayBuffer[Seq[Any]]]()
    // One always block per async reset register, (Clock, Reset, Content)
    // An alternative approach is to have one always block per combination of clock and async reset,
    // but Formality doesn't allow more than 1 statement inside async reset always blocks
    val asyncResetAlwaysBlocks = mutable.ArrayBuffer[(Expression, Expression, Seq[Any])]()
    // Used to determine type of initvar for initializing memories
    var maxMemSize: BigInt = BigInt(0)
    // maps ifdef guard to initial blocks
    val ifdefInitials: mutable.Map[String, ArrayBuffer[Seq[Any]]] = mutable.Map().withDefault { key =>
      val value = ArrayBuffer[Seq[Any]]()
      ifdefInitials(key) = value
      value
    }
    val initials = ArrayBuffer[Seq[Any]]()
    // In Verilog, async reset registers are expressed using always blocks of the form:
    // always @(posedge clock or posedge reset) begin
    //   if (reset) ...
    // There is a fundamental mismatch between this representation which treats async reset
    // registers as edge-triggered when in reality they are level-triggered.
    // When not randomized, there is no mismatch because the async reset transition at the start
    // of simulation from X to 1 triggers the posedge block for async reset.
    // When randomized, this can result in silicon-simulation mismatch when async reset is held high
    // upon power on with no clock, then async reset is dropped before the clock starts. In this
    // circumstance, the async reset register will be randomized in simulation instead of being
    // reset. To fix this, we need extra initial block logic to reset async reset registers again
    // post-randomize.
    val asyncInitials = ArrayBuffer[Seq[Any]]()
    // memories need to be initialized even when randomization is disabled
    val memoryInitials = ArrayBuffer[Seq[Any]]()
    val simulates = ArrayBuffer[Seq[Any]]()
    val formals = mutable.LinkedHashMap[Expression, ArrayBuffer[Seq[Any]]]()

    def bigIntToVLit(bi: BigInt): String =
      if (bi.isValidInt) bi.toString else s"${bi.bitLength}'d$bi"

    // declare vector type with no preset and optionally with an ifdef guard
    private def declareVectorType(
      b:        String,
      n:        String,
      tpe:      Type,
      size:     BigInt,
      info:     Info,
      ifdefOpt: Option[String]
    ): Unit = {
      val decl = Seq(b, " ", tpe, " ", n, " [0:", bigIntToVLit(size - 1), "];", info)
      if (ifdefOpt.isDefined) {
        ifdefDeclares(ifdefOpt.get) += decl
      } else {
        declares += decl
      }
    }

    // original vector type declare without initial value
    def declareVectorType(b: String, n: String, tpe: Type, size: BigInt, info: Info): Unit =
      declareVectorType(b, n, tpe, size, info, None)

    // declare vector type with initial value
    def declareVectorType(b: String, n: String, tpe: Type, size: BigInt, info: Info, preset: Expression): Unit = {
      declares += Seq(b, " ", tpe, " ", n, " [0:", bigIntToVLit(size - 1), "] = ", preset, ";", info)
    }

    val moduleTarget = CircuitTarget(circuitName).module(m.name)

    // declare with initial value
    def declare(b: String, n: String, t: Type, info: Info, preset: Expression) = t match {
      case tx: VectorType =>
        declareVectorType(b, n, tx.tpe, tx.size, info, preset)
      case tx =>
        declares += Seq(b, " ", tx, " ", n, " = ", preset, ";", info)
    }

    // original declare without initial value and optinally with an ifdef guard
    private def declare(b: String, n: String, t: Type, info: Info, ifdefOpt: Option[String]): Unit = t match {
      case tx: VectorType =>
        declareVectorType(b, n, tx.tpe, tx.size, info, ifdefOpt)
      case tx =>
        val decl = Seq(b, " ", tx, " ", n, ";", info)
        if (ifdefOpt.isDefined) {
          ifdefDeclares(ifdefOpt.get) += decl
        } else {
          declares += decl
        }
    }

    // original declare without initial value and with an ifdef guard
    private def declare(b: String, n: String, t: Type, info: Info, ifdef: String): Unit =
      declare(b, n, t, info, Some(ifdef))

    // original declare without initial value
    def declare(b: String, n: String, t: Type, info: Info): Unit =
      declare(b, n, t, info, None)

    def assign(e: Expression, infoExpr: InfoExpr): Unit =
      assign(e, infoExpr.expr, infoExpr.info)

    def assign(e: Expression, value: Expression, info: Info): Unit = {
      assigns += Seq("assign ", e, " = ", value, ";", info)
    }

    // In simulation, assign garbage under a predicate
    def garbageAssign(e: Expression, syn: Expression, garbageCond: Expression, info: Info) = {
      assigns += Seq("`ifndef RANDOMIZE_GARBAGE_ASSIGN")
      assigns += Seq("assign ", e, " = ", syn, ";", info)
      assigns += Seq("`else")
      assigns += Seq(
        "assign ",
        e,
        " = ",
        garbageCond,
        " ? ",
        rand_string(syn.tpe, "RANDOMIZE_GARBAGE_ASSIGN"),
        " : ",
        syn,
        ";",
        info
      )
      assigns += Seq("`endif // RANDOMIZE_GARBAGE_ASSIGN")
    }

    def invalidAssign(e: Expression) = {
      assigns += Seq("`ifdef RANDOMIZE_INVALID_ASSIGN")
      assigns += Seq("assign ", e, " = ", rand_string(e.tpe, "RANDOMIZE_INVALID_ASSIGN"), ";")
      assigns += Seq("`endif // RANDOMIZE_INVALID_ASSIGN")
    }

    def regUpdate(r: Expression, clk: Expression, reset: Expression, init: Expression) = {
      def addUpdate(info: Info, expr: Expression, tabs: Seq[String]): Seq[Seq[Any]] = expr match {
        case m: Mux =>
          if (m.tpe == ClockType) throw EmitterException("Cannot emit clock muxes directly")
          if (m.tpe == AsyncResetType) throw EmitterException("Cannot emit async reset muxes directly")

          val (eninfo, tinfo, finfo) = MultiInfo.demux(info)
          lazy val _if = Seq(tabs, "if (", m.cond, ") begin", eninfo)
          lazy val _else = Seq(tabs, "end else begin")
          lazy val _ifNot = Seq(tabs, "if (!(", m.cond, ")) begin", eninfo)
          lazy val _end = Seq(tabs, "end")
          lazy val _true = addUpdate(tinfo, m.tval, tab +: tabs)
          lazy val _false = addUpdate(finfo, m.fval, tab +: tabs)
          lazy val _elseIfFalse = {
            val _falsex = addUpdate(finfo, m.fval, tabs) // _false, but without an additional tab
            Seq(tabs, "end else ", _falsex.head.tail) +: _falsex.tail
          }

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
            case (t, _) if weq(t, r)              => _ifNot +: _false :+ _end
            case (_, f) if weq(f, r) =>
              m.cond.tpe match {
                case AsyncResetType => (_if +: _true :+ _else) ++ _true :+ _end
                case _              => _if +: _true :+ _end
              }
            case (_, _: Mux) => (_if +: _true) ++ _elseIfFalse
            case _ => (_if +: _true :+ _else) ++ _false :+ _end
          }
        case e => Seq(Seq(tabs, r, " <= ", e, ";", info))
      }
      if (weq(init, r)) { // Synchronous Reset
        val InfoExpr(info, e) = netlist(r)
        noResetAlwaysBlocks.getOrElseUpdate(clk, ArrayBuffer[Seq[Any]]()) ++= addUpdate(info, e, Seq.empty)
      } else { // Asynchronous Reset
        assert(reset.tpe == AsyncResetType, "Error! Synchronous reset should have been removed!")
        val tv = init
        val InfoExpr(finfo, fv) = netlist(r)
        // TODO add register info argument and build a MultiInfo to pass
        asyncResetAlwaysBlocks += (
          (
            clk,
            reset,
            addUpdate(NoInfo, Mux(reset, tv, fv, mux_type_and_widths(tv, fv)), Seq.empty)
          )
        )
      }
    }

    def update(e: Expression, value: Expression, clk: Expression, en: Expression, info: Info) = {
      val lines = noResetAlwaysBlocks.getOrElseUpdate(clk, ArrayBuffer[Seq[Any]]())
      if (weq(en, one)) lines += Seq(e, " <= ", value, ";")
      else {
        lines += Seq("if (", en, ") begin")
        lines += Seq(tab, e, " <= ", value, ";", info)
        lines += Seq("end")
      }
    }

    // Declares an intermediate wire to hold a large enough random number.
    // Then, return the correct number of bits selected from the random value
    def rand_string(t: Type, ifdefOpt: Option[String]): Seq[Any] = {
      val nx = namespace.newName("_RAND")
      val rand = VRandom(bitWidth(t))
      val tx = SIntType(IntWidth(rand.realWidth))
      declare("reg", nx, tx, NoInfo, ifdefOpt)
      val initial = Seq(wref(nx, tx), " = ", VRandom(bitWidth(t)), ";")
      if (ifdefOpt.isDefined) {
        ifdefInitials(ifdefOpt.get) += initial
      } else {
        initials += initial
      }
      Seq(nx, "[", bitWidth(t) - 1, ":0]")
    }

    def rand_string(t: Type, ifdef: String): Seq[Any] = rand_string(t, Some(ifdef))

    def rand_string(t: Type): Seq[Any] = rand_string(t, None)

    def initialize(e: Expression, reset: Expression, init: Expression) = {
      val randString = rand_string(e.tpe, "RANDOMIZE_REG_INIT")
      ifdefInitials("RANDOMIZE_REG_INIT") += Seq(e, " = ", randString, ";")
      reset.tpe match {
        case AsyncResetType =>
          asyncInitials += Seq("if (", reset, ") begin")
          asyncInitials += Seq(tab, e, " = ", init, ";")
          asyncInitials += Seq("end")
        case _ => // do nothing
      }
    }

    def initialize_mem(s: DefMemory, opt: MemoryEmissionOption): Unit = {
      if (s.depth > maxMemSize) {
        maxMemSize = s.depth
      }

      val dataWidth = bitWidth(s.dataType)
      val maxDataValue = (BigInt(1) << dataWidth.toInt) - 1

      def checkValueRange(value: BigInt, at: String): Unit = {
        if (value < 0) throw EmitterException(s"Memory ${at} cannot be initialized with negative value: $value")
        if (value > maxDataValue)
          throw EmitterException(s"Memory ${at} cannot be initialized with value: $value. Too large (> $maxDataValue)!")
      }

      opt.initValue match {
        case MemoryArrayInit(values) =>
          if (values.length != s.depth)
            throw EmitterException(
              s"Memory ${s.name} of depth ${s.depth} cannot be initialized with an array of length ${values.length}!"
            )
          val memName = LowerTypes.loweredName(wref(s.name, s.dataType))
          values.zipWithIndex.foreach {
            case (value, addr) =>
              checkValueRange(value, s"${s.name}[$addr]")
              val access = s"$memName[${bigIntToVLit(addr)}]"
              memoryInitials += Seq(access, " = ", bigIntToVLit(value), ";")
          }
        case MemoryScalarInit(value) =>
          checkValueRange(value, s.name)
          // note: s.dataType is the incorrect type for initvar, but it is ignored in the serialization
          val index = wref("initvar", s.dataType)
          memoryInitials += Seq("for (initvar = 0; initvar < ", bigIntToVLit(s.depth), "; initvar = initvar+1)")
          memoryInitials += Seq(
            tab,
            WSubAccess(wref(s.name, s.dataType), index, s.dataType, SinkFlow),
            " = ",
            bigIntToVLit(value),
            ";"
          )
        case MemoryRandomInit =>
          // note: s.dataType is the incorrect type for initvar, but it is ignored in the serialization
          val index = wref("initvar", s.dataType)
          val rstring = rand_string(s.dataType, "RANDOMIZE_MEM_INIT")
          ifdefInitials("RANDOMIZE_MEM_INIT") += Seq(
            "for (initvar = 0; initvar < ",
            bigIntToVLit(s.depth),
            "; initvar = initvar+1)"
          )
          ifdefInitials("RANDOMIZE_MEM_INIT") += Seq(
            tab,
            WSubAccess(wref(s.name, s.dataType), index, s.dataType, SinkFlow),
            " = ",
            rstring,
            ";"
          )
        case MemoryFileInlineInit(filename, hexOrBinary) =>
          val readmem = hexOrBinary match {
            case MemoryLoadFileType.Binary => "$readmemb"
            case MemoryLoadFileType.Hex    => "$readmemh"
          }
          memoryInitials += Seq(s"""$readmem("$filename", ${s.name});""")
      }
    }

    def simulate(clk: Expression, en: Expression, s: Seq[Any], cond: Option[String], info: Info) = {
      val lines = noResetAlwaysBlocks.getOrElseUpdate(clk, ArrayBuffer[Seq[Any]]())
      lines += Seq("`ifndef SYNTHESIS")
      if (cond.nonEmpty) {
        lines += Seq(s"`ifdef ${cond.get}")
        lines += Seq(tab, s"if (`${cond.get}) begin")
        lines += Seq("`endif")
      }
      lines += Seq(tab, tab, "if (", en, ") begin")
      lines += Seq(tab, tab, tab, s, info)
      lines += Seq(tab, tab, "end")
      if (cond.nonEmpty) {
        lines += Seq(s"`ifdef ${cond.get}")
        lines += Seq(tab, "end")
        lines += Seq("`endif")
      }
      lines += Seq("`endif // SYNTHESIS")
    }

    def addFormal(clk: Expression, en: Expression, stmt: Seq[Any], info: Info, msg: StringLit): Unit = {
      addFormalStatement(formals, clk, en, stmt, info, msg)
    }

    def formalStatement(op: Formal.Value, cond: Expression): Seq[Any] = {
      Seq(op.toString, "(", cond, ");")
    }

    def stop(ret: Int): Seq[Any] = Seq(if (ret == 0) "$finish;" else "$fatal;")

    def printf(str: StringLit, args: Seq[Expression]): Seq[Any] = {
      val strx = str.verilogEscape +: args.flatMap(Seq(",", _))
      Seq("$fwrite(32'h80000002,", strx, ");")
    }

    // turn strings into Seq[String] verilog comments
    def build_comment(desc: String): Seq[Seq[String]] = {
      val lines = desc.split("\n").toSeq

      if (lines.size > 1) {
        val lineSeqs = lines.tail.map {
          case ""       => Seq(" *")
          case nonEmpty => Seq(" * ", nonEmpty)
        }
        Seq("/* ", lines.head) +: lineSeqs :+ Seq(" */")
      } else {
        Seq(Seq("// ", lines(0)))
      }
    }

    def build_attribute(attrs: String): Seq[Seq[String]] = {
      Seq(Seq("(* ") ++ Seq(attrs) ++ Seq(" *)"))
    }

    // Turn ports into Seq[String] and add to portdefs
    def build_ports(): Unit = {
      def padToMax(strs: Seq[String]): Seq[String] = {
        val len = if (strs.nonEmpty) strs.map(_.length).max else 0
        strs.map(_.padTo(len, ' '))
      }

      // Turn directions into strings (and AnalogType into inout)
      val dirs = m.ports.map {
        case Port(_, name, dir, tpe) =>
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
          portDescriptions.get(name).map {
            case d =>
              portdefs += Seq("")
              portdefs ++= build_description(d)
          }

          if (i != m.ports.size - 1) {
            portdefs += Seq(dir, " ", tpe, " ", name, ",", info)
          } else {
            portdefs += Seq(dir, " ", tpe, " ", name, info)
          }
      }
    }

    def build_description(d: Seq[Description]): Seq[Seq[String]] = d.flatMap {
      case DocString(desc) => build_comment(desc.string)
      case Attribute(attr) => build_attribute(attr.string)
    }

    def build_streams(s: Statement): Unit = {
      val withoutDescription = s match {
        case DescribedStmt(d, stmt) =>
          stmt match {
            case sx: IsDeclaration =>
              declares ++= build_description(d)
            case _ =>
          }
          stmt
        case stmt => stmt
      }
      withoutDescription.foreach(build_streams)
      withoutDescription match {
        case sx @ Connect(info, loc @ WRef(_, _, PortKind | WireKind | InstanceKind, _), expr) =>
          assign(loc, expr, info)
        case sx: DefWire =>
          declare("wire", sx.name, sx.tpe, sx.info)
        case sx: DefRegister =>
          val options = emissionOptions.getRegisterEmissionOption(moduleTarget.ref(sx.name))
          val e = wref(sx.name, sx.tpe)
          if (options.useInitAsPreset) {
            declare("reg", sx.name, sx.tpe, sx.info, sx.init)
            regUpdate(e, sx.clock, sx.reset, e)
          } else {
            declare("reg", sx.name, sx.tpe, sx.info)
            regUpdate(e, sx.clock, sx.reset, sx.init)
          }
          if (!options.disableRandomization)
            initialize(e, sx.reset, sx.init)
        case sx: DefNode =>
          declare("wire", sx.name, sx.value.tpe, sx.info, sx.value)
        case sx: Stop =>
          simulate(sx.clk, sx.en, stop(sx.ret), Some("STOP_COND"), sx.info)
        case sx: Print =>
          simulate(sx.clk, sx.en, printf(sx.string, sx.args), Some("PRINTF_COND"), sx.info)
        case sx: Verification =>
          addFormal(sx.clk, sx.en, formalStatement(sx.op, sx.pred), sx.info, sx.msg)
        // If we are emitting an Attach, it must not have been removable in VerilogPrep
        case sx: Attach =>
          // For Synthesis
          // Note that this is quadratic in the number of things attached
          for (set <- sx.exprs.toSet.subsets(2)) {
            val (a, b) = set.toSeq match {
              case Seq(x, y) => (x, y)
            }
            // Synthesizable ones as well
            attachSynAssigns += Seq("assign ", a, " = ", b, ";", sx.info)
            attachSynAssigns += Seq("assign ", b, " = ", a, ";", sx.info)
          }
          // alias implementation for everything else
          attachAliases += Seq("alias ", sx.exprs.flatMap(e => Seq(e, " = ")).init, ";", sx.info)
        case sx: WDefInstanceConnector =>
          val (module, params) = moduleMap(sx.module) match {
            case DescribedMod(_, _, ExtModule(_, _, _, extname, params)) => (extname, params)
            case DescribedMod(_, _, Module(_, name, _, _))               => (name, Seq.empty)
            case ExtModule(_, _, _, extname, params)                     => (extname, params)
            case Module(_, name, _, _)                                   => (name, Seq.empty)
          }
          val ps = if (params.nonEmpty) params.map(stringify).mkString("#(", ", ", ") ") else ""
          instdeclares += Seq(module, " ", ps, sx.name, " (", sx.info)
          for (((port, ref), i) <- sx.portCons.zipWithIndex) {
            val line = Seq(tab, ".", remove_root(port), "(", ref, ")")
            if (i != sx.portCons.size - 1) instdeclares += Seq(line, ",")
            else instdeclares += line
          }
          instdeclares += Seq(");")
        case sx: DefMemory =>
          val options = emissionOptions.getMemoryEmissionOption(moduleTarget.ref(sx.name))
          val fullSize = sx.depth * (sx.dataType match {
            case GroundType(IntWidth(width)) => width
          })
          val decl = if (fullSize > (1 << 29)) "reg /* sparse */" else "reg"
          declareVectorType(decl, sx.name, sx.dataType, sx.depth, sx.info)
          initialize_mem(sx, options)
          if (sx.readLatency != 0 || sx.writeLatency != 1)
            throw EmitterException(
              "All memories should be transformed into " +
                "blackboxes or combinational by previous passses"
            )
          for (r <- sx.readers) {
            val data = memPortField(sx, r, "data")
            val addr = memPortField(sx, r, "addr")
            // Ports should share an always@posedge, so can't have intermediary wire

            declare("wire", LowerTypes.loweredName(data), data.tpe, sx.info)
            declare("wire", LowerTypes.loweredName(addr), addr.tpe, sx.info)
            // declare("wire", LowerTypes.loweredName(en), en.tpe)

            //; Read port
            assign(addr, netlist(addr))
            // assign(en, netlist(en))     //;Connects value to m.r.en
            val mem = WRef(sx.name, memType(sx), MemKind, UnknownFlow)
            val memPort = WSubAccess(mem, addr, sx.dataType, UnknownFlow)
            val depthValue = UIntLiteral(sx.depth, IntWidth(sx.depth.bitLength))
            val garbageGuard = DoPrim(Geq, Seq(addr, depthValue), Seq(), UnknownType)

            if ((sx.depth & (sx.depth - 1)) == 0)
              assign(data, memPort, sx.info)
            else
              garbageAssign(data, memPort, garbageGuard, sx.info)
          }

          for (w <- sx.writers) {
            val data = memPortField(sx, w, "data")
            val addr = memPortField(sx, w, "addr")
            val mask = memPortField(sx, w, "mask")
            val en = memPortField(sx, w, "en")
            //Ports should share an always@posedge, so can't have intermediary wire
            // TODO should we use the info here for anything?
            val InfoExpr(_, clk) = netlist(memPortField(sx, w, "clk"))

            declare("wire", LowerTypes.loweredName(data), data.tpe, sx.info)
            declare("wire", LowerTypes.loweredName(addr), addr.tpe, sx.info)
            declare("wire", LowerTypes.loweredName(mask), mask.tpe, sx.info)
            declare("wire", LowerTypes.loweredName(en), en.tpe, sx.info)

            // Write port
            assign(data, netlist(data))
            assign(addr, netlist(addr))
            assign(mask, netlist(mask))
            assign(en, netlist(en))

            val mem = WRef(sx.name, memType(sx), MemKind, UnknownFlow)
            val memPort = WSubAccess(mem, addr, sx.dataType, UnknownFlow)
            update(memPort, data, clk, AND(en, mask), sx.info)
          }

          if (sx.readwriters.nonEmpty)
            throw EmitterException(
              "All readwrite ports should be transformed into " +
                "read & write ports by previous passes"
            )
        case _ =>
      }
    }

    def emit_streams(): Unit = {
      build_description(description).foreach(emit(_))
      emit(Seq("module ", m.name, "(", m.info))
      for (x <- portdefs) emit(Seq(tab, x))
      emit(Seq(");"))

      ifdefDeclares.toSeq.sortWith(_._1 < _._1).foreach {
        case (ifdef, declares) =>
          emit(Seq("`ifdef " + ifdef))
          for (x <- declares) emit(Seq(tab, x))
          emit(Seq("`endif // " + ifdef))
      }
      for (x <- declares) emit(Seq(tab, x))
      for (x <- instdeclares) emit(Seq(tab, x))
      for (x <- assigns) emit(Seq(tab, x))
      if (attachAliases.nonEmpty) {
        emit(Seq("`ifdef SYNTHESIS"))
        for (x <- attachSynAssigns) emit(Seq(tab, x))
        emit(Seq("`elsif verilator"))
        emit(
          Seq(
            tab,
            "`error \"Verilator does not support alias and thus cannot arbirarily connect bidirectional wires and ports\""
          )
        )
        emit(Seq("`else"))
        for (x <- attachAliases) emit(Seq(tab, x))
        emit(Seq("`endif"))
      }

      for ((clk, content) <- noResetAlwaysBlocks if content.nonEmpty) {
        emit(Seq(tab, "always @(posedge ", clk, ") begin"))
        for (line <- content) emit(Seq(tab, tab, line))
        emit(Seq(tab, "end"))
      }

      for ((clk, reset, content) <- asyncResetAlwaysBlocks if content.nonEmpty) {
        emit(Seq(tab, "always @(posedge ", clk, " or posedge ", reset, ") begin"))
        for (line <- content) emit(Seq(tab, tab, line))
        emit(Seq(tab, "end"))
      }

      if (initials.nonEmpty || ifdefInitials.nonEmpty || memoryInitials.nonEmpty) {
        emit(Seq("// Register and memory initialization"))
        emit(Seq("`ifdef RANDOMIZE_GARBAGE_ASSIGN"))
        emit(Seq("`define RANDOMIZE"))
        emit(Seq("`endif"))
        emit(Seq("`ifdef RANDOMIZE_INVALID_ASSIGN"))
        emit(Seq("`define RANDOMIZE"))
        emit(Seq("`endif"))
        emit(Seq("`ifdef RANDOMIZE_REG_INIT"))
        emit(Seq("`define RANDOMIZE"))
        emit(Seq("`endif"))
        emit(Seq("`ifdef RANDOMIZE_MEM_INIT"))
        emit(Seq("`define RANDOMIZE"))
        emit(Seq("`endif"))
        emit(Seq("`ifndef RANDOM"))
        emit(Seq("`define RANDOM $random"))
        emit(Seq("`endif"))
        // the initvar is also used to initialize memories to constants
        if (memoryInitials.isEmpty) emit(Seq("`ifdef RANDOMIZE_MEM_INIT"))
        // Since simulators don't actually support memories larger than 2^31 - 1, there is no reason
        // to change Verilog emission in the common case. Instead, we only emit a larger initvar
        // where necessary
        if (maxMemSize.isValidInt) {
          emit(Seq("  integer initvar;"))
        } else {
          // Width must be able to represent maxMemSize because that's the upper bound in init loop
          val width = maxMemSize.bitLength - 1 // minus one because [width-1:0] has a width of "width"
          emit(Seq(s"  reg [$width:0] initvar;"))
        }
        if (memoryInitials.isEmpty) emit(Seq("`endif"))
        emit(Seq("`ifndef SYNTHESIS"))
        // User-defined macro of code to run before an initial block
        emit(Seq("`ifdef FIRRTL_BEFORE_INITIAL"))
        emit(Seq("`FIRRTL_BEFORE_INITIAL"))
        emit(Seq("`endif"))
        emit(Seq("initial begin"))
        emit(Seq("  `ifdef RANDOMIZE"))
        emit(Seq("    `ifdef INIT_RANDOM"))
        emit(Seq("      `INIT_RANDOM"))
        emit(Seq("    `endif"))
        // This enables testbenches to seed the random values at some time
        // before `RANDOMIZE_DELAY (or the legacy value 0.002 if
        // `RANDOMIZE_DELAY is not defined).
        // Verilator does not support delay statements, so they are omitted.
        emit(Seq("    `ifndef VERILATOR"))
        emit(Seq("      `ifdef RANDOMIZE_DELAY"))
        emit(Seq("        #`RANDOMIZE_DELAY begin end"))
        emit(Seq("      `else"))
        emit(Seq("        #0.002 begin end"))
        emit(Seq("      `endif"))
        emit(Seq("    `endif"))
        ifdefInitials.toSeq.sortWith(_._1 < _._1).foreach {
          case (ifdef, initials) =>
            emit(Seq("`ifdef " + ifdef))
            for (x <- initials) emit(Seq(tab, x))
            emit(Seq("`endif // " + ifdef))
        }
        for (x <- initials) emit(Seq(tab, x))
        for (x <- asyncInitials) emit(Seq(tab, x))
        emit(Seq("  `endif // RANDOMIZE"))
        for (x <- memoryInitials) emit(Seq(tab, x))
        emit(Seq("end // initial"))
        // User-defined macro of code to run after an initial block
        emit(Seq("`ifdef FIRRTL_AFTER_INITIAL"))
        emit(Seq("`FIRRTL_AFTER_INITIAL"))
        emit(Seq("`endif"))
        emit(Seq("`endif // SYNTHESIS"))
      }

      if (formals.keys.nonEmpty) {
        for ((clk, content) <- formals if content.nonEmpty) {
          emit(Seq(tab, "always @(posedge ", clk, ") begin"))
          for (line <- content) emit(Seq(tab, tab, line))
          emit(Seq(tab, "end"))
        }
      }

      emit(Seq("endmodule"))
    }

    /**
      * The standard verilog emitter, wraps up everything into the
      * verilog
      * @return
      */
    def emit_verilog(): DefModule = {

      build_netlist(m.body)
      build_ports()
      build_streams(m.body)
      emit_streams()
      m
    }

    /**
      * This emits a verilog module that can be bound to a module defined in chisel.
      * It uses the same machinery as the general emitter in order to insure that
      * parameters signature is exactly the same as the module being bound to
      * @param overrideName Override the module name
      * @param body the body of the bind module
      * @return A module constructed from the body
      */
    def emitVerilogBind(overrideName: String, body: String): DefModule = {
      build_netlist(m.body)
      build_ports()

      build_description(description).foreach(emit(_))

      emit(Seq("module ", overrideName, "(", m.info))
      for (x <- portdefs) emit(Seq(tab, x))

      emit(Seq(");"))
      emit(body)
      emit(Seq("endmodule"), top = 0)
      m
    }
  }

  /** Preamble for every emitted Verilog file */
  def transforms = new TransformManager(firrtl.stage.Forms.VerilogOptimized, prerequisites).flattenedTransformOrder

  def emit(state: CircuitState, writer: Writer): Unit = {
    val cs = runTransforms(state)
    val emissionOptions = new EmissionOptions(cs.annotations)
    val moduleMap = cs.circuit.modules.map(m => m.name -> m).toMap
    cs.circuit.modules.foreach {
      case dm @ DescribedMod(d, pds, m: Module) =>
        val renderer = new VerilogRender(d, pds, m, moduleMap, cs.circuit.main, emissionOptions)(writer)
        renderer.emit_verilog()
      case m: Module =>
        val renderer = new VerilogRender(m, moduleMap, cs.circuit.main, emissionOptions)(writer)
        renderer.emit_verilog()
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
          EmittedVerilogCircuitAnnotation(
            EmittedVerilogCircuit(state.circuit.main, writerToString(writer), outputSuffix)
          )
        )

      case EmitAllModulesAnnotation(a) if this.getClass == a =>
        val cs = runTransforms(state)
        val emissionOptions = new EmissionOptions(cs.annotations)
        val moduleMap = cs.circuit.modules.map(m => m.name -> m).toMap

        cs.circuit.modules.flatMap {
          case dm @ DescribedMod(d, pds, module: Module) =>
            val writer = new java.io.StringWriter
            val renderer = new VerilogRender(d, pds, module, moduleMap, cs.circuit.main, emissionOptions)(writer)
            renderer.emit_verilog()
            Some(
              EmittedVerilogModuleAnnotation(EmittedVerilogModule(module.name, writerToString(writer), outputSuffix))
            )
          case module: Module =>
            val writer = new java.io.StringWriter
            val renderer = new VerilogRender(module, moduleMap, cs.circuit.main, emissionOptions)(writer)
            renderer.emit_verilog()
            Some(
              EmittedVerilogModuleAnnotation(EmittedVerilogModule(module.name, writerToString(writer), outputSuffix))
            )
          case _ => None
        }
      case _ => Seq()
    }
    state.copy(annotations = newAnnos ++ state.annotations)
  }
}

case class VRandom(width: BigInt) extends Expression {
  def tpe = UIntType(IntWidth(width))
  def nWords = (width + 31) / 32
  def realWidth = nWords * 32
  override def serialize: String = "RANDOM"
  def mapExpr(f:      Expression => Expression): Expression = this
  def mapType(f:      Type => Type):             Expression = this
  def mapWidth(f:     Width => Width):           Expression = this
  def foreachExpr(f:  Expression => Unit):       Unit = ()
  def foreachType(f:  Type => Unit):             Unit = ()
  def foreachWidth(f: Width => Unit):            Unit = ()
}
