// See LICENSE for license details.

package firrtl

import java.io.Writer

import scala.collection.mutable

import firrtl.ir._
import firrtl.passes._
import firrtl.transforms.LegalizeAndReductionsTransform
import firrtl.annotations._
import firrtl.traversals.Foreachers._
import firrtl.PrimOps._
import firrtl.WrappedExpression._
import Utils._
import MemPortUtils.{memPortField, memType}
import firrtl.options.{Dependency, HasShellOptions, ShellOption, StageUtils, PhaseException, Unserializable}
import firrtl.stage.{RunFirrtlTransformAnnotation, TransformManager}
// Datastructures
import scala.collection.mutable.ArrayBuffer

case class EmitterException(message: String) extends PassException(message)

// ***** Annotations for telling the Emitters what to emit *****
sealed trait EmitAnnotation extends NoTargetAnnotation {
  val emitter: Class[_ <: Emitter]
}
case class EmitCircuitAnnotation(emitter: Class[_ <: Emitter]) extends EmitAnnotation
case class EmitAllModulesAnnotation(emitter: Class[_ <: Emitter]) extends EmitAnnotation

object EmitCircuitAnnotation extends HasShellOptions {

  val options = Seq(
    new ShellOption[String](
      longOption = "emit-circuit",
      toAnnotationSeq = (a: String) => a match {
        case "chirrtl"              => Seq(RunFirrtlTransformAnnotation(new ChirrtlEmitter),
                                           EmitCircuitAnnotation(classOf[ChirrtlEmitter]))
        case "high"                 => Seq(RunFirrtlTransformAnnotation(new HighFirrtlEmitter),
                                           EmitCircuitAnnotation(classOf[HighFirrtlEmitter]))
        case "middle"               => Seq(RunFirrtlTransformAnnotation(new MiddleFirrtlEmitter),
                                           EmitCircuitAnnotation(classOf[MiddleFirrtlEmitter]))
        case "low"                  => Seq(RunFirrtlTransformAnnotation(new LowFirrtlEmitter),
                                           EmitCircuitAnnotation(classOf[LowFirrtlEmitter]))
        case "verilog" | "mverilog" => Seq(RunFirrtlTransformAnnotation(new VerilogEmitter),
                                           EmitCircuitAnnotation(classOf[VerilogEmitter]))
        case "sverilog"             => Seq(RunFirrtlTransformAnnotation(new SystemVerilogEmitter),
                                           EmitCircuitAnnotation(classOf[SystemVerilogEmitter]))
        case _                      => throw new PhaseException(s"Unknown emitter '$a'! (Did you misspell it?)") },
      helpText = "Run the specified circuit emitter (all modules in one file)",
      shortOption = Some("E"),
      helpValueName = Some("<chirrtl|high|middle|low|verilog|mverilog|sverilog>") ) )

}

object EmitAllModulesAnnotation extends HasShellOptions {

  val options = Seq(
    new ShellOption[String](
      longOption = "emit-modules",
      toAnnotationSeq = (a: String) => a match {
        case "chirrtl"              => Seq(RunFirrtlTransformAnnotation(new ChirrtlEmitter),
                                           EmitAllModulesAnnotation(classOf[ChirrtlEmitter]))
        case "high"                 => Seq(RunFirrtlTransformAnnotation(new HighFirrtlEmitter),
                                           EmitAllModulesAnnotation(classOf[HighFirrtlEmitter]))
        case "middle"               => Seq(RunFirrtlTransformAnnotation(new MiddleFirrtlEmitter),
                                           EmitAllModulesAnnotation(classOf[MiddleFirrtlEmitter]))
        case "low"                  => Seq(RunFirrtlTransformAnnotation(new LowFirrtlEmitter),
                                           EmitAllModulesAnnotation(classOf[LowFirrtlEmitter]))
        case "verilog" | "mverilog" => Seq(RunFirrtlTransformAnnotation(new VerilogEmitter),
                                           EmitAllModulesAnnotation(classOf[VerilogEmitter]))
        case "sverilog"             => Seq(RunFirrtlTransformAnnotation(new SystemVerilogEmitter),
                                           EmitAllModulesAnnotation(classOf[SystemVerilogEmitter]))
        case _                      => throw new PhaseException(s"Unknown emitter '$a'! (Did you misspell it?)") },
      helpText = "Run the specified module emitter (one file per module)",
      shortOption = Some("e"),
      helpValueName = Some("<chirrtl|high|middle|low|verilog|mverilog|sverilog>") ) )

}

// ***** Annotations for results of emission *****
sealed abstract class EmittedComponent {
  def name: String
  def value: String
  def outputSuffix: String
}
sealed abstract class EmittedCircuit extends EmittedComponent
final case class EmittedFirrtlCircuit(name: String, value: String, outputSuffix: String) extends EmittedCircuit
final case class EmittedVerilogCircuit(name: String, value: String, outputSuffix: String) extends EmittedCircuit
sealed abstract class EmittedModule extends EmittedComponent
final case class EmittedFirrtlModule(name: String, value: String, outputSuffix: String) extends EmittedModule
final case class EmittedVerilogModule(name: String, value: String, outputSuffix: String) extends EmittedModule

/** Traits for Annotations containing emitted components */
sealed trait EmittedAnnotation[T <: EmittedComponent] extends NoTargetAnnotation with Unserializable {
  val value: T
}
sealed trait EmittedCircuitAnnotation[T <: EmittedCircuit] extends EmittedAnnotation[T]
sealed trait EmittedModuleAnnotation[T <: EmittedModule] extends EmittedAnnotation[T]

case class EmittedFirrtlCircuitAnnotation(value: EmittedFirrtlCircuit)
    extends EmittedCircuitAnnotation[EmittedFirrtlCircuit]
case class EmittedVerilogCircuitAnnotation(value: EmittedVerilogCircuit)
    extends EmittedCircuitAnnotation[EmittedVerilogCircuit]

case class EmittedFirrtlModuleAnnotation(value: EmittedFirrtlModule)
    extends EmittedModuleAnnotation[EmittedFirrtlModule]
case class EmittedVerilogModuleAnnotation(value: EmittedVerilogModule)
    extends EmittedModuleAnnotation[EmittedVerilogModule]

sealed abstract class FirrtlEmitter(form: CircuitForm) extends Transform with Emitter {
  def inputForm = form
  def outputForm = form

  val outputSuffix: String = form.outputSuffix

  private def emitAllModules(circuit: Circuit): Seq[EmittedFirrtlModule] = {
    // For a given module, returns a Seq of all modules instantited inside of it
    def collectInstantiatedModules(mod: Module, map: Map[String, DefModule]): Seq[DefModule] = {
      // Use list instead of set to maintain order
      val modules = mutable.ArrayBuffer.empty[DefModule]
      def onStmt(stmt: Statement): Unit = stmt match {
        case DefInstance(_, _, name, _) => modules += map(name)
        case WDefInstance(_, _, name, _) => modules += map(name)
        case _: WDefInstanceConnector => throwInternalError(s"unrecognized statement: $stmt")
        case other => other.foreach(onStmt)
      }
      onStmt(mod.body)
      modules.distinct
    }
    val modMap = circuit.modules.map(m => m.name -> m).toMap
    // Turn each module into it's own circuit with it as the top and all instantied modules as ExtModules
    circuit.modules collect { case m: Module =>
      val instModules = collectInstantiatedModules(m, modMap)
      val extModules = instModules map {
        case Module(info, name, ports, _) => ExtModule(info, name, ports, name, Seq.empty)
        case ext: ExtModule => ext
      }
      val newCircuit = Circuit(m.info, extModules :+ m, m.name)
      EmittedFirrtlModule(m.name, newCircuit.serialize, outputSuffix)
    }
  }

  override def execute(state: CircuitState): CircuitState = {
    val newAnnos = state.annotations.flatMap {
      case EmitCircuitAnnotation(a) if this.getClass == a =>
        Seq(EmittedFirrtlCircuitAnnotation(
              EmittedFirrtlCircuit(state.circuit.main, state.circuit.serialize, outputSuffix)))
      case EmitAllModulesAnnotation(a) if this.getClass == a =>
        emitAllModules(state.circuit) map (EmittedFirrtlModuleAnnotation(_))
      case _ => Seq()
    }
    state.copy(annotations = newAnnos ++ state.annotations)
  }

  // Old style, deprecated
  def emit(state: CircuitState, writer: Writer): Unit = writer.write(state.circuit.serialize)
}

// ***** Start actual Emitters *****
class ChirrtlEmitter extends FirrtlEmitter(ChirrtlForm)
class HighFirrtlEmitter extends FirrtlEmitter(HighForm)
class MiddleFirrtlEmitter extends FirrtlEmitter(MidForm)
class LowFirrtlEmitter extends FirrtlEmitter(LowForm)

case class VRandom(width: BigInt) extends Expression {
  def tpe = UIntType(IntWidth(width))
  def nWords = (width + 31) / 32
  def realWidth = nWords * 32
  def serialize: String = "RANDOM"
  def mapExpr(f: Expression => Expression): Expression = this
  def mapType(f: Type => Type): Expression = this
  def mapWidth(f: Width => Width): Expression = this
  def foreachExpr(f: Expression => Unit): Unit = Unit
  def foreachType(f: Type => Unit): Unit = Unit
  def foreachWidth(f: Width => Unit): Unit = Unit
}

class VerilogEmitter extends SeqTransform with Emitter {
  def inputForm = LowForm
  def outputForm = LowForm

  override def prerequisites =
    Dependency[LegalizeAndReductionsTransform] +:
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
  def wref(n: String, t: Type) = WRef(n, t, ExpKind, UnknownFlow)
  def remove_root(ex: Expression): Expression = ex match {
    case ex: WSubField => ex.expr match {
      case (e: WSubField) => remove_root(e)
      case (_: WRef) => WRef(ex.name, ex.tpe, InstanceKind, UnknownFlow)
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
          if (value > 0) s"$blen'd$value" else s"-${blen+1}'sd${value.abs}"
        }
      s".$name($lit)"
    case DoubleParam(name, value) => s".$name($value)"
    case StringParam(name, value) => s".${name}(${value.verilogEscape})"
    case RawStringParam(name, value) => s".$name($value)"
  }
  def stringify(tpe: GroundType): String = tpe match {
    case (_: UIntType | _: SIntType | _: AnalogType) =>
      val wx = bitWidth(tpe) - 1
      if (wx > 0) s"[$wx:0]" else ""
    case ClockType | AsyncResetType => ""
    case _ => throwInternalError(s"trying to write unsupported type in the Verilog Emitter: $tpe")
  }
  def emit(x: Any)(implicit w: Writer): Unit = { emit(x, 0) }
  def emit(x: Any, top: Int)(implicit w: Writer): Unit = {
    def cast(e: Expression): Any = e.tpe match {
      case (t: UIntType) => e
      case (t: SIntType) => Seq("$signed(",e,")")
      case ClockType => e
      case AnalogType(_) => e
      case _ => throwInternalError(s"unrecognized cast: $e")
    }
    x match {
      case (e: DoPrim) => emit(op_stream(e), top + 1)
      case (e: Mux) => {
        if (e.tpe == ClockType) {
          throw EmitterException("Cannot emit clock muxes directly")
        }
        if (e.tpe == AsyncResetType) {
          throw EmitterException("Cannot emit async reset muxes directly")
        }
        emit(Seq(e.cond," ? ",cast(e.tval)," : ",cast(e.fval)),top + 1)
      }
      case (e: ValidIf) => emit(Seq(cast(e.value)),top + 1)
      case (e: WRef) => w write e.serialize
      case (e: WSubField) => w write LowerTypes.loweredName(e)
      case (e: WSubAccess) => w write s"${LowerTypes.loweredName(e.expr)}[${LowerTypes.loweredName(e.index)}]"
      case (e: WSubIndex) => w write e.serialize
      case (e: Literal) => v_print(e)
      case (e: VRandom) => w write s"{${e.nWords}{`RANDOM}}"
      case (t: GroundType) => w write stringify(t)
      case (t: VectorType) =>
        emit(t.tpe, top + 1)
        w write s"[${t.size - 1}:0]"
      case (s: String) => w write s
      case (i: Int) => w write i.toString
      case (i: Long) => w write i.toString
      case (i: BigInt) => w write i.toString
      case (i: Info) => i match {
        case NoInfo => // Do nothing
        case ix => w.write(s" //$ix")
      }
      case (s: Seq[Any]) =>
        s foreach (emit(_, top + 1))
        if (top == 0) w write "\n"
      case x => throwInternalError(s"trying to emit unsupported operator: $x")
    }
  }

   //;------------- PASS -----------------
   def v_print(e: Expression)(implicit w: Writer) = e match {
     case UIntLiteral(value, IntWidth(width)) =>
       w write s"$width'h${value.toString(16)}"
     case SIntLiteral(value, IntWidth(width)) =>
       val stringLiteral = value.toString(16)
       w write (stringLiteral.head match {
         case '-' => s"-$width'sh${stringLiteral.tail}"
         case _ => s"$width'sh${stringLiteral}"
       })
     case _ => throwInternalError(s"attempt to print unrecognized expression: $e")
   }

   // NOTE: We emit SInts as regular Verilog unsigned wires/regs so the real type of any SInt
   // reference is actually unsigned in the emitted Verilog. Thus we must cast refs as necessary
   // to ensure Verilog operations are signed.
   def op_stream(doprim: DoPrim): Seq[Any] = {
     // Cast to SInt, don't cast multiple times
     def doCast(e: Expression): Any = e match {
       case DoPrim(AsSInt, Seq(arg), _,_) => doCast(arg)
       case slit: SIntLiteral             => slit
       case other                         => Seq("$signed(", other, ")")
     }
     def castIf(e: Expression): Any = {
       if (doprim.args.exists(_.tpe.isInstanceOf[SIntType])) {
         e.tpe match {
           case _: SIntType => doCast(e)
           case _ => throwInternalError(s"Unexpected non-SInt type for $e in $doprim")
         }
       } else {
         e
       }
     }
     def cast(e: Expression): Any = doprim.tpe match {
       case _: UIntType => e
       case _: SIntType => doCast(e)
       case _ => throwInternalError(s"Unexpected type for $e in $doprim")
     }
     def castAs(e: Expression): Any = e.tpe match {
       case _: UIntType => e
       case _: SIntType => doCast(e)
       case _ => throwInternalError(s"Unexpected type for $e in $doprim")
     }
     def a0: Expression = doprim.args.head
     def a1: Expression = doprim.args(1)
     def c0: Int = doprim.consts.head.toInt
     def c1: Int = doprim.consts(1).toInt

     def checkArgumentLegality(e: Expression): Unit = e match {
       case _: UIntLiteral | _: SIntLiteral | _: WRef | _: WSubField =>
       case DoPrim(Not, args, _,_) => args.foreach(checkArgumentLegality)
       case DoPrim(op, args, _,_) if isCast(op) => args.foreach(checkArgumentLegality)
       case DoPrim(op, args, _,_) if isBitExtract(op) => args.foreach(checkArgumentLegality)
       case _ => throw EmitterException(s"Can't emit ${e.getClass.getName} as PrimOp argument")
     }

     def checkCatArgumentLegality(e: Expression): Unit = e match {
       case DoPrim(Cat, args, _, _) => args foreach(checkCatArgumentLegality)
       case _ => checkArgumentLegality(e)
     }

     def castCatArgs(a0: Expression, a1: Expression): Seq[Any] = {
       val a0Seq = a0 match {
         case cat@DoPrim(PrimOps.Cat, args, _, _) => castCatArgs(args.head, args(1))
         case _ => Seq(cast(a0))
       }
       val a1Seq = a1 match {
         case cat@DoPrim(PrimOps.Cat, args, _, _) => castCatArgs(args.head, args(1))
         case _ => Seq(cast(a1))
       }
       a0Seq ++ Seq(",") ++ a1Seq
     }

     doprim.op match {
       case Cat => doprim.args foreach(checkCatArgumentLegality)
       case cast if isCast(cast) => // Casts are allowed to wrap any Expression
       case other => doprim.args foreach checkArgumentLegality
     }
     doprim.op match {
       case Add => Seq(castIf(a0), " + ", castIf(a1))
       case Addw => Seq(castIf(a0), " + ", castIf(a1))
       case Sub => Seq(castIf(a0), " - ", castIf(a1))
       case Subw => Seq(castIf(a0), " - ", castIf(a1))
       case Mul => Seq(castIf(a0), " * ", castIf(a1))
       case Div => Seq(castIf(a0), " / ", castIf(a1))
       case Rem => Seq(castIf(a0), " % ", castIf(a1))
       case Lt => Seq(castIf(a0), " < ", castIf(a1))
       case Leq => Seq(castIf(a0), " <= ", castIf(a1))
       case Gt => Seq(castIf(a0), " > ", castIf(a1))
       case Geq => Seq(castIf(a0), " >= ", castIf(a1))
       case Eq => Seq(castIf(a0), " == ", castIf(a1))
       case Neq => Seq(castIf(a0), " != ", castIf(a1))
       case Pad =>
         val w = bitWidth(a0.tpe)
         val diff = c0 - w
         if (w == BigInt(0) || diff <= 0) Seq(a0)
         else doprim.tpe match {
           // Either sign extend or zero extend.
           // If width == BigInt(1), don't extract bit
           case (_: SIntType) if w == BigInt(1) => Seq("{", c0, "{", a0, "}}")
           case (_: SIntType) => Seq("{{", diff, "{", a0, "[", w - 1, "]}},", a0, "}")
           case (_) => Seq("{{", diff, "'d0}, ", a0, "}")
         }
       // Because we don't support complex Expressions, all casts are ignored
       // This simplifies handling of assignment of a signed expression to an unsigned LHS value
       //   which does not require a cast in Verilog
       case AsUInt | AsSInt | AsClock | AsAsyncReset => Seq(a0)
       case Dshlw => Seq(cast(a0), " << ", a1)
       case Dshl => Seq(cast(a0), " << ", a1)
       case Dshr => doprim.tpe match {
         case (_: SIntType) => Seq(cast(a0)," >>> ", a1)
         case (_) => Seq(cast(a0), " >> ", a1)
       }
       case Shl => if (c0 > 0) Seq("{", cast(a0), s", $c0'h0}") else Seq(cast(a0))
       case Shr if c0 >= bitWidth(a0.tpe) =>
         error("Verilog emitter does not support SHIFT_RIGHT >= arg width")
       case Shr if c0 == (bitWidth(a0.tpe)-1) => Seq(a0,"[", bitWidth(a0.tpe) - 1, "]")
       case Shr => Seq(a0,"[", bitWidth(a0.tpe) - 1, ":", c0, "]")
       case Neg => Seq("-{", cast(a0), "}")
       case Cvt => a0.tpe match {
         case (_: UIntType) => Seq("{1'b0,", cast(a0), "}")
         case (_: SIntType) => Seq(cast(a0))
       }
       case Not => Seq("~", a0)
       case And => Seq(castAs(a0), " & ", castAs(a1))
       case Or => Seq(castAs(a0), " | ", castAs(a1))
       case Xor => Seq(castAs(a0), " ^ ", castAs(a1))
       case Andr => Seq("&", cast(a0))
       case Orr => Seq("|", cast(a0))
       case Xorr => Seq("^", cast(a0))
       case Cat => "{" +: (castCatArgs(a0, a1) :+ "}")
       // If selecting zeroth bit and single-bit wire, just emit the wire
       case Bits if c0 == 0 && c1 == 0 && bitWidth(a0.tpe) == BigInt(1) => Seq(a0)
       case Bits if c0 == c1 => Seq(a0, "[", c0, "]")
       case Bits => Seq(a0, "[", c0, ":", c1, "]")
       // If selecting zeroth bit and single-bit wire, just emit the wire
       case Head if c0 == 1 && bitWidth(a0.tpe) == BigInt(1) => Seq(a0)
       case Head if c0 == 1 => Seq(a0, "[", bitWidth(a0.tpe)-1, "]")
       case Head =>
         val msb = bitWidth(a0.tpe) - 1
         val lsb = bitWidth(a0.tpe) - c0
         Seq(a0, "[", msb, ":", lsb, "]")
       case Tail if c0 == (bitWidth(a0.tpe)-1) => Seq(a0, "[0]")
       case Tail => Seq(a0, "[", bitWidth(a0.tpe) - c0 - 1, ":0]")
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
  def getRenderer(descriptions: Seq[DescriptionAnnotation],
                  m: Module,
                  moduleMap: Map[String, DefModule])(implicit writer: Writer): VerilogRender = {
    val newMod = new AddDescriptionNodes().executeModule(m, descriptions)

    newMod match {
      case DescribedMod(d, pds, m: Module) => new VerilogRender(d, pds, m, moduleMap, "", new EmissionOptions(Seq.empty))(writer)
      case m: Module => new VerilogRender(m, moduleMap)(writer)
    }
  }
  
  /** 
    * Store Emission option per Target
    * Guarantee only one emission option per Target 
    */
  private[firrtl] class EmissionOptionMap[V <: EmissionOption](val df : V) extends collection.mutable.HashMap[ReferenceTarget, V] {
    override def default(key: ReferenceTarget) = df
    override def +=(elem : (ReferenceTarget, V)) : EmissionOptionMap.this.type = {
      if (this.contains(elem._1))
        throw EmitterException(s"Multiple EmissionOption for the target ${elem._1} (${this(elem._1)} ; ${elem._2})")
      super.+=(elem)
    } 
  }
  
  /** Provide API to retrieve EmissionOptions based on the provided [[AnnotationSeq]]
    * 
    * @param annotations : AnnotationSeq to be searched for EmissionOptions
    *
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

    private val emissionAnnos = annotations.collect{
      case m : SingleTargetAnnotation[ReferenceTarget] @unchecked with EmissionOption => m
    }
    // using multiple foreach instead of a single partial function as an Annotation can gather multiple EmissionOptions for simplicity
    emissionAnnos.foreach { case a :MemoryEmissionOption   => memoryEmissionOption += ((a.target,a))   case _ => }
    emissionAnnos.foreach { case a :RegisterEmissionOption => registerEmissionOption += ((a.target,a)) case _ => }
    emissionAnnos.foreach { case a :WireEmissionOption     => wireEmissionOption += ((a.target,a))     case _ => }
    emissionAnnos.foreach { case a :PortEmissionOption     => portEmissionOption += ((a.target,a))     case _ => }
    emissionAnnos.foreach { case a :NodeEmissionOption     => nodeEmissionOption += ((a.target,a))     case _ => }
    emissionAnnos.foreach { case a :ConnectEmissionOption  => connectEmissionOption += ((a.target,a))  case _ => }
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
  class VerilogRender(description: Seq[Description],
                      portDescriptions: Map[String, Seq[Description]],
                      m: Module,
                      moduleMap: Map[String, DefModule],
                      circuitName: String,
                      emissionOptions: EmissionOptions)(implicit writer: Writer) {

    def this(m: Module, moduleMap: Map[String, DefModule], circuitName: String, emissionOptions: EmissionOptions)(implicit writer: Writer) {
      this(Seq(), Map.empty, m, moduleMap, circuitName, emissionOptions)(writer)
    }
    def this(m: Module, moduleMap: Map[String, DefModule])(implicit writer: Writer) {
      this(Seq(), Map.empty, m, moduleMap, "", new EmissionOptions(Seq.empty))(writer)
    }

    val netlist = mutable.LinkedHashMap[WrappedExpression, Expression]()
    val namespace = Namespace(m)
    namespace.newName("_RAND") // Start rand names at _RAND_0
    def build_netlist(s: Statement): Unit = {
      s.foreach(build_netlist)
      s match {
        case sx: Connect => netlist(sx.loc) = sx.expr
        case sx: IsInvalid => error("Should have removed these!")
        case sx: DefNode =>
          val e = WRef(sx.name, sx.value.tpe, NodeKind, SourceFlow)
          netlist(e) = sx.value
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

    def bigIntToVLit(bi: BigInt): String =
      if (bi.isValidInt) bi.toString else s"${bi.bitLength}'d$bi"

    // declare vector type with no preset and optionally with an ifdef guard
    private def declareVectorType(b: String, n: String, tpe: Type, size: BigInt, info: Info, ifdefOpt: Option[String]): Unit = {
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
        val decl = Seq(b, " ", tx, " ", n,";",info)
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

    def assign(e: Expression, value: Expression, info: Info): Unit = {
      assigns += Seq("assign ", e, " = ", value, ";", info)
    }

    // In simulation, assign garbage under a predicate
    def garbageAssign(e: Expression, syn: Expression, garbageCond: Expression, info: Info) = {
      assigns += Seq("`ifndef RANDOMIZE_GARBAGE_ASSIGN")
      assigns += Seq("assign ", e, " = ", syn, ";", info)
      assigns += Seq("`else")
      assigns += Seq("assign ", e, " = ", garbageCond, " ? ", rand_string(syn.tpe, "RANDOMIZE_GARBAGE_ASSIGN"), " : ", syn,
                     ";", info)
      assigns += Seq("`endif // RANDOMIZE_GARBAGE_ASSIGN")
    }

    def invalidAssign(e: Expression) = {
      assigns += Seq("`ifdef RANDOMIZE_INVALID_ASSIGN")
      assigns += Seq("assign ", e, " = ", rand_string(e.tpe, "RANDOMIZE_INVALID_ASSIGN"), ";")
      assigns += Seq("`endif // RANDOMIZE_INVALID_ASSIGN")
    }

    def regUpdate(r: Expression, clk: Expression, reset: Expression, init: Expression) = {
      def addUpdate(expr: Expression, tabs: String): Seq[Seq[Any]] = expr match {
        case m: Mux =>
          if (m.tpe == ClockType) throw EmitterException("Cannot emit clock muxes directly")
          if (m.tpe == AsyncResetType) throw EmitterException("Cannot emit async reset muxes directly")

          lazy val _if     = Seq(tabs, "if (", m.cond, ") begin")
          lazy val _else   = Seq(tabs, "end else begin")
          lazy val _ifNot  = Seq(tabs, "if (!(", m.cond, ")) begin")
          lazy val _end    = Seq(tabs, "end")
          lazy val _true   = addUpdate(m.tval, tabs + tab)
          lazy val _false  = addUpdate(m.fval, tabs + tab)
          lazy val _elseIfFalse = {
            val _falsex = addUpdate(m.fval, tabs) // _false, but without an additional tab
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
            case (t, _) if weq(t, r)              =>  _ifNot +: _false                           :+ _end
            case (_, f) if weq(f, r) => m.cond.tpe match {
              case AsyncResetType =>                 (_if +: _true     :+ _else)       ++ _true  :+ _end
              case _ =>                               _if +: _true :+                               _end
            }
            case (_, _: Mux)                      => (_if    +: _true) ++ _elseIfFalse
            case _                                => (_if    +: _true  :+ _else)       ++ _false :+ _end
          }
        case e => Seq(Seq(tabs, r, " <= ", e, ";"))
      }
      if (weq(init, r)) { // Synchronous Reset
        noResetAlwaysBlocks.getOrElseUpdate(clk, ArrayBuffer[Seq[Any]]()) ++= addUpdate(netlist(r), "")
      } else { // Asynchronous Reset
        assert(reset.tpe == AsyncResetType, "Error! Synchronous reset should have been removed!")
        val tv = init
        val fv = netlist(r)
        asyncResetAlwaysBlocks += ((clk, reset, addUpdate(Mux(reset, tv, fv, mux_type_and_widths(tv, fv)), "")))
      }
    }

    def update(e: Expression, value: Expression, clk: Expression, en: Expression, info: Info) = {
      val lines = noResetAlwaysBlocks.getOrElseUpdate(clk, ArrayBuffer[Seq[Any]]())
      if (weq(en, one)) lines += Seq(e, " <= ", value, ";")
      else {
        lines += Seq("if(", en, ") begin")
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
        if(value < 0) throw EmitterException(s"Memory ${at} cannot be initialized with negative value: $value")
        if(value > maxDataValue) throw EmitterException(s"Memory ${at} cannot be initialized with value: $value. Too large (> $maxDataValue)!")
      }

      opt.initValue match {
        case MemoryArrayInit(values) =>
          if(values.length != s.depth) throw EmitterException(
            s"Memory ${s.name} of depth ${s.depth} cannot be initialized with an array of length ${values.length}!"
          )
          val memName = LowerTypes.loweredName(wref(s.name, s.dataType))
          values.zipWithIndex.foreach { case (value, addr) =>
            checkValueRange(value, s"${s.name}[$addr]")
            val access = s"$memName[${bigIntToVLit(addr)}]"
            memoryInitials += Seq(access, " = ", bigIntToVLit(value), ";")
          }
        case MemoryScalarInit(value) =>
          checkValueRange(value, s.name)
          // note: s.dataType is the incorrect type for initvar, but it is ignored in the serialization
          val index = wref("initvar", s.dataType)
          memoryInitials += Seq("for (initvar = 0; initvar < ", bigIntToVLit(s.depth), "; initvar = initvar+1)")
          memoryInitials += Seq(tab, WSubAccess(wref(s.name, s.dataType), index, s.dataType, SinkFlow),
            " = ", bigIntToVLit(value), ";")
        case MemoryRandomInit =>
          // note: s.dataType is the incorrect type for initvar, but it is ignored in the serialization
          val index = wref("initvar", s.dataType)
          val rstring = rand_string(s.dataType, "RANDOMIZE_MEM_INIT")
          ifdefInitials("RANDOMIZE_MEM_INIT") += Seq("for (initvar = 0; initvar < ", bigIntToVLit(s.depth), "; initvar = initvar+1)")
          ifdefInitials("RANDOMIZE_MEM_INIT") += Seq(tab, WSubAccess(wref(s.name, s.dataType), index, s.dataType, SinkFlow),
            " = ", rstring, ";")
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
          case "" => Seq(" *")
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
        strs map (_.padTo(len, ' '))
      }

      // Turn directions into strings (and AnalogType into inout)
      val dirs = m.ports map { case Port(_, name, dir, tpe) =>
        (dir, tpe) match {
          case (_, AnalogType(_)) => "inout " // padded to length of output
          case (Input, _) => "input "
          case (Output, _) => "output"
        }
      }
      // Turn types into strings, all ports must be GroundTypes
      val tpes = m.ports map {
        case Port(_, _, _, tpe: GroundType) => stringify(tpe)
        case port: Port => error(s"Trying to emit non-GroundType Port $port")
      }

      // dirs are already padded
      (dirs, padToMax(tpes), m.ports).zipped.toSeq.zipWithIndex.foreach {
        case ((dir, tpe, Port(info, name, _, _)), i) =>
          portDescriptions.get(name).map { case d =>
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
        case sx@Connect(info, loc@WRef(_, _, PortKind | WireKind | InstanceKind, _), expr) =>
          assign(loc, expr, info)
        case sx: DefWire =>
          declare("wire", sx.name, sx.tpe, sx.info)
        case sx: DefRegister =>
          val options = emissionOptions.getRegisterEmissionOption(moduleTarget.ref(sx.name))
          val e = wref(sx.name, sx.tpe)
          if (options.useInitAsPreset){
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
            case DescribedMod(_, _, Module(_, name, _, _)) => (name, Seq.empty)
            case ExtModule(_, _, _, extname, params) => (extname, params)
            case Module(_, name, _, _) => (name, Seq.empty)
          }
          val ps = if (params.nonEmpty) params map stringify mkString("#(", ", ", ") ") else ""
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
            throw EmitterException("All memories should be transformed into " +
                                     "blackboxes or combinational by previous passses")
          for (r <- sx.readers) {
            val data = memPortField(sx, r, "data")
            val addr = memPortField(sx, r, "addr")
            // Ports should share an always@posedge, so can't have intermediary wire

            declare("wire", LowerTypes.loweredName(data), data.tpe, sx.info)
            declare("wire", LowerTypes.loweredName(addr), addr.tpe, sx.info)
            // declare("wire", LowerTypes.loweredName(en), en.tpe)

            //; Read port
            assign(addr, netlist(addr), NoInfo) // Info should come from addr connection
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
            val clk = netlist(memPortField(sx, w, "clk"))

            declare("wire", LowerTypes.loweredName(data), data.tpe, sx.info)
            declare("wire", LowerTypes.loweredName(addr), addr.tpe, sx.info)
            declare("wire", LowerTypes.loweredName(mask), mask.tpe, sx.info)
            declare("wire", LowerTypes.loweredName(en), en.tpe, sx.info)

            // Write port
            // Info should come from netlist
            assign(data, netlist(data), NoInfo)
            assign(addr, netlist(addr), NoInfo)
            assign(mask, netlist(mask), NoInfo)
            assign(en, netlist(en), NoInfo)

            val mem = WRef(sx.name, memType(sx), MemKind, UnknownFlow)
            val memPort = WSubAccess(mem, addr, sx.dataType, UnknownFlow)
            update(memPort, data, clk, AND(en, mask), sx.info)
          }

          if (sx.readwriters.nonEmpty)
            throw EmitterException("All readwrite ports should be transformed into " +
                                     "read & write ports by previous passes")
        case _ =>
      }
    }

    def emit_streams(): Unit = {
      build_description(description).foreach(emit(_))
      emit(Seq("module ", m.name, "(", m.info))
      for (x <- portdefs) emit(Seq(tab, x))
      emit(Seq(");"))

      ifdefDeclares.toSeq.sortWith(_._1 < _._1).foreach { case (ifdef, declares) =>
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
        emit(Seq(tab, "`error \"Verilator does not support alias and thus cannot arbirarily connect bidirectional wires and ports\""))
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
        if(memoryInitials.isEmpty) emit(Seq("`ifdef RANDOMIZE_MEM_INIT"))
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
        if(memoryInitials.isEmpty) emit(Seq("`endif"))
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
        ifdefInitials.toSeq.sortWith(_._1 < _._1).foreach { case (ifdef, initials) =>
          emit(Seq("`ifdef " + ifdef))
          for (x <- initials) emit(Seq(tab, x))
          emit(Seq("`endif // " + ifdef))
        }
        for (x <- initials) emit(Seq(tab, x))
        for (x <- asyncInitials) emit(Seq(tab, x))
        emit(Seq("  `endif // RANDOMIZE"))
        for(x <- memoryInitials) emit(Seq(tab, x))
        emit(Seq("end // initial"))
        // User-defined macro of code to run after an initial block
        emit(Seq("`ifdef FIRRTL_AFTER_INITIAL"))
        emit(Seq("`FIRRTL_AFTER_INITIAL"))
        emit(Seq("`endif"))
        emit(Seq("`endif // SYNTHESIS"))
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
    val newAnnos = state.annotations.flatMap {
      case EmitCircuitAnnotation(a) if this.getClass == a =>
        val writer = new java.io.StringWriter
        emit(state, writer)
        Seq(EmittedVerilogCircuitAnnotation(EmittedVerilogCircuit(state.circuit.main, writer.toString, outputSuffix)))

      case EmitAllModulesAnnotation(a) if this.getClass == a =>
        val cs = runTransforms(state)
        val emissionOptions = new EmissionOptions(cs.annotations)
        val moduleMap = cs.circuit.modules.map(m => m.name -> m).toMap

        cs.circuit.modules flatMap {
          case dm @ DescribedMod(d, pds, module: Module) =>
            val writer = new java.io.StringWriter
            val renderer = new VerilogRender(d, pds, module, moduleMap, cs.circuit.main, emissionOptions)(writer)
            renderer.emit_verilog()
            Some(EmittedVerilogModuleAnnotation(EmittedVerilogModule(module.name, writer.toString, outputSuffix)))
          case module: Module =>
            val writer = new java.io.StringWriter
            val renderer = new VerilogRender(module, moduleMap, cs.circuit.main, emissionOptions)(writer)
            renderer.emit_verilog()
            Some(EmittedVerilogModuleAnnotation(EmittedVerilogModule(module.name, writer.toString, outputSuffix)))
          case _ => None
        }
      case _ => Seq()
    }
    state.copy(annotations = newAnnos ++ state.annotations)
  }
}

class MinimumVerilogEmitter extends VerilogEmitter with Emitter {

  override def prerequisites =
    Dependency[LegalizeAndReductionsTransform] +:
    firrtl.stage.Forms.LowFormMinimumOptimized

  override def transforms = new TransformManager(firrtl.stage.Forms.VerilogMinimumOptimized, prerequisites)
    .flattenedTransformOrder

}

class SystemVerilogEmitter extends VerilogEmitter {
  override val outputSuffix: String = ".sv"

  override def execute(state: CircuitState): CircuitState = {
    StageUtils.dramaticWarning("SystemVerilog Emitter is the same as the Verilog Emitter!")
    super.execute(state)
  }
}
