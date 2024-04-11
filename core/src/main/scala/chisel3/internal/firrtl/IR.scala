// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.firrtl

import firrtl.{ir => fir}
import chisel3._
import chisel3.internal._
import chisel3.experimental._
import chisel3.properties.{Property, PropertyType => PropertyTypeclass, Class, DynamicObject}
import _root_.firrtl.{ir => firrtlir}
import _root_.firrtl.{PrimOps, RenameMap}
import _root_.firrtl.annotations.Annotation

import scala.collection.immutable.NumericRange
import scala.math.BigDecimal.RoundingMode
import scala.annotation.nowarn
import scala.collection.mutable

// This object exists so that it can be package private and we don't have to individually mark every class
private[chisel3] object ir {

  case class PrimOp(name: String) {
    override def toString: String = name
  }

  object PrimOp {
    val AddOp = PrimOp("add")
    val SubOp = PrimOp("sub")
    val TailOp = PrimOp("tail")
    val HeadOp = PrimOp("head")
    val TimesOp = PrimOp("mul")
    val DivideOp = PrimOp("div")
    val RemOp = PrimOp("rem")
    val ShiftLeftOp = PrimOp("shl")
    val ShiftRightOp = PrimOp("shr")
    val DynamicShiftLeftOp = PrimOp("dshl")
    val DynamicShiftRightOp = PrimOp("dshr")
    val BitAndOp = PrimOp("and")
    val BitOrOp = PrimOp("or")
    val BitXorOp = PrimOp("xor")
    val BitNotOp = PrimOp("not")
    val ConcatOp = PrimOp("cat")
    val BitsExtractOp = PrimOp("bits")
    val LessOp = PrimOp("lt")
    val LessEqOp = PrimOp("leq")
    val GreaterOp = PrimOp("gt")
    val GreaterEqOp = PrimOp("geq")
    val EqualOp = PrimOp("eq")
    val PadOp = PrimOp("pad")
    val NotEqualOp = PrimOp("neq")
    val NegOp = PrimOp("neg")
    val MultiplexOp = PrimOp("mux")
    val AndReduceOp = PrimOp("andr")
    val OrReduceOp = PrimOp("orr")
    val XorReduceOp = PrimOp("xorr")
    val ConvertOp = PrimOp("cvt")
    val AsUIntOp = PrimOp("asUInt")
    val AsSIntOp = PrimOp("asSInt")
    val AsFixedPointOp = PrimOp("asFixedPoint")
    val AsIntervalOp = PrimOp("asInterval")
    val WrapOp = PrimOp("wrap")
    val SqueezeOp = PrimOp("squz")
    val ClipOp = PrimOp("clip")
    val SetBinaryPoint = PrimOp("setp")
    val IncreasePrecision = PrimOp("incp")
    val DecreasePrecision = PrimOp("decp")
    val AsClockOp = PrimOp("asClock")
    val AsAsyncResetOp = PrimOp("asAsyncReset")
  }

  sealed abstract class Arg {
    def localName: String = name
    def contextualName(ctx: Component): String = name
    def fullName(ctx:       Component): String = contextualName(ctx)
    def name: String
  }

  case class Node(id: HasId) extends Arg {
    override def contextualName(ctx: Component): String = id.getOptionRef match {
      case Some(arg) => arg.contextualName(ctx)
      case None      => id.instanceName
    }
    override def localName: String = id.getOptionRef match {
      case Some(arg) => arg.localName
      case None      => id.instanceName
    }
    def name: String = id.getOptionRef match {
      case Some(arg) => arg.name
      case None      => id.instanceName
    }
  }

  object Arg {
    def earlyLocalName(id: HasId): String = earlyLocalName(id, true)

    def earlyLocalName(id: HasId, includeRoot: Boolean): String = id.getOptionRef match {
      case Some(Index(Node(imm), Node(value))) =>
        s"${earlyLocalName(imm, includeRoot)}[${earlyLocalName(imm, includeRoot)}]"
      case Some(Index(Node(imm), arg)) => s"${earlyLocalName(imm, includeRoot)}[${arg.localName}]"
      case Some(Slot(Node(imm), name)) => s"${earlyLocalName(imm, includeRoot)}.$name"
      case Some(OpaqueSlot(Node(imm))) => s"${earlyLocalName(imm, includeRoot)}"
      case Some(arg) if includeRoot    => arg.name
      case None if includeRoot =>
        id match {
          case data: Data          => data._computeName(Some("?")).get
          case obj:  DynamicObject => obj._computeName(Some("?")).get
          case _ => "?"
        }
      case _ => "_" // Used when includeRoot == false
    }
  }

  abstract class LitArg(val num: BigInt, widthArg: Width) extends Arg {
    def forcedWidth = widthArg.known
    def width: Width = if (forcedWidth) widthArg else Width(minWidth)
    override def contextualName(ctx: Component): String = name
    // Ensure the node representing this LitArg has a ref to it and a literal binding.
    def bindLitArg[T <: Element](elem: T): T = {
      elem.bind(ElementLitBinding(this))
      elem.setRef(this)
      elem
    }

    /** Provides a mechanism that LitArgs can have their width adjusted
      * to match other members of a VecLiteral
      *
      * @param newWidth the new width for this
      * @return
      */
    def cloneWithWidth(newWidth: Width): this.type

    protected def minWidth: Int
    if (forcedWidth) {
      require(
        widthArg.get >= minWidth,
        s"The literal value ${num} was elaborated with a specified width of ${widthArg.get} bits, but at least ${minWidth} bits are required."
      )
    }
  }

  case class ILit(n: BigInt) extends Arg {
    def name: String = n.toString
  }

  case class ULit(n: BigInt, w: Width) extends LitArg(n, w) {
    def name:     String = "UInt" + width + "(0h0" + num.toString(16) + ")"
    def minWidth: Int = (if (w.known) 0 else 1).max(n.bitLength)

    def cloneWithWidth(newWidth: Width): this.type = {
      ULit(n, newWidth).asInstanceOf[this.type]
    }

    require(n >= 0, s"UInt literal ${n} is negative")
  }

  case class SLit(n: BigInt, w: Width) extends LitArg(n, w) {
    def name: String = {
      val unsigned = if (n < 0) (BigInt(1) << width.get) + n else n
      s"asSInt(${ULit(unsigned, width).name})"
    }
    def minWidth: Int = (if (w.known) 0 else 1) + n.bitLength

    def cloneWithWidth(newWidth: Width): this.type = {
      SLit(n, newWidth).asInstanceOf[this.type]
    }
  }

  /** Literal property value.
    *
    * These are not LitArgs, because not all property literals are integers.
    */
  case class PropertyLit[T, U](
    propertyType: PropertyTypeclass[_] { type Underlying = U; type Type = T },
    lit:          U)
      extends Arg {
    def name:     String = s"PropertyLit($lit)"
    def minWidth: Int = 0
    def cloneWithWidth(newWidth: Width): this.type = PropertyLit(propertyType, lit).asInstanceOf[this.type]

    /** Expose a bindLitArg API for PropertyLit, similar to LitArg.
      */
    def bindLitArg(elem: Property[T]): Property[T] = {
      elem.bind(PropertyValueBinding)
      elem.setRef(this)
      elem
    }
  }

  /** Property expressions.
    *
    * Property expressions are conceptually similar to Nodes, but only exist as a tree of Args in-memory.
    */
  case class PropExpr(sourceInfo: SourceInfo, tpe: firrtlir.PropertyType, op: firrtlir.PropPrimOp, args: List[Arg])
      extends Arg {
    // PropExpr is different from other Args, because this is only used as an internal data structure, and we never name
    // the Arg or use the name in textual FIRRTL. This is always expected to be the exp of a PropAssign, and it would be
    // an internal error to request the name.
    def name: String = throwException("Internal Error! PropExpr has no name")
  }

  case class Ref(name: String) extends Arg

  /** Arg for ports of Modules
    * @param mod the module this port belongs to
    * @param name the name of the port
    */
  case class ModuleIO(mod: BaseModule, name: String) extends Arg {
    override def contextualName(ctx: Component): String =
      if (mod eq ctx.id) name else s"${mod.getRef.name}.$name"
  }

  /** Ports of cloned modules (CloneModuleAsRecord)
    * @param mod The original module for which these ports are a clone
    * @param name the name of the module instance
    */
  case class ModuleCloneIO(mod: BaseModule, name: String) extends Arg {
    override def localName = ""
    override def contextualName(ctx: Component): String =
      // NOTE: mod eq ctx.id only occurs in Target and Named-related APIs
      if (mod eq ctx.id) localName else name
  }
  case class Slot(imm: Arg, name: String) extends Arg {
    override def contextualName(ctx: Component): String = {
      val immName = imm.contextualName(ctx)
      if (immName.isEmpty) name else s"$immName.$name"
    }
    override def localName: String = {
      val immName = imm.localName
      if (immName.isEmpty) name else s"$immName.$name"
    }
  }

  case class OpaqueSlot(imm: Node) extends Arg {
    override def contextualName(ctx: Component): String = imm.contextualName(ctx)
    override def name: String = imm.name
  }

  case class Index(imm: Arg, value: Arg) extends Arg {
    def name: String = s"[$value]"
    override def contextualName(ctx: Component): String = s"${imm.contextualName(ctx)}[${value.contextualName(ctx)}]"
    override def localName: String = s"${imm.localName}[${value.localName}]"
  }

  sealed trait ProbeDetails { this: Arg =>
    val probe: Arg
    override def name: String = s"$probe"
  }
  case class ProbeExpr(probe: Arg) extends Arg with ProbeDetails
  case class RWProbeExpr(probe: Arg) extends Arg with ProbeDetails
  case class ProbeRead(probe: Arg) extends Arg with ProbeDetails

  sealed abstract class MemPortDirection(name: String) {
    override def toString: String = name
  }
  object MemPortDirection {
    object READ extends MemPortDirection("read")
    object WRITE extends MemPortDirection("write")
    object RDWR extends MemPortDirection("rdwr")
    object INFER extends MemPortDirection("infer")
  }

  abstract class Command {
    def sourceInfo: SourceInfo
  }

  abstract class Definition extends Command {
    def id: HasId
    def name: String = id.getRef.name
  }

  case class DefPrim[T <: Data](sourceInfo: SourceInfo, id: T, op: PrimOp, args: Arg*) extends Definition

  case class DefInvalid(sourceInfo: SourceInfo, arg: Arg) extends Command

  case class DefWire(sourceInfo: SourceInfo, id: Data) extends Definition

  case class DefReg(sourceInfo: SourceInfo, id: Data, clock: Arg) extends Definition

  case class DefRegInit(sourceInfo: SourceInfo, id: Data, clock: Arg, reset: Arg, init: Arg) extends Definition

  case class DefMemory(sourceInfo: SourceInfo, id: HasId, t: Data, size: BigInt) extends Definition

  case class DefSeqMemory(
    sourceInfo:     SourceInfo,
    id:             HasId,
    t:              Data,
    size:           BigInt,
    readUnderWrite: fir.ReadUnderWrite.Value)
      extends Definition

  case class FirrtlMemory(
    sourceInfo:         SourceInfo,
    id:                 HasId,
    t:                  Data,
    size:               BigInt,
    readPortNames:      Seq[String],
    writePortNames:     Seq[String],
    readwritePortNames: Seq[String])
      extends Definition

  case class DefMemPort[T <: Data](
    sourceInfo: SourceInfo,
    id:         T,
    source:     Node,
    dir:        MemPortDirection,
    index:      Arg,
    clock:      Arg)
      extends Definition

  case class DefInstance(sourceInfo: SourceInfo, id: BaseModule, ports: Seq[Port]) extends Definition
  case class DefInstanceChoice(
    sourceInfo: SourceInfo,
    id:         HasId,
    default:    BaseModule,
    option:     String,
    choices:    Seq[(String, BaseModule)])
      extends Definition
  case class DefObject(sourceInfo: SourceInfo, id: HasId, className: String) extends Definition
  case class WhenBegin(sourceInfo: SourceInfo, pred: Arg) extends Command
  case class WhenEnd(sourceInfo: SourceInfo, firrtlDepth: Int, hasAlt: Boolean = false) extends Command
  case class AltBegin(sourceInfo: SourceInfo) extends Command
  case class OtherwiseEnd(sourceInfo: SourceInfo, firrtlDepth: Int) extends Command
  case class Connect(sourceInfo: SourceInfo, loc: Arg, exp: Arg) extends Command
  case class PropAssign(sourceInfo: SourceInfo, loc: Node, exp: Arg) extends Command
  case class Attach(sourceInfo: SourceInfo, locs: Seq[Node]) extends Command
  case class ConnectInit(sourceInfo: SourceInfo, loc: Node, exp: Arg) extends Command
  case class Stop(id: stop.Stop, sourceInfo: SourceInfo, clock: Arg, ret: Int) extends Definition

  object LayerConvention {
    sealed trait Type
    case object Bind extends Type
  }

  case class Layer(
    sourceInfo: SourceInfo,
    name:       String,
    convention: LayerConvention.Type,
    children:   Seq[Layer])

  case class LayerBlockBegin(sourceInfo: SourceInfo, layer: chisel3.layer.Layer) extends Command
  case class LayerBlockEnd(sourceInfo: SourceInfo) extends Command

  case class DefOption(
    sourceInfo: SourceInfo,
    name:       String,
    cases:      Seq[DefOptionCase])
  case class DefOptionCase(sourceInfo: SourceInfo, name: String)

  case class Port(id: Data, dir: SpecifiedDirection, sourceInfo: SourceInfo)

  case class Printf(id: printf.Printf, sourceInfo: SourceInfo, clock: Arg, pable: Printable) extends Definition

  case class ProbeDefine(sourceInfo: SourceInfo, sink: Arg, probe: Arg) extends Command
  case class ProbeForceInitial(sourceInfo: SourceInfo, probe: Arg, value: Arg) extends Command
  case class ProbeReleaseInitial(sourceInfo: SourceInfo, probe: Arg) extends Command
  case class ProbeForce(sourceInfo: SourceInfo, clock: Arg, cond: Arg, probe: Arg, value: Arg) extends Command
  case class ProbeRelease(sourceInfo: SourceInfo, clock: Arg, cond: Arg, probe: Arg) extends Command

  object Formal extends Enumeration {
    val Assert = Value("assert")
    val Assume = Value("assume")
    val Cover = Value("cover")
  }

  case class Verification[T <: VerificationStatement](
    id:         T,
    op:         Formal.Value,
    sourceInfo: SourceInfo,
    clock:      Arg,
    predicate:  Arg,
    pable:      Printable)
      extends Definition

  abstract class Component extends Arg {
    def id:    BaseModule
    def name:  String
    def ports: Seq[Port]
    val secretPorts: mutable.ArrayBuffer[Port] = id.secretPorts
  }

  case class DefTypeAlias(sourceInfo: SourceInfo, underlying: fir.Type, val name: String)

  case class DefModule(
    id:       RawModule,
    name:     String,
    isPublic: Boolean,
    layers:   Seq[chisel3.layer.Layer],
    ports:    Seq[Port],
    commands: Seq[Command])
      extends Component {
    val secretCommands: mutable.ArrayBuffer[Command] = mutable.ArrayBuffer[Command]()
  }

  case class DefBlackBox(
    id:     BaseBlackBox,
    name:   String,
    ports:  Seq[Port],
    topDir: SpecifiedDirection,
    params: Map[String, Param])
      extends Component

  case class DefIntrinsicModule(
    id:     BaseIntrinsicModule,
    name:   String,
    ports:  Seq[Port],
    topDir: SpecifiedDirection,
    params: Map[String, Param])
      extends Component

  case class DefIntrinsicExpr[T <: Data](
    sourceInfo: SourceInfo,
    intrinsic:  String,
    id:         T,
    args:       Seq[Arg],
    params:     Map[String, Param])
      extends Definition

  case class DefIntrinsic(sourceInfo: SourceInfo, intrinsic: String, args: Seq[Arg], params: Map[String, Param])
      extends Command

  case class DefClass(id: Class, name: String, ports: Seq[Port], commands: Seq[Command]) extends Component

  case class Circuit(
    name:           String,
    components:     Seq[Component],
    annotations:    Seq[ChiselAnnotation],
    renames:        RenameMap,
    newAnnotations: Seq[ChiselMultiAnnotation],
    typeAliases:    Seq[DefTypeAlias],
    layers:         Seq[Layer],
    options:        Seq[DefOption]) {

    def this(
      name:        String,
      components:  Seq[Component],
      annotations: Seq[ChiselAnnotation],
      renames:     RenameMap,
      typeAliases: Seq[DefTypeAlias],
      layers:      Seq[Layer],
      options:     Seq[DefOption]
    ) =
      this(name, components, annotations, renames, Seq.empty, typeAliases, layers, options)

    def firrtlAnnotations: Iterable[Annotation] =
      annotations.flatMap(_.toFirrtl.update(renames)) ++ newAnnotations.flatMap(
        _.toFirrtl.flatMap(_.update(renames))
      )

    def copy(
      name:        String = name,
      components:  Seq[Component] = components,
      annotations: Seq[ChiselAnnotation] = annotations,
      renames:     RenameMap = renames,
      typeAliases: Seq[DefTypeAlias] = typeAliases,
      layers:      Seq[Layer] = layers,
      options:     Seq[DefOption] = options
    ) = Circuit(name, components, annotations, renames, newAnnotations, typeAliases, layers, options)

  }

  object Circuit
      extends scala.runtime.AbstractFunction7[String, Seq[Component], Seq[ChiselAnnotation], RenameMap, Seq[
        DefTypeAlias
      ], Seq[Layer], Seq[DefOption], Circuit] {
    def unapply(c: Circuit): Option[(String, Seq[Component], Seq[ChiselAnnotation], RenameMap, Seq[DefTypeAlias])] = {
      Some((c.name, c.components, c.annotations, c.renames, c.typeAliases))
    }

    def apply(
      name:        String,
      components:  Seq[Component],
      annotations: Seq[ChiselAnnotation],
      renames:     RenameMap,
      typeAliases: Seq[DefTypeAlias] = Seq.empty,
      layers:      Seq[Layer] = Seq.empty,
      options:     Seq[DefOption] = Seq.empty
    ): Circuit =
      new Circuit(name, components, annotations, renames, typeAliases, layers, options)
  }
}
