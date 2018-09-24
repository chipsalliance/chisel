// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros
import scala.reflect.macros.blackbox.Context
import scala.collection.mutable

import chisel3.internal.Builder.pushOp
import chisel3.internal.firrtl.PrimOp._
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo._
import chisel3.internal.{Builder, InstanceId, throwException}
import firrtl.annotations._


object EnumAnnotations {
  case class EnumComponentAnnotation(target: Named, enumTypeName: String) extends SingleTargetAnnotation[Named] {
    def duplicate(n: Named) = this.copy(target = n)
  }

  case class EnumComponentChiselAnnotation(target: InstanceId, enumTypeName: String) extends ChiselAnnotation {
    def toFirrtl = EnumComponentAnnotation(target.toNamed, enumTypeName)
  }

  case class EnumDefAnnotation(enumTypeName: String, definition: Map[String, BigInt]) extends NoTargetAnnotation

  case class EnumDefChiselAnnotation(enumTypeName: String, definition: Map[String, BigInt]) extends ChiselAnnotation {
    override def toFirrtl: Annotation = EnumDefAnnotation(enumTypeName, definition)
  }
}
import EnumAnnotations._


abstract class EnumType(private val factory: EnumFactory, selfAnnotating: Boolean = true) extends Element {
  override def cloneType: this.type = factory().asInstanceOf[this.type]

  private[core] override def topBindingOpt: Option[TopBinding] = super.topBindingOpt match {
    // Translate Bundle lit bindings to Element lit bindings
    case Some(BundleLitBinding(litMap)) => litMap.get(this) match {
      case Some(litArg) => Some(ElementLitBinding(litArg))
      case _ => Some(DontCareBinding())
    }
    case topBindingOpt => topBindingOpt
  }

  private[core] def litArgOption: Option[LitArg] = topBindingOpt match {
    case Some(ElementLitBinding(litArg)) => Some(litArg)
    case _ => None
  }

  override def litOption: Option[BigInt] = litArgOption.map(_.num)
  private[core] def litIsForcedWidth: Option[Boolean] = litArgOption.map(_.forcedWidth)

  // provide bits-specific literal handling functionality here
  override private[chisel3] def ref: Arg = topBindingOpt match {
    case Some(ElementLitBinding(litArg)) => litArg
    case Some(BundleLitBinding(litMap)) => litMap.get(this) match {
      case Some(litArg) => litArg
      case _ => throwException(s"internal error: DontCare should be caught before getting ref")
    }
    case _ => super.ref
  }

  private[core] def compop(sourceInfo: SourceInfo, op: PrimOp, other: EnumType): Bool = {
    requireIsHardware(this, "bits operated on")
    requireIsHardware(other, "bits operated on")

    if(!this.typeEquivalent(other))
      throwException(s"Enum types are not equivalent: ${this.enumTypeName}, ${other.enumTypeName}")

    pushOp(DefPrim(sourceInfo, Bool(), op, this.ref, other.ref))
  }

  private[core] override def typeEquivalent(that: Data): Boolean = {
    this.getClass == that.getClass &&
    this.factory == that.asInstanceOf[EnumType].factory
  }

  // This isn't actually used anywhere (and it would throw an exception anyway). But it has to be defined since we
  // inherit it from Data.
  private[core] override def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
    this := that.asUInt
  }

  final def === (that: EnumType): Bool = macro SourceInfoTransform.thatArg
  final def =/= (that: EnumType): Bool = macro SourceInfoTransform.thatArg
  final def < (that: EnumType): Bool = macro SourceInfoTransform.thatArg
  final def <= (that: EnumType): Bool = macro SourceInfoTransform.thatArg
  final def > (that: EnumType): Bool = macro SourceInfoTransform.thatArg
  final def >= (that: EnumType): Bool = macro SourceInfoTransform.thatArg

  def do_=== (that: EnumType)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, EqualOp, that)
  def do_=/= (that: EnumType)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, NotEqualOp, that)
  def do_< (that: EnumType)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessOp, that)
  def do_> (that: EnumType)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterOp, that)
  def do_<= (that: EnumType)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessEqOp, that)
  def do_>= (that: EnumType)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterEqOp, that)

  override def do_asUInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    pushOp(DefPrim(sourceInfo, UInt(width), AsUIntOp, ref))

  protected[chisel3] override def width: Width = factory.width

  def isValid(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = {
    if (litOption.isDefined) {
      true.B
    } else {
      def muxBuilder(enums: List[EnumType]): Bool = enums match {
        case Nil => false.B
        case e :: es => Mux(this === e, true.B, muxBuilder(es))
      }

      muxBuilder(factory.all.toList)
    }
  }

  def next(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): this.type = {
    if (litOption.isDefined) {
      val index = factory.all.indexOf(this)

      if (index < factory.all.length-1)
        factory.all(index+1).asInstanceOf[this.type]
      else
        factory.all.head.asInstanceOf[this.type]
    } else {
      def muxBuilder(enums: List[EnumType], first_enum: EnumType): EnumType = enums match {
        case e :: Nil => first_enum
        case e :: e_next :: es => Mux(this === e, e_next, muxBuilder(e_next :: es, first_enum))
      }

      muxBuilder(factory.all.toList, factory.all.head).asInstanceOf[this.type]
    }
  }

  private[core] def bindToLiteral(num: BigInt, w: Width): Unit = {
    val lit = ULit(num, w)
    lit.bindLitArg(this)
  }

  override def bind(target: Binding, parentDirection: SpecifiedDirection): Unit = {
    super.bind(target, parentDirection)

    // If we try to annotate something that is bound to a literal, we get a FIRRTL annotation exception.
    // To workaround that, we only annotate enums that are not bound to literals.
    if (selfAnnotating && !litOption.isDefined) {
      annotateEnum()
    }
  }

  private def annotateEnum(): Unit = {
    annotate(EnumComponentChiselAnnotation(this, enumTypeName))

    if (!Builder.annotations.contains(factory.globalAnnotation)) {
      annotate(factory.globalAnnotation)
    }
  }

  protected def enumTypeName: String = factory.enumTypeName

  def toPrintable: Printable = FullName(this) // TODO: Find a better pretty printer
}


abstract class EnumFactory {
  class E extends EnumType(this)

  var id: BigInt = 0
  var width: Width = 0.W

  val enum_names = mutable.ArrayBuffer.empty[String]
  val enum_values = mutable.ArrayBuffer.empty[BigInt]
  val enum_instances = mutable.ArrayBuffer.empty[E]

  private[core] def globalAnnotation: EnumDefChiselAnnotation =
    EnumDefChiselAnnotation(enumTypeName, (enum_names, enum_values).zipped.toMap)

  private[core] val enumTypeName = getClass.getName.init

  def getWidth: Int = width.get

  def all: Seq[E] = enum_instances.toSeq

  def Value: E = macro EnumMacros.ValImpl
  def Value(id: UInt): E = macro EnumMacros.ValCustomImpl

  def do_Value(names: Seq[String]): E = {
    val result = new E
    enum_names ++= names.filter(!enum_names.contains(_))
    enum_instances.append(result)
    enum_values.append(id)

    // We have to use UnknownWidth here, because we don't actually know what the final width will be
    result.bindToLiteral(id, UnknownWidth())

    width = (1 max id.bitLength).W
    id += 1

    result
  }

  def do_Value(names: Seq[String], id: UInt): E = {
    // TODO: These throw ExceptionInInitializerError which can be confusing to the user. Get rid of the error, and just
    // throw an exception
    if (!id.litOption.isDefined)
      throwException(s"$enumTypeName defined with a non-literal type")
    if (id.litValue() < this.id)
      throwException(s"Enums must be strictly increasing: $enumTypeName")

    this.id = id.litValue()
    do_Value(names)
  }

  def apply(): E = new E

  def apply(n: UInt)(implicit sourceInfo: SourceInfo, connectionCompileOptions: CompileOptions): E = {
    if (n.litOption.isEmpty) {
      throwException(s"Illegal cast from non-literal UInt to $enumTypeName. Use fromBits instead")
    }

    val result = enum_instances.find(_.litValue == n.litValue)

    if (result.isEmpty) {
      throwException(s"${n.litValue}.U is not a valid value for $enumTypeName")
    } else {
      result.get
    }
  }

  def fromBits(n: UInt)(implicit sourceInfo: SourceInfo, connectionCompileOptions: CompileOptions): E = {
    if (n.litOption.isDefined) {
      apply(n)
    } else if (!n.isWidthKnown) {
      throwException(s"Non-literal UInts being cast to $enumTypeName must have a defined width")
    } else if (n.getWidth > this.getWidth) {
      throwException(s"The UInt being cast to $enumTypeName is wider than $enumTypeName's width ($getWidth)")
    } else {
      Builder.warning(s"A non-literal UInt is being cast to $enumTypeName. You can check that the value is legal by calling isValid")

      val glue = Wire(new UnsafeEnum(width))
      glue := n
      val result = Wire(new E)
      result := glue
      result
    }
  }
}


object EnumMacros {
  def ValImpl(c: Context) : c.Tree = {
    import c.universe._
    val names = getNames(c)
    q"""this.do_Value(Seq(..$names))"""
  }

  def ValCustomImpl(c: Context)(id: c.Expr[UInt]) = {
    import c.universe._
    val names = getNames(c)
    q"""this.do_Value(Seq(..$names), $id)"""
  }

  // Much thanks to Travis Brown for this solution:
  // stackoverflow.com/questions/18450203/retrieve-the-name-of-the-value-a-scala-macro-invocation-will-be-assigned-to
  def getNames(c: Context): Seq[String] = {
    import c.universe._

    val names = c.enclosingClass.collect {
      case ValDef(_, name, _, rhs)
        if rhs.pos == c.macroApplication.pos => name.decoded
    }

    if (names.isEmpty)
      c.abort(c.enclosingPosition, "Value cannot be called without assigning to an enum")

    names
  }
}


// This is an enum type that can be connected directly to UInts. It is used as a "glue" to cast non-literal UInts
// to enums.
private[chisel3] class UnsafeEnum(override val width: Width) extends EnumType(UnsafeEnum, selfAnnotating = false) {
  override def cloneType: this.type = new UnsafeEnum(width).asInstanceOf[this.type]
}
private object UnsafeEnum extends EnumFactory
