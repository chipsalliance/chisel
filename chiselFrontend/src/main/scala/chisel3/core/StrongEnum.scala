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
  /** An annotation for strong enum instances that are ''not'' inside of Vecs
    *
    * @param target the enum instance being annotated
    * @param typeName the name of the enum's type (e.g. ''"mypackage.MyEnum"'')
    */
  case class EnumComponentAnnotation(target: Named, enumTypeName: String) extends SingleTargetAnnotation[Named] {
    def duplicate(n: Named): EnumComponentAnnotation = this.copy(target = n)
  }

  case class EnumComponentChiselAnnotation(target: InstanceId, enumTypeName: String) extends ChiselAnnotation {
    def toFirrtl: EnumComponentAnnotation = EnumComponentAnnotation(target.toNamed, enumTypeName)
  }

  /** An annotation for Vecs of strong enums.
    *
    * The ''fields'' parameter deserves special attention, since it may be difficult to understand. Suppose you create a the following Vec:

    *               {{{
    *               VecInit(new Bundle {
    *                 val e = MyEnum()
    *                 val b = new Bundle {
    *                   val inner_e = MyEnum()
    *                 }
    *                 val v = Vec(3, MyEnum())
    *               }
    *               }}}
    *
    *               Then, the ''fields'' parameter will be: ''Seq(Seq("e"), Seq("b", "inner_e"), Seq("v"))''. Note that for any Vec that doesn't contain Bundles, this field will simply be an empty Seq.
    *
    * @param target the Vec being annotated
    * @param typeName the name of the enum's type (e.g. ''"mypackage.MyEnum"'')
    * @param fields a list of all chains of elements leading from the Vec instance to its inner enum fields.
    *
    */
  case class EnumVecAnnotation(target: Named, typeName: String, fields: Seq[Seq[String]]) extends SingleTargetAnnotation[Named] {
    def duplicate(n: Named) = this.copy(target = n)
  }

  case class EnumVecChiselAnnotation(target: InstanceId, typeName: String, fields: Seq[Seq[String]]) extends ChiselAnnotation {
    override def toFirrtl = EnumVecAnnotation(target.toNamed, typeName, fields)
  }

  /** An annotation for enum types (rather than enum ''instances'').
    *
    * @param typeName the name of the enum's type (e.g. ''"mypackage.MyEnum"'')
    * @param definition a map describing which integer values correspond to which enum names
    */
  case class EnumDefAnnotation(typeName: String, definition: Map[String, BigInt]) extends NoTargetAnnotation

  case class EnumDefChiselAnnotation(typeName: String, definition: Map[String, BigInt]) extends ChiselAnnotation {
    override def toFirrtl: Annotation = EnumDefAnnotation(typeName, definition)
  }
}
import EnumAnnotations._


abstract class EnumType(private val factory: EnumFactory, selfAnnotating: Boolean = true) extends Element {
  override def toString: String = {
    val bindingString = litOption match {
      case Some(value) => factory.nameOfValue(value) match {
        case Some(name) => s"($value=$name)"
        case None => s"($value=(invalid))"
      }
      case _ => bindingToString
    }
    // Use getSimpleName instead of enumTypeName because for debugging purposes the fully qualified name isn't
    // necessary (compared to for the Enum annotation), and it's more consistent with Bundle printing.
    s"${factory.getClass.getSimpleName.init}$bindingString"
  }

  override def cloneType: this.type = factory().asInstanceOf[this.type]

  private[core] def compop(sourceInfo: SourceInfo, op: PrimOp, other: EnumType): Bool = {
    requireIsHardware(this, "bits operated on")
    requireIsHardware(other, "bits operated on")

    if(!this.typeEquivalent(other)) {
      throwException(s"Enum types are not equivalent: ${this.enumTypeName}, ${other.enumTypeName}")
    }

    pushOp(DefPrim(sourceInfo, Bool(), op, this.ref, other.ref))
  }

  private[core] override def typeEquivalent(that: Data): Boolean = {
    this.getClass == that.getClass &&
    this.factory == that.asInstanceOf[EnumType].factory
  }

  private[core] override def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
    this := factory.apply(that.asUInt)
  }

  final def === (that: EnumType): Bool = macro SourceInfoTransform.thatArg
  final def =/= (that: EnumType): Bool = macro SourceInfoTransform.thatArg
  final def < (that: EnumType): Bool = macro SourceInfoTransform.thatArg
  final def <= (that: EnumType): Bool = macro SourceInfoTransform.thatArg
  final def > (that: EnumType): Bool = macro SourceInfoTransform.thatArg
  final def >= (that: EnumType): Bool = macro SourceInfoTransform.thatArg

  // scalastyle:off line.size.limit method.name
  def do_=== (that: EnumType)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, EqualOp, that)
  def do_=/= (that: EnumType)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, NotEqualOp, that)
  def do_< (that: EnumType)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessOp, that)
  def do_> (that: EnumType)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterOp, that)
  def do_<= (that: EnumType)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessEqOp, that)
  def do_>= (that: EnumType)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterEqOp, that)
  // scalastyle:on line.size.limit method.name

  override def do_asUInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    pushOp(DefPrim(sourceInfo, UInt(width), AsUIntOp, ref))

  protected[chisel3] override def width: Width = factory.width

  def isValid(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = {
    if (litOption.isDefined) {
      true.B
    } else {
      factory.all.map(this === _).reduce(_ || _)
    }
  }

  def next(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): this.type = {
    if (litOption.isDefined) {
      val index = factory.all.indexOf(this)

      if (index < factory.all.length-1) {
        factory.all(index + 1).asInstanceOf[this.type]
      } else {
        factory.all.head.asInstanceOf[this.type]
      }
    } else {
      val enums_with_nexts = factory.all zip (factory.all.tail :+ factory.all.head)
      val next_enum = SeqUtils.priorityMux(enums_with_nexts.map { case (e,n) => (this === e, n) } )
      next_enum.asInstanceOf[this.type]
    }
  }

  private[core] def bindToLiteral(num: BigInt, w: Width): Unit = {
    val lit = ULit(num, w)
    lit.bindLitArg(this)
  }

  override private[chisel3] def bind(target: Binding, parentDirection: SpecifiedDirection = SpecifiedDirection.Unspecified): Unit = {
    super.bind(target, parentDirection)

    // Make sure we only annotate hardware and not literals
    if (selfAnnotating && isSynthesizable && topBindingOpt.get.isInstanceOf[ConstrainedBinding]) {
      annotateEnum()
    }
  }

  // This function conducts a depth-wise search to find all enum-type fields within a vector or bundle (or vector of bundles)
  private def enumFields(d: Aggregate): Seq[Seq[String]] = d match {
    case v: Vec[_] => v.sample_element match {
      case b: Bundle => enumFields (b)
      case _ => Seq ()
    }
    case b: Bundle =>
      b.elements.collect {
        case (name, e: EnumType) if this.typeEquivalent(e) => Seq(Seq(name))
        case (name, v: Vec[_]) if this.typeEquivalent(v.sample_element) => Seq(Seq(name))
        case (name, b2: Bundle) => enumFields(b2).map(name +: _)
      }.flatten.toSeq
  }

  private def outerMostVec(d: Data = this): Option[Vec[_]] = {
    val currentVecOpt = d match {
      case v: Vec[_] => Some(v)
      case _ => None
    }

    d.binding match {
      case Some(ChildBinding(parent)) => outerMostVec(parent) match {
        case outer @ Some(_) => outer
        case None => currentVecOpt
      }
      case _ => currentVecOpt
    }
  }

  private def annotateEnum(): Unit = {
    val anno = outerMostVec() match {
      case Some(v) => EnumVecChiselAnnotation(v, enumTypeName, enumFields(v))
      case None => EnumComponentChiselAnnotation(this, enumTypeName)
    }

    if (!Builder.annotations.contains(anno)) {
      annotate(anno)
    }

    if (!Builder.annotations.contains(factory.globalAnnotation)) {
      annotate(factory.globalAnnotation)
    }
  }

  protected def enumTypeName: String = factory.enumTypeName

  def toPrintable: Printable = FullName(this) // TODO: Find a better pretty printer
}


abstract class EnumFactory {
  class Type extends EnumType(this)
  object Type {
    def apply(): Type = EnumFactory.this.apply()
  }

  private var id: BigInt = 0
  private[core] var width: Width = 0.W

  private case class EnumRecord(inst: Type, name: String)
  private val enum_records = mutable.ArrayBuffer.empty[EnumRecord]

  private def enumNames = enum_records.map(_.name).toSeq
  private def enumValues = enum_records.map(_.inst.litValue()).toSeq
  private def enumInstances = enum_records.map(_.inst).toSeq

  private[core] val enumTypeName = getClass.getName.init

  private[core] def globalAnnotation: EnumDefChiselAnnotation =
    EnumDefChiselAnnotation(enumTypeName, (enumNames, enumValues).zipped.toMap)

  def getWidth: Int = width.get

  def all: Seq[Type] = enumInstances

  private[chisel3] def nameOfValue(id: BigInt): Option[String] = {
    enum_records.find(_.inst.litValue() == id).map(_.name)
  }

  protected def Value: Type = macro EnumMacros.ValImpl // scalastyle:off method.name
  protected def Value(id: UInt): Type = macro EnumMacros.ValCustomImpl // scalastyle:off method.name

  protected def do_Value(names: Seq[String]): Type = {
    val result = new Type

    // We have to use UnknownWidth here, because we don't actually know what the final width will be
    result.bindToLiteral(id, UnknownWidth())

    val result_name = names.find(!enumNames.contains(_)).get
    enum_records.append(EnumRecord(result, result_name))

    width = (1 max id.bitLength).W
    id += 1

    result
  }

  protected def do_Value(names: Seq[String], id: UInt): Type = {
    // TODO: These throw ExceptionInInitializerError which can be confusing to the user. Get rid of the error, and just
    // throw an exception
    if (id.litOption.isEmpty) {
      throwException(s"$enumTypeName defined with a non-literal type")
    }
    if (id.litValue() < this.id) {
      throwException(s"Enums must be strictly increasing: $enumTypeName")
    }

    this.id = id.litValue()
    do_Value(names)
  }

  def apply(): Type = new Type

  def apply(n: UInt)(implicit sourceInfo: SourceInfo, connectionCompileOptions: CompileOptions): Type = {
    // scalastyle:off line.size.limit
    if (n.litOption.isDefined) {
      enumInstances.find(_.litValue == n.litValue) match {
        case Some(result) => result
        case None => throwException(s"${n.litValue} is not a valid value for $enumTypeName")
      }
    } else if (!n.isWidthKnown) {
      throwException(s"Non-literal UInts being cast to $enumTypeName must have a defined width")
    } else if (n.getWidth > this.getWidth) {
      throwException(s"The UInt being cast to $enumTypeName is wider than $enumTypeName's width ($getWidth)")
    } else {
      Builder.warning(s"Casting non-literal UInt to $enumTypeName. You can check that its value is legal by calling isValid")

      val glue = Wire(new UnsafeEnum(width))
      glue := n
      val result = Wire(new Type)
      result := glue
      result
    }
  }
  // scalastyle:on line.size.limit
}


private[core] object EnumMacros {
  def ValImpl(c: Context) : c.Tree = { // scalastyle:off method.name
    import c.universe._
    val names = getNames(c)
    q"""this.do_Value(Seq(..$names))"""
  }

  def ValCustomImpl(c: Context)(id: c.Expr[UInt]): c.universe.Tree = { // scalastyle:off method.name
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
        if rhs.pos == c.macroApplication.pos => name.decodedName.toString
    }

    if (names.isEmpty) {
      c.abort(c.enclosingPosition, "Value cannot be called without assigning to an enum")
    }

    names
  }
}


// This is an enum type that can be connected directly to UInts. It is used as a "glue" to cast non-literal UInts
// to enums.
private[chisel3] class UnsafeEnum(override val width: Width) extends EnumType(UnsafeEnum, selfAnnotating = false) {
  override def cloneType: this.type = new UnsafeEnum(width).asInstanceOf[this.type]
}
private object UnsafeEnum extends EnumFactory
