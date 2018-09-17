package chisel3.core

import scala.language.experimental.macros
import scala.reflect.ClassTag
import scala.reflect.runtime.currentMirror
import scala.reflect.runtime.universe.{MethodSymbol, runtimeMirror}
import scala.collection.mutable

import chisel3.internal.Builder.pushOp
import chisel3.internal.firrtl.PrimOp._
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo._
import chisel3.internal.{Builder, InstanceId, throwException}
import firrtl.annotations._

object EnumExceptions {
  case class EnumTypeMismatch(message: String) extends Exception(message)
  case class EnumHasNoCompanionObject(message: String) extends Exception(message)
  case class NonLiteralEnum(message: String) extends Exception(message)
  case class NonIncreasingEnum(message: String) extends Exception(message)
  case class IllegalDefinitionOfEnum(message: String) extends Exception(message)
  case class IllegalCastToEnum(message: String) extends Exception(message)
  case class NoEmptyConstructor(message: String) extends Exception(message)
}

object EnumAnnotations {
  case class EnumComponentAnnotation(target: Named, enumTypeName: String) extends SingleTargetAnnotation[Named] {
    def duplicate(n: Named) = this.copy(target = n)
  }

  case class EnumComponentChiselAnnotation(target: InstanceId, enumTypeName: String) extends ChiselAnnotation {
    def toFirrtl = EnumComponentAnnotation(target.toNamed, enumTypeName)
  }

  case class EnumDefAnnotation(enumTypeName: String, definition: Map[String, UInt]) extends NoTargetAnnotation

  case class EnumDefChiselAnnotation(enumTypeName: String, definition: Map[String, UInt]) extends ChiselAnnotation {
    override def toFirrtl: Annotation = EnumDefAnnotation(enumTypeName, definition)
  }
}

import EnumExceptions._
import EnumAnnotations._

abstract class EnumType(selfAnnotating: Boolean = true) extends Element {
  def cloneType: this.type = getClass.getConstructor().newInstance().asInstanceOf[this.type]

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

    checkTypeEquivalency(other)

    pushOp(DefPrim(sourceInfo, Bool(), op, this.ref, other.ref))
  }

  private[core] override def typeEquivalent(that: Data): Boolean = this.getClass == that.getClass

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

  val companionModule = currentMirror.reflect(this).symbol.companion.asModule
  val companionObject =
    try {
      currentMirror.reflectModule(companionModule).instance.asInstanceOf[StrongEnum[this.type]]
    } catch {
      case ex: java.lang.ClassNotFoundException =>
        throw EnumHasNoCompanionObject(s"$enumTypeName's companion object was not found")
    }

  private[chisel3] override def width: Width = companionObject.width

  private[core] def bindToLiteral(bits: UInt): Unit = {
    val litNum = bits.litOption.get
    val lit = ULit(litNum, width) // We must make sure to use the enum's width, rather than the UInt's width
    lit.bindLitArg(this)
  }

  override def bind(target: Binding, parentDirection: SpecifiedDirection): Unit = {
    super.bind(target, parentDirection)

    // If we try to annotate something that is bound to a literal, we get a FIRRTL annotation exception.
    // To workaround that, we only annotate enums that are not bound to literals.
    if (selfAnnotating && !litOption.isDefined)
      annotate(EnumComponentChiselAnnotation(this, enumTypeName))
  }

  private def enumTypeName: String = getClass.getName

  // TODO: See if there is a way to catch this at compile-time
  def checkTypeEquivalency(that: EnumType): Unit =
    if (!typeEquivalent(that))
      throw EnumTypeMismatch(s"${this.getClass.getName} and ${that.getClass.getName} are different enum types")

  def toPrintable: Printable = FullName(this) // TODO: Find a better pretty printer
}

// This is an enum type that can be connected directly to UInts. It is used as a "glue" to cast non-literal UInts
// to enums.
sealed private[chisel3] class UnsafeEnum(override val width: Width) extends EnumType(selfAnnotating = false) {
  override def cloneType: this.type = getClass.getConstructor(classOf[Width]).newInstance(width).asInstanceOf[this.type]
}
private object UnsafeEnum extends StrongEnum[UnsafeEnum] {
  override def checkEmptyConstructorExists(): Unit = {}
}

abstract class StrongEnum[T <: EnumType : ClassTag] {
  private var id: BigInt = 0
  private[core] var width: Width = 0.W

  private val enum_names = getEnumNames
  private val enum_values = mutable.ArrayBuffer.empty[BigInt]
  private val enum_instances = mutable.ArrayBuffer.empty[T]

  private def getEnumNames(implicit ct: ClassTag[T]): Seq[String] = {
    val mirror = runtimeMirror(this.getClass.getClassLoader)
    val reflection  = mirror.reflect(this)

    // We use Java reflection to get all the enum fields, and then we use Scala reflection to sort them in declaration
    // order. TODO: Use only Scala reflection here
    val fields = getClass.getDeclaredFields.filter(_.getType == ct.runtimeClass).map(_.getName)
    val getters = mirror.classSymbol(this.getClass).toType.members.sorted.collect {
      case m: MethodSymbol if m.isGetter => m.name.toString
    }

    getters.filter(fields.contains(_))
  }

  private def bindAllEnums(): Unit =
    (enum_instances, enum_values).zipped.foreach((inst, v) => inst.bindToLiteral(v.U(width)))

  private def createAnnotation(): Unit =
    annotate(EnumDefChiselAnnotation(enumTypeName,
      (enum_names, enum_values.map(_.U(width))).zipped.toMap))

  private def newEnum()(implicit ct: ClassTag[T]): T =
    ct.runtimeClass.newInstance.asInstanceOf[T]

  // TODO: This depends upon undocumented behavior (which, to be fair, is unlikely to change). Use reflection to find
  // the companion class's name in a more robust way.
  private val enumTypeName = getClass.getName.init

  def getWidth: BigInt = width.get

  def all: List[T] = enum_instances.toList

  def Value: T = {
    val result = newEnum()
    enum_instances.append(result)
    enum_values.append(id)

    width = (1 max id.bitLength).W
    id += 1

    // Check whether we've instantiated all the enums
    if (enum_instances.length == enum_names.length && isTopLevelConstructor) {
      bindAllEnums()
      createAnnotation()
    }

    result
  }

  def Value(id: UInt): T = {
    // TODO: These throw ExceptionInInitializerError which can be confusing to the user. Get rid of the error, and just
    // throw an exception
    if (!id.litOption.isDefined)
      throw NonLiteralEnum(s"$enumTypeName defined with a non-literal type in companion object")
    if (id.litValue() <= this.id)
      throw NonIncreasingEnum(s"Enums must be strictly increasing: $enumTypeName")

    this.id = id.litValue()
    Value
  }

  def apply(): T = newEnum()

  def apply(n: UInt)(implicit sourceInfo: SourceInfo, connectionCompileOptions: CompileOptions): T = {
    if (n.litOption.isDefined) {
      if (!enum_values.contains(n.litValue))
        throwException(s"${n.litValue}.U is not a valid value for $enumTypeName")

      val result = newEnum()
      result.bindToLiteral(n)
      result
    } else {
      Builder.warning(s"A non-literal UInt is being cast to $enumTypeName. No automatic bounds checking will be done!")

      val glue = Wire(new UnsafeEnum(width))
      glue := n
      val result = Wire(newEnum())
      result := glue
      result
    }
  }

  // StrongEnum basically has a recursive constructor. It instantiates a copy of itself internally, so that it can
  // make sure that all EnumType's inside of it were instantiated using the "Value" function. However, in order to
  // instantiate its copy, as well as to instantiate new enums, it has to make sure that it has a no-args constructor
  // as it won't know what parameters to add otherwise.

  protected def checkEmptyConstructorExists(): Unit = {
    try {
      implicitly[ClassTag[T]].runtimeClass.getDeclaredConstructor()
      getClass.getDeclaredConstructor()
    } catch {
      case ex: NoSuchMethodException => throw NoEmptyConstructor(s"$enumTypeName does not have a no-args constructor. Did you declare it inside a class?")
    }
  }

  private val isTopLevelConstructor: Boolean = {
    val stack_trace = Thread.currentThread().getStackTrace
    val constructorName = "<init>"

    stack_trace.count(se => se.getClassName.equals(getClass.getName) && se.getMethodName.equals(constructorName)) == 1
  }

  if (isTopLevelConstructor) {
    checkEmptyConstructorExists()

    val constructor = getClass.getDeclaredConstructor()
    constructor.setAccessible(true)
    val childInstance = constructor.newInstance()

    if (childInstance.enum_names.length != childInstance.enum_instances.length)
      throw IllegalDefinitionOfEnum(s"$enumTypeName defined illegally. Did you forget to call Value when defining a new enum?")
  }
}
