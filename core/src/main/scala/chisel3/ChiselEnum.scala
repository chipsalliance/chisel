// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.language.existentials
import scala.collection.mutable
import chisel3.experimental.{requireIsHardware, SourceInfo, UnlocatableSourceInfo}
import chisel3.internal.Builder.pushOp
import chisel3.internal.firrtl.ir.PrimOp._
import chisel3.internal.firrtl.ir._
import chisel3.internal.{containsProbe, throwException, Builder, BuilderContextCache, Warning, WarningID}
import chisel3.internal.binding.{Binding, ChildBinding, ConstrainedBinding}

abstract class EnumType(private[chisel3] val factory: ChiselEnum) extends Element with EnumTypeIntf {

  @deprecated("ChiselEnum annotations have been removed so selfAnnotating no longer does anything.", "Chisel 7.0")
  def this(factory: ChiselEnum, selfAnnotating: Boolean) = this(factory)

  // Use getSimpleName instead of enumTypeName because for debugging purposes
  //   the fully qualified name isn't necessary, and it's more consistent with Bundle printing.
  override def toString: String = {
    litOption match {
      case Some(value) =>
        factory.nameOfValue(value) match {
          case Some(name) => s"${factory.getClass.getSimpleName.init}($value=$name)"
          case None       => stringAccessor(s"${factory.getClass.getSimpleName.init}($value=(invalid))")
        }
      case _ => stringAccessor(s"${factory.getClass.getSimpleName.init}")
    }
  }

  override def cloneType: this.type = factory().asInstanceOf[this.type]

  private[chisel3] def compop(sourceInfo: SourceInfo, op: PrimOp, other: EnumType): Bool = {
    implicit val info: SourceInfo = sourceInfo
    requireIsHardware(this, "bits operated on")
    requireIsHardware(other, "bits operated on")

    if (!this.typeEquivalent(other)) {
      throwException(s"Enum types are not equivalent: ${this.enumTypeName}, ${other.enumTypeName}")
    }

    pushOp(DefPrim(sourceInfo, Bool(), op, this.ref, other.ref))
  }

  override protected def _fromUInt(that: UInt)(implicit sourceInfo: SourceInfo): Data =
    factory.apply(that.asUInt)

  protected def _impl_===(that: EnumType)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, EqualOp, that)
  protected def _impl_=/=(that: EnumType)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, NotEqualOp, that)
  protected def _impl_<(that:   EnumType)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, LessOp, that)
  protected def _impl_>(that:   EnumType)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, GreaterOp, that)
  protected def _impl_<=(that:  EnumType)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, LessEqOp, that)
  protected def _impl_>=(that:  EnumType)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, GreaterEqOp, that)

  // This preserves the _workaround_ for https://github.com/chipsalliance/chisel/issues/4159
  // Note that #4159 is due to _asUIntImpl below not actually padding the UInt
  // This override just ensures that if `that` has a known width, the result actually has that width
  // Put another way, this is preserving a case where #4159 does **not** occur
  // This can be deleted when Builder.useLegacyWidth is removed.
  override def _asTypeOfImpl[T <: Data](that: T)(implicit sourceInfo: SourceInfo): T = {
    that.widthOption match {
      // Note that default case will handle literals just fine
      case Some(w) =>
        val _padded = this.litOption match {
          case Some(value) =>
            value.U(w.W)
          case None =>
            val _wire = Wire(UInt(w.W))
            _wire := this.asUInt
            _wire
        }
        _padded._asTypeOfImpl(that)
      case None => super._asTypeOfImpl(that)
    }
  }

  override private[chisel3] def _asUIntImpl(first: Boolean)(implicit sourceInfo: SourceInfo): UInt = {
    this.litOption match {
      // This fixes an old bug that changes widths and thus silently changes behavior.
      // See https://github.com/chipsalliance/chisel/issues/4159.
      case Some(value) if !Builder.useLegacyWidth =>
        value.U(width)
      case _ =>
        pushOp(DefPrim(sourceInfo, UInt(width), AsUIntOp, ref))
    }
  }

  protected[chisel3] override def width: Width = factory.width

  def isValid(implicit sourceInfo: SourceInfo): Bool = {
    if (litOption.isDefined) {
      true.B
    } else {
      if (factory.isTotal) true.B else factory.all.map(this === _).reduce(_ || _)
    }
  }

  /** Test if this enumeration is equal to any of the values in a given sequence
    *
    * @param s a [[scala.collection.Seq$ Seq]] of enumeration values to look for
    * @return a hardware [[Bool]] that indicates if this value matches any of the given values
    */
  final def isOneOf(s: Seq[EnumType])(implicit sourceInfo: SourceInfo): Bool = {
    VecInit(s.map(this === _)).asUInt.orR
  }

  /** Test if this enumeration is equal to any of the values given as arguments
    *
    * @param u1 the first value to look for
    * @param u2 zero or more additional values to look for
    * @return a hardware [[Bool]] that indicates if this value matches any of the given values
    */
  final def isOneOf(
    u1: EnumType,
    u2: EnumType*
  )(
    implicit sourceInfo: SourceInfo
  ): Bool = isOneOf(u1 +: u2.toSeq)

  /** Creates circuitry that outputs True iff the Enum is equal to one of the values that has `s` in its name
    *
    * @param s the substring to search for in the Enum value's name
    */
  def contains(s: String)(implicit sourceInfo: SourceInfo): Bool =
    isOneOf(factory.allWithNames.filter(m => m._2 contains s).map(m => m._1))

  def next(implicit sourceInfo: SourceInfo): this.type = {
    if (litOption.isDefined) {
      val index = factory.all.indexOf(this)

      if (index < factory.all.length - 1) {
        factory.all(index + 1).asInstanceOf[this.type]
      } else {
        factory.all.head.asInstanceOf[this.type]
      }
    } else {
      val enums_with_nexts = factory.all.zip(factory.all.tail :+ factory.all.head)
      val next_enum = SeqUtils.priorityMux(enums_with_nexts.map { case (e, n) => (this === e, n) })
      next_enum.asInstanceOf[this.type]
    }
  }

  private[chisel3] def bindToLiteral(num: BigInt, w: Width): Unit = {
    val lit = ULit(num, w)
    lit.bindLitArg(this)
  }

  override private[chisel3] def bind(
    target:          Binding,
    parentDirection: SpecifiedDirection = SpecifiedDirection.Unspecified
  ): Unit = {
    super.bind(target, parentDirection)
  }

  // This function conducts a depth-wise search to find all enum-type fields within a vector or bundle (or vector of bundles)
  private def enumFields(d: Data): Seq[Seq[String]] = d match {
    case v: Vec[_] =>
      v.sample_element match {
        case b: Bundle => enumFields(b)
        case _ => Seq()
      }
    case b: Record =>
      b.elements.collect {
        case (name, e: EnumType) if this.typeEquivalent(e)              => Seq(Seq(name))
        case (name, v: Vec[_]) if this.typeEquivalent(v.sample_element) => Seq(Seq(name))
        case (name, b2: Bundle)                                         => enumFields(b2).map(name +: _)
      }.flatten.toSeq
  }

  private def outerMostVec(d: Data = this): Option[Vec[_]] = {
    val currentVecOpt = d match {
      case v: Vec[_] => Some(v)
      case _ => None
    }

    d.binding match {
      case Some(ChildBinding(parent)) =>
        outerMostVec(parent) match {
          case outer @ Some(_) => outer
          case None            => currentVecOpt
        }
      case _ => currentVecOpt
    }
  }

  protected def enumTypeName: String = factory.enumTypeName

  def toPrintable: Printable = {
    implicit val sourceInfo = UnlocatableSourceInfo
    val allNames = factory.allNames.zip(factory.all)
    val nameSize = allNames.map(_._1.length).max
    def leftPad(str: String): String = {
      str.reverse.padTo(nameSize, ' ').reverse
    }
    val allNamesPadded = allNames.map { case (name, value) => leftPad(name) -> value }

    val result = Wire(Vec(nameSize, UInt(8.W))).suggestName(s"_${enumTypeName}Printable")
    result.foreach(_ := '?'.U)

    for ((name, value) <- allNamesPadded) {
      when(this === value) {
        for ((r, c) <- result.zip(name)) {
          r := UInt.Lit(BigInt(c.toChar), Width())
        }
      }
    }
    result.map(Character(_)).foldLeft(p"")(_ + _)
  }
}

abstract class ChiselEnum extends ChiselEnumIntf {
  class Type extends EnumType(this)
  object Type {
    def apply(): Type = ChiselEnum.this.apply()
  }

  private var id:                    BigInt = 0
  private[chisel3] var width:        Width = 0.W
  private[chisel3] var maxUserWidth: Int = 0

  private case class EnumRecord(inst: Type, name: String)
  private val enumRecords = mutable.ArrayBuffer.empty[EnumRecord]

  private def enumNames = enumRecords.map(_.name).toSeq
  private def enumValues = enumRecords.map(_.inst.litValue).toSeq
  private def enumInstances = enumRecords.map(_.inst).toSeq

  private[chisel3] val enumTypeName = getClass.getName.init

  // Do all bitvectors of this Enum's width represent legal states?
  private[chisel3] def isTotal: Boolean = {
    (this.getWidth < 31) && // guard against Integer overflow
    (enumRecords.size == (1 << this.getWidth))
  }

  def getWidth: Int = width.get

  /** All Enum values */
  def all: Seq[Type] = enumInstances

  /** All Enum names */
  def allNames: Seq[String] = enumNames

  /** All Enum values with their names */
  def allWithNames: Seq[(Type, String)] = all.zip(allNames)

  /** All Enum values with their names, printed one per line. Compatible with gtkwave filter file */
  override def toString(): String = allWithNames.map(e => s"${e._1.litValue} ${e._2}").mkString("", "\n", "\n")

  private[chisel3] def nameOfValue(id: BigInt): Option[String] = {
    enumRecords.find(_.inst.litValue == id).map(_.name)
  }

  protected def do_Value(name: String): Type = {
    val result = new Type

    // We have to use UnknownWidth here, because we don't actually know what the final width will be
    result.bindToLiteral(id, UnknownWidth)

    enumRecords.append(EnumRecord(result, name))

    width = (1.max(id.bitLength).max(maxUserWidth)).W
    id += 1

    result
  }

  protected def do_Value(name: String, id: UInt): Type = {
    // TODO: These throw ExceptionInInitializerError which can be confusing to the user. Get rid of the error, and just
    // throw an exception
    if (id.litOption.isEmpty) {
      throwException(s"$enumTypeName defined with a non-literal type")
    }
    if (id.litValue < this.id) {
      throwException(s"Enums must be strictly increasing: $enumTypeName")
    }

    if (id.isWidthKnown) {
      maxUserWidth = maxUserWidth.max(id.getWidth)
    }
    this.id = id.litValue
    do_Value(name)
  }

  def apply(): Type = new Type

  /** Return the Enum value of which the name exactly matches the specified String. If @name does not match a valid Enum, fail */
  def apply(name: String): Type =
    allWithNames.collectFirst { case (enumValue, enumName) if enumName == name => enumValue }
      .getOrElse(throwException(s"Enum value $name is not defined"))

  private def castImpl(
    n:    UInt,
    warn: Boolean
  )(
    implicit sourceInfo: SourceInfo
  ): Type = {
    if (n.litOption.isDefined) {
      enumInstances.find(_.litValue == n.litValue) match {
        case Some(result) => result
        case None         => throwException(s"${n.litValue} is not a valid value for $enumTypeName")
      }
    } else if (!n.isWidthKnown) {
      throwException(s"Non-literal UInts being cast to $enumTypeName must have a defined width")
    } else if (n.getWidth > this.getWidth) {
      throwException(s"The UInt being cast to $enumTypeName is wider than $enumTypeName's width ($getWidth)")
    } else {
      // TODO fold this into warning filters
      if (!Builder.suppressEnumCastWarning && warn && !this.isTotal) {
        val msg =
          s"Casting non-literal UInt to $enumTypeName. You can use $enumTypeName.safe to cast without this warning."
        Builder.warning(Warning(WarningID.UnsafeUIntCastToEnum, msg))
      }
      val glue = Wire(new UnsafeEnum(width))
      glue := n
      val result = Wire(new Type)
      result := glue
      result
    }
  }

  /** Cast an [[UInt]] to the type of this Enum
    *
    * @note will give a Chisel elaboration time warning if the argument could hit invalid states
    * @param n the UInt to cast
    * @return the equivalent Enum to the value of the cast UInt
    */
  def apply(n: UInt)(implicit sourceInfo: SourceInfo): Type =
    castImpl(n, warn = true)

  /** Safely cast an [[UInt]] to the type of this Enum
    *
    * @param n the UInt to cast
    * @return the equivalent Enum to the value of the cast UInt and a Bool indicating if the
    *         Enum is valid
    */
  def safe(n: UInt)(implicit sourceInfo: SourceInfo): (Type, Bool) = {
    val t = castImpl(n, warn = false)
    (t, t.isValid)
  }
}

// This is an enum type that can be connected directly to UInts. It is used as a "glue" to cast non-literal UInts
// to enums.
private[chisel3] class UnsafeEnum(override val width: Width) extends EnumType(UnsafeEnum) {
  override def cloneType: this.type = new UnsafeEnum(width).asInstanceOf[this.type]
}
private object UnsafeEnum extends ChiselEnum

/** Suppress enum cast warnings
  *
  * Users should use [[ChiselEnum.safe <EnumType>.safe]] when possible.
  *
  * This is primarily used for casting from [[UInt]] to a Bundle type that contains an Enum.
  * {{{
  * class MyBundle extends Bundle {
  *   val addr = UInt(8.W)
  *   val op = OpEnum()
  * }
  *
  * // Since this is a cast to a Bundle, cannot use OpCode.safe
  * val bundle = suppressEnumCastWarning {
  *   someUInt.asTypeOf(new MyBundle)
  * }
  * }}}
  */
object suppressEnumCastWarning {
  def apply[T](block: => T): T = {
    val parentWarn = Builder.suppressEnumCastWarning

    Builder.suppressEnumCastWarning = true

    val res = block // execute block

    Builder.suppressEnumCastWarning = parentWarn
    res
  }
}
