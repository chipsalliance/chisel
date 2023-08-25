// SPDX-License-Identifier: Apache-2.0

package chisel3.properties

import chisel3.{ActualDirection, BaseType, MonoConnectException, RawModule, SpecifiedDirection}
import chisel3.internal.{
  checkConnect,
  throwException,
  Binding,
  Builder,
  MonoConnect,
  ObjectFieldBinding,
  PortBinding,
  PropertyValueBinding,
  ReadOnlyBinding,
  TopBinding
}
import firrtl.{ir => fir}
import chisel3.internal.{firrtl => ir}
import chisel3.experimental.{prefix, requireIsHardware, SourceInfo}
import scala.reflect.runtime.universe.{typeOf, TypeTag}
import scala.annotation.{implicitAmbiguous, implicitNotFound}
import scala.collection.immutable.SeqMap

/** PropertyType defines a typeclass for valid Property types.
  *
  * Typeclass instances will be defined for Scala types that can be used as
  * properties. This includes builtin Scala types as well as types defined in
  * Chisel.
  */
@implicitNotFound("unsupported Property type ${T}")
private[chisel3] trait PropertyType[T] {

  /** The property type coreesponding to T. This is the type parameter of the property returned by Property.apply
    */
  type Type

  /** Internal representation of T. This is the value that gets stored and bound in PropertyLit values
    */
  type Underlying

  /** Get the IR PropertyType for this PropertyType.
    */
  def getPropertyType(value: Option[T]): fir.PropertyType

  /** Get convert from the underlying representation to firrtl expression
    */
  def convert(value: Underlying, ctx: ir.Component, info: SourceInfo): fir.Expression

  /** Get convert from the raw type T to this type's internal, underlying representation
    */
  def convertUnderlying(value: T): Underlying
}

private[chisel3] trait RecursivePropertyType[T] extends PropertyType[T]

/** PropertyType where Type and Underlying are the same as T
  */
private[chisel3] trait SimplePropertyType[T] extends RecursivePropertyType[T] {
  final type Type = T
  final type Underlying = T
  def convert(value:           T): fir.Expression
  def convert(value:           T, ctx: ir.Component, info: SourceInfo): fir.Expression = convert(value)
  def convertUnderlying(value: T): T = value
}

private[chisel3] class SeqPropertyType[A, F[A] <: Seq[A], PT <: PropertyType[A]](val tpe: PT)
    extends PropertyType[F[A]] {
  type Type = F[tpe.Type]
  override def getPropertyType(value: Option[F[A]]): fir.PropertyType =
    fir.SequencePropertyType(tpe.getPropertyType(None))

  type Underlying = Seq[tpe.Underlying]
  override def convert(value: Underlying, ctx: ir.Component, info: SourceInfo): fir.Expression =
    fir.SequencePropertyValue(tpe.getPropertyType(None), value.map(tpe.convert(_, ctx, info)))
  override def convertUnderlying(value: F[A]) =
    value.map(tpe.convertUnderlying(_))
}

private[chisel3] class MapPropertyType[A, F[A] <: SeqMap[String, A], PT <: PropertyType[A]](val tpe: PT)
    extends PropertyType[F[A]] {
  type Type = F[tpe.Type]
  override def getPropertyType(value: Option[F[A]]): fir.PropertyType =
    fir.MapPropertyType(tpe.getPropertyType(None))
  type Underlying = Seq[(String, tpe.Underlying)]
  override def convert(value: Underlying, ctx: ir.Component, info: SourceInfo): fir.Expression =
    fir.MapPropertyValue(
      tpe.getPropertyType(None),
      value.map {
        case (key, value) =>
          key -> tpe.convert(value, ctx, info)
      }
    )
  override def convertUnderlying(value: F[A]): Underlying =
    value.toSeq.map { case (k, v) => k -> tpe.convertUnderlying(v) }
}

private[chisel3] trait LowPriorityPropertyTypeInstances {
  implicit def sequencePropertyTypeInstance[A, F[A] <: Seq[A]](implicit tpe: RecursivePropertyType[A]) =
    new SeqPropertyType[A, F, tpe.type](tpe) with RecursivePropertyType[F[A]]

  implicit def mapPropertyTypeInstance[A, F[A] <: SeqMap[String, A]](implicit tpe: RecursivePropertyType[A]) =
    new MapPropertyType[A, F, tpe.type](tpe) with RecursivePropertyType[F[A]]
}

/** Companion object for PropertyType.
  *
  * Typeclass instances for valid Property types are defined here, so they will
  * be in the implicit scope and available for users.
  */
private[chisel3] object PropertyType extends LowPriorityPropertyTypeInstances {
  def makeSimple[T](getType: Option[T] => fir.PropertyType, getExpression: T => fir.Expression): SimplePropertyType[T] =
    new SimplePropertyType[T] {
      def getPropertyType(value: Option[T]): fir.PropertyType = getType(value)
      def convert(value:         T):         fir.Expression = getExpression(value)
    }

  @implicitAmbiguous("unable to infer Property type. Please specify it explicitly in square brackets on the LHS.")
  implicit val intPropertyTypeInstance = makeSimple[Int](_ => fir.IntegerPropertyType, fir.IntegerPropertyLiteral(_))

  @implicitAmbiguous("unable to infer Property type. Please specify it explicitly in square brackets on the LHS.")
  implicit val longPropertyTypeInstance = makeSimple[Long](_ => fir.IntegerPropertyType, fir.IntegerPropertyLiteral(_))

  @implicitAmbiguous("unable to infer Property type. Please specify it explicitly in square brackets on the LHS.")
  implicit val bigIntPropertyTypeInstance =
    makeSimple[BigInt](_ => fir.IntegerPropertyType, fir.IntegerPropertyLiteral(_))

  @implicitAmbiguous("unable to infer Property type. Please specify it explicitly in square brackets on the LHS.")
  implicit val classPropertyTypeInstance = makeSimple[ClassType](
    value => fir.ClassPropertyType(value.get.name),
    value => fir.StringPropertyLiteral(value.name)
  )

  implicit val stringPropertyTypeInstance =
    makeSimple[String](_ => fir.StringPropertyType, fir.StringPropertyLiteral(_))

  implicit val boolPropertyTypeInstance =
    makeSimple[Boolean](_ => fir.BooleanPropertyType, fir.BooleanPropertyLiteral(_))

  implicit def propertyTypeInstance[T](implicit pte: RecursivePropertyType[T]) = new PropertyType[Property[T]] {
    type Type = pte.Type
    override def getPropertyType(value: Option[Property[T]]): fir.PropertyType = pte.getPropertyType(None)
    override def convert(value:         Underlying, ctx: ir.Component, info: SourceInfo): fir.Expression =
      ir.Converter.convert(value, ctx, info)
    type Underlying = ir.Arg
    override def convertUnderlying(value: Property[T]) = value.ref
  }

  implicit def recursiveSequencePropertyTypeInstance[A, F[A] <: Seq[A]](implicit tpe: PropertyType[A]) =
    new SeqPropertyType[A, F, tpe.type](tpe)

  implicit def recursiveMapPropertyTypeInstance[A, F[A] <: SeqMap[String, A]](implicit tpe: PropertyType[A]) =
    new MapPropertyType[A, F, tpe.type](tpe)
}

/** Property is the base type for all properties.
  *
  * Properties are similar to normal Data types in that they can be used in
  * ports, connected to other properties, etc. However, they are used to
  * describe a set of non-hardware types, so they have no width, cannot be used
  * in aggregate Data types, and cannot be connected to Data types.
  */
abstract class Property[T] extends BaseType { self =>
  protected type TT
  protected val tpe: PropertyType[TT]
  protected def value: Option[TT]

  /** Bind this node to the in-memory graph.
    */
  private[chisel3] override def bind(target: Binding, parentDirection: SpecifiedDirection): Unit = {
    this.maybeAddToParentIds(target)
    binding = target
    val resolvedDirection = SpecifiedDirection.fromParent(parentDirection, specifiedDirection)
    direction = ActualDirection.fromSpecified(resolvedDirection)
  }

  /** Clone type by simply constructing a new Property[T].
    */
  override def cloneType: this.type = new Property[T] {
    type TT = self.TT
    val tpe = self.tpe
    val value = self.value
  }.asInstanceOf[this.type]

  /** Clone type with extra information preserved.
    *
    * The only extra information present on a Property type is directionality.
    */
  private[chisel3] override def cloneTypeFull: this.type = {
    val clone = this.cloneType
    clone.specifiedDirection = specifiedDirection
    clone
  }

  /** Get the IR PropertyType for this Property.
    *
    * This delegates to the PropertyType to convert itself to an IR PropertyType.
    */
  private[chisel3] def getPropertyType: fir.PropertyType = {
    tpe.getPropertyType(value)
  }

  /** Connect a source Property[T] to this sink Property[T]
    */
  def :=(source: => Property[T])(implicit sourceInfo: SourceInfo): Unit = {
    prefix(this) {
      this.connect(source)(sourceInfo)
    }
  }

  /** Internal implementation of connecting a source Property[T] to this sink Property[T].
    */
  private def connect(source: Property[T])(implicit sourceInfo: SourceInfo): Unit = {
    requireIsHardware(this, "property to be connected to")
    requireIsHardware(source, "property to be connected from")
    this.topBinding match {
      case _: ReadOnlyBinding => throwException(s"Cannot reassign to read-only $this")
      case _ => // fine
    }

    // Get the BaseModule this connect is occuring within, which may be a RawModule or Class.
    val contextMod = Builder.referenceUserContainer

    try {
      checkConnect(sourceInfo, this, source, contextMod)
    } catch {
      case MonoConnectException(message) =>
        throwException(
          s"Connection between sink ($this) and source ($source) failed @: $message"
        )
    }

    // Add the PropAssign command directly onto the correct BaseModule subclass.
    contextMod match {
      case rm:  RawModule => rm.addCommand(ir.PropAssign(sourceInfo, this.lref, source.ref))
      case cls: Class     => cls.addCommand(ir.PropAssign(sourceInfo, this.lref, source.ref))
      case _ => throwException("Internal Error! Property connection can only occur within RawModule or Class.")
    }
  }

  /** Internal API: returns a ref that can be assigned to, if consistent with the binding.
    */
  private[chisel3] def lref: ir.Node = {
    requireIsHardware(this)
    requireVisible()
    topBindingOpt match {
      case Some(binding: ReadOnlyBinding) =>
        throwException(s"internal error: attempted to generate LHS ref to ReadOnlyBinding $binding")
      case Some(binding: TopBinding) => ir.Node(this)
      case opt => throwException(s"internal error: unknown binding $opt in generating LHS ref")
    }
  }

  /** Internal API: returns a ref, if bound.
    */
  private[chisel3] final def ref: ir.Arg = {
    requireIsHardware(this)
    requireVisible()
    topBindingOpt match {
      case Some(binding: TopBinding) => ir.Node(this)
      case opt => throwException(s"internal error: unknown binding $opt in generating RHS ref")
    }
  }

}

/** Companion object for Property.
  */
object Property {

  private[chisel3] def makeWithValueOpt[T](valueOpt: Option[T])(implicit _tpe: PropertyType[T]): Property[_tpe.Type] = {
    new Property[_tpe.Type] {
      type TT = T
      val tpe = _tpe
      val value = valueOpt
    }
  }

  /** Create a new Property based on the type T.
    */
  def apply[T]()(implicit tpe: PropertyType[T]): Property[tpe.Type] = {
    makeWithValueOpt(None)(tpe)
  }

  /** Create a new Property literal of type T.
    */
  def apply[T](lit: T)(implicit tpe: PropertyType[T]): Property[tpe.Type] = {
    val literal = ir.PropertyLit[tpe.Type, tpe.Underlying](tpe, tpe.convertUnderlying(lit))
    val result = makeWithValueOpt(None)(tpe)
    literal.bindLitArg(result)
  }
}
