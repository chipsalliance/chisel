// SPDX-License-Identifier: Apache-2.0

package chisel3
package properties

import firrtl.{ir => fir}
import firrtl.annotations.{InstanceTarget, IsMember, ModuleTarget, ReferenceTarget, Target}
import chisel3.internal._
import chisel3.internal.{firrtl => ir}
import chisel3.experimental.{prefix, requireIsHardware, SourceInfo}
import scala.reflect.runtime.universe.{typeOf, TypeTag}
import scala.annotation.{implicitAmbiguous, implicitNotFound}
import scala.collection.immutable.SeqMap
import chisel3.experimental.BaseModule
import chisel3.internal.NamedComponent

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

/** Trait for PropertyTypes that may lookup themselves up during implicit resolution
  *
  * This is to disallow nested Property types e.g. Property[Property[Property[Int]]](), Property[Property[Seq[Property[Int]]]]()
  */
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

/** This contains recursive versions of Seq and SeqMap PropertyTypes. These instances need be lower priority to prevent ambiguous implicit errors with the non-recursive versions.
  */
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
  implicit val doublePropertyTypeInstance =
    makeSimple[Double](_ => fir.DoublePropertyType, fir.DoublePropertyLiteral(_))

  @implicitAmbiguous("unable to infer Property type. Please specify it explicitly in square brackets on the LHS.")
  implicit val classPropertyTypeInstance = makeSimple[ClassType](
    value => fir.ClassPropertyType(value.get.name),
    value => fir.StringPropertyLiteral(value.name)
  )

  implicit val stringPropertyTypeInstance =
    makeSimple[String](_ => fir.StringPropertyType, fir.StringPropertyLiteral(_))

  implicit val boolPropertyTypeInstance =
    makeSimple[Boolean](_ => fir.BooleanPropertyType, fir.BooleanPropertyLiteral(_))

  implicit val pathTypeInstance = makeSimple[Path](value => fir.PathPropertyType, _.convert())

  implicit def modulePathTypeInstance[M <: BaseModule] = new RecursivePropertyType[M] {
    type Type = Path
    override def getPropertyType(value: Option[M]): fir.PropertyType = fir.PathPropertyType
    override def convert(value:         Underlying, ctx: ir.Component, info: SourceInfo): fir.Expression = value.convert()
    type Underlying = Path
    override def convertUnderlying(value: M) = Path(value)
  }

  private def dataPathTypeInstance[D <: Data] = new RecursivePropertyType[D] {
    type Type = Path
    override def getPropertyType(value: Option[D]): fir.PropertyType = fir.PathPropertyType
    override def convert(value:         Underlying, ctx: ir.Component, info: SourceInfo): fir.Expression = value.convert()
    type Underlying = Path
    override def convertUnderlying(value: D) = Path(value)
  }

  // We can't just do <: Data because Property subclasses Data
  implicit def aggregatePathTypeInstance[A <: Aggregate] = dataPathTypeInstance[A]
  implicit def elementPathTypeInstance[E <: Element] = dataPathTypeInstance[E]

  implicit def memPathTypeInstance[M <: MemBase[_]] = new RecursivePropertyType[M] {
    type Type = Path
    override def getPropertyType(value: Option[M]): fir.PropertyType = fir.PathPropertyType
    override def convert(value:         Underlying, ctx: ir.Component, info: SourceInfo): fir.Expression = value.convert()
    type Underlying = Path
    override def convertUnderlying(value: M) = Path(value)
  }

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
final class Property[T] private (tpe: PropertyType[T], valueOpt: Option[T]) extends Data { self =>

  private[chisel3] def _asUIntImpl(first: Boolean)(implicit sourceInfo: SourceInfo): chisel3.UInt = ???
  private[chisel3] def allElements: Seq[Element] = ???
  private[chisel3] def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo): Unit = ???
  private[chisel3] def firrtlConnect(that:   Data)(implicit sourceInfo: SourceInfo): Unit = ???
  def litOption:              Option[BigInt] = ???
  def toPrintable:            Printable = ???
  private[chisel3] def width: ir.Width = ???

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
  override def cloneType: this.type = new Property[T](tpe, valueOpt).asInstanceOf[this.type]

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
    tpe.getPropertyType(valueOpt)
  }

  /** Internal API: returns a ref that can be assigned to, if consistent with the binding.
    */
  private[chisel3] override def lref: ir.Node = {
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
  private[chisel3] override def ref: ir.Arg = {
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

  private[chisel3] def makeWithValueOpt[T](valueOpt: Option[T])(implicit _tpe: PropertyType[T]): Property[T] =
    new Property[T](_tpe, valueOpt)

  /** Create a new Property based on the type T.
    */
  def apply[T]()(implicit tpe: PropertyType[T]): Property[T] = {
    makeWithValueOpt(None)(tpe)
  }

  /** Create a new Property literal of type T.
    */
  def apply[T](lit: T)(implicit tpe: PropertyType[T]): Property[T] = {
    val literal = ir.PropertyLit[T, tpe.Underlying](tpe, tpe.convertUnderlying(lit))
    val result = makeWithValueOpt(None)(tpe)
    literal.bindLitArg(result)
  }
}
