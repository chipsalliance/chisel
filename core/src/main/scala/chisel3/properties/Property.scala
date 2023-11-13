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
import chisel3.experimental.BaseModule
import chisel3.internal.NamedComponent

/** PropertyType defines a typeclass for valid Property types.
  *
  * Typeclass instances will be defined for Scala types that can be used as
  * properties. This includes builtin Scala types as well as types defined in
  * Chisel.
  */
@implicitNotFound("unsupported Property type ${T}")
sealed trait PropertyType[T] {

  /** The property type coreesponding to T. This is the type parameter of the property returned by Property.apply
    */
  type Type

  /** Internal representation of T. This is the value that gets stored and bound in PropertyLit values
    */
  type Underlying

  /** Get the IR PropertyType for this PropertyType.
    */
  def getPropertyType(): fir.PropertyType

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

private[chisel3] object RecursivePropertyType {

  /** Type alias for simplifying explicit RecursivePropertyType type ascriptions */
  type Aux[T, Type0, Underlying0] = RecursivePropertyType[T] { type Type = Type0; type Underlying = Underlying0 }
}

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
  override def getPropertyType(): fir.PropertyType =
    fir.SequencePropertyType(tpe.getPropertyType())

  type Underlying = Seq[tpe.Underlying]
  override def convert(value: Underlying, ctx: ir.Component, info: SourceInfo): fir.Expression =
    fir.SequencePropertyValue(tpe.getPropertyType(), value.map(tpe.convert(_, ctx, info)))
  override def convertUnderlying(value: F[A]) =
    value.map(tpe.convertUnderlying(_))
}

/** This contains recursive versions of Seq PropertyTypes. These instances need be lower priority to prevent ambiguous implicit errors with the non-recursive versions.
  */
private[chisel3] trait LowPriorityPropertyTypeInstances {
  implicit def sequencePropertyTypeInstance[A, F[A] <: Seq[A]](
    implicit tpe: RecursivePropertyType[A]
  ): SeqPropertyType[A, F, tpe.type] with RecursivePropertyType[F[A]] =
    new SeqPropertyType[A, F, tpe.type](tpe) with RecursivePropertyType[F[A]]
}

private[chisel3] abstract class ClassTypePropertyType[T](val classType: fir.PropertyType)
    extends RecursivePropertyType[T] {
  type Type = ClassType
  override def getPropertyType(): fir.PropertyType = classType
}

private[chisel3] object ClassTypePropertyType {

  /** Type alias for simplifying explicit ClassTypePropertyType type ascriptions */
  type Aux[T, Underlying0] = ClassTypePropertyType[T] { type Underlying = Underlying0 }
}

/** Companion object for PropertyType.
  *
  * Typeclass instances for valid Property types are defined here, so they will
  * be in the implicit scope and available for users.
  */
private[chisel3] object PropertyType extends LowPriorityPropertyTypeInstances {

  /** Type alias for simplifying explicit PropertyType type ascriptions */
  type Aux[T, Type0, Underlying0] = PropertyType[T] { type Type = Type0; type Underlying = Underlying0 }

  def makeSimple[T](tpe: fir.PropertyType, getExpression: T => fir.Expression): SimplePropertyType[T] =
    new SimplePropertyType[T] {
      def getPropertyType(): fir.PropertyType = tpe
      def convert(value: T): fir.Expression = getExpression(value)
    }

  @implicitAmbiguous("unable to infer Property type. Please specify it explicitly in square brackets on the LHS.")
  implicit val intPropertyTypeInstance: SimplePropertyType[Int] =
    makeSimple[Int](fir.IntegerPropertyType, fir.IntegerPropertyLiteral(_))

  @implicitAmbiguous("unable to infer Property type. Please specify it explicitly in square brackets on the LHS.")
  implicit val longPropertyTypeInstance: SimplePropertyType[Long] =
    makeSimple[Long](fir.IntegerPropertyType, fir.IntegerPropertyLiteral(_))

  @implicitAmbiguous("unable to infer Property type. Please specify it explicitly in square brackets on the LHS.")
  implicit val bigIntPropertyTypeInstance: SimplePropertyType[BigInt] =
    makeSimple[BigInt](fir.IntegerPropertyType, fir.IntegerPropertyLiteral(_))

  @implicitAmbiguous("unable to infer Property type. Please specify it explicitly in square brackets on the LHS.")
  implicit val doublePropertyTypeInstance: SimplePropertyType[Double] =
    makeSimple[Double](fir.DoublePropertyType, fir.DoublePropertyLiteral(_))

  implicit def classTypePropertyType[T](
    implicit provider: ClassTypeProvider[T]
  ): ClassTypePropertyType.Aux[T, Nothing] =
    new ClassTypePropertyType[T](provider.classType) {
      // we rely on the fact there are no public constructors for values that provide
      // ClassTypePropertyType (AnyClassType, ClassType#Type, Property[ClassType]#ClassType)
      // so users can never create a property literal of these values so
      // these methods should never be called
      type Underlying = Nothing
      override def convert(value:           Underlying, ctx: ir.Component, info: SourceInfo): fir.Expression = ???
      override def convertUnderlying(value: T) = ???
    }

  implicit val stringPropertyTypeInstance: SimplePropertyType[String] =
    makeSimple[String](fir.StringPropertyType, fir.StringPropertyLiteral(_))

  implicit val boolPropertyTypeInstance: SimplePropertyType[Boolean] =
    makeSimple[Boolean](fir.BooleanPropertyType, fir.BooleanPropertyLiteral(_))

  implicit val pathTypeInstance: SimplePropertyType[Path] = makeSimple[Path](fir.PathPropertyType, _.convert())

  implicit def modulePathTypeInstance[M <: BaseModule]: RecursivePropertyType.Aux[M, Path, Path] =
    new RecursivePropertyType[M] {
      type Type = Path
      override def getPropertyType(): fir.PropertyType = fir.PathPropertyType
      override def convert(value: Underlying, ctx: ir.Component, info: SourceInfo): fir.Expression = value.convert()
      type Underlying = Path
      override def convertUnderlying(value: M) = Path(value)
    }

  private def dataPathTypeInstance[D <: Data]: RecursivePropertyType.Aux[D, Path, Path] = new RecursivePropertyType[D] {
    type Type = Path
    override def getPropertyType(): fir.PropertyType = fir.PathPropertyType
    override def convert(value: Underlying, ctx: ir.Component, info: SourceInfo): fir.Expression = value.convert()
    type Underlying = Path
    override def convertUnderlying(value: D) = Path(value)
  }

  // We can't just do <: Data because Property subclasses Data
  implicit def aggregatePathTypeInstance[A <: Aggregate]: RecursivePropertyType.Aux[A, Path, Path] =
    dataPathTypeInstance[A]
  implicit def elementPathTypeInstance[E <: Element]: RecursivePropertyType.Aux[E, Path, Path] = dataPathTypeInstance[E]

  implicit def memPathTypeInstance[M <: MemBase[_]]: RecursivePropertyType.Aux[M, Path, Path] =
    new RecursivePropertyType[M] {
      type Type = Path
      override def getPropertyType(): fir.PropertyType = fir.PathPropertyType
      override def convert(value: Underlying, ctx: ir.Component, info: SourceInfo): fir.Expression = value.convert()
      type Underlying = Path
      override def convertUnderlying(value: M) = Path(value)
    }

  implicit def propertyTypeInstance[T](
    implicit pte: RecursivePropertyType[T]
  ): PropertyType.Aux[Property[T], pte.Type, ir.Arg] = new PropertyType[Property[T]] {
    type Type = pte.Type
    override def getPropertyType(): fir.PropertyType = pte.getPropertyType()
    override def convert(value: Underlying, ctx: ir.Component, info: SourceInfo): fir.Expression =
      ir.Converter.convert(value, ctx, info)
    type Underlying = ir.Arg
    override def convertUnderlying(value: Property[T]) = value.ref
  }

  implicit def recursiveSequencePropertyTypeInstance[A, F[A] <: Seq[A]](
    implicit tpe: PropertyType[A]
  ): SeqPropertyType[A, F, tpe.type] =
    new SeqPropertyType[A, F, tpe.type](tpe)
}

/** Property is the base type for all properties.
  *
  * Properties are similar to normal Data types in that they can be used in
  * ports, connected to other properties, etc. However, they are used to
  * describe a set of non-hardware types, so they have no width, cannot be used
  * in aggregate Data types, and cannot be connected to Data types.
  */
sealed trait Property[T] extends Data { self =>
  sealed trait ClassType
  private object ClassType {
    implicit def classTypeProvider(
      implicit evidence: T =:= chisel3.properties.ClassType
    ): ClassTypeProvider[ClassType] = ClassTypeProvider(getPropertyType)
    implicit def propertyType(
      implicit evidence: T =:= chisel3.properties.ClassType
    ): ClassTypePropertyType.Aux[Property[ClassType] with self.ClassType, ir.Arg] =
      new ClassTypePropertyType[Property[ClassType] with self.ClassType](getPropertyType) {
        override def convert(value: Underlying, ctx: ir.Component, info: SourceInfo): fir.Expression =
          ir.Converter.convert(value, ctx, info)
        type Underlying = ir.Arg
        override def convertUnderlying(value: Property[ClassType] with self.ClassType) = value.ref
      }
  }

  protected val tpe: PropertyType[_]

  private[chisel3] def _asUIntImpl(first: Boolean)(implicit sourceInfo: SourceInfo): chisel3.UInt = {
    Builder.error(s"${this._localErrorContext} does not support .asUInt.")
    0.U
  }
  private[chisel3] def allElements: Seq[Element] = Nil
  private[chisel3] def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo): Unit = {
    Builder.error(s"${this._localErrorContext} cannot be driven by Bits")
  }
  private[chisel3] def firrtlConnect(that: Data)(implicit sourceInfo: SourceInfo): Unit = {
    that match {
      case pthat: Property[_] => MonoConnect.propConnect(sourceInfo, this, pthat, Builder.forcedUserModule)
      case other => Builder.error(s"${this._localErrorContext} cannot be connected to ${that._localErrorContext}")
    }

  }

  def litOption: Option[BigInt] = None
  def toPrintable: Printable = {
    throwException(s"Properties do not support hardware printing" + this._errorContext)
  }
  private[chisel3] def width: ir.Width = ir.UnknownWidth()

  override def typeName: String = s"Property[${tpe.getPropertyType().serialize}]"

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
    val tpe = self.tpe
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
    tpe.getPropertyType()
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

private[chisel3] sealed trait ClassTypeProvider[A] {
  val classType: fir.PropertyType
}

private[chisel3] object ClassTypeProvider {
  def apply[A](className: String) = new ClassTypeProvider[A] {
    val classType = fir.ClassPropertyType(className)
  }
  def apply[A](_classType: fir.PropertyType) = new ClassTypeProvider[A] {
    val classType = _classType
  }
}

/** Companion object for Property.
  */
object Property {

  implicit class ClassTypePropertyOps(prop: Property[ClassType]) extends AnyRef {
    // This cast should be safe, because there are no members of cls.Type to access
    def as(cls: ClassType): Property[ClassType] with cls.Type =
      prop.asInstanceOf[Property[ClassType] with cls.Type]

    // This cast should be safe, because there are no members of cls.Type to access
    def asAnyClassType: Property[ClassType] with AnyClassType =
      prop.asInstanceOf[Property[ClassType] with AnyClassType]

    // This cast should be safe, because there are no members of prop.ClassType to access
    def as(prop: Property[ClassType]): Property[ClassType] with prop.ClassType =
      prop.asInstanceOf[Property[ClassType] with prop.ClassType]
  }

  private[chisel3] def makeWithValueOpt[T](implicit _tpe: PropertyType[T]): Property[_tpe.Type] = {
    new Property[_tpe.Type] {
      val tpe = _tpe
    }
  }

  /** Create a new Property based on the type T.
    */
  def apply[T]()(implicit tpe: PropertyType[T]): Property[tpe.Type] = {
    makeWithValueOpt(tpe)
  }

  /** Create a new Property literal of type T.
    */
  def apply[T](lit: T)(implicit tpe: PropertyType[T]): Property[tpe.Type] = {
    val literal = ir.PropertyLit[tpe.Type, tpe.Underlying](tpe, tpe.convertUnderlying(lit))
    val result = makeWithValueOpt(tpe)
    literal.bindLitArg(result)
  }
}
