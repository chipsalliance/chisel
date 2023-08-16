// SPDX-License-Identifier: Apache-2.0

package chisel3.properties

import chisel3.{ActualDirection, BaseType, MonoConnectException, SpecifiedDirection}
import chisel3.internal.{checkConnect, throwException, Binding, Builder, MonoConnect, ReadOnlyBinding, TopBinding}
import chisel3.internal.{firrtl => ir}
import chisel3.experimental.{prefix, requireIsHardware, SourceInfo}
import scala.reflect.runtime.universe.{typeOf, TypeTag}
import scala.annotation.implicitNotFound

/** PropertyType defines a typeclass for valid Property types.
  *
  * Typeclass instances will be defined for Scala types that can be used as
  * properties. This includes builtin Scala types as well as types defined in
  * Chisel.
  */
@implicitNotFound("unsupported Property type ${T}")
private[chisel3] trait PropertyType[T] {

  /** Get the IR PropertyType for this PropertyType.
    */
  def getPropertyType: ir.PropertyType
}

/** Companion object for PropertyType.
  *
  * Typeclass instances for valid Property types are defined here, so they will
  * be in the implicit scope and available for users.
  */
private[chisel3] object PropertyType {
  implicit val intPropertyTypeInstance = new PropertyType[Int] {
    override def getPropertyType: ir.PropertyType = ir.IntegerPropertyType
  }

  implicit val longPropertyTypeInstance = new PropertyType[Long] {
    override def getPropertyType: ir.PropertyType = ir.IntegerPropertyType
  }

  implicit val bigIntPropertyTypeInstance = new PropertyType[BigInt] {
    override def getPropertyType: ir.PropertyType = ir.IntegerPropertyType
  }

  implicit val stringPropertyTypeInstance = new PropertyType[String] {
    override def getPropertyType: ir.PropertyType = ir.StringPropertyType
  }
}

/** Property is the base type for all properties.
  *
  * Properties are similar to normal Data types in that they can be used in
  * ports, connected to other properties, etc. However, they are used to
  * describe a set of non-hardware types, so they have no width, cannot be used
  * in aggregate Data types, and cannot be connected to Data types.
  */
class Property[T: PropertyType] extends BaseType {

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
  override def cloneType: this.type = new Property[T].asInstanceOf[this.type]

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
  private[chisel3] def getPropertyType: ir.PropertyType = {
    implicitly[PropertyType[T]].getPropertyType
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

    try {
      checkConnect(sourceInfo, this, source, Builder.referenceUserModule)
    } catch {
      case MonoConnectException(message) =>
        throwException(
          s"Connection between sink ($this) and source ($source) failed @: $message"
        )
    }

    Builder.pushCommand(ir.PropAssign(sourceInfo, this.lref, source.ref))
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

  /** Create a new Property based on the type T.
    */
  def apply[T: PropertyType](): Property[T] = {
    new Property[T]
  }

  /** Create a new Property literal of type T.
    */
  def apply[T: PropertyType](lit: T): Property[T] = {
    val literal = ir.PropertyLit[T](lit)
    val result = new Property[T]
    literal.bindLitArg(result)
  }
}
