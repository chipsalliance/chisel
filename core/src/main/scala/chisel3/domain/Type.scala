// SPDX-License-Identifier: Apache-2.0

package chisel3.domain

import chisel3.{fromIntToLiteral, Data, Element, Printable, UInt, UnknownWidth, Width}
import chisel3.experimental.SourceInfo
import chisel3.internal.{throwException, Builder}
import chisel3.properties.Property
import chisel3.internal.firrtl.ir
import scala.language.dynamics

/** Dynamic accessor for domain fields.
  *
  * This class uses Scala's Dynamic trait to enable syntax like `field.fieldName`.
  * It should be created via the `field` method on a domain.Type instance.
  */
final class FieldAccessor private[domain] (domainType: Type, sourceInfo: SourceInfo) extends Dynamic {

  /** Access a field of the domain, returning a Property of the appropriate type.
    *
    * @param fieldName the name of the field to access
    * @return a Property of the appropriate type based on the field's type
    */
  def selectDynamic(fieldName: String): Property[_] = {
    // Find the field in the domain's schema
    domainType.domain.fields.find(_._1 == fieldName) match {
      case Some((_, fieldType)) =>
        // Create the appropriate Property type using the field type's factory method
        val property = fieldType.createProperty()

        // Create a DomainSubfield IR node
        val subfieldArg = ir.DomainSubfield(sourceInfo, domainType.ref(sourceInfo), fieldName, fieldType)

        // Bind it as a property expression
        subfieldArg.bindToProperty(property)

        property
      case None =>
        Builder.error(
          s"Field '$fieldName' does not exist in domain '${domainType.domain.name}'. Available fields: ${domainType.domain.fields.map(_._1).mkString(", ")}"
        )(sourceInfo)
        // Return a dummy property to allow compilation to continue
        Property[String]()
    }
  }
}

/** A [[Data]] that is used to communicate information of a specific domain
  * kind.
  */
final class Type private[domain] (val domain: Domain) extends Element { self =>

  private[chisel3] def _asUIntImpl(first: Boolean)(implicit sourceInfo: SourceInfo): chisel3.UInt = {
    Builder.error(s"${this._localErrorContext} does not support .asUInt.")
    0.U
  }

  override protected def _fromUInt(that: UInt)(implicit sourceInfo: SourceInfo): Data = {
    Builder.exception(s"${this._localErrorContext} cannot be driven by UInt")
  }

  override def cloneType: this.type = new Type(domain).asInstanceOf[this.type]

  override def toPrintable: Printable =
    throwException(s"'domain.Type' does not support hardware printing" + this._errorContext)

  private[chisel3] def width: Width = UnknownWidth

  addDomain(domain)

  /** Accessor object for domain fields using dynamic selection.
    *
    * This allows syntax like `domainPort.field.fieldName` to access domain fields
    * and return Properties of the appropriate type.
    *
    * Example:
    * {{{
    * val A = IO(Input(ClockDomain.Type()))
    * val nameProperty = A.field.name  // Returns Property[String]
    * }}}
    */
  def field(implicit sourceInfo: SourceInfo): FieldAccessor = new FieldAccessor(this, sourceInfo)

}
