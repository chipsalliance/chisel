// SPDX-License-Identifier: Apache-2.0

package chisel3.domain

import chisel3.experimental.SourceInfo
import chisel3.internal.Builder
import chisel3.internal.firrtl.ir
import chisel3.properties.Property
import chisel3.util.simpleClassName

object Field {
  sealed trait Type {

    /** The Scala type that corresponds to this field type (e.g., scala.Int for Integer) */
    private[domain] type PropertyType

    /** Create a new Property instance of the appropriate type for this field. */
    private[domain] def createProperty(): Property[PropertyType]

    /** The expected FIRRTL PropertyType for this field type */
    private[domain] def expectedPropertyType: _root_.firrtl.ir.PropertyType

    /** The human-readable type name for error messages */
    private[domain] def typeName: scala.Predef.String
  }

  /** A boolean type */
  object Boolean extends Type {
    private[domain] type PropertyType = scala.Boolean
    private[domain] def createProperty():     Property[scala.Boolean] = Property[scala.Boolean]()
    private[domain] def expectedPropertyType: _root_.firrtl.ir.PropertyType = _root_.firrtl.ir.BooleanPropertyType
    private[domain] def typeName:             scala.Predef.String = "Boolean"
  }

  /** An integer type */
  object Integer extends Type {
    private[domain] type PropertyType = scala.Int
    private[domain] def createProperty():     Property[scala.Int] = Property[scala.Int]()
    private[domain] def expectedPropertyType: _root_.firrtl.ir.PropertyType = _root_.firrtl.ir.IntegerPropertyType
    private[domain] def typeName:             scala.Predef.String = "Int"
  }

  /** A string type */
  object String extends Type {
    private[domain] type PropertyType = scala.Predef.String
    private[domain] def createProperty():     Property[scala.Predef.String] = Property[scala.Predef.String]()
    private[domain] def expectedPropertyType: _root_.firrtl.ir.PropertyType = _root_.firrtl.ir.StringPropertyType
    private[domain] def typeName:             scala.Predef.String = "String"
  }
}

/** A [[Domain]] represents a kind information, and the schema of that
  * information, that can be associated with certain hardware in a Chisel
  * design.
  *
  * A domain is intended to represent a specific _kind_ of hardware-related
  * concern that is not captured with the digital, synchronous logic that core
  * Chisel represents.  Examples of domains are clock, reset, and power domains.
  * And while some domains are provided as part of Chisel, domain kinds are
  * intentionall user-extensible.
  *
  * To create a new user-defined domain kind, define an object that extends the
  * [[Domain]] class.  Add fields to define the domain's schema by overridding
  * the `fields` method:
  * {{{
  * import chisel3.domain.{Domain, Field}
  *
  * object FooDomain extends Domain {
  *   override def fields: Seq[(String, Field.Type)] = Seq(
  *     "bar" -> Field.Boolean,
  *     "baz" -> Field.Integer,
  *     "qux" -> Field.String
  *   )
  * }
  * }}}
  *
  * @see [[chisel3.domains.ClockDomain]]
  */
abstract class Domain()(implicit val sourceInfo: SourceInfo) { self: Singleton =>

  // The name that will be used when generating FIRRTL.
  private[chisel3] def name: String = simpleClassName(this.getClass)

  /** A sequence of name--type pairs that define the schema for this domain.
    *
    * The fields comprise the information that a user, after Verilog generation,
    * should set in order to interact with, generate collateral files related
    * to, or check the correctness of their choices for a domain.
    *
    * Alternatively, the fields are the "parameters" for the domain.  E.g., a
    * clock domain could be parameterzied by an integer frequency.  Chisel
    * itself has no knowledge of this frequency, nor does it need a frequency to
    * generate Verilog.  However, in order to generate an implementation
    * constraints file, the user must provide a frequency.
    *
    * To change the fields from the default, override this method in your
    * domain.
    * {{{
    * override def fields: Seq[(String, Field.Type)] = Seq(
    *   "foo" -> Field.Boolean
    * )
    * }}}
    */
  def fields: Seq[(String, Field.Type)] = Seq.empty

  /** Construct a type of this domain kind.
    *
    * For a given domain, this is used to create a Chisel type which can be used
    * in a port.  This is typically used to create domain type ports.
    *
    * E.g., to create a [[chisel3.domains.ClockDomain]] port, use:
    * {{{
    * import chisel3.domains.ClockDomain
    *
    * val A = IO(Input(ClockDomain.Type()))
    * }}}
    */
  final def Type() = new chisel3.domain.Type(this)

  /** Instantiate a domain with specific property values.
    *
    * This creates a domain instance similar to how Module() creates a module instance.
    * The property values must match the domain's field schema in both number and type.
    *
    * This API is intentionally very low level as it allows direct access to the
    * underlying properties which may be "unsafe" depending on how the domain
    * chooses to model information.  Therefore, this function is _not_ public,
    * but expected to be used by a domain developer by their own `apply` methods
    * for creating instances of their domains.
    *
    *
    * @param properties the property values for this domain instance, must match the domain's fields
    * @return a domain.Type representing this instantiated domain
    */
  final protected def apply(
    properties: chisel3.properties.Property[_]*
  )(implicit sourceInfo: SourceInfo): chisel3.domain.Type = {
    // Validate that the number of properties matches the number of fields.
    if (properties.length != fields.length) {
      Builder.error(
        s"Domain instantiation for '${name}' requires ${fields.length} properties but got ${properties.length}. " +
          s"Expected fields: ${fields.map { case (n, t) => s"$n: $t" }.mkString(", ")}"
      )(sourceInfo)
    }

    // Validate that each property type matches the corresponding field type.
    properties.zip(fields).zipWithIndex.foreach { case ((prop, (fieldName, fieldType)), idx) =>
      val actualPropertyType = prop.tpe.getPropertyType()

      if (actualPropertyType != fieldType.expectedPropertyType) {
        val actualTypeName = actualPropertyType.serialize
        Builder.error(
          s"Domain instantiation for '${name}': field '$fieldName' expects Property[${fieldType.typeName}] but got Property[$actualTypeName]"
        )(sourceInfo)
      }
    }

    // Create a new domain.Type instance
    val domainInstance = new chisel3.domain.Type(this)

    // Auto-generate a name for this domain instance
    domainInstance.autoSeed(name.toLowerCase)

    // Bind the domain instance as a read-only operation result in the current module
    val currentModule = Builder.referenceUserModule
    domainInstance.bind(chisel3.internal.binding.OpBinding(currentModule, Builder.currentBlock))

    // Create a DomainInstance command and push it to the Builder
    // Convert properties to their Arg references
    val propertyRefs = properties.map(_.ref(sourceInfo))
    Builder.pushCommand(ir.DomainInstance(sourceInfo, domainInstance, this, propertyRefs.toSeq))

    domainInstance
  }

}
