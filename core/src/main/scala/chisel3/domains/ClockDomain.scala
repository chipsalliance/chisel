// SPDX-License-Identifier: Apache-2.0

package chisel3.domains

import chisel3.domain.{Domain, Field}
import chisel3.experimental.UnlocatableSourceInfo
import chisel3.properties.Property

/** A Clock Domain
  *
  * This represents a collection of signals that toggle together.  This does not
  * necessarily mean that signals associated with this domain share a clock or
  * will toggle in a predictable way.  I.e., this domain can be used to describe
  * asynchronous signals or static signals (like strap pins).
  */
object ClockDomain extends Domain()(sourceInfo = UnlocatableSourceInfo) {

  /** Types of clock relationships */
  object Relationship {

    /** A clock relationship */
    sealed trait Type {
      override final def toString: String = this.getClass.getSimpleName.stripSuffix("$").toLowerCase

      final def toProperty(): Property[String] = Property(toString)
    }

    /** A synchronous relationship
      *
      * This indicates that two clocks have a deterministic phase relationship
      * and an integer frequency ratio, e.g., `1:1`, `2:1`, or `1:4`.  Both
      * clocks must be derived from the same source via integer multiplication
      * or division, e.g., using a clock divider.
      */
    object Synchronous extends Type

    /** A rational relationship
      *
      * This indicates that two clocks have a deterministic phase relationship
      * and a non-integer rational frequency ratio, e.g., `2:3`.  Both clocks
      * must be derived from a common source clock, e.g., with a
      * phase-locked-loop (PLL).
      */
    object Rational extends Type
  }

  override def fields: Seq[(String, Field.Type)] = Seq(
    "name" -> Field.String,
    "source" -> Field.String,
    "relationship" -> Field.String
  )

  /** Create a new clock domain that is asynchronous to all other domains.
    *
    *
    * @param name the name of this domain
    */
  def apply(name: String): chisel3.domain.Type =
    ClockDomain(Property(name), Property(name), Relationship.Synchronous.toProperty())

  /** Create a derived clock domain.
    *
    * @param synchronousTo the source domain from which this clock domain is derived
    * @param suffix a naming suffix to apply to this domain
    * @see [[synchronous]]
    */
  @deprecated(
    message = "use the more exact `synchronous` or `rational` to create a derived clock",
    since = "Chisel 7.12.0"
  )
  def derived(synchronousTo: chisel3.domain.Type, suffix: String): chisel3.domain.Type =
    synchronous(synchronousTo, suffix)

  /** Create a new clock domain with a synchronous relationship to another clock domain.
    *
    * @param source the domain with which the new domain has a synchronous relationship
    * @param suffix a naming suffix to apply to this domain
    */
  def synchronous(source: chisel3.domain.Type, suffix: String): chisel3.domain.Type =
    ClockDomain(
      Property.concat(source.field.name.asInstanceOf[Property[String]], Property(suffix)),
      source.field.name,
      Relationship.Synchronous.toProperty()
    )

  /** Create a new clock domain with a rational relationship to another clock domain.
    *
    * @param source the domain with which the new domain has a rational relationship
    * @param suffix a naming suffix to apply to this domain
    */
  def rational(source: chisel3.domain.Type, suffix: String): chisel3.domain.Type =
    ClockDomain(
      Property.concat(source.field.name.asInstanceOf[Property[String]], Property(suffix)),
      source.field.name,
      Relationship.Rational.toProperty()
    )

}
