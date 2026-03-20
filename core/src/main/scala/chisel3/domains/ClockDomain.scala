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

  override def fields: Seq[(String, Field.Type)] = Seq(
    "name" -> Field.String,
    "synchronousTo" -> Field.String
  )

  /** Create a new clock domain that is asynchronous to all other domains.
    *
    *
    * @param name the name of this domain
    */
  def apply(name: String): chisel3.domain.Type = ClockDomain(Property(name), Property(""))

  /** Create a new clock domain that is derived from another clock domain.
    *
    * This implies that this new domain is synchronous to the other other.
    *
    * @param synchronousTo the domain that this clock domain is derived from
    * @param suffix a naming suffix to apply to this domain
    */
  def derived(synchronousTo: chisel3.domain.Type, suffix: String): chisel3.domain.Type =
    ClockDomain(
      Property.concat(synchronousTo.field.name.asInstanceOf[Property[String]], Property("_div2")),
      synchronousTo.field.name
    )

}
