// SPDX-License-Identifier: Apache-2.0

package chisel3.domain

import chisel3.experimental.SourceInfo
import chisel3.util.simpleClassName

object Field {
  sealed trait Type

  /** A boolean type */
  object Boolean extends Type

  /** An integer type */
  object Integer extends Type

  /** A string type */
  object String extends Type
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

}
