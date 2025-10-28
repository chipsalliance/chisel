// SPDX-License-Identifier: Apache-2.0

package chisel3.domains

import chisel3.domain.{Domain, Field}
import chisel3.experimental.UnlocatableSourceInfo

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
    "period" -> Field.Integer
  )

}
