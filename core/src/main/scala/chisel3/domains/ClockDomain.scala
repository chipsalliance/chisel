// SPDX-License-Identifier: Apache-2.0

package chisel3.domains

import chisel3.domain.{Domain, Field}
import chisel3.experimental.UnlocatableSourceInfo

object ClockDomain extends Domain()(sourceInfo = UnlocatableSourceInfo) {

  override def fields: Seq[(String, Field.Type)] = Seq(
    "name" -> Field.String
  )

}
